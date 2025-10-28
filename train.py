import os
from typing import Iterable
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from config import get_default_config, reset_some_value
from model import build_model, build_discriminator
from argparse import ArgumentParser
from data import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR
from loss import LossModule
from validate import validate_model
import tqdm
import sys
from utils import EMAMeter


def parse_args():
    parser = ArgumentParser("HS Super Resolution Training Script")
    parser.add_argument('--shm', action="store_true", help="Using shared memory data")
    parser.add_argument("--cfg", type=str, default=None, help="config file to merge")
    parser.add_argument('-d', "--dataset_root", type=str, default=None, help="dataset_root")
    args = parser.parse_args()
    return args


def setup(rank, world_size, port:int = None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' if port is None else str(port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_opt_scheduler(cfg, model: nn.Module):
    params = (p for p in model.parameters() if p.requires_grad)
    if cfg.TRAIN.OPTIMIZER == "adam":
        opt = torch.optim.Adam(params, lr=cfg.TRAIN.LR, betas=[0.5, 0.5])
    elif cfg.TRAIN.OPTIMIZER == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.TRAIN.LR)
    else:
        raise NotImplementedError

    if cfg.TRAIN.SCHEDULER == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg.TRAIN.ITERS, eta_min=1e-8
        )
    elif cfg.TRAIN.SCHEDULER == "multi_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, [cfg.TRAIN.ITERS // 2, cfg.TRAIN.ITERS * 4 // 5], gamma=0.1
        )
    else:
        raise NotImplementedError
    
    if cfg.TRAIN.WARMUP:
        warm_up_sche = torch.optim.lr_scheduler.LinearLR(opt, 0.3, 1.0, total_iters=5000)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [warm_up_sche, scheduler], milestones=[5000])

    return opt, scheduler

def train_on_one_dev(rank, world_size, cfg):
    torch.set_float32_matmul_precision('high')
    # build model to be trained
    model = build_model(cfg)
    model.to(rank)
    
    use_multiple_gpu = world_size > 1

    # load pretrained model
    if len(cfg.MODEL.PRETRAIN) > 0:
        model.load_pretrained(cfg.MODEL.PRETRAIN)
    
    # load saved_ckpt
    if len(cfg.MODEL.LOAD) > 0:
        ckpt = torch.load(cfg.MODEL.LOAD)["model_state"]
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt.items()})
        
    if len(cfg.MODEL.LOAD_BB) > 0:
        model.load_and_freeze_backbone(cfg.MODEL.LOAD_BB)

    # if cfg.RESUME:
        
    
    if use_multiple_gpu:
        # model = torch.nn.DataParallel(model, [0,1]) # legacy, use distributed data parallel instead
        setup(rank, world_size)
        model = DDP(model, [rank], find_unused_parameters=True)

    # model = torch.compile(model)
    model.train()
    
    if "ADV" in cfg.TRAIN.LOSS and "GAN" in cfg.MODEL.TYPE:
        discrim_net = build_discriminator(cfg)
        discrim_net.to(rank)
        discrim_net.train()
    else:
        discrim_net = None

    # build data loaders
    train_loader, valid_loader = get_dataloader(cfg, rank=rank, world_size=world_size)

    # build summary writer
    writer = SummaryWriter(log_dir=cfg.TRAIN.LOG) if rank == 0 else None

    # build loss functions
    loss_mod = LossModule(cfg)
    loss_mod.to(rank)
    
    # build optimizer
    opt, scheduler = get_opt_scheduler(cfg, model)
    if discrim_net is not None:
        opt_D, scheduler_D = get_opt_scheduler(cfg, discrim_net)
    else:
        opt_D, scheduler_D = None, None

    swa_model = None
    if cfg.TRAIN.SWA.START >= 0:
        # swa_start = cfg.TRAIN.SWA.START
        # swa_decay = cfg.TRAIN.SWA.DECAY
        # swa_model = AveragedModel(model)        
        swa_lr = SWALR(optimizer=opt, swa_lr=cfg.TRAIN.SWA.LR)
    

    train_iter = iter(train_loader)
    best_psnr = 0.0
    loss_dict_ema = {}

    for global_step in tqdm.tqdm(range(1, cfg.TRAIN.ITERS + 1)):
        
        try:
            data = next(train_iter)
        except StopIteration:
            if hasattr(train_loader.dataset, 'set_use_cache'):
                train_loader.dataset.set_use_cache(True)
            train_iter = iter(train_loader)
            data = next(train_iter)

        hs_tensor, rgb_tensor = data
        hs_tensor = hs_tensor.to(rank)
        rgb_tensor = rgb_tensor.to(rank)

        pred_hs = model(rgb_tensor)

        if type(pred_hs) in (tuple, list):
            pred_hs, error_feat = pred_hs
        else:
            error_feat = None

        loss_dict = loss_mod(pred_hs, hs_tensor, discrim_net, error_feat)
        loss_all = loss_dict["ALL"]
        # if D exist, do not update
        if isinstance(discrim_net, nn.Module):
            for param in discrim_net.parameters():
                param.requires_grad = False
        opt.zero_grad()
        loss_all.backward()

        # add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        opt.step()          

        scheduler.step()
        # Then update Discriminator Networks
        loss_D = None
        if discrim_net is not None:
            assert opt_D is not None
            assert scheduler_D is not None
            if isinstance(discrim_net, nn.Module):
                for param in discrim_net.parameters():
                    param.requires_grad = True
            loss_D = loss_mod.get_GAN_D_loss(pred_hs, hs_tensor, discrim_net)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            scheduler_D.step()

        # average the loss_dict    
        for name, value in loss_dict.items():
            if not name in loss_dict_ema.keys():
                loss_dict_ema[name] = EMAMeter()                
            loss_dict_ema[name].update(value.detach().item())

        # log the losses
        if global_step % 500 == 0 and rank == 0:
            for loss_name, loss_value_ema in loss_dict_ema.items():
                loss_value = loss_value_ema.avg
                writer.add_scalar(
                    "LOSS_%s" % loss_name, loss_value, global_step=global_step
                )
            if loss_D is not None:
                writer.add_scalar(
                    "LOSS_%s" % "ADV_D", loss_D, global_step=global_step
                )
            writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step=global_step)
        
        # SWA model
        if global_step % 1000 == 0 and cfg.TRAIN.SWA.START >= 0 and global_step > cfg.TRAIN.SWA.START:
            if swa_model is None:
                swa_model = AveragedModel(model)
            swa_model.update_parameters(model)
            swa_lr.step(global_step)

        # validate cycle
        if global_step % 1000 == 0 and rank == 0:
            # validate_model
            valid_model = swa_model if cfg.TRAIN.SWA.START > 0 and global_step > cfg.TRAIN.SWA.START else model
            valid_psnr = validate_model(cfg, valid_model, valid_loader, writer, global_step=global_step, pad=cfg.VALIDATE_PAD)
            if valid_psnr > best_psnr:
                if cfg.TRAIN.SWA.START > 0 and global_step > cfg.TRAIN.SWA.START:
                    model_save = swa_model.module
                else:
                    model_save = model
                if use_multiple_gpu:
                    state = model_save.module.state_dict()
                else:
                    state = model_save.state_dict()
                save_state = {"model_state":state, "global_step": global_step}
                torch.save(save_state, os.path.join(cfg.TRAIN.LOG, "BestPSNR.pth"))

            model.train()
            if hasattr(valid_loader.dataset, 'set_use_cache'):
                valid_loader.dataset.set_use_cache(True)


        # save model
        if global_step % 3000 == 0 and rank == 0:
            if cfg.TRAIN.SWA.START > 0 and global_step > cfg.TRAIN.SWA.START:
                model_save = swa_model.module
            else:
                model_save = model
            if use_multiple_gpu:
                state = model_save.module.state_dict()
            else:
                state = model_save.state_dict()
            save_state = {"model_state":state, "global_step": global_step}
            torch.save(save_state, os.path.join(cfg.TRAIN.LOG, "Iter%05d.pth"%(global_step // 1000)))
    


def main():
    cfg = get_default_config()
    args = parse_args()
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
    if args.dataset_root is not None:
        cfg.DATASET.ROOT = args.dataset_root
    cfg = reset_some_value(cfg)
    # print(cfg)
    torch.set_float32_matmul_precision('high') # just for no warnings
    # legacy
    gpu_num = torch.cuda.device_count()
    use_multiple_gpu = True if gpu_num > 1 else False
    
    # save the config file from the main process
    yaml_text = str(cfg)
    os.makedirs(cfg.TRAIN.LOG, exist_ok=True)
    with open(os.path.join(cfg.TRAIN.LOG, 'config.yaml'), 'w') as f:
        f.writelines(yaml_text)

    if use_multiple_gpu:
        mp.spawn(train_on_one_dev, args=(gpu_num,cfg,), nprocs=gpu_num, join=True)
    else:
        world_size = 1 if args.shm else -1
        train_on_one_dev(0, world_size, cfg)


if __name__ == "__main__":
    # def trace(frame, event, arg):
    #     print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    #     return trace
    # sys.settrace(trace)
    main()
