import math
import os
from unittest import result
import cv2
from typing import Iterable, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchmetrics.functional as tmf
import torchmetrics.functional.image as tmfi
import logging
from config import get_default_config, reset_some_value
from data import get_valid_load
from model import build_model
from yacs.config import CfgNode as CN
import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import io
import warnings
import gc
from torch.utils.tensorboard._utils import figure_to_image
from utils import infer_model_by_patchPG
from metrics import get_metrics
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Validation script")
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Saving directory of the output NPY files.",
    )
    parser.add_argument(
        "--saveerr",
        type=str,
        default=None,
        help="Saving directory of the histo/coor images.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="The validation dataset to be loaded"
    )

    parser.add_argument("--load", type=str)
    args = parser.parse_args()
    return args


def mean_non_zero_attn(attn: torch.Tensor, thres: float = 0.05):
    b, p, dict_size = attn.shape
    attn = attn.reshape(b * p, dict_size)
    non_zero = attn > thres
    mean_non_zero = torch.mean(torch.sum(non_zero.to(torch.float32), dim=1))
    return mean_non_zero


def unnormalize_hs(hs_tensor, hs_mean, hs_std):
    """_summary_

    Args:
        hs_mean (_type_): _description_
        hs_std (_type_): _description_
    """
    n_channel = hs_tensor.shape[1]
    hs_mean = (
        torch.from_numpy(
            np.array(hs_mean).astype(np.float32).reshape(1, n_channel, 1, 1)
        ).to(hs_tensor.device)
        if isinstance(hs_mean, Iterable)
        else hs_mean
    )

    hs_std = (
        torch.from_numpy(
            np.array(hs_std).astype(np.float32).reshape(1, n_channel, 1, 1)
        ).to(hs_tensor.device)
        if isinstance(hs_std, Iterable)
        else hs_std
    )

    out = hs_tensor * hs_std + hs_mean
    out = torch.clamp(out, 0.0, 1.0)
    return out


def draw_histogram_err(arr_list: List[np.ndarray], save_path: str = None):
    arr_all = np.concatenate([a.reshape(-1) for a in arr_list])
    figure = plt.figure()
    plt.hist(arr_all, bins=1000)
    plt.savefig(save_path)
    # img = figure_to_image(figure, close=True)
    # if save_path is not None:
    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(save_path, img_bgr)
    return None


def draw_scatter_point(hs_err_list, feat_err_list, save_path):
    arr_err_all = np.concatenate([a.reshape(-1) for a in hs_err_list])
    arr_feat_all = np.concatenate([a.reshape(-1) for a in feat_err_list])

    # compute the cov
    pd_efr = pd.Series(arr_feat_all)
    pd_ehr = pd.Series(arr_err_all)
    for met in ['kendall', 'spearman', 'pearson']:
        coor = pd_efr.corr(pd_ehr, method=met)
        print(met, ' CORR: ', coor)

    assert len(arr_err_all) == len(arr_feat_all)

    fig = plt.figure()
    ax = plt.axes()
    plt.hist2d(
        arr_err_all,
        arr_feat_all,
        bins=50,
        cmin=0,
        cmap="jet",
        # norm=mpl.colors.LogNorm(),
        range=((0, 0.2), (0.0, 0.08))
    )

    # sctt = ax.scatter(arr_err_all, arr_feat_all, cmap='jet')
    fontdict={'family' : 'CMU Serif', 'size'   : 14}
    ax.set_xlabel("HSI Reconstruction RMSE", fontdict=fontdict)
    ax.set_ylabel("Uncertainty Score", fontdict=fontdict)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    plt.colorbar(ax=ax, shrink=0.5, aspect=10)
    plt.savefig(save_path)
    # img = figure_to_image(fig, close=True)
    # if save_path is not None:
    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(save_path, img_bgr)
    img = None
    return img

def _visualize_efrehr(in_tensor, norm_factor_max = 0.2):
    assert isinstance(in_tensor, np.ndarray)
    in_tensor = np.clip(in_tensor, 0, norm_factor_max) / norm_factor_max # cut out the big values, does not count
    in_tensor = np.clip(in_tensor * 255, 0, 255).astype(np.uint8)
    in_color = cv2.applyColorMap(in_tensor, cv2.COLORMAP_JET).astype(np.float32) / 255.0
    # in_color = (in_tensor > 0.1).astype(np.float32)
    in_color = cv2.cvtColor(in_color, cv2.COLOR_BGR2RGB)
    return in_color

@torch.no_grad()
def validate_model(
    cfg: CN,
    model: nn.Module,
    data_loader: DataLoader,
    writer: SummaryWriter = None,
    global_step: int = 0,
    pad: Tuple[int] = (0, 0),
    save_dir: str = None,
    save_err_img: str = None,
    save_vis_img: str = None,
):
    if save_dir is not None:  # stands for save function
        os.makedirs(save_dir, exist_ok=True)
    # Hate the warnings from torchmetrics
    warnings.filterwarnings("ignore", category=FutureWarning)
    model.eval()
    function_dict = {
        "mrae": tmf.mean_absolute_percentage_error,
        "sam": tmfi.spectral_angle_mapper,
        "psnr": tmfi.peak_signal_noise_ratio,
        "ssim": tmf.structural_similarity_index_measure,
        "uiqi": tmfi.universal_image_quality_index,
        "ergas": tmfi.error_relative_global_dimensionless_synthesis,
        # "mse": tmf.mean_squared_error,
        # "mean_nonzero": mean_non_zero_attn,
        # "mean_nonzero": mean_non_zero_attn,
    }

    record_idx = (13, 19, 17)

    use_padding = np.any(np.array(pad) > 0) and len(pad) == 4
    adapt_padding = np.all(np.array(pad) < 0) and len(pad) == 4
    result_dict = {}
    for key in function_dict.keys():
        result_dict[key] = list()
    result_dict['mean_us'] = list()
    result_dict['area_us'] = list()

    print("=========Validating=========")
    save_count = 0
    vad_count = 0
    valid_record = None

    error_feat_list = []
    error_hsi_list = []

    for hs_tensor, rgb_tensor, *mask_tensor_list in tqdm.tqdm(data_loader):
        hs_tensor = hs_tensor.cuda()
        rgb_tensor = rgb_tensor.cuda()
        # print(rgb_tensor.shape)
        if len(mask_tensor_list) > 0:
            mask_tensor = mask_tensor_list[0].cuda()
        else:
            mask_tensor = None

        if use_padding:
            rgb_tensor = torch.nn.functional.pad(rgb_tensor, pad, "constant", 0.0)
            # print(rgb_tensor.shape)
        elif adapt_padding:
            h_r, w_r = rgb_tensor.shape[2:4]
            h_p, w_p = [math.ceil(dim / 16.0) * 16 - dim for dim in (h_r, w_r)]
            pad = [w_p // 2, (w_p - w_p // 2), h_p // 2, (h_p - h_p // 2)]
            rgb_tensor = torch.nn.functional.pad(rgb_tensor, pad, "constant", 0.0)

        if cfg.VALIDATE_PATCH > 0:
            hs_pred = infer_model_by_patchPG(model, rgb_tensor, cfg.VALIDATE_PATCH)
        else:
            hs_pred = model(rgb_tensor)

        if type(hs_pred) in (list, tuple):
            hs_pred, err_feat = hs_pred
        else:
            err_feat = None

        if use_padding or adapt_padding:
            h_, w_ = hs_pred.shape[2], hs_pred.shape[3]
            hs_pred = hs_pred[:, :, pad[2] : (h_ - pad[3]), pad[0] : (w_ - pad[1])]
            if err_feat is not None:
                err_feat = err_feat[
                    :, :, pad[2] : (h_ - pad[3]), pad[0] : (w_ - pad[1])
                ]
            rgb_tensor = rgb_tensor[
                :, :, pad[2] : (h_ - pad[3]), pad[0] : (w_ - pad[1])
            ]

        # before unnormalize
        if save_err_img is not None and vad_count in record_idx:
            if err_feat is not None:
                err_norm_2 = torch.linalg.norm(err_feat, ord=2, dim=1, keepdim=True)
                error_feat_list.append(err_norm_2.detach().cpu().numpy())
            err_hs = torch.linalg.norm(hs_tensor - hs_pred, ord=2, dim=1, keepdim=True)
            error_hsi_list.append(err_hs.detach().cpu().numpy())

        # unnormalized teh tensors
        hs_pred = unnormalize_hs(
            hs_pred,
            cfg.DATASET.TRANSFORMS.NORMAL_HS_MEAN,
            cfg.DATASET.TRANSFORMS.NORMAL_HS_STD,
        )
        hs_tensor = unnormalize_hs(
            hs_tensor,
            cfg.DATASET.TRANSFORMS.NORMAL_HS_MEAN,
            cfg.DATASET.TRANSFORMS.NORMAL_HS_STD,
        )
        rgb_tensor = unnormalize_hs(
            rgb_tensor,
            cfg.DATASET.TRANSFORMS.NORMAL_RGB_MEAN,
            cfg.DATASET.TRANSFORMS.NORMAL_RGB_STD,
        )

        # if vad_count in record_idx and writer is not None:
        #     valid_record = add_one_sample_val(
        #         valid_record, rgb_tensor, hs_tensor, hs_pred, err_feat
        #     )

        if save_dir is not None and isinstance(hs_pred, torch.Tensor):
            hs_pred_save = hs_pred[0].detach().cpu().clamp(0, 1).numpy()
            hs_pred_save = np.transpose(hs_pred_save, (0, 2, 1))  # 31, 512, 482
            hs_pred_all_valid = np.all(np.linalg.norm(hs_pred_save, axis=0) > 0)
            # print(f'output: {hs_pred_all_valid}')
            np.save(os.path.join(save_dir, "%05d.npy" % save_count), hs_pred_save)

            hsierr_norm_2 = torch.linalg.norm(
                hs_pred - hs_tensor, ord=2, dim=1, keepdim=False
            )
            hsierr_norm_2 = hsierr_norm_2[0].detach().cpu().numpy()
            
            err_norm_2 = [hsierr_norm_2]

            if err_feat is not None:
                err_feat_n2 = torch.linalg.norm(err_feat, ord=2, dim=1, keepdim=False)
                err_feat_n2 = err_feat_n2[0].detach().cpu().numpy()
                err_norm_2.append(err_feat_n2)

                # mean norm 2
                result_dict['mean_us'].append(np.mean(err_feat_n2))
                result_dict['area_us'].append(np.mean((err_feat_n2 > 0.1).astype(np.float32)))

            rgb_np = rgb_tensor[0].detach().cpu().numpy()
            rgb_np = np.transpose(rgb_np, (1, 2, 0)).astype(np.float32)
            rgb_h = rgb_np * 0.5

            ## this part for binary visualization
            # abv_norm_2 = [n2 > 0.1 for n2 in err_norm_2]
            # err_vis = [
            #     (
            #         np.stack(
            #             [
            #                 abv.astype(np.float32),
            #                 np.zeros_like(abv),
            #                 np.zeros_like(abv),
            #             ],
            #             axis=-1,
            #         )
            #         * 0.5
            #         + rgb_h
            #     )
            #     for abv in abv_norm_2
            # ]
            err_vis = [
                _visualize_efrehr(tt, 0.2) for tt in err_norm_2
            ]
            
            
            err_vis.append(rgb_np)
            err_vis_uint8 = [
                np.round(vis * 255).clip(0, 255).astype(np.uint8) for vis in err_vis
            ]
            cv2.imwrite(
                os.path.join(save_dir, "%04d_rgb.png" % vad_count),
                cv2.cvtColor(err_vis_uint8[-1], cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(save_dir, "%04d_ehr.png" % vad_count),
                cv2.cvtColor(err_vis_uint8[0], cv2.COLOR_RGB2BGR),
            )
            if len(err_vis_uint8) > 2:
                cv2.imwrite(
                    os.path.join(save_dir, "%04d_efr.png" % vad_count),
                    cv2.cvtColor(err_vis_uint8[1], cv2.COLOR_RGB2BGR),
                )

            save_count += 1

        for name, func in function_dict.items():
            # if name == "mean_nonzero":
            #     # if attn is not None:
            #     #     metric = func(attn, 0.05)
            #     else:
            #         continue
            # else:
            hs_pred = hs_pred.detach().clone()
            hs_tensor = hs_tensor.detach().clone()
            metric = func(hs_pred, hs_tensor)
            metric = metric.detach().cpu().item()
            if not np.isnan(metric):
                result_dict[name].append(metric)
            else:
                print(f'Idx: {vad_count}, metric: {name} is NAN')

        vad_count += 1

    if valid_record is not None and writer is not None:
        writer.add_image("VAL_ERR", valid_record, global_step, dataformats="HWC")

    # valid_record = np.round(valid_record * 255).astype(np.uint8)
    # valid_record = cv2.cvtColor(valid_record, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("Errors.png", valid_record)

    logging_info = "Metrics at global step %d:" % global_step
    for name, v_list in result_dict.items():
        assert isinstance(name, str)
        if len(v_list) == 0:
            continue
        value = np.mean(v_list)
        logging_info += "\t%s: %.4f" % (name.upper(), value)
        if writer is not None:
            writer.add_scalar("VAL_" + name.upper(), value, global_step=global_step)

    sorted_psnr = np.sort(result_dict["psnr"])
    top30_psnr = np.mean(sorted_psnr[25:])
    last20_psnr = np.mean(sorted_psnr[:25])
    logging_info += "  TOP25PSNR: %.4f, LAST25: %.4f" % (top30_psnr, last20_psnr)
    print(logging_info)

    if save_dir is not None:
        np.save(os.path.join(save_dir, "metrics.npy"), result_dict)

    if save_err_img is not None:
        if len(error_feat_list) > 0:
            img_err_feat = draw_histogram_err(
                error_feat_list, os.path.join(save_err_img, "Feat_Error_Hist.png")
            )
        if len(error_hsi_list) > 0:
            img_err_hsi = draw_histogram_err(
                error_hsi_list, os.path.join(save_err_img, "HSI_Error_Hist.png")
            )
        if len(error_feat_list) > 0 and len(error_hsi_list) > 0:
            img_cor = draw_scatter_point(
                error_hsi_list,
                error_feat_list,
                os.path.join(save_err_img, "FEAT_HIS_COOR.png"),
            )
    
    # return the PSNR as validation
    return np.mean(result_dict["psnr"])


def add_one_sample_val(
    valid_record: torch.Tensor,
    rgb_tensor: torch.Tensor,
    hs_tensor: torch.Tensor,
    hs_pred: torch.Tensor,
    err_feat: torch.Tensor,
):
    hs_err = torch.linalg.norm(hs_tensor - hs_pred, ord=2, dim=1, keepdim=True)
    if err_feat is not None:
        feat_err = torch.linalg.norm(err_feat, ord=2, dim=1, keepdim=True)
    else:
        feat_err = None

    # hs_err = torch.log(hs_err + 1.0)
    # feat_err = torch.log(feat_err + 1.0)

    hs_hmnp = draw_heatmap_th(hs_err, scale=0.65)
    rgb_tensor = rgb_tensor.detach().cpu().numpy()[0]
    rgb_tensor = np.transpose(rgb_tensor, (1, 2, 0))
    rgb_tensor = np.pad(
        rgb_tensor, ((2, 2), (2, 2), (0, 0)), mode="constant", constant_values=1.0
    )
    if feat_err is not None:
        feat_hmnp = draw_heatmap_th(feat_err, scale=1.0)
        # print(rgb_tensor.shape,hs_hmnp.shape, feat_hmnp.shape)
        one_sample = np.concatenate((rgb_tensor, hs_hmnp, feat_hmnp), axis=1)
    else:
        one_sample = np.concatenate((rgb_tensor, hs_hmnp), axis=1)

    if valid_record is None:
        valid_record = one_sample
    else:
        valid_record = np.concatenate((valid_record, one_sample), axis=0)

    return valid_record


def draw_heatmap_th(ii: torch.Tensor, pad: int = 2, scale: float = 5.0):
    cmap_np = [
        0,
        0,
        0.515625000000000,
        0,
        0,
        0.531250000000000,
        0,
        0,
        0.546875000000000,
        0,
        0,
        0.562500000000000,
        0,
        0,
        0.578125000000000,
        0,
        0,
        0.593750000000000,
        0,
        0,
        0.609375000000000,
        0,
        0,
        0.625000000000000,
        0,
        0,
        0.640625000000000,
        0,
        0,
        0.656250000000000,
        0,
        0,
        0.671875000000000,
        0,
        0,
        0.687500000000000,
        0,
        0,
        0.703125000000000,
        0,
        0,
        0.718750000000000,
        0,
        0,
        0.734375000000000,
        0,
        0,
        0.750000000000000,
        0,
        0,
        0.765625000000000,
        0,
        0,
        0.781250000000000,
        0,
        0,
        0.796875000000000,
        0,
        0,
        0.812500000000000,
        0,
        0,
        0.828125000000000,
        0,
        0,
        0.843750000000000,
        0,
        0,
        0.859375000000000,
        0,
        0,
        0.875000000000000,
        0,
        0,
        0.890625000000000,
        0,
        0,
        0.906250000000000,
        0,
        0,
        0.921875000000000,
        0,
        0,
        0.937500000000000,
        0,
        0,
        0.953125000000000,
        0,
        0,
        0.968750000000000,
        0,
        0,
        0.984375000000000,
        0,
        0,
        1,
        0,
        0.0156250000000000,
        1,
        0,
        0.0312500000000000,
        1,
        0,
        0.0468750000000000,
        1,
        0,
        0.0625000000000000,
        1,
        0,
        0.0781250000000000,
        1,
        0,
        0.0937500000000000,
        1,
        0,
        0.109375000000000,
        1,
        0,
        0.125000000000000,
        1,
        0,
        0.140625000000000,
        1,
        0,
        0.156250000000000,
        1,
        0,
        0.171875000000000,
        1,
        0,
        0.187500000000000,
        1,
        0,
        0.203125000000000,
        1,
        0,
        0.218750000000000,
        1,
        0,
        0.234375000000000,
        1,
        0,
        0.250000000000000,
        1,
        0,
        0.265625000000000,
        1,
        0,
        0.281250000000000,
        1,
        0,
        0.296875000000000,
        1,
        0,
        0.312500000000000,
        1,
        0,
        0.328125000000000,
        1,
        0,
        0.343750000000000,
        1,
        0,
        0.359375000000000,
        1,
        0,
        0.375000000000000,
        1,
        0,
        0.390625000000000,
        1,
        0,
        0.406250000000000,
        1,
        0,
        0.421875000000000,
        1,
        0,
        0.437500000000000,
        1,
        0,
        0.453125000000000,
        1,
        0,
        0.468750000000000,
        1,
        0,
        0.484375000000000,
        1,
        0,
        0.500000000000000,
        1,
        0,
        0.515625000000000,
        1,
        0,
        0.531250000000000,
        1,
        0,
        0.546875000000000,
        1,
        0,
        0.562500000000000,
        1,
        0,
        0.578125000000000,
        1,
        0,
        0.593750000000000,
        1,
        0,
        0.609375000000000,
        1,
        0,
        0.625000000000000,
        1,
        0,
        0.640625000000000,
        1,
        0,
        0.656250000000000,
        1,
        0,
        0.671875000000000,
        1,
        0,
        0.687500000000000,
        1,
        0,
        0.703125000000000,
        1,
        0,
        0.718750000000000,
        1,
        0,
        0.734375000000000,
        1,
        0,
        0.750000000000000,
        1,
        0,
        0.765625000000000,
        1,
        0,
        0.781250000000000,
        1,
        0,
        0.796875000000000,
        1,
        0,
        0.812500000000000,
        1,
        0,
        0.828125000000000,
        1,
        0,
        0.843750000000000,
        1,
        0,
        0.859375000000000,
        1,
        0,
        0.875000000000000,
        1,
        0,
        0.890625000000000,
        1,
        0,
        0.906250000000000,
        1,
        0,
        0.921875000000000,
        1,
        0,
        0.937500000000000,
        1,
        0,
        0.953125000000000,
        1,
        0,
        0.968750000000000,
        1,
        0,
        0.984375000000000,
        1,
        0,
        1,
        1,
        0.0156250000000000,
        1,
        0.984375000000000,
        0.0312500000000000,
        1,
        0.968750000000000,
        0.0468750000000000,
        1,
        0.953125000000000,
        0.0625000000000000,
        1,
        0.937500000000000,
        0.0781250000000000,
        1,
        0.921875000000000,
        0.0937500000000000,
        1,
        0.906250000000000,
        0.109375000000000,
        1,
        0.890625000000000,
        0.125000000000000,
        1,
        0.875000000000000,
        0.140625000000000,
        1,
        0.859375000000000,
        0.156250000000000,
        1,
        0.843750000000000,
        0.171875000000000,
        1,
        0.828125000000000,
        0.187500000000000,
        1,
        0.812500000000000,
        0.203125000000000,
        1,
        0.796875000000000,
        0.218750000000000,
        1,
        0.781250000000000,
        0.234375000000000,
        1,
        0.765625000000000,
        0.250000000000000,
        1,
        0.750000000000000,
        0.265625000000000,
        1,
        0.734375000000000,
        0.281250000000000,
        1,
        0.718750000000000,
        0.296875000000000,
        1,
        0.703125000000000,
        0.312500000000000,
        1,
        0.687500000000000,
        0.328125000000000,
        1,
        0.671875000000000,
        0.343750000000000,
        1,
        0.656250000000000,
        0.359375000000000,
        1,
        0.640625000000000,
        0.375000000000000,
        1,
        0.625000000000000,
        0.390625000000000,
        1,
        0.609375000000000,
        0.406250000000000,
        1,
        0.593750000000000,
        0.421875000000000,
        1,
        0.578125000000000,
        0.437500000000000,
        1,
        0.562500000000000,
        0.453125000000000,
        1,
        0.546875000000000,
        0.468750000000000,
        1,
        0.531250000000000,
        0.484375000000000,
        1,
        0.515625000000000,
        0.500000000000000,
        1,
        0.500000000000000,
        0.515625000000000,
        1,
        0.484375000000000,
        0.531250000000000,
        1,
        0.468750000000000,
        0.546875000000000,
        1,
        0.453125000000000,
        0.562500000000000,
        1,
        0.437500000000000,
        0.578125000000000,
        1,
        0.421875000000000,
        0.593750000000000,
        1,
        0.406250000000000,
        0.609375000000000,
        1,
        0.390625000000000,
        0.625000000000000,
        1,
        0.375000000000000,
        0.640625000000000,
        1,
        0.359375000000000,
        0.656250000000000,
        1,
        0.343750000000000,
        0.671875000000000,
        1,
        0.328125000000000,
        0.687500000000000,
        1,
        0.312500000000000,
        0.703125000000000,
        1,
        0.296875000000000,
        0.718750000000000,
        1,
        0.281250000000000,
        0.734375000000000,
        1,
        0.265625000000000,
        0.750000000000000,
        1,
        0.250000000000000,
        0.765625000000000,
        1,
        0.234375000000000,
        0.781250000000000,
        1,
        0.218750000000000,
        0.796875000000000,
        1,
        0.203125000000000,
        0.812500000000000,
        1,
        0.187500000000000,
        0.828125000000000,
        1,
        0.171875000000000,
        0.843750000000000,
        1,
        0.156250000000000,
        0.859375000000000,
        1,
        0.140625000000000,
        0.875000000000000,
        1,
        0.125000000000000,
        0.890625000000000,
        1,
        0.109375000000000,
        0.906250000000000,
        1,
        0.0937500000000000,
        0.921875000000000,
        1,
        0.0781250000000000,
        0.937500000000000,
        1,
        0.0625000000000000,
        0.953125000000000,
        1,
        0.0468750000000000,
        0.968750000000000,
        1,
        0.0312500000000000,
        0.984375000000000,
        1,
        0.0156250000000000,
        1,
        1,
        0,
        1,
        0.984375000000000,
        0,
        1,
        0.968750000000000,
        0,
        1,
        0.953125000000000,
        0,
        1,
        0.937500000000000,
        0,
        1,
        0.921875000000000,
        0,
        1,
        0.906250000000000,
        0,
        1,
        0.890625000000000,
        0,
        1,
        0.875000000000000,
        0,
        1,
        0.859375000000000,
        0,
        1,
        0.843750000000000,
        0,
        1,
        0.828125000000000,
        0,
        1,
        0.812500000000000,
        0,
        1,
        0.796875000000000,
        0,
        1,
        0.781250000000000,
        0,
        1,
        0.765625000000000,
        0,
        1,
        0.750000000000000,
        0,
        1,
        0.734375000000000,
        0,
        1,
        0.718750000000000,
        0,
        1,
        0.703125000000000,
        0,
        1,
        0.687500000000000,
        0,
        1,
        0.671875000000000,
        0,
        1,
        0.656250000000000,
        0,
        1,
        0.640625000000000,
        0,
        1,
        0.625000000000000,
        0,
        1,
        0.609375000000000,
        0,
        1,
        0.593750000000000,
        0,
        1,
        0.578125000000000,
        0,
        1,
        0.562500000000000,
        0,
        1,
        0.546875000000000,
        0,
        1,
        0.531250000000000,
        0,
        1,
        0.515625000000000,
        0,
        1,
        0.500000000000000,
        0,
        1,
        0.484375000000000,
        0,
        1,
        0.468750000000000,
        0,
        1,
        0.453125000000000,
        0,
        1,
        0.437500000000000,
        0,
        1,
        0.421875000000000,
        0,
        1,
        0.406250000000000,
        0,
        1,
        0.390625000000000,
        0,
        1,
        0.375000000000000,
        0,
        1,
        0.359375000000000,
        0,
        1,
        0.343750000000000,
        0,
        1,
        0.328125000000000,
        0,
        1,
        0.312500000000000,
        0,
        1,
        0.296875000000000,
        0,
        1,
        0.281250000000000,
        0,
        1,
        0.265625000000000,
        0,
        1,
        0.250000000000000,
        0,
        1,
        0.234375000000000,
        0,
        1,
        0.218750000000000,
        0,
        1,
        0.203125000000000,
        0,
        1,
        0.187500000000000,
        0,
        1,
        0.171875000000000,
        0,
        1,
        0.156250000000000,
        0,
        1,
        0.140625000000000,
        0,
        1,
        0.125000000000000,
        0,
        1,
        0.109375000000000,
        0,
        1,
        0.0937500000000000,
        0,
        1,
        0.0781250000000000,
        0,
        1,
        0.0625000000000000,
        0,
        1,
        0.0468750000000000,
        0,
        1,
        0.0312500000000000,
        0,
        1,
        0.0156250000000000,
        0,
        1,
        0,
        0,
        0.984375000000000,
        0,
        0,
        0.968750000000000,
        0,
        0,
        0.953125000000000,
        0,
        0,
        0.937500000000000,
        0,
        0,
        0.921875000000000,
        0,
        0,
        0.906250000000000,
        0,
        0,
        0.890625000000000,
        0,
        0,
        0.875000000000000,
        0,
        0,
        0.859375000000000,
        0,
        0,
        0.843750000000000,
        0,
        0,
        0.828125000000000,
        0,
        0,
        0.812500000000000,
        0,
        0,
        0.796875000000000,
        0,
        0,
        0.781250000000000,
        0,
        0,
        0.765625000000000,
        0,
        0,
        0.750000000000000,
        0,
        0,
        0.734375000000000,
        0,
        0,
        0.718750000000000,
        0,
        0,
        0.703125000000000,
        0,
        0,
        0.687500000000000,
        0,
        0,
        0.671875000000000,
        0,
        0,
        0.656250000000000,
        0,
        0,
        0.640625000000000,
        0,
        0,
        0.625000000000000,
        0,
        0,
        0.609375000000000,
        0,
        0,
        0.593750000000000,
        0,
        0,
        0.578125000000000,
        0,
        0,
        0.562500000000000,
        0,
        0,
        0.546875000000000,
        0,
        0,
        0.531250000000000,
        0,
        0,
        0.515625000000000,
        0,
        0,
        0.500000000000000,
        0,
        0,
    ]
    cmap_np = np.array(cmap_np).reshape(256, 3)
    # print(cmap_np)

    cmap_th = torch.from_numpy(cmap_np)
    # print(cmap_th.shape)
    ii = ii.detach().cpu().squeeze()
    # ii = (ii - ii.min()) / (ii.max() - ii.min() + 1e-10)
    # ii = torch.log(ii + 1.0)
    # ii = (ii - ii.min()) / (ii.max() - ii.min() + 1e-10)
    ii = ii / scale
    ii = (ii * 255).clamp(0, 255).long()
    # print(ii.shape)
    oo = cmap_th[ii]
    # time.sleep(10)
    oo = oo.numpy()
    oo = np.pad(
        oo, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=1.0
    )
    return oo


"""not using since the memory leak caused by matplotlib when drawing a lot of plots
"""


def draw_heatmap(tt: torch.Tensor):
    tt = tt.squeeze()
    assert len(tt.shape) == 2
    tt = tt.cpu().detach().numpy()

    plt.clf()
    fig = plt.figure()
    seaborn.heatmap(tt)
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, format="raw")
        hm_np = np.frombuffer(io_buf.getvalue(), np.uint8).reshape(
            int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1
        )
    hm_np = hm_np[..., :3]  # only record 3 channels
    fig.clear()
    plt.clf()
    plt.close()

    del fig
    # plt.clf()
    # plt.cla()

    # plt.close()
    gc.collect()
    return hm_np


def main():
    cfg = get_default_config()
    args = parse_args()
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
    
    cfg = reset_some_value(cfg)
    if args.dataset is None:
        dataloader = get_valid_load(cfg)
    else:
        dataloader = get_valid_load(cfg, args.dataset)
    model = build_model(cfg)
    # model.cuda()

    # load the parameters
    if args.load is not None:
        ckpt = torch.load(args.load)["model_state"]
        try:
            model.load_state_dict(ckpt)
        except:
            model.load_state_dict(
                {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}, strict=True
            )
    model.cuda()

    # writer = SummaryWriter('log/temp')
    validate_model(
        cfg,
        model,
        dataloader,
        pad=cfg.VALIDATE_PAD,
        save_dir=args.save,
        save_err_img=args.saveerr,
    )


if __name__ == "__main__":
    main()
