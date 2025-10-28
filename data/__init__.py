from yacs.config import CfgNode as CN
from .arad_re import ARADDatasetRe
from .icvl_list import ICVLFTDataset
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from .transforms import get_transforms
from .utils import DSBalanceSampler, DistDSBalanceSampler


def get_dataloader(cfg: CN, rank: int = 0, world_size: int = 0):
    # build transforms
    train_transforms = get_transforms(cfg, True)
    valid_transforms = get_transforms(cfg, False)
    # print(rank, world_size)
    ds_names = str(cfg.DATASET.NAME)
    ds_names = ds_names.split("+")

    train_ds_all = []
    valid_ds_all = []
    for name in ds_names:
        if name == "ARAD_RE":
            ds_train = ARADDatasetRe(
                cfg.DATASET.ROOT,
                True,
                train_transforms,
                cfg.DATASET.PRELOAD,
                rank=rank,
                world_size=world_size,
            )
            ds_valid = ARADDatasetRe(
                cfg.DATASET.ROOT,
                False,
                valid_transforms,
                cfg.DATASET.PRELOAD,
                rank=rank,
                world_size=world_size,
            )
        elif name == "ICVL":
            cfg_fttarget = cfg.DATASET.FTTARGET
            ft_target = cfg_fttarget if len(cfg_fttarget) > 0 else None
            ds_train = ICVLFTDataset(
                "/mnt/data/ICVL",
                True,
                train_transforms,
                True,
                rank=rank,
                world_size=world_size,
                test_set=ft_target,
                random_ratio=cfg.DATASET.RANDOMRATIO,
            )
            ds_valid = ICVLFTDataset("/mnt/data/ICVL", False, valid_transforms, False)
        else:
            raise NotImplementedError

        train_ds_all.append(ds_train)
        valid_ds_all.append(ds_valid)

    if len(ds_names) > 1:
        train_sampler = (
            DistDSBalanceSampler(train_ds_all, rank=rank, drop_last=True)
            if world_size > 1
            else DSBalanceSampler(train_ds_all)
        )
        train_ds_all = ConcatDataset(train_ds_all)
        valid_ds_all = valid_ds_all[-1]

    else:
        train_ds_all = train_ds_all[0]
        valid_ds_all = valid_ds_all[0]
        train_sampler = (
            DistributedSampler(ds_train, rank=rank, shuffle=True, drop_last=True)
            if world_size > 1
            else None
        )

    batch_size = cfg.DATASET.BATCH_SIZE
    shuffle = False if world_size > 0 or len(ds_names) > 1 else True

    train_loader = DataLoader(
        train_ds_all, batch_size, shuffle=shuffle, sampler=train_sampler
    )
    valid_loader = DataLoader(valid_ds_all, 1, shuffle=False)
    return train_loader, valid_loader


def get_valid_load(cfg: CN, name: str = None):
    # valid_transforms = None
    valid_transforms = get_transforms(cfg, False)
    name = cfg.DATASET.NAME if name is None else name

    if name == "ARAD_RE":
        ds_valid = ARADDatasetRe(
            cfg.DATASET.ROOT, False, valid_transforms, cfg.DATASET.PRELOAD
        )
    elif name == "ICVL":
        ds_valid = ICVLFTDataset("/mnt/data/ICVL", False, valid_transforms, False)
    
    else:
        raise NotImplementedError

    valid_loader = DataLoader(ds_valid, 1, shuffle=False)
    return valid_loader
