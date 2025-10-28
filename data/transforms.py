from .abc import HSSRTransform
import numpy as np
import scipy as sp
import torch
from typing import Iterable, List, Union, Tuple
from yacs.config import CfgNode as CN


class Compose(HSSRTransform):
    def __init__(
        self, transforms: Union[List[HSSRTransform], Tuple[HSSRTransform]]
    ) -> None:
        """
        Compose several transforms into one
        Args:
            transforms (Union[List[HSSRTransform], Tuple[HSSRTransform]]): List or Tuple of transforms to be composed.
        """
        super().__init__()
        self.transforms = transforms

    def transform(self, hs_tensor: np.ndarray, rgb_tensor: np.ndarray):
        for trans in self.transforms:
            hs_tensor, rgb_tensor = trans.transform(hs_tensor, rgb_tensor)

        # To reduce the
        hs_tensor = np.ascontiguousarray(hs_tensor)
        rgb_tensor = np.ascontiguousarray(rgb_tensor)

        return hs_tensor, rgb_tensor


class CenterCrop(HSSRTransform):
    def __init__(self, crop_size: int = 256) -> None:
        super().__init__()
        self.crop_size = crop_size

    def transform(self, hs_tensor: np.ndarray, rgb_tensor: np.ndarray):
        """_summary_

        Args:
            hs_tensor (np.ndarray): _description_
            rgb_tensor (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        _channels, h, w = hs_tensor.shape
        assert (
            h >= self.crop_size and w >= self.crop_size
        ), "Crop size ({}) should be smaller to the mat size {}x{}".format(
            self.crop_size, h, w
        )
        if self.crop_size <= 0:
            return hs_tensor, rgb_tensor

        _color, h_img, w_img = rgb_tensor.shape
        assert h_img == h and w_img == w

        start_h = (h - self.crop_size) // 2
        start_w = (w - self.crop_size) // 2

        hs_tensor = np.ascontiguousarray(
            hs_tensor[
                :,
                start_h : start_h + self.crop_size,
                start_w : start_w + self.crop_size,
            ]
        )
        rgb_tensor = np.ascontiguousarray(
            rgb_tensor[
                :,
                start_h : start_h + self.crop_size,
                start_w : start_w + self.crop_size,
            ]
        )

        return hs_tensor, rgb_tensor


class RandomCrop(HSSRTransform):
    def __init__(self, crop_size: int = 256) -> None:
        super().__init__()
        self.crop_size = crop_size

    def transform(self, hs_tensor: np.ndarray, rgb_tensor: np.ndarray):
        """_summary_

        Args:
            hs_tensor (np.ndarray): _description_
            rgb_tensor (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        _channels, h, w = hs_tensor.shape
        assert (
            h >= self.crop_size and w >= self.crop_size
        ), "Crop size ({}) should be smaller to the mat size {}x{}".format(
            self.crop_size, h, w
        )
        if self.crop_size <= 0:
            return hs_tensor, rgb_tensor

        _color, h_img, w_img = rgb_tensor.shape
        assert h_img == h and w_img == w

        start_h = np.random.randint(0, h - self.crop_size)
        start_w = np.random.randint(0, w - self.crop_size)

        hs_tensor = np.ascontiguousarray(
            hs_tensor[
                :,
                start_h : start_h + self.crop_size,
                start_w : start_w + self.crop_size,
            ]
        )
        rgb_tensor = np.ascontiguousarray(
            rgb_tensor[
                :,
                start_h : start_h + self.crop_size,
                start_w : start_w + self.crop_size,
            ]
        )

        return hs_tensor, rgb_tensor


class Normalize(HSSRTransform):
    def __init__(
        self,
        hs_mean: Union[float, Iterable[float]] = 0.5,
        hs_std: Union[float, Iterable[float]] = 0.5,
        img_mean: Union[float, Iterable[float]] = 0.5,
        img_std: Union[float, Iterable[float]] = 0.5,
    ) -> None:
        super().__init__()
        def _init_value(x, value, length):
            if isinstance(x, Iterable) and len(x) == 0:
                x = [value] * length
            return x
        hs_mean = _init_value(hs_mean, 0.0, 31)
        hs_std = _init_value(hs_std, 1.0, 31)
        img_mean = _init_value(img_mean, 0.0, 3)
        img_std = _init_value(img_std, 1.0, 3)
        self._hs_mean = (
            np.array(hs_mean).reshape(31, 1, 1)
            if isinstance(img_mean, Iterable)
            else img_mean
        )
        self._hs_std = (
            np.array(hs_std).reshape(31, 1, 1)
            if isinstance(img_mean, Iterable)
            else img_mean
        )
        self._img_mean = (
            np.array(img_mean).reshape(3, 1, 1)
            if isinstance(img_mean, Iterable)
            else img_mean
        )
        self._img_std = (
            np.array(img_std).reshape(3, 1, 1)
            if isinstance(img_std, Iterable)
            else img_std
        )

    def transform(self, hs_tensor: np.ndarray, rgb_tensor: np.ndarray):
        hs_tensor = (hs_tensor - self._hs_mean) / self._hs_std
        rgb_tensor = (rgb_tensor - self._img_mean) / self._img_std
        hs_tensor = hs_tensor.astype(np.float32)
        rgb_tensor = rgb_tensor.astype(np.float32)
        return hs_tensor, rgb_tensor


class RandomFlipHorizontal(HSSRTransform):
    def __init__(self, flip_rate: float = 0.5) -> None:
        super().__init__()
        self._flip_rate = flip_rate

    def transform(self, hs_tensor: np.ndarray, rgb_tensor: np.ndarray):
        if np.random.uniform() < self._flip_rate:
            hs_tensor = np.flip(hs_tensor, axis=2)
            rgb_tensor = np.flip(rgb_tensor, axis=2)
        return hs_tensor, rgb_tensor


class RandomFlipVeritcal(HSSRTransform):
    def __init__(self, flip_rate: float = 0.5) -> None:
        super().__init__()
        self._flip_rate = flip_rate

    def transform(self, hs_tensor: np.ndarray, rgb_tensor: np.ndarray):
        if np.random.uniform() < self._flip_rate:
            hs_tensor = np.flip(hs_tensor, axis=1)
            rgb_tensor = np.flip(rgb_tensor, axis=1)
        return hs_tensor, rgb_tensor


class RandomTranspose(HSSRTransform):
    def __init__(self, trans_rate: float = 0.5) -> None:
        super().__init__()
        self._trans_rate = trans_rate

    def transform(self, hs_tensor: np.ndarray, rgb_tensor: np.ndarray):
        if np.random.uniform() < self._trans_rate:
            hs_tensor = np.transpose(hs_tensor, (0, 2, 1))
            rgb_tensor = np.transpose(rgb_tensor, (0, 2, 1))
        return hs_tensor, rgb_tensor


def get_transforms(cfg: CN, is_train: bool = True):
    if is_train:
        transforms = Compose(
            [
                RandomCrop(cfg.DATASET.TRANSFORMS.RANDOM_CROP),
                RandomFlipHorizontal(cfg.DATASET.TRANSFORMS.FLIP_HORI_RATE),
                RandomFlipVeritcal(cfg.DATASET.TRANSFORMS.FLIP_VERTI_RATE),
                RandomTranspose(cfg.DATASET.TRANSFORMS.TRANSPOSE_RATE),
                Normalize(
                    cfg.DATASET.TRANSFORMS.NORMAL_HS_MEAN,
                    cfg.DATASET.TRANSFORMS.NORMAL_HS_STD,
                    cfg.DATASET.TRANSFORMS.NORMAL_RGB_MEAN,
                    cfg.DATASET.TRANSFORMS.NORMAL_RGB_STD,
                ),
            ]
        )
    else:
        transforms = Compose(
            [
                # CenterCrop(cfg.DATASET.TRANSFORMS.RANDOM_CROP),
                Normalize(
                    cfg.DATASET.TRANSFORMS.NORMAL_HS_MEAN,
                    cfg.DATASET.TRANSFORMS.NORMAL_HS_STD,
                    cfg.DATASET.TRANSFORMS.NORMAL_RGB_MEAN,
                    cfg.DATASET.TRANSFORMS.NORMAL_RGB_STD,
                )
            ]
        )
    return transforms


if __name__ == "__main__":
    pass
