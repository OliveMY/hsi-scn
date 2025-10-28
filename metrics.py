import enum
import io
from math import isinf, pi
import numpy as np
import cv2
import torch
from typing import Union
import matplotlib.pyplot as plt


def _convert_2_np(img_tensor: Union[np.ndarray, torch.Tensor]):
    if isinstance(img_tensor, torch.Tensor):
        # convert the img_tensor to numpy array
        img_tensor = img_tensor.squeeze().detach().cpu().numpy()

    assert (
        len(img_tensor.shape) == 2
    ), "Input tensor should be in 2 dimensions. Please check the input."
    return img_tensor


def area_above_thres(img_tensor: Union[np.ndarray, torch.Tensor], thres: float):
    img_tensor = _convert_2_np(img_tensor)
    mask = img_tensor >= thres
    area = np.sum(mask.astype(np.float32))
    mask = mask.astype(np.uint8) * 255
    return area, mask


def max_area_above_thres(img_tensor: Union[np.ndarray, torch.Tensor], thres: float):
    img_tensor = _convert_2_np(img_tensor)
    mask = img_tensor > thres
    output_mask = np.zeros_like(mask, dtype=np.uint8)

    contours, hier = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return 0, output_mask

    areas = [cv2.contourArea(cont) for cont in contours]
    max_area = np.max(areas)
    max_idx = np.argmax(areas)
    cv2.fillPoly(output_mask, [contours[max_idx]], 0xFF)

    return max_area, output_mask


def adaptive_thresholding(
    img_tensor: Union[np.ndarray, torch.Tensor], bins: int = 300, method: str = "otsu"
):
    """This methods implements three adaptive thresholding algos

    Args:
        img_tensor (Union[np.ndarray, torch.Tensor]): the input image tensor
        bins (int, optional): _description_. Defaults to 100.
        method (str, optional): choose in ['otsu', 'kittler', 'RC']. Defaults to 'otsu'.
    """
    assert method in (
        "otsu",
        "kittler",
        "RC",
        "hard_thres",
    ), "Unsupported mode. Supported methods are ['otsu', 'kittler', 'RC']"
    img_tensor = _convert_2_np(img_tensor)
    if method == "otsu":
        img_min, img_max = np.min(img_tensor), np.max(img_tensor)
        bin_edges = (
            np.arange(bins).astype(np.float32) * (img_max - img_min) / bins + img_min
        )
        all_fn = [-np.inf]

        for thres_candidate in bin_edges[1:-1]:
            pixels_1 = img_tensor[img_tensor < thres_candidate]
            pixels_2 = img_tensor[img_tensor >= thres_candidate]
            fn = (
                np.size(pixels_1)
                * np.size(pixels_2)
                * ((np.mean(pixels_1) - np.mean(pixels_2)) ** 2)
            )
            all_fn.append(fn)

        max_idx = np.argmax(all_fn)
        thres = bin_edges[max_idx]
        mask = img_tensor >= thres
    elif method == "kittler":
        img_min, img_max = np.min(img_tensor), np.max(img_tensor)
        bin_edges = (
            np.arange(bins).astype(np.float32) * (img_max - img_min) / bins + img_min
        )
        pixel_num = np.size(img_tensor)
        all_fn = [np.inf]
        for thres_candidate in bin_edges[1:-1]:
            pixels_1 = img_tensor[img_tensor < thres_candidate]
            pixels_2 = img_tensor[img_tensor >= thres_candidate]
            p1_t, p2_t = np.size(pixels_1) / pixel_num, np.size(pixels_2) / pixel_num
            if p1_t > 1.0e-6 and p2_t > 1.0e-6:
                sigma1_t, sigma2_t = np.std(pixels_1), np.std(pixels_2)
                fn = 1 + 2 * (
                    p1_t * np.log(sigma1_t / p1_t) + p2_t * np.log(sigma2_t / p2_t)
                )
                if np.isinf(fn):
                    fn = np.inf
            else:
                fn = np.inf
            all_fn.append(fn)
        # print(all_fn)
        min_idx = np.argmin(all_fn)
        thres = bin_edges[min_idx]
        mask = img_tensor >= thres
    elif method == "RC":
        max_iter = 200
        tol = 0.05
        thres = np.mean(img_tensor)
        for _ in range(max_iter):
            bg_mask = img_tensor < thres
            mean_bg = np.mean(img_tensor[bg_mask])
            mean_fg = np.mean(img_tensor[~bg_mask])
            new_thres = (mean_bg + mean_fg) / 2.0
            if np.abs(new_thres - thres) < tol:
                break
            else:
                thres = new_thres
        mask = img_tensor >= thres
    else:
        thres = 0.1
        mask = img_tensor >= thres

    return thres, mask


def get_metrics(
    img_tensor: Union[np.ndarray, torch.Tensor],
    save_name: str = None,
    psnr: float = None,
    hsi_err_tensor: Union[np.ndarray, torch.Tensor] = None,
    draw: bool = False,
):
    # methods = ["otsu", "kittler", "RC", "hard_thres"]
    methods = ["hard_thres"]
    img_tensor = _convert_2_np(img_tensor)
    img_show = img_tensor / np.max(img_tensor)
    # rc('text', usetex=True)
    if draw:
        plt.clf()
        plt.cla()
        h, w = img_tensor.shape
        font = {"size": 4}
        plt.subplot(2, 2, 1)
        plt.imshow(img_show, cmap="binary_r")
        if psnr is not None:
            plt.text(0, -10, "PSNR: %.4f" % psnr, fontdict=font)

    metric_dict = {}

    for idx, method in enumerate(methods):
        thres, mask = adaptive_thresholding(img_tensor, bins=256, method=method)
        area = np.mean(mask.astype(np.float32))
        mean_fg = np.mean(img_tensor[mask])
        metric_dict[method + "_thres"] = thres
        metric_dict[method + "_area"] = area
        metric_dict[method + "_meanfg"] = mean_fg

        if draw:
            img_c = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB)
            img_c = img_c * 0.5
            img_c[:, :, 0] = img_c[:, :, 0] + mask.astype(np.float32) * 0.5
            img_c[:, :, 1] = img_c[:, :, 1] + (1 - mask.astype(np.float32)) * 0.5
            plt.subplot(2, 2, idx + 2)
            plt.imshow(img_c)
            txt = method + ": Thres: %4f, Mean FG: %.4f, Area: %.4f" % (
                thres,
                mean_fg,
                area,
            )
            plt.text(0, -10, txt, fontdict=font)

            if save_name is not None:
                plt.savefig(save_name, dpi=300)

    return metric_dict


if __name__ == "__main__":
    # a = np.random.random(size=(5,5))
    # a = cv2.resize(a, (512,512))
    # a = cv2.GaussianBlur(a, [7,7], 1.0)
    # area, mask = max_area_above_thres(a, 0.3)
    # area2, mask2 = area_above_thres(a,0.3)
    # out = np.concatenate([np.round(a*255).astype(np.uint8), mask, mask2], axis=0)
    # cv2.imwrite('asdf.png', out)
    # print(area, area2)
    # img1 = np.random.randn(512,512) * (15/255) + (50/255)
    # img2 = np.random.randn(512,512) * (15/255) + (150/255)
    # mask = np.random.uniform(0, 1, (512,512)) > 0.5
    # img = img1.clip(0,1) * mask + img2.clip(0,1) * (1-mask)

    img = cv2.imread("a.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    thres, mask = adaptive_thresholding(img, 300, method="otsu")
    print(thres)
    # cv2.imwrite('ostu.png', mask.astype(np.uint8) * 255)
    draw_metrics(img)
