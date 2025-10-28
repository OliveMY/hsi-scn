from typing import List
from matplotlib import axis
import torch
import numpy as np
import math


class EMAMeter(object):
    def __init__(self, period = 10, start_step = 0, weigth_new=0.2) -> None:
        self.period = period
        self.step = start_step
        self.weight_new = weigth_new
        self.avg = 0
    
    def update(self, value):
        next_step = self.step + 1
        if self.step < self.period:
            self.avg = (self.avg * self.step + value) / next_step
        else:
            self.avg = self.weight_new * value + self.avg * (1 - self.weight_new)
        self.step = next_step
    
    def get_newest_value(self):
        return self.avg

class PatchG(object):
    def __init__(self, patch_size: int, stride: int):
        self.patch_size = patch_size
        self.stride = stride

    @staticmethod
    def _comp_start_point(dim: int, idx: int, patch_size: int, stride: int):
        start_p = idx * stride
        start_p = dim - patch_size if start_p + patch_size > dim else start_p
        return start_p

    @staticmethod
    def _comp_n_blocks_1d(dim: int, patch_size: int, stride: int):
        if dim <= patch_size:
            n_dim = 1
        else:
            n_dim = 1 + (dim - patch_size) // stride
            n_dim = n_dim + 1 if (dim - patch_size) % stride > 0 else n_dim
        return n_dim

    def unfold(self, tensor_in: torch.Tensor):
        assert len(tensor_in.shape) == 4, "Input tensor must be 4 dimension"
        b, c, h, w = tensor_in.shape

        n_h = PatchG._comp_n_blocks_1d(h, self.patch_size, self.stride)
        n_w = PatchG._comp_n_blocks_1d(w, self.patch_size, self.stride)
        output_tensor = torch.empty(
            (b, c, self.patch_size, self.patch_size, n_h * n_w),
            dtype=tensor_in.dtype,
            device=tensor_in.device,
        )

        for i in range(n_h):
            start_h = PatchG._comp_start_point(h, i, self.patch_size, self.stride)
            for j in range(n_w):
                start_w = PatchG._comp_start_point(w, j, self.patch_size, self.stride)
                output_tensor[..., i * n_w + j] = tensor_in[
                    :,
                    :,
                    start_h : start_h + self.patch_size,
                    start_w : start_w + self.patch_size,
                ]
        return output_tensor

    def fold(self, tensor_in: torch.Tensor, out_size: List[int]):
        assert len(tensor_in.shape) == 5
        assert len(out_size) == 2, "Out_size should be a list with 2 integers [h,w]"
        b, c, p_h, p_w, n_p = tensor_in.shape
        assert p_h == self.patch_size
        assert p_w == self.patch_size
        h, w = out_size
        output = torch.zeros(
            (b, c, h, w), dtype=tensor_in.dtype, device=tensor_in.device
        )
        divisor = torch.zeros(
            (1, 1, h, w), dtype=tensor_in.dtype, device=tensor_in.device
        )
        dummy_ones = torch.ones(
            (1, 1, p_h, p_w), dtype=tensor_in.dtype, device=tensor_in.device
        )

        def _get_gaussian_kernel_2d(
            k_size: List[int], sigma: float, device: torch.device
        ):
            assert len(k_size) == 2
            k_1d_h = torch.arange(0, k_size[0], dtype=torch.float32, device=device)
            k_1d_w = torch.arange(0, k_size[1], dtype=torch.float32, device=device)
            mid_h, mid_w = (k_size[0] - 1) / 2.0, (k_size[1] - 1) / 2.0
            k_1d_h = torch.exp((k_1d_h - mid_h) ** 2 / 2 / sigma / sigma)
            k_1d_w = torch.exp((k_1d_w - mid_w) ** 2 / 2 / sigma / sigma)
            k_2d = torch.matmul(k_1d_h.unsqueeze(-1), k_1d_w.unsqueeze(0))
            k_2d = k_2d / torch.sum(k_2d)
            return k_2d

        g_k = _get_gaussian_kernel_2d([p_h, p_w], float(p_h) * 2, tensor_in.device)
        g_k = g_k.reshape(1, 1, p_h, p_w)
        n_h = PatchG._comp_n_blocks_1d(h, self.patch_size, self.stride)
        n_w = PatchG._comp_n_blocks_1d(w, self.patch_size, self.stride)

        assert n_p == n_h * n_w, "The number of patches are incorrect! Please check"
        for idx in range(n_p):
            idx_h, idx_w = idx // n_w, idx % n_w
            start_h = PatchG._comp_start_point(h, idx_h, p_h, self.stride)
            start_w = PatchG._comp_start_point(w, idx_w, p_w, self.stride)
            end_h, end_w = start_h + p_h, start_w + p_w
            output[:, :, start_h:end_h, start_w:end_w] += tensor_in[..., idx]  # * g_k
            divisor[:, :, start_h:end_h, start_w:end_w] += dummy_ones
        output = output / divisor
        return output


@torch.no_grad()
def infer_model_by_patch(
    model: torch.nn.Module, input_tensor: torch.Tensor, patch_size: int = 224
):
    b, c, h, w = input_tensor.shape
    info = dict(kernel_size=(patch_size, patch_size), dilation=1, stride=36)
    unfold = torch.nn.Unfold(**info)
    fold = torch.nn.Fold(output_size=(h, w), **info)
    divisor = fold(
        unfold(
            torch.ones((1, 1, h, w), device=input_tensor.device, dtype=torch.float32)
        )
    )
    unfolded_tensor = (
        unfold(input_tensor)
        .reshape(b, c, patch_size, patch_size, -1)
        .permute(0, 4, 1, 2, 3)
    )
    num_patches = unfolded_tensor.shape[1]
    unfolded_tensor = unfolded_tensor.reshape(
        b * num_patches, c, patch_size, patch_size
    )

    output = []
    for idx in range(math.ceil(b * num_patches / 8)):  # infer with only batch_size 8
        tt = unfolded_tensor[idx * 8 : (idx + 1) * 8].detach()
        model_out = model(tt)
        if isinstance(model_out, torch.Tensor):
            model_out = [model_out.detach()]
        else:
            model_out = [
                (
                    mm.detach()
                    if ii == 0
                    else torch.linalg.norm(mm, ord=2, dim=1, keepdim=True).detach()
                )
                for (ii, mm) in enumerate(model_out)
            ]
        output.append(model_out)

    all_output = []

    num_output = len(output[0])
    for i in range(num_output):
        out_i = torch.concatenate([oo[i] for oo in output], dim=0)
        out_i = out_i.reshape(b, num_patches, -1).permute(0, 2, 1)
        out_i = fold(out_i)
        out_i = out_i / divisor
        all_output.append(out_i)

    if num_output == 1:
        return all_output[0]
    else:
        return all_output


@torch.no_grad()
def infer_model_by_patchPG(
    model: torch.nn.Module, input_tensor: torch.Tensor, patch_size: int = 224, patch_stride = None
):
    if patch_stride is None:
        patch_stride = patch_size // 4
    PG = PatchG(patch_size, patch_stride)
    b, c, h, w = input_tensor.shape

    unfolded_tensor = PG.unfold(input_tensor).permute(0, 4, 1, 2, 3)
    num_patches = unfolded_tensor.shape[1]
    unfolded_tensor = unfolded_tensor.reshape(
        b * num_patches, c, patch_size, patch_size
    )

    output = []
    for idx in range(math.ceil(b * num_patches / 8)):  # infer with only batch_size 8
        tt = unfolded_tensor[idx * 8 : (idx + 1) * 8].detach()
        model_out = model(tt)
        if isinstance(model_out, torch.Tensor):
            model_out = [model_out.detach()]
        else:
            model_out = [mm.detach() for mm in model_out]
        output.append(model_out)

    all_output = []

    num_output = len(output[0])
    for i in range(num_output):
        out_i = torch.concatenate([oo[i] for oo in output], dim=0)
        out_i = out_i.reshape(b, num_patches, -1, patch_size, patch_size).permute(
            0, 2, 3, 4, 1
        )
        out_i = PG.fold(out_i, [h, w])
        all_output.append(out_i)

    if num_output == 1:
        return all_output[0]
    else:
        return all_output


if __name__ == "__main__":

    class DummyModule(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.conv1 = torch.nn.Conv2d(3, 31, 1, 1, 0)

        def forward(self, in_tensor):
            return in_tensor, self.conv1(in_tensor)

    # dummy_input = torch.randn(8, 3, 512, 512)
    # # model = torch.nn.Conv2d(3, 31, 1, 1, 0)
    # model = DummyModule()
    # out = infer_model_by_patch(model, dummy_input, 224)
    # print(out[0].shape, out[1].shape)
    pg = PatchG(256, 128)
    dummy_input = torch.randn(1, 3, 482, 512)
    model = DummyModule()
    ppp = pg.unfold(dummy_input)
    out = pg.fold(ppp, [482, 512])
    # print(dummy_input, out)
    print(torch.all(torch.abs(out - dummy_input) < 1e-6))

    out_infer1, out_infer2 = infer_model_by_patchPG(model, dummy_input, 224)
    print(torch.all(torch.abs(out_infer1 - dummy_input) < 1e-6))
    print(out_infer2.shape)
