import math
from cv2 import mean
import torch
import torch.nn as nn


class LearnalbeSoftShrink(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.theta = nn.Parameter(torch.ones(()) * 0.1, requires_grad=True)

    def forward(self, in_tensor: torch.Tensor):
        x = in_tensor / self.theta
        x = torch.nn.functional.softshrink(x, 1.0)
        x = x * self.theta
        return x


class LISTA(nn.Module):
    def __init__(
        self, in_channels: int, dict_size: int, n_iters: int = 3, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_iters = n_iters
        # self.lambda_shrink = lambda_shrink
        self.conv_We = nn.Conv2d(
            in_channels, dict_size, 1, 1, 0, bias=False
        )  # conv1*1 to multiply We
        # torch.nn.init.normal_(self.conv_We.weight, mean=0.0, std=0.5/(math.sqrt(self.conv_We.weight.size(1))))
        self.soft_shrink_0 = LearnalbeSoftShrink()
        if n_iters > 0:
            self.conv_S = nn.ModuleList(
                [
                    nn.Conv2d(dict_size, dict_size, 1, 1, 0, bias=False)
                    for _ in range(n_iters)
                ]
            )
            self.soft_shrinks = nn.ModuleList(
                [LearnalbeSoftShrink() for _ in range(n_iters)]
            )
        else:
            self.conv_S = None

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """
        in_tensor: b*c*h*w
        """
        logits = self.conv_We(in_tensor)
        B = logits
        logits = self.soft_shrink_0(logits)
        # print(self._sparsity(logits))WWW
        for i in range(self.n_iters):
            # feat = torch.nn.functional.softshrink(logits, self.lambda_shrink)
            feat = self.conv_S[i](logits)
            feat = B + feat
            logits = self.soft_shrinks[i](feat)
        return logits

    def _sparsity(self, in_tensor: torch.Tensor):
        sp = (torch.abs(in_tensor) > 0).to(torch.int32)
        sp = torch.sum(sp, dim=1)
        mean_sp = torch.mean(sp.to(torch.float32))
        return mean_sp

class ELISTA(nn.Module):

    def __init__(self, in_channels: int, dict_size: int, n_iters: int = 3, shared_weights = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iters = n_iters

        self.conv_We = nn.Conv2d(
            in_channels, dict_size, 1, 1, 0, bias=False
        )  # conv1*1 to multiply We
        # torch.nn.init.normal_(self.conv_We.weight, mean=0.0, std=0.5/(math.sqrt(self.conv_We.weight.size(1))))
        self.soft_shrink_0 = LearnalbeSoftShrink()
        assert n_iters > 0
        if shared_weights:
            shared_S1 = nn.Conv2d(dict_size, dict_size, 1, 1, 0, bias=False)
            self.conv_S1 = nn.ModuleList(
                [
                    shared_S1 for _ in range(n_iters)
                ]
            )
            shared_st1 = LearnalbeSoftShrink()
            self.soft_shrinks1 = nn.ModuleList(
                [shared_st1 for _ in range(n_iters)]
            )
            shared_S2 = nn.Conv2d(dict_size, dict_size, 1, 1, 0, bias=False)
            self.conv_S2 = nn.ModuleList(
                [
                    shared_S2
                    for _ in range(n_iters)
                ]
            )
            shared_st2 = LearnalbeSoftShrink()
            self.soft_shrinks2 = nn.ModuleList(
                [shared_st2 for _ in range(n_iters)]
            )
            
        else:
            self.conv_S1 = nn.ModuleList(
                [
                    nn.Conv2d(dict_size, dict_size, 1, 1, 0, bias=False) for _ in range(n_iters)
                ]
            )
            self.soft_shrinks1 = nn.ModuleList(
                [LearnalbeSoftShrink() for _ in range(n_iters)]
            )
            self.conv_S2 = nn.ModuleList(
                [
                    nn.Conv2d(dict_size, dict_size, 1, 1, 0, bias=False)
                    for _ in range(n_iters)
                ]
            )
            self.soft_shrinks2 = nn.ModuleList(
                [LearnalbeSoftShrink() for _ in range(n_iters)]
            )
        
    
    def forward(self, in_tensor: torch.Tensor):
        logits = self.conv_We(in_tensor)
        B = logits
        logits = self.soft_shrink_0(logits)
        for i in range(self.n_iters):
            feat = self.conv_S1[i](logits)
            feat = B + feat
            logits_half = self.soft_shrinks1[i](feat)
            feat = self.conv_S2[i](logits_half) + logits + B
            logits = self.soft_shrinks2[i](feat)
        return logits

        


# class PUDLE(nn.Module):
#     def __init__(
#         self, in_channels: int, dict_size: int, n_iters: int = 3, *args, **kwargs
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.n_iters = n_iters
#         self.dict_size = dict_size
#         self._dict = nn.Parameter(
#             torch.randn((dict_size, in_channels, 1, 1), dtype=torch.float32)
#             / math.sqrt(dict_size)
#         )  # normalize
#         self.soft_shrink_0 = torch.nn.Softshrink(1.0)

#     def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         in_tensor: b*c*h*w
#         """
#         y = torch.nn.functional.conv2d(in_tensor, self._dict, None, 1, 0)
#         z = self.soft_shrink_0(y)
#         # print(self._sparsity(z))
#         _DT = self._dict.squeeze()
#         _DTD = torch.matmul(_DT, _DT.T).unsqueeze(-1).unsqueeze(-1)

#         for i in range(self.n_iters):
#             # feat = torch.nn.functional.softshrink(logits, self.lambda_shrink)
#             tmp = z - torch.nn.functional.conv2d(z, _DTD)
#             z = self.soft_shrink_0(y + tmp)
#             # print(self._sparsity(z))
#         recons = torch.nn.functional.conv2d(z, self._dict.transpose(0, 1))
#         return z, recons

#     def _sparsity(self, in_tensor: torch.Tensor):
#         sp = (torch.abs(in_tensor) > 0).to(torch.int32)
#         sp = torch.sum(sp, dim=1)
#         mean_sp = torch.mean(sp.to(torch.float32))
#         return mean_sp


class CoupleSparseCodingLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_iters: int,
        out_dim: int,
        dict_size: int,
        method: str = "lista",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.method = method
        if method.lower() == 'lista':
            self.lista = LISTA(in_channels, dict_size, n_iters)
            self.conv_recons_feat = nn.Conv2d(dict_size, in_channels, 1, 1, 0, bias=False)
            # target_std = 1/(math.sqrt(self.conv_recons_feat.weight.size(1)))
            # torch.nn.init.uniform_(self.conv_recons_feat.weight, a=-target_std, b=target_std)
        elif method.lower() == 'elista':
            self.lista = ELISTA(in_channels, dict_size, n_iters)
            self.conv_recons_feat = nn.Conv2d(dict_size, in_channels, 1, 1, 0, bias=False)
        else:
            print('Using PUDLE instead of LISTA')
            # self.pudle = PUDLE(in_channels, dict_size, n_iters)
            raise RuntimeError("PUDLE deprecated.")
        self.conv_recons_hs = nn.Conv2d(dict_size, out_dim, 1, 1, 0, bias=False)
        # torch.nn.init.normal_(self.conv_recons_hs.weight, mean=0.0, std=1/(math.sqrt(self.conv_recons_hs.weight.size(1))))

    def forward(
        self, in_feat, return_logits: bool = False, return_recons_feat: bool = False
    ):
        if self.method == 'lista':
            # if self.training:
            #     logits = self.lista(in_feat)
            #     in_feat_sg = in_feat.detach() # add stop_gradient
            #     logits_feat = self.lista(in_feat_sg)
            #     recons_feat = self.conv_recons_feat(logits_feat)
            # else:
            logits = self.lista(in_feat)
            recons_feat = self.conv_recons_feat(logits)
        elif self.method == 'elista':
            # if self.training:
            #     logits = self.lista(in_feat)
            #     in_feat_sg = in_feat.detach() # add stop_gradient
            #     logits_feat = self.lista(in_feat_sg)
            #     recons_feat = self.conv_recons_feat(logits_feat)
            # else:
            logits = self.lista(in_feat)
            recons_feat = self.conv_recons_feat(logits)
        else:
            logits, recons_feat = self.pudle(in_feat)
        recons_hs = self.conv_recons_hs(logits)

        ret = recons_hs
        if return_logits:
            ret = [ret, logits]
        if return_recons_feat:
            if isinstance(ret, list):
                ret.append(recons_feat)
            else:
                ret = [ret, recons_feat]
        return ret


if __name__ == "__main__":
    dummy_input = torch.randn(1, 128, 224, 224)
    layer = CoupleSparseCodingLayer(128, 5, 31, 100, method='pudle')
    hs, logits, recons_feat = layer(dummy_input, True, True)
    print(hs.shape, logits.shape, recons_feat.shape)
    # pudle = PUDLE(128, 500, 3)
    # # pudle = LISTA(128, 500, 3)
    # z, recons = pudle(dummy_input)
    # print(z, recons)
