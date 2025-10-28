import math
import torch.nn as nn
from .srcgan import UnetGenerator
import torch
import functools
from .utils import LayerNorm2d
from .mstpp import MST
from .csc import CoupleSparseCodingLayer
from .vision_transformer import vit_huge


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, add_extra:bool = False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.add_extra = add_extra
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.GELU()
        downnorm = norm_layer(inner_nc)
        uprelu = nn.GELU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                # model = down + [submodule] + up + [nn.Dropout(0.1)]
                up = up + [nn.Dropout(0.1)]
            else:
                model = down + [submodule] + up

        if innermost and add_extra:
            self.model = nn.ModuleList([nn.Sequential(*down), nn.Sequential(*up)])
        else:
            self.model = nn.Sequential(*model)
        # self.submodule = submodule

    def forward(self, x, extra_feat:torch.Tensor = None):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        d = self.model[0](x)
        if not self.innermost:
            feat = self.submodule(d, extra_feat)
        else:
            if self.add_extra:
                assert extra_feat is not None
                feat = d + extra_feat
            else:
                feat = d
        
        up = self.model[1](feat)

        if self.outermost:
            return up
        else:
            return torch.cat([x, up], 1)



class UnetBlock(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, add_extra:bool=False):
        # super().__init__(input_nc, output_nc, 4, ngf, norm_layer, use_dropout)
        super().__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, add_extra=add_extra)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, add_extra=add_extra)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, add_extra=add_extra)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, norm_layer=norm_layer, outermost=True, add_extra=add_extra)
    
    def forward(self,in_tensor, extra_feat:torch.Tensor=None):
        res = in_tensor
        return self.model(in_tensor, extra_feat) + res

class UnetHSI(nn.Module):
    def __init__(self, ngf:int, num_blks:int = 4, blk_type:str = 'unet', final_layer = 'conv', extra_feat:str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1)

        if extra_feat is not None:
            if extra_feat == 'ijepa':
                self.extra_feat_model = vit_huge(16)
                self.extra_feat_model.load_state_dict(torch.load('/home/rudolfmaxx/projects/ijepa/pretrained/Encoder_only_IN1K-vit.h.16-448px-300e.pth.tar', map_location=torch.device('cpu')), strict=False)
            else:
                raise NotImplementedError
            self._freeze_extra_feat_model()
            self.extra_feat_model.eval()
        else:
            self.extra_feat_model = None

        if blk_type == 'unet':
            self.blks = nn.ModuleList([
             UnetBlock(ngf, ngf, ngf, nn.InstanceNorm2d, add_extra=(extra_feat is not None)) for _ in range(num_blks)
            ])
        elif blk_type == 'mst':
            self.blks = nn.ModuleList([
             MST(in_dim=ngf, out_dim=ngf, dim=ngf, stage=2, num_blocks=[1,1,1]) for _ in range(num_blks)])
        if final_layer == 'conv':
            self.final_conv = nn.Conv2d(ngf, 31, 1, 1, bias=False)
            self.use_csc = False
        elif final_layer == 'csc':
            self.final_conv = CoupleSparseCodingLayer(ngf, 5, 31, 300)
            self.use_csc = True
        elif final_layer == 'cscv2':
            self.final_conv = CoupleSparseCodingLayer(ngf, 3, 31, 300, method='elista')
            self.use_csc = True
        else:
            raise NotImplementedError
        
        
    
    def forward(self, in_tensor):
        res_f = self.conv1(in_tensor)
        if self.extra_feat_model is not None:
            with torch.no_grad():
                extra_feat = self.extra_feat_model(in_tensor)
            b, hw, c = extra_feat.shape
            h_n_w = int(math.sqrt(hw))
            extra_feat = extra_feat.reshape(b, h_n_w, h_n_w, c).permute(0, 3, 1, 2)
        else:
            extra_feat = None

        f = res_f
        # print(torch.std_mean(f))
        for blk in self.blks:
            f = blk(f)
            # f = f + f_next
            # print(torch.std_mean(f))
        # f = f + res_f
        if self.use_csc:
            f_o, logits, recons_feat = self.final_conv(f, True, True) # return logits and recons_feat
            # print("recons", torch.std_mean(recons_feat))
            # print(torch.mean((logits == 0).to(torch.float32)))
            error_feat = recons_feat - f
        else:
            f_o = self.final_conv(f)
            error_feat = None
        return f_o, error_feat
    

    # just to all the way eval pre-trained transformer
    def train(self, mode:bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not name == 'extra_feat_model':
                module.train(mode)
        return self
    
    def _freeze_extra_feat_model(self):
        self.extra_feat_model.requires_grad_(False)
    
    def load_and_freeze_backbone(self, load_dir):
        ckpt = torch.load(load_dir)["model_state"]
        no_final_layer = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items() if "final_conv" not in k}
        self.load_state_dict(no_final_layer, strict=False)
        
        for name, param in self.named_parameters():
            if not "final_conv" in name:
                param.requires_grad = False

        n_frozen = sum(not p.requires_grad for p in self.parameters())
        n_total = sum(1 for _ in self.parameters())
        print(f"✅ {n_frozen}/{n_total} parameters frozen.")



if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    
    dummy_input = torch.randn((1,3,224,224), dtype=torch.float32)
    print(torch.std_mean(dummy_input))
    # model = UnetHSI(128, 4, extra_feat='ijepa').train()
    model = UnetHSI(128, 4, final_layer='csc', extra_feat=None).train()
    hs_pred, feat_recons = model(dummy_input)
    print(hs_pred.shape)
    print(torch.std_mean(hs_pred))
    print(torch.std_mean(feat_recons))
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                            print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
