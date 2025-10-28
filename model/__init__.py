from yacs.config import CfgNode as CN
from .mstpp import *
from .srcgan import SRCGAN_G, SRCGAN_D
from .unet import UnetHSI
from .hscnnpp import HSCNN_Plus
from .hsgan import Generator, Discriminator

def build_model(cfg:CN):
    if cfg.MODEL.TYPE == 'MSTPP':
        model = MST_Plus_Plus()
    elif cfg.MODEL.TYPE == "SRCGAN":
        model = SRCGAN_G()
    elif cfg.MODEL.TYPE == 'HSISCN_L': # HSISCN with LISTA
        model = UnetHSI(128, 4, blk_type='unet', final_layer='csc', extra_feat=None)
    elif cfg.MODEL.TYPE == 'HSISCN_E': # HSISCN with ELISTA
        model = UnetHSI(128, 4, blk_type='unet', final_layer='cscv2', extra_feat=None)
    elif cfg.MODEL.TYPE == 'HSCNNPP':
        model = HSCNN_Plus()
    elif cfg.MODEL.TYPE == 'HSGAN':
        model = Generator(None)
    else:
        raise NotImplementedError
    return model

def build_discriminator(cfg:CN):
    if cfg.MODEL.TYPE == "SRCGAN":
        model = SRCGAN_D()
    if cfg.MODEL.TYPE == "HSGAN":
        model = Discriminator(None)
    else:
        raise NotImplementedError
    return model