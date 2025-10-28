import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """Define different GAN objectives.
    Adopted from Cycle-GAN implementation
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        # self.register_buffer('real_label', torch.tensor(target_real_label))
        # self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = torch.ones_like(prediction, device=prediction.device)
        else:
            target_tensor = torch.zeros_like(prediction, device=prediction.device)
        return target_tensor

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class LossModule(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        loss_setup = cfg.TRAIN.LOSS
        self.loss_modules = []
        self.loss_names = []
        self.loss_weights = []
        for i in range(len(loss_setup) // 2):
            name = loss_setup[i * 2]
            weight = loss_setup[i * 2 + 1]
            if name == "MRAE":
                loss_module = lambda x, gt: torch.mean(torch.abs(x - gt) / torch.abs(gt).clamp_min(0.1))
            elif name == "MSE":
                loss_module = torch.nn.MSELoss()
            elif name == "ADV":
                # loss_module = GANLoss('vanilla')
                loss_module = GANLoss('lsgan')
            elif name == "L1":
                loss_module = torch.nn.L1Loss()
            elif name == "WD_ATTN":
                loss_module = torch.nn.L1Loss()
            elif name == "WD_ERR_FEAT":
                loss_module = torch.nn.MSELoss()
            else:
                raise NotImplementedError
            
            self.loss_modules.append(loss_module)
            self.loss_names.append(name)
            self.loss_weights.append(float(weight))
        
    def forward(self, pred:torch.Tensor, gt: torch.Tensor, model_D: nn.Module = None, err_feat: torch.Tensor = None):

        loss_all = 0.0 # loss for the generator
        return_dict = {}
        for loss_mod, name, weight in zip(self.loss_modules, self.loss_names, self.loss_weights):
            name_add = name
            if name == "ADV": # GAN LOSS
                assert model_D is not None, "Define discriminator when using GAN loss!"
                # just compute the loss to update Generator
                name_add = "ADV_G"
                loss = loss_mod(model_D(pred), True) # For generator, label True
            # elif name == "WD_ATTN":
            #     assert attn is not None, "Input attention using attention weight decay!"
            #     loss = loss_mod(attn, torch.zeros_like(attn))
            elif name == "WD_ERR_FEAT":
                assert err_feat is not None, "Input attention using attention weight decay!"
                loss = loss_mod(err_feat, torch.zeros_like(err_feat))
            else:
                loss = loss_mod(pred, gt)
            loss_all += weight * loss
            return_dict[name_add] = loss
        return_dict['ALL'] = loss_all
        return return_dict

    def get_GAN_D_loss(self, pred:torch.Tensor, gt: torch.Tensor, model_D: nn.Module = None):
        idx = self.loss_names.index("ADV")
        loss_mod = self.loss_modules[idx]
        loss_D_real = loss_mod(model_D(gt), True)
        loss_D_fake = loss_mod(model_D(pred.detach()), False) # detach in case the backward function called
        loss = (loss_D_fake + loss_D_real) * 0.5
        return loss