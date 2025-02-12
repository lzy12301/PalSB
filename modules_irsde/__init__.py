from .DenoisingUNet_arch import ConditionalUNet, ControlledConditionalUNet, ControlUNet, UNet
from .DenoisingNAFNet_arch import ConditionalNAFNet
from .FNO import FNO1d, FNO2d, FNO3d

import torch


def get_conditional_unet(opt):
    if 'diffusion' in opt.model_name.lower():
        noise_levels = torch.linspace(opt.t0, opt.T, opt.num_scales, device=opt.device) * opt.num_scales
        kwargs = dict(
            in_nc=opt.in_channels,
            out_nc=opt.in_channels,
            nf=opt.nf,
            # depth=len(opt.ch_mult),
            ch_mul=opt.ch_mult,
            upscale=1,
            noise_levels=noise_levels
                    )
        return ConditionalUNet(**kwargs)
    elif 'fno2d' in opt.model_name:
        kwargs = dict(
            num_channels=opt.in_channels,
            width=opt.nf,
            modes1=opt.modes1,
            modes2=opt.modes2,
            initial_step=1,
            num_para=2,
                    )
        return FNO2d(**kwargs)
    elif 'unet' in opt.model_name:
        noise_levels = torch.linspace(opt.t0, opt.T, opt.num_scales, device=opt.device) * opt.num_scales
        kwargs = dict(
            in_nc=opt.in_channels,
            out_nc=opt.in_channels,
            nf=opt.nf,
            # depth=len(opt.ch_mult),
            ch_mul=opt.ch_mult,
            upscale=1,
            noise_levels=noise_levels
                    )
        return UNet(**kwargs)
    else:
        raise ValueError('Unknown model name!')


def get_control_unet(opt):
    noise_levels = torch.linspace(opt.t0, opt.T, opt.num_scales, device=opt.device) * opt.num_scales
    kwargs = dict(
        in_nc=opt.in_channels,
        out_nc=opt.in_channels,
        nf=opt.nf,
        depth=len(opt.ch_mult),
        upscale=1,
        noise_levels=noise_levels
                  )
    return ControlledConditionalUNet(**kwargs), ControlUNet(**kwargs)
