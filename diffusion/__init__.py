from .i2sb import Diffusion, make_beta_schedule

def get_diffusion(opt):
    betas = make_beta_schedule(n_timestep=opt.num_scales, linear_end=opt.beta_max / opt.num_scales)
    return Diffusion(betas, device=opt.device)