import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np


lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class MatchingRewardNeg(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            r = weights * loss
        else:
            r = loss

        return r.mean(-1)


class PhysRewardNeg(nn.Module):
    def __init__(self, type='vorticity', loss_type='l2', is_weighted=False, **kwargs):
        super().__init__()
        self.type = type
        self.is_weighted = is_weighted
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')
        if 'num_grid_bound' in kwargs.keys():
            self.num_grid_bound = kwargs['num_grid_bound']
            del kwargs['num_grid_bound']
        else: 
            self.num_grid_bound = None
        if type == 'vorticity':
            self.phys_loss = lambda x: phys_loss_kol(x, **kwargs)
        elif type == 'rdgs':
            self.phys_loss = lambda x: phys_loss_rdgs(x, **kwargs)
        elif type == 'cylinder':
            self.phys_loss = lambda x: eq_loss_cont(x, **kwargs)

    def forward(self, predict, target, weights=None):

        if self.num_grid_bound is not None:
            n = self.num_grid_bound
            predict = predict[..., n:-n, n:-n]
        res = self.phys_loss(predict)
        loss = self.loss_fn(res, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            r = weights * loss
        else:
            r = loss

        return r.mean(-1)


def eq_loss_cont(w_, dx=6.256e-4/6e-3):
    # w [b t h w]
    w = w_.clone()
    w.requires_grad_(True)
    u = w[:, ::2]
    v = w[:, 1::2]
    # ux = (u[..., 2:, 1:-1] - u[..., :-2, 1:-1])/(2*dx)
    # vy = (v[..., 1:-1, 2:] - v[..., 1:-1, :-2])/(2*dx)
    ux = (u[..., 1:-1, 2:] - u[..., 1:-1, :-2])/(2*dx)
    vy = (v[..., 2:, 1:-1] - v[..., :-2, 1:-1])/(2*dx)   

    residual = ux + vy
    return residual


def phys_loss_kol(w, re=1000.0, dt=1/32):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f
    # residual_loss = (residual**2).mean()
    return residual


def phys_loss_rdgs(w, dt=1., dx=1.):
    laplace = Conv2dDerivative(
            DerFilter = lapl_op,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').to(w.device)
    # forward/backward derivative operator 
    dt = Conv1dDerivative(
        DerFilter = [[[-1/2, 0, 1/2]]],
        resol = (dt),
        kernel_size = 3,
        name = 'partial_t').to(w.device)

    fwd_dt = Conv1dDerivative(
        DerFilter = [[[-3/2, 2, -1/2]]],
        resol = (dt),
        kernel_size = 3,
        name = 'forward_partial_t').to(w.device)

    bwd_dt = Conv1dDerivative(
        DerFilter = [[[1/2, -2, 3/2]]],
        resol = (dt),
        kernel_size = 3,
        name = 'backward_partial_t').to(w.device)

    # w.shape = [B, (T C) H W]
    u = w[:, ::2]
    v = w[:, 1::2]
    num_b, num_t, num_h, num_w = u.shape

    u = einops.rearrange(u, 'b t h w -> (b t) () h w')
    v = einops.rearrange(v, 'b t h w -> (b t) () h w')
    lap_u = einops.rearrange(laplace(u), '(b t) c h w -> t b c h w', b=num_b)
    lap_v = einops.rearrange(laplace(v), '(b t) c h w -> t b c h w', b=num_b)

    u = einops.rearrange(w[:, ::2, 2:-2, 2:-2], 'b t h w -> (h w b) () t')
    v = einops.rearrange(w[:, 1::2, 2:-2, 2:-2], 'b t h w -> (h w b) () t')

    u_t = dt(u)
    # u_t0 = fwd_dt(u[:, :, 0:3])
    # u_tn = bwd_dt(u[:, :, -3:])
    # u_t = torch.cat([u_t0, u_t, u_tn], dim=2)
    u_t = einops.rearrange(u_t, '(h w b) c t -> t b c h w', b=num_b, h=num_h-4)

    v_t = dt(v)
    # v_t0 = fwd_dt(v[:, :, 0:3])
    # v_tn = bwd_dt(v[:, :, -3:])
    # v_t = torch.cat([v_t0, v_t, v_tn], dim=2)
    v_t = einops.rearrange(v_t, '(h w b) c t -> t b c h w', b=num_b, h=num_h-4)

    u = einops.rearrange(w[:, ::2, 2:-2, 2:-2], 'b t h w -> t b () h w')
    v = einops.rearrange(w[:, 1::2, 2:-2, 2:-2], 'b t h w -> t b () h w')

    # make sure the dimensions consistent
    lap_u = lap_u[1:-1]
    lap_v = lap_v[1:-1]
    u = u[1:-1]
    v = v[1:-1]
    assert lap_u.shape == u_t.shape
    assert u_t.shape == v_t.shape
    assert lap_u.shape == u.shape
    assert lap_v.shape == v.shape

    Du, Dv = 0.16, 0.08
    f, k = 0.06, 0.062
    f_u = Du*lap_u - u*v**2 + f*(1-u) - u_t
    f_v = Dv*lap_v + u*v**2 - (f+k)*v - v_t

    return einops.rearrange(torch.cat([f_u, f_v], dim=2), 't b c h w -> b (t c) h w')



def bound_loss(w, type='periodic'):
    if 'periodic' in type.lower():
        return 
