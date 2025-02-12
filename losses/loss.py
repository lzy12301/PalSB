import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys


class MatchingLoss(nn.Module):
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
            loss = weights * loss

        return loss.mean()


class PhysLoss(nn.Module):
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
        if type == 'vorticity':
            self.phys_loss = lambda x: voriticity_residual(x, return_residual=True, **kwargs)

    def forward(self, predict, target, weights=None):

        res = self.phys_loss(predict)
        loss = self.loss_fn(res, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


def voriticity_residual(w, re=1000.0, dt=1/32, return_residual=False):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    with torch.enable_grad():
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
        if return_residual:
            return residual
        residual_loss = (residual**2).mean()
        dw = torch.autograd.grad(residual_loss, w)[0]
    return dw.detach()


def bound_loss(w, bound_type='periodic'):
    if 'periodic' in bound_type.lower():
        res_h = ((w[..., -1, :] - w[..., 0, :])**2).mean()
        res_w = ((w[..., -1] - w[..., 0])**2).mean()
        return res_h + res_w
    else:
        raise ValueError('Unknown boundary type!')
