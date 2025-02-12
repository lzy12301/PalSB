import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from einops import rearrange


def get_corrupt_func(opt, **kwargs):
    if 'sr' in opt.corrupt_method.lower():
        if hasattr(opt, 'interp_method'):
            recon_method = opt.interp_method
        else:
            recon_method = 'bicubic'
        if 'scatter' in opt.corrupt_method.lower():
            return lambda x, y: corrupt_sr_scatter(y, x.device, **kwargs)
        else:
            if 'noise' in kwargs.keys():
                del kwargs['noise']
                return lambda x, y, noise: corrupt_sr(x, scale=opt.scale, recon_method='bicubic', noise=noise, **kwargs)
            else:
                return lambda x, y: corrupt_sr(x, scale=opt.scale, recon_method='bicubic', **kwargs)
    elif 'random_points' in opt.corrupt_method.lower():
        mask_ratio = np.random.rand()*abs(opt.masks[-2]-opt.masks[-3]) + min(opt.masks[-2], opt.masks[-3])
        masks, _ = mask_gen([int(opt.masks[-1]), opt.in_channels, opt.crop_size, opt.crop_size], mask_ratio=mask_ratio)
        grid_x, grid_y = np.meshgrid(range(opt.crop_size), range(opt.crop_size), indexing='ij')
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        nearest_indices_list = []
        unmasked_indices_list = []
        for i_m in tqdm(range(masks.shape[0]), total=opt.masks[-1], desc='building masks for random points corruption!'):
            h_unmasked_indices = []
            h_nearest_indices = []
            try:
                for i_c in range(masks.shape[1]):
                    unmasked_indices = np.argwhere(masks[i_m, i_c]==True)
                    vor = Voronoi(unmasked_indices)
                    tree = cKDTree(vor.points)
                    _, nearest_indices = tree.query(grid_points)
                    h_nearest_indices.append(nearest_indices)
                    h_unmasked_indices.append(unmasked_indices)
            except:
                continue
            nearest_indices_list.append(h_nearest_indices)
            unmasked_indices_list.append(h_unmasked_indices)
        if 'noise' in kwargs.keys():
            del kwargs['noise']
            return lambda x, y, noise: corrupt_random_points(x, unmasked_indices_list, np.stack(nearest_indices_list), noise=noise)
        else:
            return lambda x, y: corrupt_random_points(x, unmasked_indices_list, np.stack(nearest_indices_list))


def get_corrupt_sr(scale, recon_method='bicubic', **kwargs):
    if 'noise' in kwargs.keys():
        return lambda x, noise: corrupt_sr(x, scale=scale, recon_method=recon_method, noise=noise)
    else:
        return lambda x: corrupt_sr(x, scale=scale, recon_method=recon_method, noise=None)


def corrupt_sr_scatter(y, device):
    # x_recon = F.interpolate(y, scale_factor=scale, mode='bilinear', align_corners=False)
    return y['lr'].to(device), y['lr'].to(device)


def corrupt_sr(x, scale, recon_method, noise=None, bound_func=None):
    x_ = x.clone()
    if noise is not None:
        x_ = x_ + torch.randn_like(x_)*noise
    # x_lr = F.avg_pool2d(x_, scale, stride=scale)
    x_lr = x_[..., ::scale, ::scale]
    if bound_func is not None:
        x_lr = bound_func(x_lr)
    if recon_method == 'nearest':
        x_recon = F.interpolate(x_lr, scale_factor=scale, mode=recon_method)
    else:
        x_recon = F.interpolate(x_lr, scale_factor=scale, mode=recon_method, align_corners=False)
    return x_lr, x_recon


def corrupt_rp(x, mask_ratio=0.995, bound_func=None, noise=None, seed=None):
    shape = x.shape
    x_ = x.copy() if isinstance(x, np.ndarray) else x.clone()
    if noise is not None:
        x_ += np.random.randn(*x_.shape)*noise if isinstance(x, np.ndarray) else torch.randn_like(x_)*noise
    x_ = x_.reshape(-1, *shape[-2:])
    x_lr = []
    x_recon = []
    for s in x_:
        mask, _ = mask_gen(s.shape, mask_ratio, seed=seed)
        if isinstance(x_, torch.Tensor):
            mask = torch.from_numpy(mask).to(x_.device)
        h = mask*s
        if bound_func is not None:
            h = bound_func(h)
            mask = bound_func(mask)
        x_lr.append(h)
        x_recon.append(voronoi_interp(h, mask))
    stack_fn = np.stack if isinstance(x_, np.ndarray) else torch.stack
    return rearrange(stack_fn(x_lr), '(b c) h w -> b c h w', b=shape[0]), rearrange(stack_fn(x_recon), '(b c) h w -> b c h w', b=shape[0])


def corrupt_random_points(x, unmasked_indices_list, nearest_indices_list, noise=None, bound_func=None):
    assert len(unmasked_indices_list) == len(nearest_indices_list)
    x_ = x.clone()
    if noise is not None:
        x_ = x_ + torch.randn_like(x_)*noise
    ind_choose = np.random.choice(len(unmasked_indices_list), len(x_))
    x_lr = torch.zeros_like(x_)
    x_recon = x_.clone()
    for i, ind in enumerate(ind_choose):
        for c in range(len(unmasked_indices_list[0])):
            values = x_[i, c, unmasked_indices_list[ind][c][:, 0], unmasked_indices_list[ind][c][:, 1]]
            x_recon[i, c] = values[nearest_indices_list[ind][c]].reshape(x_[i, c].shape)
            x_lr[i, c, unmasked_indices_list[ind][c][:, 0], unmasked_indices_list[ind][c][:, 1]] = values
    return x_lr, x_recon
