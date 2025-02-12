from .era5 import DatasetERA5SR
from .kol import DatasetKolSR
from .channel import DatasetChannelSR
from .sst import DatasetSSTSR
from .rd_gs import DatasetRDGSSR
from .cylinder2d import DatasetCy2d
import torch
import numpy as np


def get_dataset(opt, log, data, train=True):
    kwags = dict(opt=opt, log=log, data=data, train=train)
    if 'era5' in opt.data.lower():
        return DatasetERA5SR(**kwags)
    elif 'kol' in opt.data.lower():
        return DatasetKolSR(**kwags)
    elif 'channel' in opt.data.lower():
        return DatasetChannelSR(**kwags)
    elif 'sst' in opt.data.lower():
        return DatasetSSTSR(**kwags)
    elif 'rdgs' in opt.data.lower():
        return DatasetRDGSSR(**kwags)
    elif 'cylinder' in opt.data.lower():
        return DatasetCy2d(**kwags)
    return 


def get_bound_func(bound_type='periodic', num_grid=3):
    if 'periodic' in bound_type.lower():
        return lambda x: bound_func_periodic(x, num_grid)
    elif 'dirichlet' in bound_type.lower():
        return
    elif 'neumann' in bound_type.lower():
        return
    elif 'robin' in bound_type.lower():
        return
    else:
        raise ValueError('Unknown type of boundary condition!')


def bound_func_periodic(field, num_grid=3):
    roll_fn = torch.roll if isinstance(field, torch.Tensor) else np.roll
    cat_fn = torch.cat if isinstance(field, torch.Tensor) else np.concatenate
    if isinstance(field, torch.Tensor):
        field_ = cat_fn([roll_fn(field, num_grid, -2)[..., :num_grid, :].clone().detach(), field, roll_fn(field, -num_grid, -2)[..., -num_grid:, :].clone().detach()], -2)
        field_ = cat_fn([roll_fn(field_, num_grid, -1)[..., :num_grid].clone().detach(), field_, roll_fn(field_, -num_grid, -1)[..., -num_grid:].clone().detach()], -1)
    else:
        field_ = cat_fn([roll_fn(field, num_grid, -2)[..., :num_grid, :], field, roll_fn(field, -num_grid, -2)[..., -num_grid:, :]], -2)
        field_ = cat_fn([roll_fn(field_, num_grid, -1)[..., :num_grid], field_, roll_fn(field_, -num_grid, -1)[..., -num_grid:]], -1)
    return field_


def bound_func_dirichlet(field, num_grid=3, vb=0, num_dim=2):
    if isinstance(field, np.ndarray):
        field = torch.tensor(field)
    field_ = torch.nn.functional.pad(field, (num_grid for _ in range(num_dim*2)), mode='constant', value=vb)
    return field_ if isinstance(field, torch.Tensor) else field_.numpy()
