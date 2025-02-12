from .loss import *
from .reward import PhysRewardNeg, MatchingRewardNeg, Conv1dDerivative, Conv2dDerivative, lapl_op
import os


def get_loss_func(opt):
    loss = MatchingLoss(loss_type='l1', is_weighted=False)
    if opt.phys_loss > 1e-12:
        if os.path.exists(opt.results_path + '/mean_std.npy'):
            mean, std = np.load(opt.results_path + '/mean_std.npy')
            scalar_inv = lambda x: x*std+mean
        else:
            scalar_inv = lambda x: x

        if 'kol' in opt.data:
            phys_type = 'vorticity'
            kwargs = dict(re=1000, dt=1/32)
        loss_phys = PhysLoss(phys_type, loss_type='l2', is_weighted=False, **kwargs)
    if 'scatter' in opt.corrupt_method:
        return lambda pred, gt, y: loss(pred*y['masks'].to(pred.device), gt) if opt.phys_loss < 1e-12 else\
             loss(pred*y['masks'].to(pred.device), gt) + opt.phys_loss * loss_phys(scalar_inv(pred), torch.tensor(0.).to(pred.device))
    else:
        return lambda pred, gt, y: loss(pred, gt) if opt.phys_loss < 1e-12 else\
             loss(pred, gt) + opt.phys_loss * loss_phys(scalar_inv(pred), torch.tensor(0.).to(pred.device))


def get_reward_func(opt, **addargs):
    reward = MatchingRewardNeg(loss_type='l1', is_weighted=False)
    if os.path.exists(opt.results_path + '/mean_std.npy'):
        mean, std = np.load(opt.results_path + '/mean_std.npy')
        scalar_inv = lambda x: x*std+mean
    else:
        scalar_inv = lambda x: x

    if 'kol' in opt.data:
        phys_type = 'vorticity'
        kwargs = dict(re=1000, dt=1/32, **addargs)
    elif 'rdgs' in opt.data:
        phys_type = 'rdgs'
        kwargs = dict(dx=1., dt=1., **addargs)
    elif 'cylinder' in opt.data:
        phys_type = 'cylinder'
        kwargs = dict(dx=6.256e-4/6e-3)
    reward_phys = PhysRewardNeg(phys_type, loss_type='l2', is_weighted=False, **kwargs)
    if 'scatter' in opt.corrupt_method:
        return lambda pred, gt, y: (reward(pred*y['masks'].to(pred.device), gt), reward_phys(scalar_inv(pred), torch.tensor(0.).to(pred.device))/(scalar_inv(pred)**2).mean())
    else:
        return lambda pred, gt, y: (reward(pred, gt), reward_phys(scalar_inv(pred), torch.tensor(0.).to(pred.device))/(scalar_inv(pred)**2).mean())


def get_control_func(opt):
    if 'kol' in opt.data:
        return lambda x: voriticity_residual(x, re=1000, dt=1/32)
