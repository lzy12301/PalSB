import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from logger import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch.optim import AdamW, lr_scheduler
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
import json
from collections import namedtuple
import argparse
import importlib

from datasets import get_dataset, get_bound_func
from modules_irsde import get_conditional_unet
from diffusion import get_diffusion
from corruptions import get_corrupt_func
from losses import get_loss_func, get_reward_func
from sampling import get_sampling_func, space_indices


parser = argparse.ArgumentParser(description='PalSB test')
parser.add_argument('--data', type=str,                             default='kol',                          help='data name')
parser.add_argument('--version', type=str,                          default='v30_snet',                       help='version of model')
parser.add_argument('--results_path', type=str,                     default='__results__',                  help='version of model')
parser.add_argument('--data_location', type=str,                    default="/media/group3/lzy/Data/kol/kf_2d_re1000_256_40seed_bthwc.npy",)
parser.add_argument('--gpu', type=str,                              default="0",                            help='whether to use gpu')
parser.add_argument('--num_epoches', type=int,                      default=10,                             help='number of outer epoches')
parser.add_argument('--num_inner_epoches', type=int,                default=10,                             help='number of inner epoches')
parser.add_argument('--num_tiny_epoches', type=int,                 default=10,                             help='number of inner-inner epoches')
parser.add_argument('--num_sample_steps_train', type=int,           default=10,                             help='sample steps')
parser.add_argument('--small_batch_size', type=int,                 default=32,                             help='batch size for outer loop')
parser.add_argument('--tiny_batch_size', type=int,                  default=4,                              help='batch size for inner loop')
parser.add_argument('--phys_loss', type=float,                      default=0.5,                            help='weight for physics-informed loss')
parser.add_argument('--match_loss', type=float,                     default=10.,                            help='weight for matching loss')
parser.add_argument('--lr', type=float,                             default=1e-5,                           help='learning rate')
parser.add_argument('--lr_step', type=int,                          default=10,                             help='decay step of learning rate')
parser.add_argument('--lr_gamma', type=float,                       default=0.99,                           help='decay factor of learning rate')
parser.add_argument('--ema_rate', type=float,                       default=0.99,                           help='ema rate')
parser.add_argument('--noise_level', type=float,                    default=0.,                             help='learning rate')
parser.add_argument('--K_sg', type=int,                             default=1,                              help='where to truncate the backprop.')
parser.add_argument('--num_grid_bound', type=int,                   default=2,                              help='whether to use BA in finetune')
parser.add_argument('--masks', type=float, nargs='+',               default=[0.98, 0.995, 1000],            help='parameters for mask')
args = parser.parse_args()


def train(opt, log):

    if opt.bound_extend is not None:
        reward_func = get_reward_func(opt, num_grid_bound=args.num_grid_bound)
        bound_func = get_bound_func(opt.bound_extend, num_grid=args.num_grid_bound)
        crop_bound = lambda x: x[..., args.num_grid_bound*opt.scale:-args.num_grid_bound*opt.scale, args.num_grid_bound*opt.scale:-args.num_grid_bound*opt.scale]
    else:
        reward_func = get_reward_func(opt)
        bound_func = None
        crop_bound = lambda x: x

    if 'diffusion' in opt.model_name:
        diffusion = get_diffusion(opt)
    corrupt_func = get_corrupt_func(opt, noise=True, bound_func=bound_func)
    net = get_conditional_unet(opt)
    # ema = ExponentialMovingAverage(net.parameters(), decay=ema_rate)
    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=0)
    sched = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    checkpoint_path = opt.results_path + '/checkpoint.pt'
    checkpoint_save_path = opt.results_path + '/checkpoint_tunePF.pt'      # _tunephys
    if opt.continue_training:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        log.info(f"[Net] Loaded network ckpt: {checkpoint_path}!")
        # ema.load_state_dict(checkpoint["ema"])
        # log.info(f"[Ema] Loaded ema ckpt: {checkpoint_path}!")
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # sched.load_state_dict(checkpoint["sched"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opt.device)

    net = nn.DataParallel(net)
    # net.to(opt.device)
    # ema.to(opt.device)

    data = np.load(opt.data_location)
    dataset_train = get_dataset(opt, log, data, train=True)
    dataset_val = get_dataset(opt, log, data, train=False)

    dataloader_train = DataLoader(dataset_train, batch_size=args.small_batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=args.small_batch_size, shuffle=True)
    logging = {'reward': [], 'reward_match':[], 'reward_phys': [], 'baseline':[], 'unclipped_loss':[], 'clipped_loss':[], }
    def _updata_logging(**kwargs):
        for key in kwargs.keys():
            if key in logging.keys():
                logging[key].append(kwargs[key])

    buffer_rewards = []
    dataloader_train_iter = iter(dataloader_train)
    for epoch in range(args.num_epoches):
        net.eval()
        x0, y = next(dataloader_train_iter)
        x0 = x0.to(opt.device)
        _, x1 = corrupt_func(x0, y, noise=args.noise_level)
        cond = x1.clone()

        with torch.no_grad():
            if 'diffusion' in opt.model_name:
                sampling_func = get_sampling_func(net, None, num_scales=opt.num_scales, nfe=args.num_sample_steps_train, diffusion=diffusion, log_count=args.num_sample_steps_train, device=opt.device)       # ema
                steps = range(opt.num_scales)[-args.num_sample_steps_train-1:]
                # steps = space_indices(opt.num_scales, num_sample_steps_train+1)
                steps = steps[::-1]
                xs, x0s, log_probs = sampling_func(x1, cond, cal_log_prob=True)     # B T C H W, B T C H W, B T
                # np.save(opt.results_path + '/generated_samples.npy', torch.stack([xs, x0s]).numpy())
                x0s = crop_bound(x0s)
                rewards_match, rewards_phys = reward_func(x0s[:, -1].to(opt.device), x0, cond)
                rewards = -(args.match_loss * rewards_match + args.phys_loss * rewards_phys)       # B
                if epoch == 0:
                    baseline = rewards.mean()*0.9
                else:
                    baseline = baseline*0.9 + rewards.mean()*0.1
                buffer_rewards.append((rewards, rewards_match, rewards_phys))
            elif 'fno' in opt.model_name.lower():
                X, Y = torch.meshgrid(torch.linspace(0, 1, x0.shape[-2]), torch.linspace(0, 1, x0.shape[-1]))
                grid = torch.stack([X, Y], axis=-1).float().to(device)
                x0 = net(x1.permute(0, 2, 3, 1), grid.repeat(len(x0), 1, 1, 1)).squeeze(-2).permute(0, 3, 1, 2)
            else:
                x0 = net(x1)

        for epoch_inner in range(args.num_inner_epoches):
            x0, y = next(dataloader_train_iter)
            x0 = x0.to(opt.device)
            _, x1 = corrupt_func(x0, y, noise=args.noise_level)
            cond = x1.clone()
            with tqdm(range(args.num_tiny_epoches)) as tqdm_setting:
                for i in range(args.num_tiny_epoches):
                    indices = np.random.choice(len(x0), args.tiny_batch_size, replace=False)
                    optimizer.zero_grad()
                    if 'diffusion' in opt.model_name:
                        pair_steps = zip(steps[1:], steps[:-1])
                        xt = x1[indices].to(opt.device)
                        for j, (nprev, n) in enumerate(pair_steps):
                            if j < len(steps)-2-args.K_sg:
                                net.eval()
                                with torch.no_grad():
                                    x0_pred = net(xt, x1[indices], torch.tensor([nprev]).to(opt.device))
                                    xt, log_probs = diffusion.p_posterior(nprev, n, xt, x0_pred, ot_ode=False, cal_log_prob=True)
                            # elif j == len(steps)-2-K_sg:
                            #     net.train()
                            #     x0_pred = net(xt, x1[indices], torch.tensor([nprev]).to(opt.device))
                            #     xt, log_probs = diffusion.p_posterior(nprev, n, xt, x0_pred, ot_ode=False, cal_log_prob=True)
                            else:
                                net.train()
                                x0_pred = net(xt, x1[indices], torch.tensor([nprev]).to(opt.device))
                                xt, log_probs = diffusion.p_posterior(nprev, n, xt, x0_pred, ot_ode=False, cal_log_prob=True)
                    elif 'fno' in opt.model_name:
                        X, Y = torch.meshgrid(torch.linspace(0, 1, x0.shape[-2]), torch.linspace(0, 1, x0.shape[-1]))
                        grid = torch.stack([X, Y], axis=-1).float().to(device)
                        x0_pred = net(x1[indices].permute(0, 2, 3, 1), grid.repeat(args.tiny_batch_size, 1, 1, 1)).squeeze(-2).permute(0, 3, 1, 2)
                    else:
                        x0_pred = net(x1[indices])
                    x0_pred = crop_bound(x0_pred)
                    rewards_match, rewards_phys = reward_func(x0_pred, x0[indices], cond)
                    loss = (args.match_loss * rewards_match + args.phys_loss * rewards_phys).mean()
                    rewards = -loss.item()
                    loss.backward()
                    optimizer.step()
                    # ema.update()
                    sched.step()

                    '''logging'''
                    lr_curr = optimizer.param_groups[0]['lr']
                    tqdm_setting.set_description(f'training {i}/{args.num_tiny_epoches} | lr: {lr_curr:.2e} | loss: {loss.item():.2e}')
                    tqdm_setting.update(1)
            log.info(f'epoch: {epoch}_{epoch_inner}/{args.num_epoches}_{args.num_inner_epoches} | loss: {loss.item():.4f} | '+\
                     f'loss_match: {rewards_match.mean().item():.4f} | loss_phys: {rewards_phys.mean().item():.4f}')
            # logging = {'reward': [], 'reward_match':[], 'reward_phys': [], 'baseline':[], 'baseline':[], 'unclipped_loss':[], 'clipped_loss':[], }
            _updata_logging(
                reward=rewards, 
                reward_match=rewards_match.mean().item(),
                reward_phys=rewards_phys.mean().item(),
                # baseline=baseline.item(),
            )
        '''save model'''
        torch.save({
            "net": net.module.state_dict(),
            # "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sched": sched.state_dict() if sched is not None else sched,
        }, checkpoint_save_path)
        np.save(opt.results_path + '/tuneRL_logging.npy', logging)
        log.info(f"Saved latest({epoch=}) checkpoint to {checkpoint_save_path=}!")


if __name__ == "__main__":
    cuda = args.gpu is not None
    device = 'cuda' if cuda else 'cpu'

    '''create results folder'''
    path = args.results_path + '/' + args.data + '_' + args.version

    used_para = dict(
        continue_training=True,
        device=device,
        results_path=path,
        data_location=args.data_location,
        dropout=0,
        bound_extend=None,      # 'periodic'
        masks=args.masks,
        )

    '''load option file'''
    opt_path = path + '/opt.json'
    with open(opt_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        config['crop_size'] = config['image_size']
        for key in used_para.keys():
            config[key] = used_para[key]
    OPT_class = namedtuple('OPT_class', config.keys())
    config = OPT_class(**config)

    log = Logger(0, path)
    log.info('**************************************')
    log.info('           start training !           ')
    log.info('**************************************')
    train(config, log)
