import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from logger import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch.optim import AdamW, lr_scheduler
import numpy as np
from tqdm import tqdm
import json
from collections import namedtuple

from datasets import get_dataset
from modules_irsde import get_conditional_unet
from diffusion import get_diffusion
from corruptions import get_corrupt_func
from losses import get_loss_func

from configs.config_kol_sr import config


def train(opt, log):
    if 'diffusion' in opt.model_name.lower():
        diffusion = get_diffusion(opt)
    corrupt_func = get_corrupt_func(opt, noise=True)
    net = get_conditional_unet(opt)
    ema = ExponentialMovingAverage(net.parameters(), decay=opt.ema_rate)
    loss_func = get_loss_func(opt)
    optimizer = AdamW(net.parameters(), lr=opt.lr, weight_decay=0)
    sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)

    checkpoint_path = opt.results_path + '/checkpoint.pt'
    checkpoint_save_path = opt.results_path + '/checkpoint.pt'      # _tunephys
    if opt.continue_training:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        log.info(f"[Net] Loaded network ckpt: {checkpoint_path}!")
        ema.load_state_dict(checkpoint["ema"])
        log.info(f"[Ema] Loaded ema ckpt: {checkpoint_path}!")
        optimizer.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["sched"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opt.device)

    net = nn.DataParallel(net)
    # net.to(opt.device)
    ema.to(opt.device)

    data = np.load(opt.data_location)
    dataset_train = get_dataset(opt, log, data, train=True)
    dataset_val = get_dataset(opt, log, data, train=False)
    dataloader_train = DataLoader(dataset_train, batch_size=opt.small_batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=opt.small_batch_size, shuffle=True)

    net.train()
    i_outer = 0
    i_inner = 0
    n_inner_loop = opt.batch_size // opt.small_batch_size
    for i, (x0, y) in enumerate(dataloader_train):      # x0 refers to the high-res label data, while y can be any thing that you want to use in corrupt_func
        if i % n_inner_loop == 0:
            optimizer.zero_grad()
        
        x0 = x0.to(opt.device)
        _, x1 = corrupt_func(x0, y, noise=opt.noise_level)

        if 'diffusion' in opt.model_name.lower():
            step = torch.randint(0, opt.num_scales, (x0.shape[0],))
            xt = diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
            x0_pred = net(xt, x1, step)
        else:
            if 'fno' in opt.model_name.lower():
                X, Y = torch.meshgrid(torch.linspace(0, 1, x0.shape[-2]), torch.linspace(0, 1, x0.shape[-1]))
                grid = torch.stack([X, Y], axis=-1).float().to(device)
                x0_pred = net(x1.permute(0, 2, 3, 1), grid.repeat(len(x0), 1, 1, 1)).squeeze(-2).permute(0, 3, 1, 2)
            else:
                x0_pred = net(x1)

        loss = loss_func(x0_pred, x0, y)
        loss.backward()
        i_inner += 1

        if i_inner == n_inner_loop:
            optimizer.step()
            ema.update()
            sched.step()
            i_outer += 1
            i_inner = 0
            if i_outer == opt.n_iter:
                break
        
            '''logging'''
            if i_outer % opt.print_freq == 0:
                lr_curr = optimizer.param_groups[0]['lr']
                log.info(f'training {i_outer}/{opt.n_iter} | lr: {lr_curr:.2e} | loss: {loss.item():.4f}')
            
            '''save model'''
            if i_outer % 500 == 0:
                torch.save({
                    "net": net.module.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict() if sched is not None else sched,
                }, checkpoint_save_path)
                log.info(f"Saved latest({i_outer=}) checkpoint to {checkpoint_path=}!")
    return


if __name__ == "__main__":
    config.cuda = config.gpu is not None
    if config.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    config.device = device

    '''create results folder'''
    path = config.results_path + '/' + config.data + '_' + config.version
    config.results_path = path

    used_para = dict(
        batch_size=config.batch_size,
        small_batch_size=config.small_batch_size,
        crop_size=config.crop_size,        # config.image_size
        n_iter=config.n_iter,
        data_location=config.data_location,
        phys_loss=config.phys_loss,
        noise_level=0.1
        )

    if not os.path.exists(path):
        os.mkdir(path)
    if not config.continue_training:
        with open(config.results_path + "/opt.json", mode="w") as f:
            json.dump(config.__dict__, f, indent=4)
    else:
        '''load option file'''
        opt_path = path + '/opt.json'
        with open(opt_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config['continue_training'] = True
            for key in used_para.keys():
                config[key] = used_para[key]
        OPT_class = namedtuple('OPT_class', config.keys())
        config = OPT_class(**config)

    log = Logger(0, path)
    log.info('**************************************')
    log.info('           start training !           ')
    log.info('**************************************')
    train(config, log)
