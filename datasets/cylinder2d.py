import numpy as np
import h5py
import scipy.io as sio
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange
import torch.nn as nn
import torch


class DatasetCy2d(Dataset):
    def __init__(self, opt, log, data, train=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_h = data.shape[-3]
        self.num_w = data.shape[-2]
        self.num_c = data.shape[-1]
        if self.num_b > 10:
            data = data[:int(self.num_b*opt.train_portion)] if train else data[int(self.num_b*opt.train_portion):]
        else:
            data = data[:, :int(self.num_t*opt.train_portion)] if train else data[:, int(self.num_t*opt.train_portion):]
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.crop_size = opt.crop_size
        self.length = opt.num_train if train else opt.num_val
        self.scale = opt.scale
        self.num_frames = opt.num_frames if hasattr(opt, 'num_frames') else 1

        self.mean = np.mean(data)
        self.std = np.std(data)
        np.save(opt.results_path + '/mean_std.npy', [self.mean, self.std])
        self.data = (data-self.mean)/self.std
        self.data = torch.from_numpy(self.data)
        log.info(f"[Dataset] Built Kolmogorov flow dataset for {'training' if train else 'evaluating'} and number of frames {self.num_frames}!")

        # m = nn.AvgPool2d(scale, stride=scale)
        # temp = rearrange(self.data, 'b t h w c -> (b t) c h w')
        # self.data_low = m(temp)
        # self.data_low = nn.functional.interpolate(self.data_low, size=(self.num_h, self.num_w), mode='bicubic', align_corners=False)
        # self.data_low = rearrange(self.data_low, '(b t) c h w -> b t h w c', b=self.num_b)

    def __getitem__(self, item):
        # i_b = int(item%self.num_b)
        # i_t = int(item//self.num_b)
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames+1, 1)[0]
        i_h = np.random.choice(self.num_h-self.crop_size+1, 1)[0]
        i_w = np.random.choice(self.num_w-self.crop_size+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames, i_h:i_h+self.crop_size, i_w:i_w+self.crop_size]
        # x_low = self.data_low[i_b, i_t, i_h:i_h+self.crop_size, i_w:i_w+self.crop_size]
        # x = x.permute(2, 0, 1)
        x = rearrange(x, 't h w c -> (t c) h w')
        # x_low = x_low.permute(2, 0, 1)
        return x.float(), 1.       # B C H W {'lr': torch.from_numpy(x_low).float(), 'hr': torch.from_numpy(x).float()}

    def __len__(self):
        return self.length