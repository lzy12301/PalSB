import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset


class DatasetSSTSR(Dataset):
    def __init__(self, opt, log, data, train=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_h = data.shape[-3]
        self.num_w = data.shape[-2]
        self.num_c = data.shape[-1]
        data = data[:, :int(self.num_t*opt.train_portion)] if train else data[:, int(self.num_t*opt.train_portion):]
        self.num_t = len(data[0])
        self.crop_size = opt.crop_size
        self.length = opt.num_train if train else opt.num_val
        self.scale = opt.scale
        data, masks = data[..., :1], data[..., 1:]

        self.mean = np.mean(data[0, 0, ..., 0][np.where(abs(data[0, 0, ..., 0])>1e-12)])
        self.std = np.std(data[0, 0, ..., 0][np.where(abs(data[0, 0, ..., 0])>1e-12)])
        np.save(opt.results_path + '/mean_std.npy', [self.mean, self.std])
        self.data = ((data-self.mean)/self.std)*masks
        self.data = torch.from_numpy(self.data)

        log.info(f"[Dataset] Building interpolated low-res data for {'training' if train else 'evaluating'}!")
        data_lr_path = opt.data_location.split('train_')[0] + 'train_lr_' + opt.data_location.split('train_')[-1]
        data_lr = np.load(data_lr_path)
        data_lr = data_lr[:, :self.num_t] if train else data_lr[:, self.num_t:]
        data_lr = (data_lr-self.mean)/self.std
        data_lr = rearrange(data_lr, 'b t h w c -> (b t) c h w')
        self.data_lr = nn.functional.interpolate(torch.from_numpy(data_lr), scale_factor=int(self.num_w/data_lr.shape[-1]), mode='bilinear', align_corners=False)
        self.data_lr = rearrange(self.data_lr, '(b t) c h w -> b t h w c', b=self.num_b)
        self.data_lr = {'lr': self.data_lr, 'masks': torch.from_numpy(masks)}
        log.info(f"[Dataset] Built SST dataset for {'training' if train else 'evaluating'}!")

    def __getitem__(self, item):
        # i_b = int(item%self.num_b)
        # i_t = int(item//self.num_b)
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t, 1)[0]
        i_h = np.random.choice(self.num_h-self.crop_size+1, 1)[0]
        i_w = np.random.choice(self.num_w-self.crop_size+1, 1)[0]
        x = self.data[i_b, i_t, i_h:i_h+self.crop_size, i_w:i_w+self.crop_size]
        x_low = self.data_lr['lr'][i_b, i_t, i_h:i_h+self.crop_size, i_w:i_w+self.crop_size]
        mask = self.data_lr['masks'][i_b, i_t, i_h:i_h+self.crop_size, i_w:i_w+self.crop_size]
        x = x.permute(2, 0, 1)
        x_low = x_low.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        return x.float(), {'lr': x_low.float(), 'masks': mask.long()}       # B C H W {'lr': torch.from_numpy(x_low).float(), 'hr': torch.from_numpy(x).float()}

    def __len__(self):
        return self.length
