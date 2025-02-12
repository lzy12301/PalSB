import argparse


# Parameters
parser = argparse.ArgumentParser(description='RD-GS SR demo')
parser.add_argument('--continue_training', type=bool, default=True, help='continue training or not')

parser.add_argument('--data', type=str, default='rdgs', help='GMM, Gaussian, cifar10, ns1, ns2, ns3, ns4')
parser.add_argument('--data_location', type=str,
                    default='/home/lzy/projects_dir/generative_model/data/RD-GS/2DGS_IC1_2x3001x256x256.npy',
                    help='data location, /home/lzy/projects_dir/generative_model/data/kol/kol_train_256_Re1000_freq4.npy')
parser.add_argument('--results_path', type=str, default='./__results__', help='GMM, Gaussian, cifar10, ns1, ns2, ns3, ns4')
parser.add_argument('--version', type=str, default='v2_rp', help='model version')
parser.add_argument('--is_train', type=bool, default=True, help='training or evaluating')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of data for training')
parser.add_argument('--num_train', type=int, default=100000000, help='number of samples to be generated using PFGM')
parser.add_argument('--num_val', type=int, default=100000000, help='number of samples to be generated using PFGM')
parser.add_argument('--n_iter', type=int, default=100001, help='number of samples to be generated using PFGM')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=1e-4, help='ns1: 2e-4, ns2: 2e-5, fno: 5e-5')
parser.add_argument('--lr_step', type=int, default=1000, help='')
parser.add_argument('--lr_gamma', type=float, default=0.99, help='')
parser.add_argument('--batch_size', type=int, default=64, help='total batch size')
parser.add_argument('--small_batch_size', type=int, default=64, help='small batch size for inner loops')
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--phys_loss', type=float, default=0, help='')
parser.add_argument('--noise_level', type=float, default=0.1, help='')

# image
parser.add_argument('--image_size', type=int, default=256, help='32 for cifar10, 64 for ns1, 512 for ns2')
parser.add_argument('--crop_size', type=int, default=64, help='32 for cifar10, 64 for ns1, 512 for ns2')
parser.add_argument('--in_channels', type=int, default=6)
parser.add_argument('--num_frames', type=int, default=3)

# SR setting
parser.add_argument('--corrupt_method', type=str, default='random_points', help='SR, SR_scatter, random_points')
parser.add_argument('--scale', type=int, default=8, help='downscaling factor for SR')
parser.add_argument('--masks', type=list, nargs='+', default=[0.98, 0.995, 10000], help='range of masked ratio for random_points')

# model
parser.add_argument('--model_name', type=str, default='ncsnpp', help='ncsnpp')
parser.add_argument('--sigma_min', type=float, default=0.01)
parser.add_argument('--sigma_max', type=float, default=50)
parser.add_argument('--num_scales', type=int, default=1000)
parser.add_argument('--t0', type=float, default=1e-4)
parser.add_argument('--T', type=float, default=1)
parser.add_argument('--beta_min', type=float, default=0.1)
parser.add_argument('--beta_max', type=float, default=0.3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embedding_type', type=str, default='positional')
parser.add_argument('--scale_by_sigma', type=bool, default=True)
parser.add_argument('--sigma_begin', type=int, default=90)
parser.add_argument('--ema_rate', type=float, default=0.995, help='ns1: 0.999, ns2: 0.9999')
parser.add_argument('--normalization', type=str, default='GroupNorm')
parser.add_argument('--nonlinearity', type=str, default='swish')
parser.add_argument('--nf', type=int, default=32, help='128 for all, 16 for autoregressive')
parser.add_argument('--ch_mult', type=tuple, default=(1, 2, 2, 2),
                    help='ns1: (1, 2, 2, 2), '
                         'ns2: (1, 1, 2, 2, 4, 4)')
parser.add_argument('--num_res_blocks', type=int, default=4)
parser.add_argument('--attn_resolutions', type=tuple, default=(16, ))

# diffusion
parser.add_argument('--ot_ode', type=int, default=0)

config = parser.parse_args()        # args=[]
# config_nb = parser.parse_args(args=[])