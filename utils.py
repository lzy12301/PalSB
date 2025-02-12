import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from scipy.spatial import Voronoi, cKDTree


def plot_field(fields, row, col, dpi=100, q_range=None, save_fig=None):
    figsize = (col, row)
    fig, axes = plt.subplots(row, col, tight_layout=True, figsize=figsize, dpi=dpi)
    fields = fields.reshape(row, col, *fields.shape[1:])
    for i in range(row):
        for j in range(col):
            field = fields[i, j]
            pc = axes[i, j].pcolormesh(field, cmap='RdBu_r')
            if q_range is not None:
                pc.set_clim(q_range)
            axes[i, j].axis('off')
            axes[i, j].set_aspect(1)
    plt.show()
    if save_fig is not None:
        fig.savefig('./results/'+save_fig)


def cal_rmse(gt, pred, normalize=True, reduct='sum'):
    # reduct = 'sum' or 'mean' etc.
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    rmse = []
    for a, b in zip(gt, pred):
        if normalize:
            coeff = 1./lib_name.sqrt(reduct_fn(a**2))
        else:
            coeff = 1.
        rmse.append(coeff*lib_name.sqrt(reduct_fn((a-b)**2)))
    return np.array(rmse) if isinstance(a, np.ndarray) else torch.tensor(rmse)


def cal_mse(gt, pred, normalize=True):
    rmse = []
    for a, b in zip(gt, pred):
        if normalize:
            if isinstance(a, np.ndarray):
                coeff = 1./np.sqrt(np.sum(a**2))
            else:
                coeff = 1./(a**2).mean().sqrt()
        else:
            coeff = 1.
        rmse.append(coeff*np.sqrt(np.mean((a-b)**2)) if isinstance(a, np.ndarray) else coeff*((a-b)**2).mean().sqrt())
    return np.array(rmse) if isinstance(a, np.ndarray) else rmse


def kes_from_vorticity(w, dx, dy, num_bins=30):
    """
    Calculates the kinetic energy spectrum of a 2D velocity field.
    Args:
        w (ndarray): 2D array containing the y-component of the vorticity field.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
    Returns:
        k (ndarray): 1D array containing the wavenumber values.
        energy_spectrum (ndarray): 1D array containing the kinetic energy spectrum values.
    """
    # Calculate the wavenumber values
    w_hat = np.fft.fft2(w)
    Nx, Ny = w.shape
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(kx ** 2 + ky ** 2)
    # Calculate the kinetic energy spectrum
    energy_spectrum = 0.5 * (np.abs(w_hat) ** 2)
    energy_spectrum = energy_spectrum.flatten()
    k = k.flatten()
    # Bin the kinetic energy spectrum values by wavenumber
    # num_bins = int(np.ceil(np.max(k) / (2 * np.pi / dx)))
    num_bins = num_bins
    bin_edges = np.linspace(0, np.max(k), num_bins + 1)
    digitized = np.digitize(k, bin_edges)
    bin_means = np.zeros(num_bins)
    for i in range(num_bins):
        bin_means[i] = np.mean(energy_spectrum[digitized == i])
    return bin_edges[:-1], bin_means


def kes_plot(vors, dx, dy, start, end, num_bins=30, desc=None, is_plot=False, savefig=None):
    E_mean = [0 for _ in range(len(vors))]
    k, _ = kes_from_vorticity(vors[0][0], dx=dx, dy=dy, num_bins=num_bins)
    for t in range(start, end):
        for i, vor in enumerate(vors):
            _, E = kes_from_vorticity(vor[t], dx=dx, dy=dy, num_bins=num_bins)
            E_mean[i] += E
    total = end - start
    E_mean = [i/total for i in E_mean]
    E_mean_k5 = [i*k**5 for i in E_mean]
    if is_plot:
        fig1, ax1 = plt.subplots()
        for i in range(len(vors)):
            ax1.loglog(k, E_mean[i], label=desc[i])
        plt.xlabel('Wavenumber')
        plt.ylabel('Kinetic Energy Spectrum')
        plt.legend()
        plt.show()
        if savefig is not None:
            fig1.savefig('./results/energy_spectrum_time_series.pdf')
            fig1.savefig('./results/energy_spectrum_time_series.png')

        fig2, ax2 = plt.subplots()
        for i in range(len(vors)):
            ax2.loglog(k, E_mean_k5[i], label=desc[i])
        plt.xlabel('Wavenumber')
        plt.ylabel('Kinetic Energy Spectrum')
        plt.legend()
        plt.show()
        if savefig is not None:
            fig2.savefig('./results/energy_spectrumk5_time_series.pdf')
            fig2.savefig('./results/energy_spectrumk5_time_series.png')
    return E_mean, E_mean_k5


def nearest_interp(matrix, mask):
    """
    Completes a masked matrix using nearest-neighbor interpolation.

    Args:
        matrix (ndarray): 2D array containing the masked matrix.
        mask (ndarray): 2D bool-type array

    Returns:
        completed_matrix (ndarray): 2D array containing the completed matrix.
    """

    # Find the indices of the masked points
    masked_indices = np.argwhere(mask==False)
    unmasked_indices = np.argwhere(mask==True)

    # Loop over each masked point and fill it in using the nearest-neighbor value
    completed_matrix = matrix.copy()
    for index in masked_indices:
        # Find the indices of the neighboring points
        # index = np.argmin((i - unmasked_indices[:, 0])**2+(j - unmasked_indices[:, 1])**2)
        i_nearest = np.argmin(np.sum((unmasked_indices-index)**2, axis=1))

        # Fill in the masked point with the value of its nearest neighbor
        # completed_matrix[i, j] = matrix[unmasked_indices[index, 0], unmasked_indices[index, 1]]
        index = tuple(index.reshape(-1, 1))
        index_nearest = tuple(unmasked_indices[i_nearest].reshape(-1, 1))
        completed_matrix[index] = matrix[index_nearest]

    return completed_matrix


def voronoi_interp(matrix, mask):
    """
    Completes a masked matrix using voronoi-tessellation interpolation, compatible with
    both numpy.ndarray and torch.Tensor.

    Args:
        matrix (ndarray or Tensor): 2D array containing the masked matrix.
        mask (ndarray or Tensor): 2D bool-type array or tensor.

    Returns:
        completed_matrix (ndarray or Tensor): 2D array or tensor containing the completed matrix.
    """
    
    is_tensor = torch.is_tensor(matrix)

    # Convert to NumPy if input is Tensor
    if is_tensor:
        device = matrix.device
        matrix_np = matrix.detach().cpu().numpy()
        mask_np = mask.numpy()
    else:
        matrix_np = matrix
        mask_np = mask

    # Find the indices of the masked points
    unmasked_indices = np.argwhere(mask_np == True)
    vor = Voronoi(unmasked_indices)
    values = matrix_np[unmasked_indices[:, 0], unmasked_indices[:, 1]]

    # Loop over each masked point and fill it in using the nearest-neighbor value
    grid_x, grid_y = np.meshgrid(range(matrix_np.shape[0]), range(matrix_np.shape[1]), indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    tree = cKDTree(vor.points)
    _, indexes = tree.query(grid_points)
    voronoi_matrix = values[indexes].reshape(grid_x.shape)

    # Convert back to Tensor if the input was Tensor
    if is_tensor:
        return torch.from_numpy(voronoi_matrix).to(device)
    else:
        return voronoi_matrix
    

def voronoi_interp_gpu(matrix, mask):
    """
    Completes a masked matrix using voronoi-tessellation interpolation, optimized for
    torch.cuda.Tensor.

    Args:
        matrix (Tensor): 2D tensor containing the masked matrix on GPU.
        mask (Tensor): 2D bool-type tensor on GPU.

    Returns:
        Tensor: 2D tensor containing the completed matrix on GPU.
    """
    assert torch.is_tensor(matrix) and matrix.is_cuda, "matrix must be a CUDA tensor"
    assert torch.is_tensor(mask) and mask.is_cuda, "mask must be a CUDA tensor"

    # Find the indices of the masked points and transfer them to CPU for Voronoi
    unmasked_indices = torch.nonzero(mask, as_tuple=False).cpu().numpy()
    values = matrix.cpu().numpy()[unmasked_indices[:, 0], unmasked_indices[:, 1]]

    # Compute Voronoi tessellation on CPU
    vor = Voronoi(unmasked_indices)
    tree = cKDTree(vor.points)

    # Prepare grid points on GPU
    grid_x, grid_y = torch.meshgrid(torch.arange(matrix.shape[0], device=matrix.device), 
                                    torch.arange(matrix.shape[1], device=matrix.device), indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).cpu().numpy()

    # Compute nearest neighbors on CPU
    _, indexes = tree.query(grid_points)

    # Transfer the interpolated values back to GPU
    voronoi_matrix = torch.tensor(values[indexes], device=matrix.device).view(matrix.shape)

    return voronoi_matrix


def mask_gen(input_shape, mask_ratio=0.5, seed=None):
    m = np.ones(input_shape)

    indices = [np.arange(i) for i in input_shape]
    I = np.meshgrid(*indices, indexing='ij')
    indices = np.array([index.reshape(-1) for index in I]).transpose(1, 0)
    num_pixel = len(indices)
    if seed is None:
        i_indices = np.random.choice(num_pixel, int(mask_ratio*num_pixel), replace=False)
    else:
        rng = np.random.RandomState(seed)
        i_indices = rng.choice(num_pixel, int(mask_ratio * num_pixel), replace=False)
    indices = indices[i_indices]
    m[tuple(indices.transpose(1, 0))] = 0
    m = m.astype(bool)
    return m, indices


def cal_correlation(gt, pred, standardize=True, reduct='sum'):
    # standardize: whether to substract mean value of input data
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    cossim = []
    for a, b in zip(gt, pred):
        if standardize:
            a_mean = lib_name.mean(a)
            b_mean = lib_name.mean(b)
        else:
            a_mean = 0.
            b_mean = 0.
        a_norm = lib_name.sqrt(reduct_fn(a**2))
        b_norm = lib_name.sqrt(reduct_fn(b**2))
        cossim.append(reduct_fn((a-a_mean).reshape(-1)*(b-b_mean).reshape(-1))/(a_norm*b_norm))
    return np.array(cossim) if isinstance(a, np.ndarray) else torch.tensor(cossim)


def vor_cal(u, v, grid_num, x_range):
    dx = (x_range[1]-x_range[0])/grid_num
    vor = (v[:-1, 1:]-v[:-1, :-1])/dx-(u[1:, :-1]-u[:-1, :-1])/dx
    return vor


def vor_cal_batch(x, grid_num, x_range, reverse=False, method='diff_1st', is_stagger=True):
    # method: 'diff_1st', 'spectral'
    vor = []
    for v in x:
        vx, vy = (v[1], v[0]) if reverse else (v[0], v[1])
        if 'diff_1st' in method:
            vor.append(vor_cal(vx, vy, grid_num, x_range))
        elif 'spectral' in method:
            vor.append(vor_cal_spectral(vx, vy, is_stagger=is_stagger))
        else:
            raise NotImplementedError('No such method for vorticity calculation!')
    return np.array(vor)


def vor_cal_spectral(u, v, is_stagger=True):
    if is_stagger:
        # for staggered grid arrangement, we interpolate velocities from cell faces to cell centres
        u = 0.5 * (u + np.roll(u, 1, axis=1))
        v = 0.5 * (v + np.roll(v, -1, axis=0))
    k_max = len(u)//2
    k = np.concatenate([np.arange(0, k_max, 1), np.arange(-k_max, 0, 1)])
    k_x, k_y = np.meshgrid(k, k)
    F_u = np.fft.fft2(u)
    F_v = np.fft.fft2(v)
    # F_ux = 1j * k_x * F_u
    F_uy = 1j * k_y * F_u
    F_vx = 1j * k_x * F_v
    # F_vy = 1j * k_y * F_v
    # ux = np.fft.ifft2(F_ux)
    uy = np.fft.irfft2(F_uy[..., :k_max+1])
    vx = np.fft.irfft2(F_vx[..., :k_max+1])
    # vy = np.fft.ifft2(F_vy)
    return vx - uy


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


import os
def create_dir(base_folder, seed):
    i = 0
    while(True):
        path = base_folder + f'/experiment_{seed}_No{i}'
        if os.path.exists(path):
            i += 1
        else:
            os.mkdir(path)
            del i
            break
    return path
