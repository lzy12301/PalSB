# Physics-aligned Schr\"{o}dinger Bridge

This is a demo code for the conference paper "Physics-aligned field reconstruction with diffusionn bridge" submitted to the ICLR 2025, which is a diffusion-based method for physical field reconstruction that can generates high-quality contents with enhanced physical compliance.

This code is based on these two repositories: https://github.com/NVlabs/I2SB ([I2SB](https://arxiv.org/abs/2302.05872)) and https://github.com/Algolzw/image-restoration-sde ([IRSDE](https://proceedings.mlr.press/v202/luo23b.html)).

Since the whole datsets for all three cases can be vary large, here, we only provide a demo for Kolmogorov flow.

## 1. Install Dependencies

To install the dependencies, please enter your own virtual environment, and then run

    pip install -r requirements.txt

To install pytorch, use the following command

    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 torch-ema==0.3 --extra-index-url https://download.pytorch.org/whl/cu116

or you can follow the instruction at the official website: https://pytorch.org/get-started/previous-versions/ to install the suitable version of pytorch.

## 2. Download Dataset

Before running this code, you need to download the training and testing datasets for Kolmogorov flow from the link provided in paper https://doi.org/10.1016/j.jcp.2023.111972. Then you need to rearrange and resave the .npy file from dimension 'B\*T\*H\*W' to dimension 'B\*T\*H\*W\*C', where the extra dimension C=1.

## 3. Pretrain

If you don't want to retrain the model, you can download the fully trained checkpoints from Google Drive after the paper's acceptance, where the folders 'kol_pretrain_fi' and 'kol_pretrain_ri' contains the checkpoints for FI task and RI task, respectively. After downloading, please put these two folders into the path './\_\_results__/', and then you can directly go to the step 5 for further testing.

To train the model on the fourier-space interpolation task, run

    CUDA_VISIBLE_DEVICES=0 python ./train.py --data kol --version <version> --corrupt_method SR --interp_method bicubic --batch_size 64 --small_batch_size 64 --crop_size 64 --n_iter 100001 --noise_level 0.05 --data_location <path_to_the_dataset>

To train the model on the real-space interpolation task, run

    CUDA_VISIBLE_DEVICES=0 python ./train.py --data kol --version <version> --corrupt_method random_points --masks 0.98 0.995 10000 --batch_size 64 --small_batch_size 64 --crop_size 64 --n_iter 100001 --noise_level 0 --data_location <path_to_the_dataset>

where you can change \<version\> to any words you like to denote this run, and change <path_to_the_dataset> to your own path of the dataset.

## 4. Finetune

After pretraining stage, run the following code for physics-aligned finetuning.

For fourier-space interpolation task, run

    CUDA_VISIBLE_DEVICES=0 python ./fine_tune.py --data kol --version <version> --phys_loss 0.5 --match_loss 10. --data_location <path_to_the_dataset>

For real-space interpolation task, run

    CUDA_VISIBLE_DEVICES=0 python ./fine_tune.py --data kol --version <version> --phys_loss 0.5 --match_loss 10. --masks 0.98 0.995 1000 --data_location <path_to_the_dataset>

Note that, since the finetuning is based on the pretrained model, the 'version' here should be consistent with the choice in the pretraining stage.

## 5. Test

To test the performance of the model, we provide notebooks (namely test_kol_fi.ipynb and test_kol_ri.ipynb) in the root path of this project to interactively show the results for FI and RI tasks, respectively. Remember to change the name of 'version' in the first block of the notebook file to specify the checkpoint you want to check.

If you want to run the pretrained models provided by us, directly run the notebook block-by-block once the pretrained checkpoints are downloaded and properly put into the correct place.


## Citation

```
@inproceedings{li2025physicsaligned,
    title       =   {Physics-aligned field reconstruction with diffusion bridge},
    author      =   {Zeyu Li and Hongkun Dou and Shen Fang and Wang Han and Yue Deng and Lijun Yang},
    booktitle   =   {The Thirteenth International Conference on Learning Representations},
    year        =   {2025},
    url         =   {https://openreview.net/forum?id=D042vFwJAM}
}

```