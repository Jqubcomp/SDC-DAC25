
from diffusion_trainer import Trainer1D
from architecture import UB_Diff
import torch
import numpy as np
import random
import wandb 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Diffusion Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-c', '--checkpoint_path', default='../encoder_decoder/checkpoints/training/check_point.pth', type=str, help='checkpoint path')
    parser.add_argument('-s', '--save_path', default='./gen_data', type=str, help='save path')
    parser.add_argument('-n', '--num_samples', default=500, type=int, help='number of samples')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('-p', '--project', default='CFA', type=str, help='wandb project name')
    parser.add_argument('-pn', '--proj_name', default='SDC_DAC_CFA2', type=str, help='wandb run name')
    parser.add_argument('-td', '--train-data', default='./seismic_data/', type=str, help='train data path')
    parser.add_argument('-tl', '--train-label', default='./velocity_map/', type=str, help='train label path')
    parser.add_argument('-da', '--dataset', default='curvefault-a', type=str, help='dataset name')
    parser.add_argument('-nd', '--num_data', default=48000, type=int, help='number of training data')
    parser.add_argument('-ts', '--time_steps', default=256, type=int, help='diffusion time steps')
    parser.add_argument('-lr', '--learning_rate', default=8e-5, type=float, help='learning rate')
    parser.add_argument('-ns', '--num_steps', default=150000, type=int, help='number of training steps')
    parser.add_argument('-sse', '--save_and_sample_every', default=30000, type=int, help='save and sample every')
    parser.add_argument('-rf', '--results_folder', default='./checkpoints', type=str, help='results folder')
    parser.add_argument('--latent_dim', default=128, type=int, help='latent dimension')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    setup_seed(0)
    args = parse_args()
    if args.use_wandb:
        wandb.init(
        # set the wandb project where this run will be logged
            project= args.project,
            name = args.proj_name,
        )
    
    model = UB_Diff(
        1, 
        time_steps=args.time_steps,
        checkpoint_path = args.checkpoint_path, 
        dim_mults=(1,2,2,4),
        objective = 'pred_v', # or 'pred_noise' or 'pred_x0'
        dim5 = args.latent_dim,
        use_wandb = args.use_wandb
        )
    
    trainer = Trainer1D(
        model,
        args.train_data,
        args.train_label,
        args.dataset,
        args.num_data, #all vel can be used to train
        train_lr=args.learning_rate,
        train_num_steps=args.num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder=args.results_folder,
        save_and_sample_every=args.save_and_sample_every
    )

    trainer.train()
