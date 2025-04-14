from ddpm_1d import GaussianDiffusion1D

import torch
import torch.nn.functional as F


class GaussianDiffusion1DDefault(GaussianDiffusion1D):
    def __init__(self, model, seq_length, objective, betas,  time_scale=1, gamma = 0, use_wandb = False):
        super(GaussianDiffusion1DDefault, self).__init__(model=model, seq_length=seq_length, betas=betas, objective = objective, time_scale=time_scale, use_wandb=use_wandb)
        self.gamma = gamma