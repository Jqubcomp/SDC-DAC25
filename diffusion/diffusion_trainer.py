import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count
from torch.utils.data import RandomSampler, DataLoader, random_split


import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import S_dataset

from torch.optim import Adam

from torchvision import transforms as T, utils

import matplotlib.pyplot as plt
import transforms as Tr

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

import numpy as np
from ddpm_1d import has_int_squareroot, cycle, exists, num_to_groups, default, extract, convert_image_to_fn
from torchvision.transforms import Compose
import json
import sys
import os
from experiment_log import PytorchExperimentLogger


class Trainer1D(object):
    def __init__(
        self,
        geoFusion_model,
        seis_folder,
        vel_folder,
        dataset,
        num_data,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        k = 1,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.exp_logger = PytorchExperimentLogger('./log', "elog", ShowTerminal=True)

        with open('dataset_config.json') as f:
            try:
                ctx = json.load(f)[dataset]
            except KeyError:
                print('Unsupported dataset.')
                sys.exit()

        self.ctx = ctx

        transform_data = Compose([
            Tr.LogTransform(k=k),
            Tr.MinMaxNormalize(Tr.log_transform(ctx['data_min'], k=k), Tr.log_transform(ctx['data_max'], k=k))
        ])
        transform_label = Compose([
            Tr.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
        ])


        self.model = geoFusion_model
        self.channels = geoFusion_model.diffusion.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        if dataset not in ['flatfault-a', 'curvefault-a', 'flatfault-b', 'curvefault-b']:
            fault_fam = False
        else:
            fault_fam = True
        train_dataset = S_dataset(seis_folder, vel_folder, transform_data, transform_label, pre_load=True, fault_fam=fault_fam)
        train_size = num_data
        test_size = len(train_dataset) - train_size
        dataset_train, _ = random_split(train_dataset, [train_size, test_size])

        print("Training data:", len(dataset_train))
        assert len(dataset_train) == train_size

        print('Creating data loaders')
        train_sampler = RandomSampler(dataset_train)
        dl = DataLoader(
            dataset_train, batch_size=train_batch_size, shuffle = True, num_workers=cpu_count(), pin_memory=True, drop_last=False)


        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(geoFusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(geoFusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    def save_image(self, images, s_path, v_path, milestone, idx=0): #0 means during training
        v, s = self.model.decode(images)

        all_v = Tr.minmax_denormalize(v, self.ctx['label_min'], self.ctx['label_max'])
        all_s = Tr.tonumpy_denormalize(s, self.ctx['data_min'], self.ctx['data_max'])

        all_v = all_v.cpu().numpy()

        save_path_v = str(v_path) + f'/gen_vel-{milestone}-{idx}.npy'
        save_path_s = str(s_path) + f'/gen_seis-{milestone}-{idx}.npy'

        np.save(save_path_v, all_v)
        np.save(save_path_s, all_s)




    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        print(f"run on {device}")
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.exp_logger.print(f"step: {self.step}")


                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data, label = next(self.dl)
                    label = label.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(label)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                self.exp_logger.print(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.diffusion.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)
                        self.save_image(all_images, os.path.join(self.results_folder, 'seis'), os.path.join(self.results_folder, 'vel'), milestone)
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
