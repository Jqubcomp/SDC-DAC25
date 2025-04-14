import itertools

import torch
import torch.nn as nn

from encoder import Encoder_v, Decoder_V, TDecoder_S
from vsnet import VSNet
import copy
from ddpm_1d import Unet1D
from v_diffusion import GaussianDiffusion1DDefault
from ddpm_1d import cosine_beta_schedule


class UB_Diff(nn.Module):
    def __init__(self, in_channels, num_samples = 16, checkpoint_path = None, dim_mults = (1, 2, 2, 2), time_scale = 1, objective = 'pred_v', time_steps = 256, dim5 = 512, use_wandb = False):
        super(UB_Diff, self).__init__()
        self.in_channels = in_channels
        self.dim5 = dim5
        self.load_encoder(checkpoint_path)

        self.unet = Unet1D(dim = self.encoder.dim5, channels = 1, dim_mults=dim_mults)
        betas = cosine_beta_schedule(timesteps=time_steps)
        self.diffusion = GaussianDiffusion1DDefault(model=self.unet, seq_length=self.encoder.dim5, betas=betas,time_scale = time_scale, objective = objective, use_wandb=use_wandb)

        self.load_decoder(checkpoint_path)
        self.leaky_relu = nn.LeakyReLU(0.2)


        self.step = 0
        self.num_samples = num_samples
        self.channels = self.diffusion.channels

    def load_encoder(self, checkpoint_path, freeze = True):
        self.encoder = Encoder_v(in_channels=self.in_channels, checkpoint=checkpoint_path, dim5 = self.dim5)

        original_params = copy.deepcopy(self.encoder.state_dict())
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
            model = VSNet(in_channels= 1, out_channels_s=5, out_channels_v=1, dim5=self.dim5)
            self.out_channels_s = model.out_channels_s
            self.out_channels_v = model.out_channels_v

            model.load_state_dict(checkpoint['model'])
            self.encoder.load_state_dict(model.encoder.state_dict())

            
            loaded_params = self.encoder.state_dict()
            params_match = all(torch.equal(a, b) for a, b in zip(original_params.values(), loaded_params.values()))

            if params_match:
                print("load failed")
                raise ValueError
            else:
                print("params changes")

        if freeze:
            self.encoder = self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("encoder froze")

    def load_decoder(self, checkpoint_path, freeze = True, dd_v = 1, dd_s = 5,  dh_v = 70, dh_s = 1000, dw = 70, vit_latent_dim = 128):
        self.decoder_v = Decoder_V(dim5=dd_v * vit_latent_dim, out_channels=dd_v)
        self.decoder_s = TDecoder_S(dd = dd_s, origin_h=dh_s, origin_w=dw, laten_dim=vit_latent_dim)

        self.fc_v = nn.Linear(in_features=self.encoder.dim5, out_features=dd_v * vit_latent_dim)
        self.fc_s = nn.Linear(in_features=self.encoder.dim5, out_features=dd_s * vit_latent_dim)

        self.batch_norm_v = nn.BatchNorm2d(dd_v * vit_latent_dim)
        self.batch_norm_s = nn.BatchNorm2d(dd_s * vit_latent_dim)

        self.vit_latent_dim = vit_latent_dim

        original_params_v = copy.deepcopy(self.decoder_v.state_dict())
        original_params_s = copy.deepcopy(self.decoder_s.state_dict())
        original_params_fc_s = copy.deepcopy(self.fc_s.state_dict())
        original_params_fc_v = copy.deepcopy(self.fc_v.state_dict())
        original_params_b_s = copy.deepcopy(self.batch_norm_s.state_dict())
        original_params_b_v = copy.deepcopy(self.batch_norm_v.state_dict())

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
            model = VSNet(in_channels= 1, out_channels_s=5, out_channels_v=1, dim5=self.dim5)
            model.load_state_dict(checkpoint['model'])

            self.decoder_v.load_state_dict(model.decoder_v.state_dict())
            self.decoder_s.load_state_dict(model.decoder_s.state_dict())

            self.fc_v.load_state_dict(model.fc_v.state_dict())
            self.fc_s.load_state_dict(model.fc_s.state_dict())

            self.batch_norm_v.load_state_dict(model.batch_norm_v.state_dict())
            self.batch_norm_s.load_state_dict(model.batch_norm_s.state_dict())
            loaded_params_v = self.decoder_v.state_dict()
            loaded_params_s = self.decoder_s.state_dict()
            loaded_params_fc_s = self.fc_s.state_dict()
            loaded_params_fc_v = self.fc_v.state_dict()

            loaded_params_b_s = self.batch_norm_s.state_dict()
            loaded_params_b_v = self.batch_norm_v.state_dict()

            original_params = {**original_params_v, **original_params_s, **original_params_fc_s, **original_params_fc_v, **original_params_b_s, **original_params_b_v}
            loaded_params = {**loaded_params_v, **loaded_params_s, **loaded_params_fc_s, **loaded_params_fc_v, **loaded_params_b_s, **loaded_params_b_v}

            params_match = all(torch.equal(a, b) for a, b in zip(original_params.values(), loaded_params.values()))

            if params_match:
                print("s load failed")
                raise ValueError
            else:
                print("s params changes")

        if freeze:
            self.decoder_v = self.decoder_v.eval()
            self.decoder_s = self.decoder_s.eval()

            self.fc_v = self.fc_v.eval()
            self.fc_s = self.fc_s.eval()

            self.batch_norm_v = self.batch_norm_v.eval()
            self.batch_norm_s = self.batch_norm_s.eval()

            for param in itertools.chain(self.decoder_v.parameters(), self.decoder_s.parameters(), self.fc_v.parameters(), self.fc_s.parameters(), self.batch_norm_v.parameters(), self.batch_norm_s.parameters()):
                param.requires_grad = False
            print("decoder froze")



    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.shape[0], 1, -1)
        
        loss = self.diffusion(z)


        return loss

    def decode(self, z):
        z = z.view(z.shape[0], -1)

        z_v = self.fc_v(z)
        z_s = self.fc_s(z)

        z_v = z_v.view(z.shape[0], self.out_channels_v * self.vit_latent_dim, 1, 1)
        z_s = z_s.view(z.shape[0], self.out_channels_s * self.vit_latent_dim, 1, 1)

        z_v = self.batch_norm_v(z_v)
        z_v = self.leaky_relu(z_v)
        z_s = self.batch_norm_s(z_s)
        z_s = self.leaky_relu(z_s)

        v = self.decoder_v(z_v)
        s = self.decoder_s(z_s)

        return v, s


