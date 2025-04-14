import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import VisionTransformer as ViT


NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln':nn.LayerNorm}

class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel = 3, stride = 1, padding = 1, norm = 'bn', dropout = None):
        super(ConvBlock, self).__init__()

        layers = [nn.Conv2d(in_chan, out_chan, kernel, stride, padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))
        layers.append(nn.ReLU())


        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel = 2, stride = 2, padding = 0, output_padding = 0, norm = 'bn', dropout = None):
        super(DeconvBlock, self).__init__()

        layers = [nn.ConvTranspose2d(in_chan, out_chan, kernel_size=kernel,stride=stride,padding=padding,output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))

        layers.append(nn.ReLU())

        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_chan, out_chan, kernel = 3, stride = 1, padding = 1, norm = 'bn'):
        super(ConvBlock_Tanh, self).__init__()

        layers = [nn.Conv2d(in_chan, out_chan, kernel, stride, padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))
        layers.append(nn.Tanh())


        # if dropout:
        #     layers.append(nn.Dropout2d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)


class Encoder_v(nn.Module):
    def __init__(self, in_channels, dim1 = 32, dim2 = 64, dim3 = 128, dim4 = 256, dim5 = 512, checkpoint = None):
        super(Encoder_v, self).__init__()
        self.dim5 = dim5
        self.module_list1 = []
        self.module_list2 = []
        self.convBlock1_1 = ConvBlock(in_channels, dim1, kernel=1, stride=1, padding=0) # 70 * 70
        self.convBlock1_2 = ConvBlock(dim1, dim1, kernel=3, stride=1, padding=1)

        self.convBlock2_1 = ConvBlock(dim1, dim2, kernel=3, stride=2, padding=1) # 35 * 35
        self.convBlock2_2 = ConvBlock(dim2, dim2, kernel=1, stride=1, padding=0) # 35 * 35

        self.convBlock3_1 = ConvBlock(dim2, dim2, kernel=3, stride=1, padding=1) # 35* 35
        self.convBlock3_2 = ConvBlock(dim2, dim2, kernel=1, stride=1, padding=0)  # 35* 35

        self.convBlock4_1 = ConvBlock(dim2, dim3, kernel=3, stride=2, padding=1) # 18 * 18
        self.convBlock4_2 = ConvBlock(dim3, dim3, kernel=1, stride=1, padding=0) # 18 * 18

        self.convBlock5_1 = ConvBlock(dim3, dim3, kernel=3, stride=1, padding=0) # 16 * 16
        self.convBlock5_2 = ConvBlock(dim3, dim3, kernel=1, stride=1, padding=0) # 16 * 16


        self.module_list1.append(self.convBlock1_1)
        self.module_list1.append(self.convBlock1_2)
        self.module_list1.append(self.convBlock2_1)
        self.module_list1.append(self.convBlock2_2)
        self.module_list1.append(self.convBlock3_1)
        self.module_list1.append(self.convBlock3_2)
        self.module_list1.append(self.convBlock4_1)
        self.module_list1.append(self.convBlock4_2)
        self.module_list1.append(self.convBlock5_1)
        self.module_list1.append(self.convBlock5_2)


        self.convBlock6_1 = ConvBlock(dim3, dim4, kernel=3, stride=2, padding=1) # 8*8
        self.convBlock6_2 = ConvBlock(dim4, dim4, kernel=1, stride=1, padding=0) #8*8

        self.convBlock7_1 = ConvBlock(dim4, dim4, kernel=3, stride=2, padding=1) #4*4
        self.convBlock7_2 = ConvBlock(dim4, dim4, kernel=3, stride=2, padding=1) #2*2

        self.convBlock8 = ConvBlock_Tanh(dim4, dim5, kernel=3, stride=2, padding=1) # 1*1

        self.module_list2.append(self.convBlock6_1)
        self.module_list2.append(self.convBlock6_2)
        self.module_list2.append(self.convBlock7_1)
        self.module_list2.append(self.convBlock7_2)
        self.module_list2.append(self.convBlock8)

        self.module_list1 = nn.ModuleList(self.module_list1)
        self.module_list2 = nn.ModuleList(self.module_list2)

        if checkpoint:
            self.checkpoint = checkpoint



    def forward(self, x):
        for module in self.module_list1:
            x = module(x)

        for module in self.module_list2:
            x = module(x)

        return x

    def load_model(self):
        cpt = torch.load(self.checkpoint)

        loaded_module_list1_state_dict = cpt['module_list1']
        loaded_module_list2_state_dict = cpt['module_list2']

        self.module_list1.load_state_dict(loaded_module_list1_state_dict)
        self.module_list2.load_state_dict(loaded_module_list2_state_dict)




class Decoder_V(nn.Module):
    def __init__(self, out_channels, dim1 = 32, dim2 = 64, dim3 = 128, dim4 = 256, dim5 = 512):
        super(Decoder_V, self).__init__()

        self.deconvBlock1_1 = DeconvBlock(dim5, dim5, kernel=5)  # (None, 512, 5, 5)
        self.deconvBlock1_2 = ConvBlock(dim5, dim5, kernel=3, stride=1) #(None, 512, 5, 5)

        self.deconvBlock2_1 = DeconvBlock(dim5, dim4, kernel=4, stride=2, padding=1) #(None, 256, 10, 10)
        self.deconvBlock2_2 = ConvBlock(dim4, dim4, kernel=3, stride=1) #(None, 256, 10, 10)

        self.deconvBlock3_1 = DeconvBlock(dim4, dim3, kernel=4, stride=2, padding=1) #(None, 128, 20, 20)
        self.deconvBlock3_2 = ConvBlock(dim3, dim3, kernel=3, stride=1) #(None, 128, 20, 20)

        self.deconvBlock4_1 = DeconvBlock(dim3, dim2, kernel=4, stride=2, padding=1) #(None, 64, 40, 40)
        self.deconvBlock4_2 = ConvBlock(dim2, dim2, kernel=3, stride=1) #(None, 64, 40, 40)

        self.deconvBlock5_1 = DeconvBlock(dim2, dim1, kernel=4, stride=2, padding=1) #(None, 32, 80, 80)
        self.deconvBlock5_2 = ConvBlock(dim1, dim1, kernel=3, stride=1) #(None, 32, 80, 80)

        self.deconvBlock6 = ConvBlock_Tanh(dim1, out_channels, kernel=3, stride=1, padding=1) #(None, 32, 80, 80)

        self.blocks = []
        self.blocks.append(self.deconvBlock1_1)
        self.blocks.append(self.deconvBlock1_2)
        self.blocks.append(self.deconvBlock2_1)
        self.blocks.append(self.deconvBlock2_2)
        self.blocks.append(self.deconvBlock3_1)
        self.blocks.append(self.deconvBlock3_2)
        self.blocks.append(self.deconvBlock4_1)
        self.blocks.append(self.deconvBlock4_2)
        self.blocks.append(self.deconvBlock5_1)
        self.blocks.append(self.deconvBlock5_2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = F.pad(x,[-5,-5,-5,-5], mode='constant', value=0)
        x = self.deconvBlock6(x)
        return x

class TDecoder_S(nn.Module):
    def __init__(self, dd, origin_h, origin_w, depth = 2, laten_dim=64, latent_h = 1, latent_w = 1):
        super(TDecoder_S, self).__init__()

        self.dd = dd
        self.dh = latent_h
        self.dw = latent_w

        self.origin_h = origin_h
        self.origin_w = origin_w

        self.laten_dim = laten_dim
        self.depth = depth

        if self.depth:
            self.decoder = ViT(img_size=(self.dh, self.dw), patch_size=1, in_chans=self.laten_dim * self.dd,
                               depth=self.depth, n_heads=8, mlp_ratio=self.dd, embed_dim=self.laten_dim * self.dd)
        self.mlp = nn.Linear(self.laten_dim * self.dd, self.origin_h * self.origin_w * self.dd)

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        x = x.view(x.shape[0], self.dd, self.origin_h, self.origin_w)
        
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    decoder = TDecoder_S(dd=5, origin_h=1000, origin_w=70, laten_dim=128).to(device)
    print(decoder)
    test_data = torch.randn(2, 640, 1, 1).to(device) 
    
    output = decoder(test_data)
    print("Output shape:", output.shape)