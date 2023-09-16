import torch
import torch.nn as nn

import numpy as np
from pyts.image import GramianAngularField

class BasicLinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicLinearBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class AutoEncoder(nn.Module):
    def __init__(self, sequence_length, compress_ratio=16):
        super(AutoEncoder, self).__init__()

        num_layer = 1
        while True:
            if compress_ratio // 2**(num_layer) == 1: break
            else: num_layer += 1

        self.encoder = nn.ModuleList([BasicLinearBlock(sequence_length // 2 ** (layer_idx), sequence_length // 2 ** (layer_idx + 1)) for layer_idx in range(num_layer)])
        self.decoder = nn.ModuleList([BasicLinearBlock(sequence_length // 2 ** (num_layer - layer_idx), sequence_length // 2 ** (num_layer - layer_idx - 1)) for layer_idx in range(num_layer)])

    def forward(self, x):
        for encoder_block in self.encoder:
            x = encoder_block(x)

        latent_sequence = x

        for decoder_block in self.decoder:
            x = decoder_block(x)

        recon_sequence = x

        return latent_sequence, recon_sequence

class OurModel(nn.Module):
    def __init__(self, num_channels, sequence_length=4096, compress_ratio=16):
        super(OurModel, self).__init__()
        self.num_channels = num_channels

        self.compress_autoencoder = AutoEncoder(sequence_length=sequence_length, compress_ratio=compress_ratio)
        self.ts_image_encoder = GramianAngularField()

    def forward(self, x):
        ts_image_list = []
        recon_sequence_list = []

        for channel_idx in range(self.num_channels):
            latent_sequence, recon_sequence = self.compress_autoencoder(x[:, channel_idx, :])
            ts_image_list.append(self.ts_image_encoder.transform(latent_sequence.cpu().detach().numpy()))
            recon_sequence_list.append(recon_sequence)

        ts_image_mean = torch.mean(torch.tensor(np.array(ts_image_list)).to(x.get_device()), dim=0)
        ##### 이 다음 부터 서형이가 GPT-3 + Classification 코드 짜주세요


if __name__=='__main__':
    model = OurModel(num_channels=12).cuda()
    inp = torch.randn(2, 12, 4096).cuda()
    # print(model)
    oup = model(inp)