import torch
import numpy as np
import torch.nn as nn
from typing import Tuple, List
from huggingface_hub import PyTorchModelHubMixin

from .transformer import TransformerBlock
from ..env import AttrDict


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


def mag_pha_stft(
    y, hann_window, n_fft, hop_size, win_size, compress_factor=1.0, center=True
):
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1] + (1e-10), stft_spec[:, :, :, 0] + (1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(
    mag, pha, hann_window, n_fft, hop_size, win_size, compress_factor=1.0, center=True
):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    wav = torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )

    return wav


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super().__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2**i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
                nn.Conv2d(
                    h.dense_channel * (i + 1),
                    h.dense_channel,
                    kernel_size,
                    dilation=(dilation, 1),
                ),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super().__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

        self.dense_block = DenseBlock(h, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)  # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft // 2 + 1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        return x


class TSTransformerBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.time_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPSENet(
    PyTorchModelHubMixin,
    nn.Module,
    pipeline_tag="audio-to-audio",
    tags=["arxiv:2308.08926", "speech", "audio", "denoising", "speech-enhancement"],
    license="mit",
    repo_url="https://github.com/yxlu-0102/MP-SENet",
):
    def __init__(self, h: AttrDict | dict, num_tsblocks=4):
        super().__init__()
        if isinstance(h, dict):
            h = AttrDict(h)
        self.h = h
        self.sampling_rate = h.sampling_rate
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSTransformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(h))

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

        self.register_buffer(
            "hann_window", torch.hann_window(h.win_size), persistent=False
        )

    def __call__(
        self, inputs: np.array, segment_size: int | None = None
    ) -> Tuple[np.array, int, List[str]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default sampled at `self.sampling_rate`.
                The shape of this array is `T`, where `T` is the time axis
        Return:
            A :obj:`tuple` containing:
              - :obj:`np.array`:
                 The return shape of the array must be `C'`x`T'`
              - a :obj:`int`: the sampling rate as an int in Hz.
              - a :obj:`List[str]`: the annotation for each out channel.
                    This can be the name of the instruments for audio source separation
                    or some annotation for speech enhancement. The length must be `C'`.
        """
        if segment_size is None:
            if "segment_size" in self.h:
                segment_size = self.h.segment_size
            else:
                segment_size = 32000

        device = next(self.parameters()).device

        inputs = torch.tensor(inputs).to(device)
        norm_factor = torch.sqrt(len(inputs) / torch.sum(inputs**2.0))
        inputs = (inputs * norm_factor).unsqueeze(0)

        segments = []
        for start in range(0, inputs.size(1), segment_size):
            end = min(start + segment_size, inputs.size(1))
            segment = inputs[:, start:end]
            noisy_amp, noisy_pha, _ = mag_pha_stft(
                segment,
                self.hann_window,
                self.h.n_fft,
                self.h.hop_size,
                self.h.win_size,
                self.h.compress_factor,
            )

            amp_g, pha_g, _ = self.forward(noisy_amp, noisy_pha)

            audio_g = mag_pha_istft(
                amp_g,
                pha_g,
                self.hann_window,
                self.h.n_fft,
                self.h.hop_size,
                self.h.win_size,
                self.h.compress_factor,
            )
            segments.append(audio_g)

        audio_g = torch.cat(segments, dim=-1)
        audio_g = audio_g / norm_factor
        audio_g = audio_g.squeeze().detach().cpu().numpy()

        return audio_g, self.h.sampling_rate, ["denoised_audio"]

    def forward(self, noisy_amp, noisy_pha):  # [B, F, T]
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(
            0, 3, 2, 1
        )  # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)

        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack(
            (
                denoised_amp * torch.cos(denoised_pha),
                denoised_amp * torch.sin(denoised_pha),
            ),
            dim=-1,
        )

        return denoised_amp, denoised_pha, denoised_com
