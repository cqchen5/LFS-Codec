# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import torch
import numpy as np
import torch.nn as nn

from .conv import StreamingConv1d, StreamingConvTranspose1d
from .streaming import StreamingContainer, StreamingAdd

import sys
sys.path.append('~/moshi_try/')

from utils.compile import torch_compile_lazy


class SEANetResnetBlock(StreamingContainer):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamingConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.add = StreamingAdd()
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamingConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        u, v = self.shortcut(x), self.block(x)
        return self.add(u, v)

LRELU_SLOPE = 0.1
class GlobalTokenEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.pad = (kernel_size - stride) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, stride, self.pad, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, stride, self.pad, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden_channels, out_channels, kernel_size, stride, self.pad, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        self.fn = nn.Sequential(
            # # 2 layers
            # nn.Linear(out_channels, hidden_channels),
            # nn.LeakyReLU(LRELU_SLOPE),
            # nn.Linear(hidden_channels, out_channels),
            # nn.LeakyReLU(LRELU_SLOPE),
            # 1 layer
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.BatchNorm1d(out_channels),
        )
    def forward(self, x):
        """
        x --- [B, in_channels, T]
        out -- [B, out_channels]
        """
        # x_mask = torch.unsqueeze(sequence_mask(
        #     x_lengths, x.size(2)), 1).to(x.dtype)
        # x = self.conv(x) * x_mask
        x = self.conv(x)
        x = torch.mean(x, dim=2) # [B, out_channels]
        x = self.fn(x)
        return x


class SEANetEncoder(StreamingContainer):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
        mask_fn (nn.Module): Optional mask function to apply after convolution layers.
        mask_position (int): Position of the mask function, with mask_position == 0 for the first convolution layer,
            mask_position == 1 for the first conv block, etc.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        mask_fn: tp.Optional[nn.Module] = None,
        mask_position: tp.Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
            self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            StreamingConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        model1: tp.List[nn.Module] = []
        model2: tp.List[nn.Module] = []

        if mask_fn is not None and mask_position == 0:
            model += [mask_fn]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            if i <= 1:
                block_norm = "none" if self.disable_norm_outer_blocks >= i + 2 else norm
                # Add residual layers
                for j in range(n_residual_layers):
                    model += [
                        SEANetResnetBlock(
                            mult * n_filters,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            norm=block_norm,
                            norm_params=norm_params,
                            activation=activation,
                            activation_params=activation_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                        )
                    ]

                # Add downsampling layers
                model += [
                    act(**activation_params),
                    StreamingConv1d(
                        mult * n_filters,
                        mult * n_filters * 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                    ),
                ]
                mult *= 2
                if mask_fn is not None and mask_position == i + 1:
                    model += [mask_fn]
            elif i==2:
                block_norm = "none" if self.disable_norm_outer_blocks >= i + 2 else norm
                # Add residual layers
                for j in range(n_residual_layers):
                    model1 += [
                        SEANetResnetBlock(
                            mult * n_filters,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            norm=block_norm,
                            norm_params=norm_params,
                            activation=activation,
                            activation_params=activation_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                        )
                    ]

                # Add downsampling layers
                model1 += [
                    act(**activation_params),
                    StreamingConv1d(
                        mult * n_filters,
                        mult * n_filters * 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                    ),
                ]
                mult *= 2
                if mask_fn is not None and mask_position == i + 1:
                    model1 += [mask_fn]
            else:
                block_norm = "none" if self.disable_norm_outer_blocks >= i + 2 else norm
                # Add residual layers
                for j in range(n_residual_layers):
                    model2 += [
                        SEANetResnetBlock(
                            mult * n_filters,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            norm=block_norm,
                            norm_params=norm_params,
                            activation=activation,
                            activation_params=activation_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                        )
                    ]

                # Add downsampling layers
                model2 += [
                    act(**activation_params),
                    StreamingConv1d(
                        mult * n_filters,
                        mult * n_filters * 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                    ),
                ]
                mult *= 2
                if mask_fn is not None and mask_position == i + 1:
                    model2 += [mask_fn]

        model2 += [
            act(**activation_params),
            StreamingConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.gfc = [256, 128, 256, 3, 1]
        self.GlobalTokenEncoder = GlobalTokenEncoder(self.gfc[0], self.gfc[1], self.gfc[2], self.gfc[3], self.gfc[4])

    @torch_compile_lazy
    def forward(self, x, x_global, speaker_embedding):
        feat = self.model(x)
        feat2 = self.model(x_global)
        # breakpoint()
        global_features = self.GlobalTokenEncoder(feat)
        global_features2 = self.GlobalTokenEncoder(feat2)
        global_features2 = global_features2.detach()
        feat = feat - global_features.unsqueeze(-1).repeat(1, 1, feat.shape[-1])

        # breakpoint()
        feat = self.model1(feat)
        feat = feat - speaker_embedding.unsqueeze(-1).repeat(1, 1, feat.shape[-1])
        
        y = self.model2(feat)
        return y, global_features, global_features2


class SEANetDecoder(StreamingContainer):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
            self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamingConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        model2: tp.List[nn.Module] = []
        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            if i <= 1:
                block_norm = (
                    "none"
                    if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
                    else norm
                )
                # Add upsampling layers
                model += [
                    act(**activation_params),
                    StreamingConvTranspose1d(
                        mult * n_filters,
                        mult * n_filters // 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                    ),
                ]
                # Add residual layers
                for j in range(n_residual_layers):
                    model += [
                        SEANetResnetBlock(
                            mult * n_filters // 2,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            activation=activation,
                            activation_params=activation_params,
                            norm=block_norm,
                            norm_params=norm_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                        )
                    ]

                mult //= 2
                
            elif i == 2:
                block_norm = (
                    "none"
                    if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
                    else norm
                )
                # Add upsampling layers
                model2 += [
                    act(**activation_params),
                    StreamingConvTranspose1d(
                        mult * n_filters,
                        mult * n_filters,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                    ),
                ]
                # Add residual layers
                for j in range(n_residual_layers):
                    model2 += [
                        SEANetResnetBlock(
                            mult * n_filters,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            activation=activation,
                            activation_params=activation_params,
                            norm=block_norm,
                            norm_params=norm_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                        )
                    ]
            else:
                block_norm = (
                    "none"
                    if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
                    else norm
                )
                # Add upsampling layers
                model2 += [
                    act(**activation_params),
                    StreamingConvTranspose1d(
                        mult * n_filters,
                        mult * n_filters // 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                    ),
                ]
                # Add residual layers
                for j in range(n_residual_layers):
                    model2 += [
                        SEANetResnetBlock(
                            mult * n_filters // 2,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            activation=activation,
                            activation_params=activation_params,
                            norm=block_norm,
                            norm_params=norm_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                        )
                    ]

                mult //= 2
        # Add final layers
        model2 += [
            act(**activation_params),
            StreamingConv1d(
                64,
                channels,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model2 += [final_act(**final_activation_params)]

        self.model = nn.Sequential(*model)
        self.model2 = nn.Sequential(*model2)

    @torch_compile_lazy
    def forward(self, z, global_features, speaker_embedding):
        z += speaker_embedding.unsqueeze(-1).repeat(1, 1, z.shape[-1])
        emb = self.model(z)
        emb += global_features.unsqueeze(-1).repeat(1, 1, emb.shape[-1])
        y = self.model2(emb)
        return y
