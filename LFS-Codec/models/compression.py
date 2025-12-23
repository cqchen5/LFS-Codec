# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this file is adapted from encodec.py in https://github.com/facebookresearch/audiocraft
# released under the following license.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models. In particular, provides the implementation
for Mimi. Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import typing as tp

import torch
from torch import nn
import random

import sys
sys.path.append('~/moshi_try/')

from modules.resample import ConvDownsample1d, ConvTrUpsample1d
from modules.streaming import StreamingModule, State
from utils.compile import no_compile, CUDAGraphed
from .timbre import TorchMelSpectrogram
from .ecapa_tdnn import ECAPA_TDNN as SpeakerEncoder


logger = logging.getLogger()
import torch.nn.functional as F

def orthogonal_regularization(W, lambda_ortho=0.02):
    d = W.size(0)                   
    I = torch.eye(d, device=W.device)
    WtW = torch.matmul(W.T, W)      
    ortho_loss = lambda_ortho * F.mse_loss(WtW, I) 
    return ortho_loss

class Quantizer_module(torch.nn.Module):    
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)    #self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
        # for p in self.embedding.parameters():
        #     p.requires_grad = False
        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)

    def forward(self, x):
        quant_codebook = self.embedding_proj(self.embedding.weight)
        W_loss = orthogonal_regularization(self.embedding_proj.weight)
        # compute Euclidean distance
        d = torch.sum(x ** 2, 1, keepdim=True) + torch.sum(quant_codebook ** 2, 1) \
            - 2 * torch.matmul(x, quant_codebook.T)
        min_indicies = torch.argmin(d, 1)
        z_q = F.embedding(min_indicies, quant_codebook).view(x.shape)
        return z_q, min_indicies, W_loss


class Quantizer(torch.nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        assert 512 % 2 == 0
        self.quantizer_modules = nn.ModuleList([
            Quantizer_module(2048, 512 // 2)
            for _ in range(2)
        ])
        self.quantizer_modules2 = nn.ModuleList([
            Quantizer_module(2048, 512 // 2)
            for _ in range(2)
        ])
        self.quantizer_modules3 = nn.ModuleList([
            Quantizer_module(2048, 512 // 2)
            for _ in range(2)
        ])
        self.quantizer_modules4 = nn.ModuleList([
            Quantizer_module(2048, 512 // 2)
            for _ in range(2)
        ])
        # self.h = h
        self.distill_proj = torch.nn.Conv1d(    # For Distill
                256, 1024, 1, bias=False
            )
        self.codebook_loss_lambda = 1.0  
        self.commitment_loss_lambda = 0.25 
        self.residul_layer = 4
        self.n_code_groups = 2

    def for_one_step(self, xin, idx):
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, 512)
        x = torch.split(x, 512 // 2, dim=-1)
        min_indicies = []
        z_q = []
        W_loss = []
        if idx == 0:
            for i, (_x, m) in enumerate(zip(x, self.quantizer_modules)):
                _z_q, _min_indicies, w_loss = m(_x)
                if i == 0:
                    quantized_feature = self.distill_proj(_z_q.reshape(xin.size(0),-1,256).permute(0, 2, 1))
                z_q.append(_z_q)
                W_loss.append(w_loss)
                min_indicies.append(_min_indicies)  #B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
                + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            # print(loss)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies, quantized_feature, W_loss
        elif idx == 1:
            for _x, m in zip(x, self.quantizer_modules2):
                _z_q, _min_indicies, w_loss = m(_x)
                z_q.append(_z_q)
                W_loss.append(w_loss)
                min_indicies.append(_min_indicies)  #B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
                + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies, W_loss
        elif idx == 2:
            for _x, m in zip(x, self.quantizer_modules3):
                _z_q, _min_indicies, w_loss = m(_x)
                z_q.append(_z_q)
                W_loss.append(w_loss)
                min_indicies.append(_min_indicies)  #B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
                + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies, W_loss
        else:
            for _x, m in zip(x, self.quantizer_modules4):
                _z_q, _min_indicies, w_loss = m(_x)
                z_q.append(_z_q)
                W_loss.append(w_loss)
                min_indicies.append(_min_indicies)  #B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
                + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies, W_loss

    def forward(self, xin):
        #B, C, T
        quantized_out = 0.0
        residual = xin
        all_losses = []
        all_indices = []
        W_total_loss = []
        for i in range(self.residul_layer):
            if i == 0:
                quantized, loss, indices, quantized_feature, W_loss = self.for_one_step(residual, i)  # 
            else:
                quantized, loss, indices, W_loss = self.for_one_step(residual, i)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.extend(indices)  # 
            all_losses.append(loss)
            W_total_loss.extend(W_loss)
        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)
        W_total_loss = torch.stack(W_total_loss)
        W_Loss = torch.mean(W_total_loss)
        return quantized_out, loss, all_indices, quantized_feature, W_Loss

    def embed(self, x):
        #idx: N, T, 4
        #print('x ', x.shape)
        quantized_out = torch.tensor(0.0, device=x.device)
        x = torch.split(x, 1, 2) 
        #print('x.shape ', len(x),x[0].shape)
        for i in range(self.residul_layer):
            ret = []
            if i == 0:
                for j in range(self.n_code_groups):
                    q = x[j]
                    embed = self.quantizer_modules[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                #print(ret.shape)
                quantized_out = quantized_out + ret
            else:
                for j in range(self.n_code_groups):
                    q = x[j + self.n_code_groups]
                    embed = self.quantizer_modules2[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                quantized_out = quantized_out + ret
        return quantized_out.transpose(1, 2)  #N, C, T


class CompressionModel(StreamingModule[State]):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) : ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...


@dataclass
class _MimiState:
    graphed_tr_enc: CUDAGraphed | None
    graphed_tr_dec: CUDAGraphed | None

    def reset(self):
        pass


class MimiModel(CompressionModel[_MimiState]):
    """Mimi model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (float): Final frame rate of the quantized representatiopn.
        encoder_frame_rate (float): frame rate of the encoder model. Note that if `frame_rate != encopder_frame_rate`,
            the latent will be resampled linearly to match the desired `frame_rate` before and after quantization.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        encoder_transformer (nn.Module or None): optional transformer for the encoder.
        decoder_transformer (nn.Module or None): optional transformer for the decoder.
        resample_method (str): method to use for resampling the latent space before the quantizer.
        upsample_channel_wise_bug (bool): controls whether the upsampling is channel wise.
            Defaults to true to reproduce bug in original implementation.
        freeze_encoder: whether to freeze the encoder weights.
        freeze_quantizer: whether to freeze the quantizer weights.
        freeze_quantizer_level: If positive, freeze the quantizer up to this level.
        torch_compile_encoder_decoder (bool): if True, uses torch.compile on the encoder / decoder.
            Deactivated by default for training as this is incompatible at the moment with weight norm.
            See https://github.com/pytorch/pytorch/issues/121902
            Also this seems to work well with 2.2.0, but completely fail with 2.4.0.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: Quantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: tp.Optional[nn.Module] = None,
        decoder_transformer: tp.Optional[nn.Module] = None,
        resample_method: str = "interpolate",
        upsample_channel_wise_bug: bool = True,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
        torch_compile_encoder_decoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder
        self.mel_spectrogram = TorchMelSpectrogram()
        self.speaker_encoder = SpeakerEncoder(80, 512,
                                              channels=[256, 256, 256, 256, 768],
                                              kernel_sizes=[5, 3, 3, 3, 1],
                                              dilations=[1, 2, 3, 4, 1],
                                              attention_channels=64,
                                              res2net_scale=2,
                                              se_channels=64,
                                              global_context=True,
                                              batch_norm=False)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.encoder_transformer is not None:
                for p in self.encoder_transformer.parameters():
                    p.requires_grad = False
            for name, p in self.quantizer.named_parameters():
                if name.endswith("input_proj.weight"):
                    p.requires_grad = False
        # if freeze_quantizer:
        #     self.quantizer.ema_frozen_(True)
        # self.freeze_quantizer = freeze_quantizer
        # self.freeze_quantizer_level = (
        #     freeze_quantizer_level
        #     if freeze_quantizer_level > 0
        #     else self.quantizer.num_codebooks
        # )

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

        assert resample_method in [
            "interpolate",
            "conv",
            "avg_pool",
        ], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method
        if encoder_frame_rate != frame_rate:
            assert not (
                causal and resample_method == "interpolate"
            ), "Cannot interpolate with causal model."
            if resample_method in ["conv", "avg_pool"]:
                assert (
                    self.encoder_frame_rate > self.frame_rate
                ), "Cannot upsample with conv."
                downsample_stride = self.encoder_frame_rate / self.frame_rate
                assert downsample_stride == int(
                    downsample_stride
                ), f"Only integer strides are supported, got {downsample_stride}"
                learnt = resample_method == "conv"
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                )
                if freeze_encoder:
                    for p in self.downsample.parameters():
                        p.requires_grad = False
                self.upsample = ConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                )

    def _init_streaming_state(self, batch_size: int) -> _MimiState:
        device = next(self.parameters()).device
        disable = device.type != 'cuda'
        graphed_tr_dec = None
        graphed_tr_enc = None
        if self.encoder_transformer is not None:
            graphed_tr_enc = CUDAGraphed(self.encoder_transformer, disable=disable)
        if self.decoder_transformer is not None:
            graphed_tr_dec = CUDAGraphed(self.decoder_transformer, disable=disable)
        return _MimiState(graphed_tr_enc, graphed_tr_dec)
    
    def _random_clip(self, sequences, lengths, max_ratio=0.75, min_ratio=0.25, n_segments=3, min_length=100):
        truncated_lengths = (
            lengths * (torch.rand_like(lengths.float()) * (max_ratio - min_ratio) + min_ratio)
        ).long()
        min_length = max(min_length, truncated_lengths.max())
        # breakpoint()
        new_sequences, new_lengths = [], []
        for seq, org_len, new_len in zip(sequences, lengths, truncated_lengths):
            # Clip
            start = random.randint(0, org_len - new_len)
            seg = seq[start : start + int(new_len)]

            # Shuffle
            segment_length = seg.shape[0] // n_segments
            seg = seg[: seg.shape[0] // n_segments * n_segments]
            slices = [
                seg[i: i + segment_length]
                for i in range(0, seg.shape[0], segment_length)
            ]
            random.shuffle(slices)
            seg = torch.cat(slices, dim=0)
            
            if seg.shape[0] < min_length:
                seg = torch.cat([seg] * (min_length // seg.shape[0] + 1))[: min_length]
            new_sequences.append(seg)
            new_lengths.append(new_len)

        new_sequences = torch.stack(new_sequences, dim=0)
        new_lengths = torch.tensor(new_lengths, device=new_sequences.device)
        return new_sequences, new_lengths

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.cardinality

    def _to_framerate(self, x: torch.Tensor):
        # Convert from the encoder frame rate to the overall framerate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        # print("frame_rate",frame_rate)
        # print("new_frame_rate",new_frame_rate)
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.downsample(x)

    def _to_encoder_framerate(self, x: torch.Tensor):
        # Convert from overall framerate to the encoder frame rate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.upsample(x)

    @property
    def _context_for_encoder_decoder(self):
        if self.torch_compile_encoder_decoder:
            return nullcontext()
        else:
            return no_compile()

    def forward(self, x: torch.Tensor, x_global: torch.Tensor):# -> QuantizedResult:

        wav_len = torch.tensor(144000)
        # print(x_global.shape)
        mel, mel_length = self.mel_spectrogram(x_global, wav_len)
        mel_length = mel_length.unsqueeze(0).repeat(x_global.size(0), 1)
        cond, cond_length = self._random_clip(mel, mel_length)
        speaker_embedding = self.speaker_encoder(cond, cond_length)

        assert x.dim() == 3
        length = x.shape[-1]
        extra_metrics: tp.Dict[str, torch.Tensor] = {}
###########loss_w#############
        loss_w = torch.tensor([0.0], device=x.device, requires_grad=True)
        W_Loss = torch.tensor([0.0], device=x.device, requires_grad=True)

        with self._context_for_encoder_decoder:
            emb, global_features, global_features2 = self.encoder(x, x_global, speaker_embedding)
            global_features_compare_loss = 1-F.cosine_similarity(global_features, global_features2, dim=1).mean()
            # print(torch.max(emb))
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
            # print(torch.max(emb))
        emb = self._to_framerate(emb)
        expected_length = self.frame_rate * length / self.sample_rate
        # Checking that we have the proper length given the advertised frame rate.
        assert abs(emb.shape[-1] - expected_length) < 1, (
            emb.shape[-1],
            expected_length,
        )
        # emb = emb.contiguous()
        # print(emb.shape)
        # breakpoint()
        emb, loss_w, c, quantized_feature, W_Loss = self.quantizer(emb) 
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)

        with self._context_for_encoder_decoder:
            out = self.decoder(emb, global_features, speaker_embedding)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        # q_res.x = out
        # q_res.metrics.update(extra_metrics)
        if self.training:
            return out,loss_w, quantized_feature, global_features_compare_loss, W_Loss
        else:
            # print("out:",out.shape)
            return out
        
    def _encode_to_unquantized_latent(self, x: torch.Tensor, x_global, speaker_embedding) -> torch.Tensor:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings.
        """
        assert (
            x.dim() == 3
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"
        state = self._streaming_state
        with self._context_for_encoder_decoder:
            emb, feat, _ = self.encoder(x, x_global, speaker_embedding)
        if self.encoder_transformer is not None:
            if state is None:
                (emb,) = self.encoder_transformer(emb)
            else:
                assert state.graphed_tr_enc is not None
                (emb,) = state.graphed_tr_enc(emb)
        emb = self._to_framerate(emb)
        return emb, feat

    def encode(self, x: torch.Tensor, x_global) -> torch.Tensor:
        """Encode the given input tensor to quantized representation.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes (torch.Tensor): an int tensor of shape [B, K, T]
                with K the number of codebooks used and T the timestep.
        """
        wav_len = torch.tensor(x.shape[-1])
        mel, mel_length = self.mel_spectrogram(x, wav_len)
        mel_length = mel_length.unsqueeze(0).repeat(x.size(0), 1)
        mel_length = mel_length.squeeze(0)
        # cond, cond_length = self._random_clip(mel, mel_length)
        # breakpoint()
        speaker_embedding = self.speaker_encoder(mel, mel_length)

        emb, feat = self._encode_to_unquantized_latent(x, x_global, speaker_embedding)
        codes, _, _, _, _ = self.quantizer(emb)
        return codes, feat, speaker_embedding

    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Projects a batch of waveforms to latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Embeddings, either quantized or not.
        """
        emb = self._encode_to_unquantized_latent(x)
        if not quantize:
            return emb
        else:
            codes = self.quantizer.encode(emb)
            return self.decode_latent(codes)

    def decode(self, codes: torch.Tensor, feat, speaker_embedding):
        """Decode the given codes to a reconstructed representation.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        state = self._streaming_state
        # emb = self.decode_latent(codes)
        emb = self._to_encoder_framerate(codes)
        if self.decoder_transformer is not None:
            if state is None:
                (emb,) = self.decoder_transformer(emb)
            else:
                assert state.graphed_tr_dec is not None
                (emb,) = state.graphed_tr_dec(emb)
        with self._context_for_encoder_decoder:
            out = self.decoder(emb, feat, speaker_embedding)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class WrapperCompressionModel(CompressionModel[State]):
    """Base API for CompressionModel wrappers that do not depend on external frameworks."""

    def __init__(self, model: CompressionModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode(codes)

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode_latent(codes)

    def set_num_codebooks(self, n: int):
        self.model.set_num_codebooks(n)

    @property
    def quantizer(self):
        return self.model.quantizer

    @property
    def channels(self) -> int:
        return self.model.channels

    @property
    def frame_rate(self) -> float:
        return self.model.frame_rate

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.cardinality

    @property
    def num_codebooks(self) -> int:
        return self.model.num_codebooks

    @property
    def total_codebooks(self) -> int:
        return self.model.total_codebooks
