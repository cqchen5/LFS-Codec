import os
import random

import librosa
import pandas as pd
import torch
import audioread

import logging
logger = logging.getLogger(__name__)

from utils_en import convert_audio
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence

import lmdb
import sys
from datum_pb2 import Datum
from transformers import AutoModel,  Wav2Vec2FeatureExtractor
import torch.nn as nn

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None,mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        # load lmdb
        if mode == 'train':
            self.lmdb_file = config.datasets.lmdb_train_file
        else:
            self.lmdb_file = config.datasets.lmdb_test_file
        self.keys = [line.strip() for line in open(self.lmdb_file+".key", "rt").readlines()]
        self.mode = mode
        self.transform = transform
        self.fixed_length = config.datasets.fixed_length
        self.segment_size = config.datasets.segment_size
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels
        self.downsample_rate = 1920

        self.wavlm_sample_rate = 16000
        self.target_layer = 'notavg'

    def open_lmdb(self):
        self.lmdb_env = lmdb.open(self.lmdb_file, readonly=True, lock=False)
        self.txn = self.lmdb_env.begin(write=False)
        random.shuffle(self.keys)

    def parse_datum(self, buff):
        datum = Datum()
        datum.ParseFromString(buff)
        audio = np.frombuffer(datum.audio, dtype=np.int16)
        audio = audio.astype(np.float32)/ 32768.0
        return {'audio':audio}

    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.keys) > self.fixed_length else len(self.keys)  

    def get(self, idx=None):
        """uncropped, untransformed getter with random sample feature"""
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        if idx is not None and idx > len(self.keys):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))
        key = self.keys[idx]
        cursor = self.txn.cursor()
        cursor.set_key(key.encode())
        sample = self.parse_datum(cursor.value())
        audio = sample['audio']
        audio = torch.from_numpy(audio)
        sr = 24000

        # add channel dimension IF loaded audio was mono
        waveform = torch.as_tensor(audio)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            waveform = waveform.expand(self.channels, -1)

        return waveform, sr

    def __getitem__(self, idx):
        # waveform, sample_rate = torchaudio.load(self.audio_files.iloc[idx, :].values[0])
        # """you can preprocess the waveform's sample rate to save time and memory"""
        # if sample_rate != self.sample_rate:
        #     waveform = convert_audio(waveform, sample_rate, self.sample_rate, self.channels)
        # load lmdb
        if not hasattr(self, 'txn'):
            pid = os.getpid()
            print(pid)
            self.open_lmdb()
        key = self.keys[idx]
        cursor = self.txn.cursor()
        cursor.set_key(key.encode())
        audio = self.parse_datum(cursor.value())['audio']    
        sr = 24000
        audio = torch.as_tensor(audio)
       
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            sr = self.sample_rate

        if audio.size(-1) > self.segment_size:
            if self.mode == 'test':
                audio = audio[:self.segment_size]
                audio = audio.unsqueeze(0)
                return audio, audio
            
            # lmdb
            max_audio_start = audio.size(-1) - self.segment_size
            max_audio_global_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio_global_start = random.randint(0, max_audio_global_start)
            audio_raw = audio[audio_start:audio_start+self.segment_size]
            audio_global = audio[audio_global_start:audio_global_start+self.segment_size]
        else:
            if self.mode == 'train':
                audio_raw = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')
                audio_global = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')

        audio_raw = audio_raw.unsqueeze(0)
        audio_global = audio_global.unsqueeze(0)
        return audio_raw, audio_global


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors = []
    tensors1 = []
    for waveform, waveform1 in batch:
        tensors += [waveform]
        tensors1 += [waveform1]
    # Group the list of tensors into a batched tensor
    # breakpoint()
    tensors = pad_sequence(tensors)
    tensors1 = pad_sequence(tensors1)
    return tensors, tensors1