from torchprofile import profile_macs

#from huggingface_hub import hf_hub_download
import torch
import torchaudio
# from moshi.models import LMGen
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
from models import loaders
import os
import tqdm

#mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi_weight = '/train20/sppro/permanent/cqchen5/mimi_ckpt/test206/checkpoints/bs12_cut48000_length0_epoch14_step310000_lr0.0003.pt'
print(mimi_weight)
model_checkpoint = torch.load(mimi_weight, map_location='cpu')
mimi = loaders.get_mimi('', device='cpu')
mimi.load_state_dict(model_checkpoint['model_state_dict'])
mimi.eval()

f = "/train20/sppro/permanent/cqchen5/test/test_speech/121_121726_000025_000001.wav"


wav, sr = torchaudio.load(f)

wav = wav.unsqueeze(0)
print(wav.shape)

profile_macs(mimi,wav)