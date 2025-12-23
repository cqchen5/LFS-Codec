#from huggingface_hub import hf_hub_download
import torch
import torchaudio

import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
from models import loaders
import os
import tqdm

#mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi_weight = ''
print(mimi_weight)
model_checkpoint = torch.load(mimi_weight, map_location='cpu')
mimi = loaders.get_mimi('', device='cpu')
mimi.load_state_dict(model_checkpoint['model_state_dict'])
mimi.eval()

speech_dir = ""

wav_files = [f for f in os.listdir(speech_dir) if f.endswith('.wav')]
output_dir = ""

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for f in wav_files:

    input_fire = os.path.join(speech_dir, f)
    output_fire = os.path.join(output_dir, f)
    wav, sr = torchaudio.load(input_fire)
    wav = torchaudio.transforms.Resample(sr, 24000)(wav)

    wav = wav.unsqueeze(0)
    print(wav.shape)

    with torch.no_grad():
        codes, feat, speaker_embedding = mimi.encode(wav, wav)  # [B, K = 8, T]
        print(codes.shape)
        decoded = mimi.decode(codes, feat, speaker_embedding)
        print(decoded.shape)
        
    decoded = decoded.detach().numpy()
    write(output_fire, mimi.sample_rate, decoded.astype(np.float32))