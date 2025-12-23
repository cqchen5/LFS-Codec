from transformers import AutoModel,  Wav2Vec2FeatureExtractor
from pathlib import Path
import torchaudio
import torch
import yaml
import argparse
from tqdm import tqdm
import random
import numpy as np
import os

import torch.nn as nn

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path',default="")
    parser.add_argument('--audio_dir', type=str, help='Audio folder path',default="")
    parser.add_argument('--rep_dir', type=str, help='Path to save representation files',default="")
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='flac')
    args = parser.parse_args()
    exts = args.exts.split(',')
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    sample_rate = 16000
    model_file = ""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)
    model = AutoModel.from_pretrained(model_file).eval().to(device)
    target_layer = 'notavg'   
    path = Path(args.audio_dir)
    file_list = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]

    train_file_list = "" 
    segment_size = cfg['datasets']['segment_size']
    # random.seed(args.split_seed)
    # random.shuffle(file_list)
    # print(f'A total of {len(file_list)} samples will be processed, and {valid_set_size} of them will be included in the validation set.')
    print(f'A total of {len(file_list)} samples will be processed')
    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):
            wav, sr = torchaudio.load(audio_file)
            # print(wav.shape)

            if wav.size(-1) < segment_size: #24000
                wav = torch.nn.functional.pad(wav, (0, segment_size - wav.size(-1)), 'constant')

            if sr != sample_rate:   #16000
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
                print(wav.shape)
            
            input_values = feature_extractor(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
            # print(input_values.shape)
            ouput = model(input_values.to(model.device), output_hidden_states=True)
            # print(ouput.hidden_states[-3].shape)
            if target_layer == 'avg':
                rep = torch.mean(torch.stack(ouput.hidden_states), axis=0)
            else:
                rep = ouput.hidden_states[-1]
            pooling_layer = nn.AvgPool1d(kernel_size=4, stride=4)
            rep = rep.permute(0, 2, 1)
            rep = pooling_layer(rep)
            # print(rep.shape)
            rep_file = audio_file.replace(args.audio_dir, args.rep_dir).split('.')[0] + '.wavlm.npy'
            rep_sub_dir = '/'.join(rep_file.split('/')[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir)
            np.save(rep_file, rep.detach().cpu().numpy())
            with open(train_file_list, 'a+') as f:
                f.write(f'{audio_file}\t{rep_file}\n')
            
            