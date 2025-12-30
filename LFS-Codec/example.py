import argparse
import torch
import torchaudio
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
from models import loaders
import os
import tqdm

def main():
    # 1. 定义命令行参数解析
    parser = argparse.ArgumentParser(description="Run Mimi audio encoding/decoding with custom paths.")
    
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model checkpoint file (.pth or .pt)')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing input .wav files')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save processed audio files')
    
    args = parser.parse_args()

    # 2. 使用参数加载模型
    print(f"Loading model from: {args.model_path}")
    
    # 注意：这里假设 loaders.get_mimi 不需要路径参数即可初始化结构，或者接受空字符串
    # 如果 get_mimi 需要根据配置加载，请根据实际情况调整
    mimi = loaders.get_mimi('', device='cpu') 
    
    model_checkpoint = torch.load(args.model_path, map_location='cpu')
    # 处理可能的 key 不匹配问题 (有些 checkpoint 可能包含 'model_state_dict'，有些直接是 state_dict)
    if 'model_state_dict' in model_checkpoint:
        mimi.load_state_dict(model_checkpoint['model_state_dict'])
    else:
        mimi.load_state_dict(model_checkpoint)
        
    mimi.eval()

    # 3. 准备输入输出目录
    speech_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(speech_dir):
        print(f"Error: Input directory '{speech_dir}' does not exist.")
        return

    wav_files = [f for f in os.listdir(speech_dir) if f.endswith('.wav')]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Found {len(wav_files)} wav files. Starting processing...")

    # 4. 循环处理文件
    for f in tqdm.tqdm(wav_files): # 添加 tqdm 显示进度条
        input_file = os.path.join(speech_dir, f)
        output_file = os.path.join(output_dir, f)
        
        # 加载音频
        wav, sr = torchaudio.load(input_file)
        
        # 重采样到 24k (Mimi 通常需要 24k)
        if sr != 24000:
            wav = torchaudio.transforms.Resample(sr, 24000)(wav)

        # 增加 batch 维度 [1, C, T]
        wav = wav.unsqueeze(0)

        with torch.no_grad():
            # 编码与解码
            # 注意：mimi.encode 通常只需要一个输入，这里原代码传了两个 wav，保留原样
            codes, feat, speaker_embedding = mimi.encode(wav, wav) 
            decoded = mimi.decode(codes, feat, speaker_embedding)
            
        # 后处理并保存
        decoded = decoded.squeeze(0).cpu().numpy() # 移除 batch 维度 [C, T]
        
        # scipy write 需要数据形状为 [T, C] 或 [T]，所以如果是立体声需要转置
        if decoded.ndim == 2 and decoded.shape[0] < decoded.shape[1]: 
            decoded = decoded.T
            
        write(output_file, mimi.sample_rate, decoded.astype(np.float32))
        
    print("Processing complete.")

if __name__ == "__main__":
    main()
