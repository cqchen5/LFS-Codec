LFS-Codec: Low-Frame-Rate Speech Coding with Regularized Codebook Remapping and Dual-Branch Timbre Decoupling

<div align="center">

</div>

# 📖 简介 (Introduction)

LFS-Codec 是一种专为资源受限的多媒体应用设计的低帧率（Low-Frame-Rate）神经语音编解码器。针对降低帧率通常导致码本利用率不足和音色细节丢失的问题，我们提出了两种正交的增强机制，在 12.5 fps 的超低帧率下实现了卓越的重建质量、可懂度和说话人相似度。

主要贡献包括：

LFS-Codec: 一个基于 Group-Residual VQ 的高效编解码器，仅需 84 GMACs 计算量。

PLCR (Parametric Linear Codebook Remapping): 通过参数化线性重映射和动态解冻策略，最大化码本利用率并优化语义蒸馏。

DBTD (Dual-Branch Timbre Decoupling): 双分支音色解耦模块，利用 ECAPA-TDNN 和 TIRE 分别提取说话人声纹和韵律风格，有效分离内容与音色。

# 🚀 核心方法 (Methods)

1. 总体架构

LFS-Codec 基于 Mimi codec 架构，包含因果 SeaNet 编码器和转置卷积解码器。我们在量化瓶颈处引入了 PLCR 和 DBTD 模块以增强低帧率下的表现。

2. Parametric Linear Codebook Remapping (PLCR)

为了解决低帧率下的表征退化问题，PLCR 引入了：

正交正则化 (Orthogonal Regularization): 防止投影矩阵退化，保持维度完整性。

语义锚定策略 (Semantic Anchoring Strategy): 利用 WavLM 作为教师模型，仅对第一层 RVQ 的第一个量化器进行语义蒸馏，释放剩余量化器用于捕捉声学细节。

动态调度 (Dynamic Scheduling): 训练初期冻结码本，后期解冻以稳定对齐。

3. Dual-Branch Timbre Decoupling (DBTD)

为了在压缩内容的同时保留说话人特征，DBTD 采用双分支结构：

ECAPA-TDNN 分支: 提取短时说话人声纹 (Fingerprints)。

TIRE 分支: 学习句子级时不变表征 (Style/Prosody)。

操作: 在编码器侧显式减去这些特征，在解码器侧重新注入，强制 VQ 瓶颈专注于语言内容。

# 📊 实验结果 (Results)

我们在 LibriTTS 和 AISHELL-3 数据集上进行了广泛评估。LFS-Codec 在 12.5 fps 下显著优于现有的 High-Frame-Rate (HFR) 适配模型和原生 Low-Frame-Rate (LFR) 模型。



# 🛠️ 安装 (Installation)

建议使用 Anaconda 创建虚拟环境：

conda create -n lfscodec python=3.9
conda activate lfscodec

安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

安装其他依赖
pip install -r requirements.txt


依赖项 (Requirements)

主要依赖包括但不限于：

torchaudio

numpy

scipy

soundfile

speechbrain (用于 ECAPA-TDNN)

transformers (用于 WavLM)

# 📂 数据准备 (Data Preparation)

本项目支持 LibriTTS (英语) 和 AISHELL-3 (中文) 数据集。

下载数据集并解压。



# 🖥️ 使用方法 (Usage)

1. 训练 (Training)

修改 configs/lfs_codec.json 中的配置路径，然后运行：

python train_multi_gpu.py


注意:

训练总迭代次数建议为 300k。

PLCR 层的动态解冻（Unfreezing）将在 30k 步时自动触发。

2. 推理 (Inference / Reconstruction)

使用训练好的模型对音频进行编码和解码：

python example.py \
    --checkpoint_path checkpoints/lfs_base_12.5fps/best_model.pth \
    --input_wav samples/input.wav \
    --output_wav samples/output.wav


# 📝 协议 (License)

本项目采用 MIT License 开源协议。
