\<div align="center"\>

# **LFS-Codec: Low-Frame-Rate Speech Coding with Regularized Codebook Remapping and Dual-Branch Timbre Decoupling**

**\[ICME Submission\]**

**在 12.5 fps 超低帧率下实现高质量语音重构**

[English](https://www.google.com/search?q=./README_EN.md) | [简体中文](https://www.google.com/search?q=./README.md)

\</div\>

## **📖 简介 (Introduction)**

**LFS-Codec** 是一种专为资源受限的多媒体应用设计的低帧率（Low-Frame-Rate, LFR）神经语音编解码器。针对传统模型在降低帧率时常出现的**码本利用率不足**和**音色细节丢失**问题，我们提出了两种正交的增强机制，在 **12.5 fps** 的超低帧率下实现了卓越的重建质量、可懂度和说话人相似度。

### **✨ 核心特性**

* **⚡ 极致高效**: 仅需 **84 GMACs** 计算量，适合边缘设备部署。  
* **🎯 PLCR (Parametric Linear Codebook Remapping)**: 通过参数化线性重映射和动态解冻策略，最大化码本利用率并优化语义蒸馏。  
* **🎭 DBTD (Dual-Branch Timbre Decoupling)**: 双分支音色解耦模块，利用 ECAPA-TDNN 和 TIRE 分别提取声纹与韵律，有效实现内容与音色的分离。

## **🏗️ 架构概览 (Architecture)**

\<div align="center"\>  
\<\!-- 请替换为您的架构图链接，例如: docs/assets/architecture.png \--\>  
\<img src="https://www.google.com/search?q=https://via.placeholder.com/800x300%3Ftext%3DLFS-Codec%2BArchitecture%2BDiagram" alt="LFS-Codec Architecture" width="100%"\>  
\</div\>  
LFS-Codec 基于 Mimi codec 架构，包含因果 SeaNet 编码器和转置卷积解码器。

1. **编码器侧**: 引入 DBTD 模块显式减去音色特征，迫使 VQ 瓶颈专注于语言内容。  
2. **量化器**: 采用 PLCR 策略，仅对第一层 RVQ 进行强语义蒸馏。  
3. **解码器侧**: 重新注入音色特征以恢复高质量的语音波形。

## **📊 性能表现 (Performance)**

我们在 **LibriTTS** (英语) 和 **AISHELL-3** (中文) 数据集上进行了广泛评估。LFS-Codec 在 12.5 fps 下显著优于现有的 High-Frame-Rate (HFR) 适配模型和原生 Low-Frame-Rate (LFR) 模型。

| Model                       | Frame Rate (fps) | GMACs   | NISQA (Quality) ↑ | SIM (Similarity) ↑ | WER (Intelligibility) ↓ |
| :-------------------------- | :--------------- | :------ | :---------------- | :----------------- | :---------------------- |
| **HFR Baselines (Adapted)** |                  |         |                   |                    |                         |
| SimVQ                       | 12.5             | 12G     | 1.459             | 0.368              | 72.75%                  |
| DAC                         | 12.5             | 172G    | 2.955             | 0.478              | 19.36%                  |
| **LFR Baselines**           |                  |         |                   |                    |                         |
| Mimi \[11\]                 | 12.5             | 69G     | 4.098             | 0.694              | 4.66%                   |
| SNAC \[9\]                  | 47               | 112G    | 4.189             | 0.727              | 4.61%                   |
| **LFS-Codec (Ours)**        | **12.5**         | **84G** | **4.411**         | **0.742**          | **3.95%**               |

**注**: 更详细的消融实验结果（PLCR 与 DBTD 的有效性验证）请参考论文 Table II。

## **🛠️ 安装指南 (Installation)**

建议使用 Anaconda 创建独立的虚拟环境：

\# 1\. 克隆仓库  
git clone \[https://github.com/your\_username/LFS-Codec.git\](https://github.com/your\_username/LFS-Codec.git)  
cd LFS-Codec

\# 2\. 创建环境  
conda create \-n lfscodec python=3.9  
conda activate lfscodec

\# 3\. 安装依赖 (根据您的 CUDA 版本调整 PyTorch 安装命令)  
pip install torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/cu118\](https://download.pytorch.org/whl/cu118)  
pip install \-r requirements.txt

## **🚀 快速开始 (Quick Start)**

### **1\. 数据准备 (Data Preparation)**

本项目支持 LibriTTS 和 AISHELL-3 数据集。请按以下结构组织数据，并运行预处理脚本：

python preprocess.py \--dataset libritts \--in\_dir /path/to/LibriTTS \--out\_dir ./filelists

### **2\. 模型训练 (Training)**

修改 configs/lfs\_12.5fps.json 中的路径配置，然后启动训练：

python train.py \\  
    \--config configs/lfs\_12.5fps.json \\  
    \--model lfs\_codec \\  
    \--name lfs\_experiment\_v1

**提示**: PLCR 的动态解冻（Unfreezing）策略将在 30k 步时自动触发，无需人工干预。

### **3\. 推理与重建 (Inference)**

使用预训练模型对音频进行编解码重建：

python inference.py \\  
    \--checkpoint checkpoints/lfs\_experiment\_v1/best\_model.pth \\  
    \--input\_file assets/demo\_input.wav \\  
    \--output\_dir results/

### **4\. 提取离散 Token (Tokenization)**

如果您只需要提取语音的离散编码（用于训练下游 LLM）：

python tokenize.py \--checkpoint ... \--input\_file ... \--output\_file tokens.npy

## **🎧 试听样例 (Audio Samples)**

请访问我们的 Demo Page 收听更多重建音频样本：  
👉 LFS-Codec Demo Page

## **📜 引用 (Citation)**

如果您觉得本工作对您的研究有帮助，请引用我们的论文：

@inproceedings{anonymous2024lfs,  
  title={Low-Frame-Rate Speech Coding with Regularized Codebook Remapping and Dual-Branch Timbre Decoupling},  
  author={Anonymous Authors},  
  booktitle={ICME},  
  year={2024}  
}

## **📄 协议 (License)**

本项目代码采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源协议。
