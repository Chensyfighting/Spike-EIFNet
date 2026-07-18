# Spike-EIFNet
Spike-EIFNet: Lightweight Spike-driven Event-Image Fusion Network for Accurate and Efficient Semantic Segmentation

![SEINet Overview](figs/SEINet_Overview.png)

This repository is an official PyTorch implementation of our recent work: Spike-EIFNet, accepted by **IEEE Transactions on Neural Networks and Learning Systems (TNNLS'26)**.

# Abstract
Semantic segmentation is critical for intelligent robotics to understand complex environments. While CNN-based models on RGB images achieve high performance, their accuracy drops in fast-motion or low-light scenes. Fortunately, event cameras, with high temporal resolution and low latency, offer robust perception in such challenging conditions. Many event-image fusion methods attempt to combine the complementary strengths of both modalities, but most adopt simple fusion strategies without considering inter-modal correlations or designing computationally expensive architectures, resulting in degraded accuracy and high energy costs. 
To overcome these limitations, we propose Spike-EIFNet, a lightweight spiking neural network (SNN)-based event-image fusion network that leverages the complementary strengths of multi-modal fusion and the energy-efficient spike-driven computation. Specifically, to reduce computation cost for lightweight, Spike-EIFNet adopts a dual-branch SNN encoder to process events and images in parallel.

Then, to improve the segmentation accuracy with enhanced feature interaction, we introduce a spike-driven cross-modal fusion (SCMF) module, consisting of a modality-aware fine-grained extraction (MFE) stage to capture dynamic cues from events and spatial details from images, followed by a cross-modal interaction and fusion (CIF) stage for effective feature alignment. Finally, a lightweight feature enhancement (LFE) module is proposed to further refine feature representations and facilitate deep-shallow feature fusion. Extensive experiments demonstrate that Spike-EIFNet achieves 67.34\% and 58.09\% mIoU on the DDD17 and DSEC-Semantic datasets, while consuming 72.83$\times$ and 100.26$\times$ less energy, respectively. Compared with ANN-based methods, Spike-EIFNet significantly reduces energy consumption, while among SNN-based methods, it achieves the highest segmentation accuracy with a favorable accuracy-efficiency trade-off.

# Contribution
- We propose Spike-EIFNet, a lightweight SNN-based event-image fusion network for semantic segmentation, which combines a spike-driven dual-branch backbone with an efficient modality-specific fusion strategy to achieve a favorable accuracy-efficiency trade-off.

- We design an SCMF module to fuse complementary features accurately, consisting of an MFE stage for modality-specific feature capture, and a CIF stage for aligning and fusing image semantics with event dynamics.

- We introduce an LFE module that enriches fused representations by capturing informative cues across channel and spatial dimensions, strengthening both shallow encoder and deep decoder features.

- Experiments show that Spike-EIFNet achieves 67.34% and 58.09% mIoU on two benchmark datasets, while reducing energy consumption by up to 72.83× and 100.26×, reaching the highest accuracy among SNN-based methods while maintaining superior efficiency in both computation and energy usage.

# Dataset
To proceed, please download the DDD17/DSEC-SEMANTIC dataset on your own.
