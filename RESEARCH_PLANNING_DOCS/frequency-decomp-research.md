# Frequency decomposition approaches transform domain generalization through spectral analysis

Frequency domain methods have emerged as a powerful paradigm for domain generalization and adaptation, with extensive research demonstrating that separating high-frequency texture/style components from low-frequency shape/content components enables robust cross-domain performance. The field has produced multiple dual-stream architectures specifically tested on Office-Home dataset, established theoretical foundations linking spectral analysis to domain shift, and extended these methods to modern vision-language models like CLIP.

## Foundational frequency domain methods establish core principles

The breakthrough **FDA: Fourier Domain Adaptation** paper (Yang & Soatto, CVPR 2020) established the fundamental principle that low-frequency amplitude captures style/appearance while phase preserves semantic structure. Their remarkably simple approach swaps low-frequency amplitude components between source and target domains using FFT, achieving state-of-the-art results on GTA5→CityScapes (50.45% mIOU) without any training. The mathematical formulation uses a mask Mβ to control frequency region size (typically β ≤ 0.15), with domain adaptation achieved through x^{s→t} = F^{-1}([Mβ ◦ FA(xt) + (1-Mβ) ◦ FA(xs), FP(xs)]).

Building on this foundation, the **FACT framework** (Xu et al., CVPR 2021) introduced Amplitude Mix (AM) strategy, linearly interpolating amplitude spectra while preserving phase: Â(x) = (1-λ)A(x) + λA(x'), where λ ~ U(0,η). Combined with co-teacher regularization using momentum-updated models, FACT achieved **81.5% on Digits-DG** and **84.51% on PACS** with ResNet18. The key insight that phase information contains high-level semantics robust to domain shifts has influenced numerous subsequent approaches.

Recent advances include **PhaMa** (2023) introducing patch contrastive learning for phase spectrum matching, **HMCA** (2025) using multi-spectral attention for high-frequency components, and **FAD** (2024) implementing frequency diversion adapters with band-specific processing. The **Spectral Batch Normalization** method (2023) normalizes features in frequency domain rather than spatial domain, inserting stochastic noise and enabling channel weighting for improved generalization. Mathematical foundations rely on 2D DFT: F(u,v) = ΣΣ f(x,y) * e^{-j2π(ux/M + vy/N)}, with amplitude A(u,v) = |F(u,v)| and phase P(u,v) = arctan(I(u,v)/R(u,v)).

## Dual-stream architectures separate style and content through frequency

The **Domain Generalization via Frequency-domain-based Feature Disentanglement and Interaction (FFDI)** paper (Wang et al., ACM MM 2022) presents the most directly relevant dual-stream architecture for Office-Home dataset. Their encoder-decoder structure disentangles high-frequency components (preserving edge structure and semantic information) from low-frequency components (containing smooth structure susceptible to domain shifts). The Information Interaction Mechanism (IIM) fuses frequency features while Frequency-domain-based Data Augmentation (FDAG) enhances training. **FFDI achieved state-of-the-art results on Office-Home** using ResNet18, with detailed accuracy numbers for Art, Clipart, Product, and Real World domain transitions.

The **Dual-stream Feature Augmentation for Domain Generalization** architecture (2024) implements dual-path feature disentanglement with domain-invariant and domain-specific encoders. Using hard feature perturbation with semantic consistency and adversarial mask modules for causal information mining, it achieves **86.80% average accuracy on PACS**, outperforming existing methods. The architecture combines domain-related feature augmentation for transferability with causal-related augmentation for discriminability, training through contrastive learning on dual-stream augmented features.

**FNSDA (Fourier Neural Simulator for Dynamical Adaptation)** (2024) introduces parameter-efficient adaptation in frequency domain using learnable filters that decompose Fourier modes into shared dynamics and system-specific discrepancy components. Remarkably, it requires only **0.088K-0.096K parameters** for adaptation versus 0.035M-1.162M for baseline methods. Common architectural patterns include dual encoders for domain-invariant versus domain-specific features, FFT/DCT/wavelet transforms for component separation, learnable filtering for automatic frequency mode partitioning, and feature fusion through contrastive learning or attention mechanisms.

## High versus low frequency decomposition reveals distinct roles

Research consistently demonstrates that **high-frequency components encode texture, edges, fine details, and style information**, while **low-frequency components capture shape, global structure, content, and domain-specific characteristics**. The FAD paper (2024) explicitly states that low frequencies capture global shape/layout while high frequencies encode fine-grained textures/edges, implementing band-wise adaptation with dedicated convolutional branches using kernel sizes matched to spectral scales.

The **Combined Spatial and Frequency Dual Stream Network** integrates multi-scale frequency decomposition modules for high-frequency trace extraction using DCT transformation with channel attention for adaptive frequency extraction. **FADS-Net** implements separate raw input and frequency recalibration streams, with adaptive frequency recalibration modules enhancing data-driven frequency processing. The **Time and Frequency Synergy approach (TFDA)** mines temporal and frequency features through dual-branch networks with time-frequency consistency constraints for domain shift reduction.

Style-content disentanglement methods leverage these frequency characteristics effectively. The **FreMixer module** (AAAI Conference) disentangles frequency spectrum of content/style based on different frequency bands and patterns. The **Unveiling Advanced Frequency Disentanglement** paper uses Adaptive Convolutional Composition Aggregation (ACCA) for low-frequency processing and Laplace Decoupled Restoration Model (LDRM) for high-frequency components, with low-frequency consistency constraints decomposing restoration into simpler tasks.

## CLIP and vision-language models benefit from frequency adaptation

The **FrogDogNet** paper (Gunduboina et al., 2024) represents the most significant application of frequency methods to CLIP, integrating Fourier frequency filtering for remote sensing domain generalization. Their **Fourier Filter Block (FFB) selectively retains 350 out of 512 low-frequency components** while preserving structural information, combined with Remote Sensing Prompt Alignment Loss for aligning learned prompts with RS-specific initializations. The architecture combines projection networks, self-attention, and FFT-based filtering, consistently outperforming state-of-the-art prompt learning methods across PatternNet, RSICD, RESISC45, and MLRSNet datasets.

While direct frequency analysis applications to CLIP remain limited, related work demonstrates potential. The **CLIP the Divergence** paper (2024) achieves **+10.3% on Office-Home** and **+24.3% on DomainNet** using CLIP for domain divergence measurement. **SpectralKD** (2024) uses spectral analysis to understand Vision Transformers and optimize knowledge distillation, revealing information concentration patterns and similar spectral encoding across architectures. The **Enriching visual feature representations** paper incorporates DFT, DCT, DHT, and Hadamard Transform, achieving **4.8% improvement in CIDEr scores** for image captioning.

Research gaps include limited work on multi-modal frequency analysis, scaling to larger vision-language models, theoretical understanding of frequency decomposition benefits in vision-language contexts, and real-time applications for efficient frequency processing. Vision Transformer backbones (ViT-B/16, ViT-L/14) generally show better frequency-domain compatibility than CNN architectures.

## Style-content disentanglement through spectral decomposition advances rapidly

Multiple approaches demonstrate effective style-content separation using frequency analysis. The **Frequency-Auxiliary One-Shot Domain Adaptation** paper (2024) introduces Low-Frequency Fusion Module (LFF-Module) preserving domain-sharing information through low-frequency features, and High-Frequency Guide Module (HFG-Module) focusing on domain-specific information using high-frequency guidance, applying discrete wavelet transform for one-shot generative domain adaptation.

**UnZipLoRA** decomposes single images into subject (content) and style via disentangled LoRAs, combining prompt separation, column separation, and block separation strategies with parameter-efficient fine-tuning. **FANeRV** implements Wavelet Frequency Upgrade Blocks separating high/low frequency components with Frequency Separation Feature Boosting (FSFB) modules and Time-Modulated Gated Feed-Forward Networks (TGFN) for enhanced high-frequency detail recovery in video neural representations.

The theoretical connection between style transfer and domain adaptation, demonstrated by Li et al. showing Gram matrix matching equals MMD minimization, provides foundation for cross-domain applications. Practical implementations use Power Spectral Density via FFT: P(f) = |DFT(autocorr(x))|², phase-amplitude separation for style features, and STFT log-magnitude processing for multi-modal transfer.

## Neural network adapters combine with frequency transforms effectively

The **Frequency Adaptation and Diversion (FAD)** framework (2024) exemplifies successful integration, treating different frequency bands as semantically distinct components with discrete Fourier transform application on intermediate features. Each frequency band adapts using dedicated convolutional branches with tailored kernel sizes, achieving consistent improvements on Meta-Dataset benchmark for both seen and unseen domains.

**FNSDA** demonstrates extreme parameter efficiency through learnable filters in frequency domain, requiring two orders of magnitude fewer parameters than baseline methods. The integration of frequency analysis with LoRA-style adapters remains underexplored, presenting significant research opportunities. Available implementations include FNSDA (GitHub), DFA dual-stream feature augmentation, and various frequency decomposition libraries for PyTorch/TensorFlow.

## Frequency domain adaptation and spectral generalization mature as fields

The terminology "frequency domain adaptation" and "spectral domain generalization" now represents established research areas. **Class Aware Frequency Transformation (CAFT/CAFT++)** (2024) utilizes pseudo label based class consistent low-frequency swapping, outperforming MixUp and other augmentation strategies by bringing source and target domains closer through frequency manipulation. The finding that class-aware frequency transformation exceeds uniform frequency manipulation effectiveness guides future research.

**Temporal-Spectral Domain Adaptation Network (TSDAN)** (2024) implements dual-branch architecture with temporal CNN and novel spectral neural network branches, narrowing both temporal and spectral feature shifts using Sinkhorn divergence for bearing fault diagnosis applications. The **Block-wise DCT** approach (2024) for deepfake detection captures frequency domain artifacts while retaining spatial semantics through inter/intra-block multi-scale frequency-convolutional networks with hierarchical cross-modal fusion.

## Phase and amplitude manipulation enables fine-grained control

Phase preservation proves critical for maintaining semantic content across all surveyed methods. FDA's success stems from preserving phase while swapping amplitude, FACT's Amplitude Mix strategy maintains phase integrity, and multiple papers demonstrate that phase contains high-level semantics robust to domain shifts. Amplitude manipulation controls style transfer intensity, with typical parameters like β ≤ 0.15 for FDA preventing artifacts.

Advanced manipulation techniques include learnable phase-amplitude separation, multi-band transfer combining predictions from multiple frequency scales, and adaptive frequency band selection based on task requirements. The **StyleTime** paper uses Power Spectral Density and autocorrelation with FFT for time series style transfer, while audio spectrogram methods apply STFT with independent phase-amplitude processing.

## Style transfer insights inspire domain generalization breakthroughs

The **Text-Driven Generative Domain Adaptation with Spectral Consistency Regularization** (ICCV 2023) uses spectral analysis of Hessian matrices preventing mode collapse, with eigenvalue analysis H(z) = J^T(z)J(z) maintaining diversity during adaptation. This spectral consistency regularization preserves domain-invariant spectral properties, directly bridging style transfer and domain generalization.

**Adaptive Instance Normalization (AdaIN)** variants extend to frequency domain: AdaIN(x,y) = σ(y)((x-μ(x))/σ(x)) + μ(y), with multi-scale AdaIN for different frequency bands and frequency-aware normalization combining AdaIN with spectral features. The theoretical connection between MMD-based domain adaptation and frequency domain alignment, established through Gram matrix analysis, provides mathematical foundation for cross-domain transfer.

Inspired approaches include spectral domain alignment using consistency losses, frequency-aware AdaIN operating on frequency statistics, phase-amplitude disentanglement for cross-domain transfer, and progressive adaptation across frequency scales. Implementation ideas span Fourier domain generalization losses, spectral consistency regularization, and frequency-adaptive normalization: FreqAdaIN(x, y) = IFFT(AdaIN(FFT(x), FFT(y))).

## Major conferences showcase accelerating innovation

CVPR, ICCV, ECCV, and NeurIPS 2020-2025 produced numerous frequency-based domain adaptation papers. **ECCV 2024** featured "High-Fidelity and Transferable NeRF Editing by Frequency Decomposition" enabling transferable editing across scene types, and "G2fR: Frequency Regularization in Grid-based Feature Encoding" improving cross-domain performance. **CVPR 2024** presented "Towards Progressive Multi-Frequency Representation for Image Warping" with multi-scale frequency analysis enabling gradual domain adaptation.

Performance benchmarks demonstrate consistent superiority: GTA5→CityScapes achieves 50.45% mIOU with FDA-MBT, SYNTHIA→CityScapes reaches 52.5% mIOU, PACS dataset shows 84.51% (ResNet18) and 88.15% (ResNet50) with FACT, **Office-Home achieves 66.56% average accuracy**, and Digits-DG reaches 81.5%. Frequency domain methods consistently outperform spatial-only approaches while maintaining computational efficiency advantages over adversarial methods.

## Office-Home implementations validate dual-stream frequency approaches

The search for existing dual-stream frequency decomposition on Office-Home reveals **FFDI (2022) as the most directly relevant implementation**, using encoder-decoder architecture to disentangle high and low-frequency features with Information Interaction Mechanism. CAFT/CAFT++ (2024) shows comprehensive Office-Home results using Fourier Domain Adaptation with class-aware transformations and low-frequency swapping. Enhanced Dynamic Feature Representation Learning Framework (2023) explicitly tests Fourier-based dynamic residual feature representation on Office-Home, showing superior performance.

FAMLP implements adaptive Fourier filter layers with learnable frequency filters and low-rank enhancement modules, achieving 3-9% improvements over state-of-the-art on Office-Home. Implementation details typically use leave-one-domain-out protocol across Art, Clipart, Product, and Real World domains, ResNet18/50 backbones, 224x224 image preprocessing, with frequency threshold parameters around r=25 for decomposition.

The comprehensive research reveals frequency decomposition methods as a mature, effective approach for domain generalization with established theoretical foundations, practical implementations achieving state-of-the-art results, and clear pathways for future research combining spectral analysis with modern architectures and vision-language models.