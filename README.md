# FADA-CLIP: Frequency-Aware Dual-Stream Adaptation for CLIP

This repository implements **Frequency-Aware Dual-Stream Adaptation (FADA)**, a novel approach for improving CLIP's domain generalization performance on the Office-Home dataset by decomposing features into frequency domains and applying stream-specific adaptation.

## Overview

FADA addresses CLIP's weakness in artistic domains (Art ~75-80% vs Real-World ~85-90%) by:

1. **Frequency Decomposition**: Separating CLIP features into high-frequency (texture/style) and low-frequency (shape/content) components
2. **Dual-Stream Adaptation**: Applying different adaptation strategies to each frequency stream
3. **Information Interaction**: Fusing streams using spatial attention mechanisms
4. **Frequency Augmentation**: Training with frequency-domain data augmentation

## Key Innovation

Unlike existing methods that remove or ignore problematic features, FADA separates and processes them appropriately:
- **Low-frequency stream**: Preserves semantic content with minimal adaptation
- **High-frequency stream**: Neutralizes style variations with aggressive adaptation
- **Fusion mechanism**: Combines streams using learnable attention

## Target Performance

- **Current CLIP baseline**: 82.4% on Office-Home
- **Current SOTA**: 87.0% (attention head purification)
- **FADA target**: 87-89% through frequency-aware processing

## Project Structure

```
fourier-clip/
├── DomainBed/                    # Modified DomainBed framework
│   ├── domainbed/algorithms/     # Includes FADA algorithm
│   └── domainbed/datasets/       # Office-Home dataset loader
├── RESEARCH_PLANNING_DOCS/       # Research documentation
├── REFERENCE_RESOURCES/          # Implementation references
└── experiments/                  # Experiment configurations
```

## Research Documentation

- [`high-level-implementation-plan.md`](RESEARCH_PLANNING_DOCS/high-level-implementation-plan.md) - 12-week implementation roadmap
- [`frequency-decomp-research.md`](RESEARCH_PLANNING_DOCS/frequency-decomp-research.md) - Comprehensive literature review
- [`office-home-research-motivation.md`](RESEARCH_PLANNING_DOCS/office-home-ressearch-motivation.md) - Dataset analysis and challenges

## Implementation Timeline

**Phase 1 (Days 1-3)**: Environment setup and baseline verification  
**Phase 2 (Days 4-7)**: Core frequency decomposition module  
**Phase 3 (Days 8-11)**: Adapter design and integration  
**Phase 4 (Days 12-15)**: Training pipeline and optimization  
**Phase 5 (Days 16-18)**: Ablation studies and final evaluation  

## Academic Context

This work builds on:
- **FDA** (Yang & Soatto, CVPR 2020): Fourier domain adaptation principles
- **FFDI** (Wang et al., ACM MM 2022): Dual-stream frequency architectures
- **CLIP-Adapter** (Gao et al.): Parameter-efficient CLIP adaptation

## Citation

```bibtex
@misc{malik2024fada,
  title={FADA-CLIP: Frequency-Aware Dual-Stream Adaptation for Domain Generalization},
  author={Salman Malik},
  year={2024},
  institution={University of Cambridge}
}
```

## MPhil Thesis - University of Cambridge

This repository contains the implementation for an MPhil Computer Science thesis investigating frequency-domain approaches to vision-language model adaptation for domain generalization.