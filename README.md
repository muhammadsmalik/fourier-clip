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

## Implementation Approach

**Framework**: Dassl + OpenAI CLIP (following proven CLIP-Adapter pattern)
- **Base**: CLIP-Adapter architecture (simple, proven to work)
- **Extension**: Frequency decomposition + dual-stream adapters
- **Benefits**: Direct research lineage and comparability

## Current Status

**Phase 1**: Environment Setup ✅  
**Phase 2**: FADA-CLIP Algorithm Implementation (in progress)
- Following CLIP-Adapter's proven pattern
- Adding frequency decomposition and dual-stream architecture
- Target: 87-89% on Office-Home (vs 82.4% CLIP baseline)

## Execution

- **`fada_clip_colab.ipynb`**: Main execution notebook for Google Colab
  - Clean setup following CLIP-Adapter approach
  - Environment setup and dataset loading
  - Training and evaluation pipeline
  - Development workflow and git integration

## Target Performance

- **Current CLIP baseline**: 82.4% on Office-Home
- **Current SOTA**: 87.0% (attention head purification)
- **FADA target**: 87-89% through frequency-aware processing

## Project Structure

```
fourier-clip/
├── fada_clip_colab.ipynb        # 🚀 Main Colab execution file
├── DomainBed/                   # Modified DomainBed framework
│   ├── domainbed/algorithms.py  # FADA-CLIP algorithm implementation
│   └── domainbed/datasets.py    # Office-Home dataset support
├── RESEARCH_PLANNING_DOCS/      # Comprehensive research documentation
├── REFERENCE_RESOURCES/         # Implementation references (FDA, FFDI, CLIP-Adapter)
└── data/                        # Office-Home dataset (auto-downloaded)
```

## Research Documentation

- [`high-level-implementation-plan.md`](RESEARCH_PLANNING_DOCS/high-level-implementation-plan.md) - 12-week implementation roadmap
- [`frequency-decomp-research.md`](RESEARCH_PLANNING_DOCS/frequency-decomp-research.md) - Comprehensive literature review
- [`office-home-research-motivation.md`](RESEARCH_PLANNING_DOCS/office-home-ressearch-motivation.md) - Dataset analysis and challenges

## Quick Start

1. **Open in Google Colab**: Upload `fada_clip_colab.ipynb`
2. **Run setup cells**: Install dependencies and clone repository
3. **Verify baseline**: Test CLIP + DomainBed integration
4. **Train FADA**: Execute training once algorithm is implemented
5. **Evaluate**: Compare results across Office-Home domains

## Implementation Timeline

**Phase 1**: Environment setup ✅  
**Phase 2**: Core frequency decomposition module (current)
- Implement FDA-based frequency decomposer
- 2D FFT decomposition with configurable threshold
- Integration with CLIP feature extraction

**Phase 3**: Adapter design and integration  
- Low-frequency adapter (content preservation)
- High-frequency adapter (style neutralization) 
- Dual-stream architecture following CLIP-Adapter pattern

**Phase 4**: Training pipeline and optimization
- FFDI-based information interaction mechanism
- Frequency consistency and contrastive losses
- FDAG frequency-domain data augmentation

**Phase 5**: Ablation studies and evaluation
- Component-wise analysis
- Comparison with CLIP baseline and SOTA methods
- Results analysis across Office-Home domains  

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