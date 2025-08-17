# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository implements **FADA-CLIP (Frequency-Aware Dual-Stream Adaptation for CLIP)**, an MPhil Computer Science thesis project at University of Cambridge. The goal is to improve CLIP's domain generalization on Office-Home dataset from 82.4% to 87-89% by decomposing features into frequency domains and applying stream-specific adaptation.

## Session Handover - Current Status

### What We've Accomplished
1. **✅ Repository Setup**: Created GitHub repo with comprehensive research documentation
2. **✅ Framework Decision**: Chose to follow CLIP-Adapter approach using DomainBed + OpenAI CLIP
3. **✅ Environment Setup**: Created minimal `fada_clip_colab.ipynb` that properly uses DomainBed
4. **✅ Architecture Planning**: Designed FADA extension of CLIP-Adapter with frequency decomposition

### Critical Design Decisions Made
- **Framework**: DomainBed + OpenAI CLIP (not Hugging Face) following proven CLIP-Adapter pattern
- **Approach**: Extend CLIP-Adapter with frequency decomposition rather than building from scratch
- **Execution**: Use DomainBed's scripts (`train.py`, `collect_results.py`) rather than custom training loops
- **Implementation**: Add algorithms to `DomainBed/domainbed/algorithms.py` following framework patterns

### Immediate Next Steps
1. **Priority 1**: Implement `CLIPZeroShot` algorithm in `DomainBed/domainbed/algorithms.py`
   - Add 'CLIPZeroShot' to `ALGORITHMS` list (line 60)
   - Implement class following the reference pattern in `REFERENCE_RESOURCES/CLIP_ADAPTER_IMPLEMENTATION.md`
   - Test baseline performance (~82.4% expected)

2. **Priority 2**: Implement `FADA_CLIP` algorithm extending CLIPZeroShot
   - Add frequency decomposition module based on `REFERENCE_RESOURCES/FDA_IMPLEMENTATION.md`
   - Add dual-stream adapters for high/low frequency components
   - Add information interaction mechanism from `REFERENCE_RESOURCES/FDFI.md`

### Where Everything Is Located
- **Main execution**: `fada_clip_colab.ipynb` (minimal, uses DomainBed properly)
- **Implementation target**: `DomainBed/domainbed/algorithms.py` (add new algorithms here)
- **Research docs**: `RESEARCH_PLANNING_DOCS/` (comprehensive planning and literature review)
- **Reference implementations**: `REFERENCE_RESOURCES/` (CLIP-Adapter, FDA, FFDI patterns)
- **Framework**: `DomainBed/` (handles datasets, training, evaluation automatically)

## Core Architecture

FADA-CLIP extends the proven CLIP-Adapter pattern with frequency decomposition:

1. **Base Framework**: DomainBed + OpenAI CLIP (following CLIP-Adapter approach)
2. **Frequency Decomposition**: FDA-based module separating high/low frequency components  
3. **Dual-Stream Adapters**: Separate adaptation strategies for texture (high-freq) vs content (low-freq)
4. **Information Interaction**: FFDI-based fusion mechanism using spatial attention

The implementation follows DomainBed's algorithm pattern where all algorithms inherit from the `Algorithm` base class in `DomainBed/domainbed/algorithms.py`.

## Key Commands

### Training and Evaluation
```bash
# Train FADA-CLIP with leave-one-domain-out protocol
python DomainBed/domainbed/scripts/train.py \
    --data_dir ./data \
    --dataset OfficeHome \
    --algorithm FADA_CLIP \
    --test_env 0 \
    --output_dir ./outputs/fada_clip_art

# Test CLIP baseline
python DomainBed/domainbed/scripts/train.py \
    --data_dir ./data \
    --dataset OfficeHome \
    --algorithm CLIPZeroShot \
    --test_env 0 \
    --steps 100 \
    --output_dir ./outputs/clip_baseline_test

# Collect and analyze results
python DomainBed/domainbed/scripts/collect_results.py \
    --input_dir ./outputs/
```

### Development Workflow
```bash
# Main execution environment
# Use fada_clip_colab.ipynb in Google Colab

# Repository updates
git pull origin main  # Get latest changes
```

## Implementation Status

- **Phase 1**: Environment setup ✅
- **Phase 2**: Core frequency decomposition module (current)
- **Phase 3**: Adapter design and integration (pending)
- **Phase 4**: Training pipeline and optimization (pending) 
- **Phase 5**: Ablation studies and evaluation (pending)

## Algorithm Implementation Pattern

New algorithms must be added to `DomainBed/domainbed/algorithms.py`:

1. Add algorithm name to `ALGORITHMS` list
2. Implement class inheriting from `Algorithm` base class
3. Implement required methods: `__init__()`, `update()`, `predict()`
4. Follow CLIP-Adapter pattern for CLIP integration

Example structure:
```python
class FADA_CLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Load CLIP model, add frequency decomposer and adapters
        
    def update(self, minibatches, unlabeled=None):
        # Training step with frequency-aware losses
        
    def predict(self, x):
        # Forward pass through dual-stream architecture
```

## Reference Materials

Key implementation references in `REFERENCE_RESOURCES/`:
- `CLIP_ADAPTER_IMPLEMENTATION.md`: Proven CLIP-Adapter pattern to follow
- `FDA_IMPLEMENTATION.md`: Fourier Domain Adaptation core FFT operations
- `FDFI.md`: Dual-stream architecture and information interaction mechanism

Research documentation in `RESEARCH_PLANNING_DOCS/`:
- `high-level-implementation-plan.md`: 12-week implementation roadmap
- `frequency-decomp-research.md`: Comprehensive literature review
- `office-home-ressearch-motivation.md`: Dataset analysis and challenges

## Expected Performance Targets

- CLIP Baseline: ~82.4% on Office-Home
- Current SOTA: ~87.0% (attention head purification)
- FADA Target: 87-89% through frequency-aware processing
- Key challenge: Art/Clipart domains perform worst (~75-80% vs Real-World ~85-90%)

## DomainBed Integration Notes

- Office-Home dataset is automatically downloaded by DomainBed
- Leave-one-domain-out evaluation protocol is standard (4 domains: Art, Clipart, Product, Real-World)
- All dependencies are handled by DomainBed's installation
- Results collection and analysis use DomainBed's built-in scripts
- Training loops, data loading, and evaluation metrics are provided by the framework

## Important Context from Previous Session

### Research Motivation
- Office-Home is particularly challenging: 65 classes vs PACS's 7 classes
- CLIP has known weakness in artistic domains (Art ~75-80% vs Real-World ~85-90%)
- Frequency decomposition addresses this by separating style (high-freq) from content (low-freq)
- Target is to reach 87-89% performance (current SOTA is 87.0% via attention head purification)

### Technical Approach Decided
- **Base**: Use CLIP-Adapter's simple but proven pattern (just 60 lines of core code)
- **Extension**: Add FDA-based frequency decomposition before adapter
- **Architecture**: Dual-stream processing with separate adapters for each frequency component
- **Fusion**: FFDI's Information Interaction Mechanism using spatial attention

### Key Files to Reference
- `REFERENCE_RESOURCES/CLIP_ADAPTER_IMPLEMENTATION.md`: Shows exact CLIP-Adapter pattern to follow
- `REFERENCE_RESOURCES/FDA_IMPLEMENTATION.md`: Core FFT operations for frequency decomposition
- `REFERENCE_RESOURCES/FDFI.md`: Dual-stream architecture and fusion mechanism
- `RESEARCH_PLANNING_DOCS/high-level-implementation-plan.md`: Detailed 12-week roadmap

### Development Workflow
- User will work in Google Colab using `fada_clip_colab.ipynb`
- All actual algorithm implementation goes in `DomainBed/domainbed/algorithms.py`
- Use `git pull origin main` to get updates
- Test with short runs first (`--steps 100`) before full training

### Current Blockers Resolved
- ✅ Framework choice: DomainBed + OpenAI CLIP (not Hugging Face)
- ✅ Execution method: DomainBed scripts (not custom training loops)
- ✅ Repository structure: Clean separation of execution vs implementation
- ✅ Dependencies: All handled by DomainBed installation

**Next session should start with implementing CLIPZeroShot baseline in algorithms.py**

## CRITICAL DEVELOPMENT PRINCIPLE

⚠️ **NEVER implement multiple components at once** ⚠️

**Development Rules:**
1. **One component at a time** - Never add multiple modules simultaneously
2. **Test each component separately** - Create standalone tests before integration
3. **Verify functionality** - Ensure each component works as expected before moving on
4. **Incremental integration** - Add components one by one to the main system
5. **Debug early** - If something doesn't work, fix it immediately before adding more

**Example Workflow:**
1. Implement `CLIPZeroShot` baseline → Test → Verify performance (~82.4%)
2. Add frequency decomposition module → Test standalone → Integrate
3. Add low-frequency adapter → Test → Integrate  
4. Add high-frequency adapter → Test → Integrate
5. Add fusion mechanism → Test → Integrate

**Why This Matters:**
- Implementing 5-10 things at once makes debugging extremely difficult
- Each component must be verified to work before building on top of it
- Incremental development allows for early detection and fixing of issues
- don't need to commit colab ipynb notebook to github