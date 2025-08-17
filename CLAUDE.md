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
**IMPORTANT**: Run commands from DomainBed directory using python -m module format:

```bash
# Change to DomainBed directory first
%cd DomainBed

# Test CLIP baseline (short run for testing)
!python -m domainbed.scripts.train \
    --data_dir ../data \
    --dataset OfficeHome \
    --algorithm CLIPZeroShot \
    --test_env 0 \
    --steps 100 \
    --output_dir ../../outputs/clip_baseline_test

# Test FADA-CLIP Phase 1 (should behave like CLIPZeroShot)
!python -m domainbed.scripts.train \
    --data_dir ../data \
    --dataset OfficeHome \
    --algorithm FADA_CLIP \
    --test_env 0 \
    --steps 10 \
    --output_dir ../../outputs/fada_clip_phase1_test

# Full FADA-CLIP training (once implementation is complete)
!python -m domainbed.scripts.train \
    --data_dir ../data \
    --dataset OfficeHome \
    --algorithm FADA_CLIP \
    --test_env 0 \
    --output_dir ../../outputs/fada_clip_art

# Collect and analyze results
!python -m domainbed.scripts.collect_results \
    --input_dir ../../outputs/
```

### Useful Bash Scripts

**Create these scripts in the project root directory for easy testing:**

**1. `test_clip_baseline.sh` - Test CLIPZeroShot on all domains:**
```bash
#!/bin/bash
cd DomainBed
echo "Testing CLIPZeroShot on all 4 domains..."

for env in 0 1 2 3; do
    echo "Running test_env $env..."
    python -m domainbed.scripts.train \
        --data_dir ../data \
        --dataset OfficeHome \
        --algorithm CLIPZeroShot \
        --test_env $env \
        --steps 100 \
        --output_dir ../../outputs/clip_baseline_env$env
done

echo "Collecting results..."
python -m domainbed.scripts.collect_results \
    --input_dir ../../outputs/
```

**2. `test_fada_clip.sh` - Test FADA-CLIP on all domains:**
```bash
#!/bin/bash
cd DomainBed
echo "Testing FADA-CLIP on all 4 domains..."

for env in 0 1 2 3; do
    echo "Running FADA-CLIP test_env $env..."
    python -m domainbed.scripts.train \
        --data_dir ../data \
        --dataset OfficeHome \
        --algorithm FADA_CLIP \
        --test_env $env \
        --steps 100 \
        --output_dir ../../outputs/fada_clip_env$env
done

echo "Collecting results..."
python -m domainbed.scripts.collect_results \
    --input_dir ../../outputs/
```

**3. `quick_test.sh` - Quick functionality test:**
```bash
#!/bin/bash
cd DomainBed
echo "Quick FADA-CLIP functionality test..."

python -m domainbed.scripts.train \
    --data_dir ../data \
    --dataset OfficeHome \
    --algorithm FADA_CLIP \
    --test_env 0 \
    --steps 10 \
    --output_dir ../../outputs/quick_test
```

### Development Workflow

**IMPORTANT: All experiments run on Google Colab, not locally**

```bash
# Local development (Claude Code environment)
# - Implement algorithms in DomainBed/domainbed/algorithms.py
# - Update hyperparameters in DomainBed/domainbed/hparams_registry.py  
# - Commit and push to GitHub

# Experiment execution (Google Colab)
# - Pull latest changes from GitHub
# - Run training scripts using the bash scripts provided
# - Test implementations with quick_test.sh before full runs

# Repository updates
git pull origin main  # Get latest changes before running experiments
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

## CRITICAL LESSONS LEARNED - CLIP BASELINE PERFORMANCE

### What Caused 72% → 82.4% Performance Jump
**NEVER repeat these mistakes:**

1. **Wrong CLIP Model Version** 
   - ❌ **Mistake**: Using ViT-B/32 or smaller models
   - ✅ **Fix**: Always use ViT-B/16 for better performance
   - 📈 **Impact**: ~7-8% improvement

2. **Wrong Image Preprocessing**
   - ❌ **Mistake**: Using ImageNet normalization instead of CLIP's native preprocessing
   - ✅ **Fix**: Always use `clip.load(model)[1]` for CLIP's preprocessing pipeline
   - 📈 **Impact**: ~2-3% improvement

3. **Missing Class Name Integration**
   - ❌ **Mistake**: Hardcoding class names or using wrong class names
   - ✅ **Fix**: Automatically extract from dataset structure in `datasets.py`
   - 📈 **Impact**: ~1-2% improvement

### CRITICAL Configuration Requirements:
```python
# In hparams_registry.py:
_hparam('clip_backbone', 'ViT-B/16', lambda r: r.choice(['ViT-B/16']))  # NOT ViT-B/32
_hparam('clip_transform', True, lambda r: True)  # ESSENTIAL for CLIP preprocessing

# In datasets.py:
class_names = [f.name for f in os.scandir(os.path.join(root, environments[0])) if f.is_dir()]
hparams['class_names'] = sorted(class_names)  # Dynamic class extraction
```

**Next session should start with implementing FADA-CLIP extension**

## CRITICAL DEVELOPMENT PRINCIPLES

⚠️ **NEVER implement custom functions without checking existing libraries first** ⚠️

### Implementation Risk Minimization Rules:

1. **Library-First Approach**
   - ✅ **ALWAYS check** if CLIP, PyTorch, NumPy, or other libraries already provide the function
   - ✅ **Search documentation** thoroughly before writing custom code
   - ✅ **Use existing implementations** from proven codebases when available
   - ❌ **NEVER assume** a function doesn't exist - verify first

2. **Reference Implementation Priority**
   - ✅ **Use proven patterns** from REFERENCE_RESOURCES/ whenever possible
   - ✅ **Copy-adapt existing code** rather than writing from scratch
   - ✅ **Minimal custom code** - only when absolutely necessary for novel algorithm
   - ❌ **Avoid reinventing** standard operations (FFT, attention, etc.)

3. **Incremental Development**
   - ✅ **One component at a time** - Never add multiple modules simultaneously
   - ✅ **Test each component separately** - Create standalone tests before integration
   - ✅ **Verify functionality** - Ensure each component works as expected before moving on
   - ❌ **No bulk implementation** - Avoid implementing 5-10 things at once

### Example Safe Implementation Workflow:
```
1. Check PyTorch/CLIP docs for existing FFT functions → Use torch.fft if available
2. Check REFERENCE_RESOURCES/ for proven patterns → Copy-adapt existing code
3. Implement ONE component → Test standalone → Verify works
4. Integrate to main system → Test integration → Verify no regression
5. Only then move to next component
```

### Pre-Implementation Checklist:
- [ ] Searched PyTorch documentation for existing functions
- [ ] Searched CLIP library documentation  
- [ ] Checked NumPy/SciPy for standard operations
- [ ] Reviewed REFERENCE_RESOURCES/ for proven patterns
- [ ] Identified minimal custom code needed
- [ ] Plan to implement one component at a time

### Current Status Update:
- ✅ **CLIPZeroShot baseline**: 82.5% performance achieved (confirmed)
- ✅ **CLIP preprocessing**: Using native CLIP transforms
- ✅ **Class name extraction**: Dynamic from dataset structure
- ✅ **FADA-CLIP Phase 1**: Working skeleton implemented, behaves identically to CLIPZeroShot
- ✅ **Ready for Phase 2**: Frequency decomposition implementation

### CLIPZeroShot Baseline Results (Leave-One-Domain-Out):
```
Algorithm             A         C         P         R         Avg                  
CLIPZeroShot          83.1      68.2      89.3      89.5      82.5
```
- **Art (A)**: 83.1% - Best performance in artistic domain (unexpected!)
- **Clipart (C)**: 68.2% - Weakest domain (as expected from literature)  
- **Product (P)**: 89.3% - Strong performance on product images
- **Real-World (R)**: 89.5% - Strongest domain (as expected)
- **Target**: Improve to 87-89% average through frequency-aware processing