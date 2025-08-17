## Immediate Next Steps - Action Plan

### Today/Tomorrow: Foundation & Resources

#### Step 1: Gather Critical Implementation References (2-3 hours)
**You need to fetch:**
1. **FDA GitHub repo**: `https://github.com/YanchaoYang/FDA`
   - Look for `FDA_demo.py` - core FFT swap implementation
   - Check `utils/freq_space_interpolation.py`

2. **CLIP-Adapter source**: `https://github.com/gaopengcuhk/CLIP-Adapter`
   - Focus on `clip_adapter.py` - where adapters insert into CLIP
   - Check training loop in `train.py`

3. **FFDI paper details** (ArXiv 2201.08029)
   - Fetch Section 3.2 (Frequency Disentanglement Module)
   - Section 3.3 (Information Interaction Mechanism)
   - Table with Office-Home results

#### Step 2: Set Up Development Environment (1-2 hours)
```bash
# Create environment
conda create -n fada python=3.9
conda activate fada

# Core dependencies
pip install torch torchvision transformers
pip install clip-openai timm
pip install numpy scipy scikit-learn

# Office-Home dataset tools
pip install gdown  # for downloading
```

#### Step 3: Download Office-Home & Verify Baseline (2-3 hours)
```python
# Quick baseline test script
import clip
import torch
from torchvision import datasets, transforms

# Load CLIP
model, preprocess = clip.load("ViT-B/16", device="cuda")

# Test on one Office-Home domain
# Should get ~82.4% average
```

### Week 1: Core Implementation

#### Monday-Tuesday: FFT Module Development
**Deliverable**: `frequency_decomposer.py`
```python
class FrequencyDecomposer(nn.Module):
    def __init__(self, threshold=0.1):
        # Implement 2D FFT decomposition
        # Based on FDA paper
        
    def decompose(self, features):
        # Returns low_freq, high_freq
```

#### Wednesday-Thursday: CLIP Integration Study
**Deliverable**: Analysis document
- Test where to insert decomposer (try 3 locations):
  1. After patch embedding (layer 0)
  2. After middle transformer block (layer 6)
  3. After final transformer block (layer 11)
- Measure impact on baseline accuracy

#### Friday: Dual-Stream Architecture
**Deliverable**: `dual_stream_clip.py`
```python
class DualStreamCLIP(nn.Module):
    def __init__(self, clip_model, decompose_at_layer=6):
        # Implement dual-stream forward pass
```

### Week 2: Adapter Design

#### Monday-Tuesday: Basic Adapter Implementation
**Deliverable**: `frequency_adapters.py`
- Low-frequency adapter (smaller, ~0.1M params)
- High-frequency adapter (larger, ~0.5M params)
- Use LoRA-style or bottleneck design

#### Wednesday-Thursday: Fusion Mechanism
**Deliverable**: Stream fusion module
- Test 3 fusion strategies:
  1. Simple addition
  2. Learned weighted sum
  3. Cross-attention fusion

#### Friday: Complete Pipeline
**Deliverable**: `train_fada.py`
- End-to-end training script
- Implement Office-Home evaluation protocol

### Week 3: Training & Optimization

#### Monday-Tuesday: Loss Functions
**Deliverable**: `losses.py`
- Frequency consistency loss
- Contrastive loss between streams
- Total loss balancing

#### Wednesday-Thursday: Training Strategy
**Deliverable**: Optimized training loop
- Implement SWAD
- Add frequency-based augmentation
- Hyperparameter tuning

#### Friday: Initial Results
**Checkpoint**: First complete results on Office-Home

### Week 4: Refinement & Ablations

#### Monday-Tuesday: Ablation Studies
**Deliverable**: Ablation table
- Component-wise analysis
- Frequency threshold sensitivity
- Adapter size impact

#### Wednesday-Thursday: SOTA Integration
**Deliverable**: Combined method
- Add attention head purification
- Integrate with prompt learning
- Ensemble strategies

#### Friday: Final Evaluation
**Deliverable**: Complete results and analysis

## Critical Decision Points

### This Weekend - Need Your Input:
1. **Which FDA variant to use?**
   - Original FDA (simple, proven)
   - FACT's Amplitude Mix (more sophisticated)
   - Custom hybrid approach

2. **Adapter architecture preference?**
   - LoRA (parameter efficient, proven with CLIP)
   - Bottleneck adapter (more traditional)
   - Convolution-based (better for frequency features?)

3. **Compute resources available?**
   - Single GPU: Focus on ViT-B/16
   - Multiple GPUs: Can try ViT-L/14
   - Limited: Use smaller ResNet-based CLIP

## Risk Mitigation Strategies

### If frequency decomposition doesn't improve CLIP:
- **Pivot 1**: Apply to intermediate features instead of final
- **Pivot 2**: Use wavelet transform instead of FFT
- **Pivot 3**: Focus on attention map decomposition

### If training is unstable:
- Freeze more CLIP layers
- Reduce learning rate for frequency modules
- Add stronger regularization

### If we're stuck at ~85% (below SOTA):
- Combine with existing SOTA methods
- Try multi-scale frequency decomposition
- Add test-time adaptation

## Deliverables Timeline

| Week | Deliverable | Success Metric |
|------|------------|----------------|
| 1 | Working FFT decomposer + CLIP integration | Baseline maintained (82%) |
| 2 | Complete dual-stream with adapters | Improvement to 84%+ |
| 3 | Optimized training pipeline | Reach 85-86% |
| 4 | Final method with ablations | Target 87-88% |

## Next Immediate Action (Right Now):

**Please fetch:**
1. FDA source code from GitHub (specifically the FFT implementation)
2. FFDI paper Section 3 (the technical details)
3. One working CLIP adapter implementation

Then I can write the exact FFT decomposer module tailored to CLIP's feature format. Should we start with fetching the FDA code?