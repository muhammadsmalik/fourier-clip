## Confidence Assessment: **70% chance of achieving SOTA (>87%)**

### Why I'm confident:
1. **Clear gap exists**: No one has combined frequency decomposition with CLIP adapters specifically for Office-Home
2. **Addresses known weakness**: CLIP struggles with Art/Clipart domains (75-80%) where frequency methods excel
3. **Complementary approaches**: Frequency decomposition is orthogonal to current SOTA methods (attention head purification), so they can be combined
4. **Strong baseline**: Starting from CLIP's 82.4% with proven 4-5% improvements from frequency methods

### Why I'm cautious:
1. **FFDI already exists**: They did frequency disentanglement on Office-Home, though with ResNet not CLIP
2. **Marginal gains**: Current SOTA (87%) leaves only ~13% room for improvement
3. **Integration complexity**: Combining frequency streams with CLIP's transformer architecture is non-trivial

## High-Level Implementation Plan

### Phase 1: Baseline Reproduction (Week 1-2)
**Goal**: Reproduce CLIP baseline and understand existing methods

1. **Set up Office-Home evaluation pipeline**
   - Implement leave-one-domain-out protocol
   - Verify CLIP zero-shot baseline (should get ~82.4%)

2. **Study key implementations** - I need you to fetch:
   - FFDI source code (if available) or paper implementation details
   - FDA (Fourier Domain Adaptation) GitHub repo
   - CLIP adapter implementations (CLIP-Adapter, Tip-Adapter)

### Phase 2: Core Dual-Stream Architecture (Week 3-5)
**Goal**: Build the frequency decomposition module for CLIP features

1. **Frequency Decomposition Module**
   ```
   CLIP Image Encoder → Feature Maps → FFT → 
   ├─ Low-Freq Stream (shape/content)
   └─ High-Freq Stream (texture/style)
   ```

2. **Key design decisions needed**:
   - Where to insert frequency decomposition (after which CLIP layer?)
   - Frequency threshold parameter (β value from FDA paper)
   - 2D FFT on spatial features vs 1D FFT on token sequences

### Phase 3: Stream-Specific Adaptation (Week 6-8)
**Goal**: Design learnable adapters for each frequency stream

1. **Low-Frequency Adapter** (Domain-invariant)
   - Lightweight LoRA-style adapter
   - Focus on preserving semantic content
   - Possibly frozen or minimally tuned

2. **High-Frequency Adapter** (Domain-specific)
   - More aggressive adaptation
   - Learn to neutralize style variations
   - Potentially multiple adapters for different domains

3. **Fusion Mechanism**
   - Attention-based fusion vs simple addition
   - Learnable weighting between streams

### Phase 4: Training Strategy (Week 9-10)
**Goal**: Optimize training procedure

1. **Loss Functions**:
   - Classification loss
   - Frequency consistency loss (preserve low-freq across domains)
   - Contrastive loss between streams
   - Optional: Spectral regularization

2. **Training Tricks**:
   - SWAD for flat minima
   - Frequency-based augmentation
   - Progressive frequency threshold adjustment

### Phase 5: Ablation & Optimization (Week 11-12)
**Goal**: Fine-tune and validate contributions

1. **Ablation studies**:
   - Single vs dual stream
   - Different frequency thresholds
   - Various adapter configurations
   - With/without attention head purification

2. **Combine with SOTA techniques**:
   - Integrate attention head purification
   - Add prompt learning
   - Ensemble with other methods

## What I Need From You:

### Critical Resources:
1. **FFDI paper's implementation details** (Section 3.2-3.3 about their encoder-decoder)
2. **FDA source code** from GitHub (for FFT implementation reference)
3. **CLIP-Adapter or Tip-Adapter source code** (for adapter integration patterns)

### Nice to Have:
1. **FACT framework code** (for Amplitude Mix implementation)
2. **Attention Head Purification paper details** (to understand integration potential)
3. **Any existing CLIP + frequency analysis code** (even if not for Office-Home)

### Key Technical Questions to Resolve:

1. **Where in CLIP to insert frequency decomposition?**
   - After patch embedding?
   - After specific transformer blocks?
   - On the final [CLS] token?

2. **How to handle CLIP's sequential token format?**
   - Reshape to 2D for spatial FFT?
   - Use 1D FFT on token sequence?
   - Apply to attention maps instead?

3. **Adapter architecture specifics?**
   - Parallel vs sequential adapter insertion
   - Shared vs separate adapters per stream
   - Parameter budget (how many trainable params?)

The core novelty would be: **"Frequency-Aware Dual-Stream Adaptation (FADA) for CLIP"** - the first to decompose CLIP features into frequency domains and apply stream-specific adaptation for domain generalization.

Should we start by fetching the FDA and FFDI implementation details to understand the technical specifics better?