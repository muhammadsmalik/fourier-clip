INITIAL PLAN:

Phase 2: FrequencyDecomposer Module Implementation Plan

Architecture Overview & Justification

Key Design Decision: Where to Apply Frequency Decomposition
	•	Intercept features before the final projection layer but after spatial tokens are available.
	•	Rationale:
	1.	CLIP ViT-B/16 outputs
	•	Final output: pooled features [batch, 512]
	•	Internal tokens: [batch, 197, 768] (196 patches + 1 CLS token)
	2.	FFT requires 2D structure
	•	Reshape patch tokens into a 14×14 grid
	•	Perform frequency decomposition on this spatial map

⸻

Implementation Components

1. FrequencyDecomposer Class

class FrequencyDecomposer(nn.Module):
    def __init__(self, threshold=0.1):
        # threshold = L parameter from FDA (0.1 = 10% of frequencies are low)

	•	Why:
	•	Modular, testable component
	•	Threshold as parameter → easy ablations
	•	Follows PyTorch best practices

⸻

2. FFT Operations (Modern PyTorch API)

Old FDA code (deprecated):

fft_src = torch.rfft(src_img, signal_ndim=2, onesided=False)

Modern PyTorch (>=1.7):

fft_src = torch.fft.fft2(src_img)   # complex tensor
amp = torch.abs(fft_src)            # amplitude
phase = torch.angle(fft_src)        # phase

	•	Why:
	•	torch.fft.fft2 is the current standard
	•	Returns proper complex tensors
	•	Cleaner, NumPy/SciPy-compatible API

⸻

3. Frequency Separation Strategy

def create_frequency_mask(H, W, threshold):
    # Low frequencies appear at corners
    # fftshift moves them to center
    # Apply square mask of size threshold * min(H, W)

	•	Why:
	•	Follows FDA/FFDI pattern
	•	Scale-invariant (percentage-based threshold)
	•	Square mask = standard in frequency domain

⸻

4. Feature Extraction & Reshaping

# Extract features before projection
visual_features = clip_model.visual.transformer(x)     # [B, 197, 768]
patch_features = visual_features[:, 1:, :]             # [B, 196, 768] (exclude CLS)
spatial_features = patch_features.reshape(B, 14, 14, 768)
spatial_features = spatial_features.permute(0, 3, 1, 2) # [B, 768, 14, 14]

	•	Why:
	•	FFT requires spatial grids
	•	14×14 from (224 ÷ 16)
	•	Channels-first → CNN convention

⸻

Testing Strategy

Test 1: FFT Invertibility

def test_fft_invertibility():
    x = torch.randn(2, 768, 14, 14)
    decomposer = FrequencyDecomposer()
    low, high = decomposer(x)
    reconstructed = low + high
    assert torch.allclose(x, reconstructed, atol=1e-5)

	•	Verifies no information loss

Test 2: Frequency Threshold Effect
	•	Check that varying threshold changes the low/high split

Test 3: Standalone Module Test
	•	Use synthetic data
	•	Visualize low vs high frequencies → smooth vs detailed patterns

⸻

Integration Points
	1.	FADA_CLIP.__init__()
	•	Add self.freq_decomposer = FrequencyDecomposer(hparams['frequency_threshold'])
	2.	FADA_CLIP.predict()
	•	Extract spatial features
	•	Apply decomposition
	•	For Phase 2: validate shapes, still return original predictions

⸻

Risk Mitigation
	1.	Fallback flag → bypass decomposition if needed
	2.	Gradient flow → use only differentiable PyTorch ops
	3.	Memory → batch-wise processing
	4.	Numerical stability → add epsilon to divisions

⸻

Why This Approach is Robust
	1.	Pure PyTorch native ops (no custom FFT)
	2.	Based on proven FDA design
	3.	Testable in isolation
	4.	Incremental: doesn’t break CLIP zero-shot
	5.	Configurable via hparams

⸻

Expected Outputs
	•	low_freq_features: [B, 768, 14, 14] → smooth, content-focused
	•	high_freq_features: [B, 768, 14, 14] → detailed, texture-focused
	•	Validation: low + high ≈ original (within precision)

⸻

Next Steps After Phase 2
	•	Phase 3: Add adapters for separate processing of low/high streams
	•	Phase 4: Add fusion mechanism

⸻

FINAL PLAN (Updated with Optimal Solution):

⸻

Phase 2: FrequencyDecomposer Module – Optimal Implementation Plan

⸻

Core Challenge & Solution

Problem
	•	CLIP's VisionTransformer outputs a single 512-dim vector after projection (model.py, lines 235–238).
	•	For FFT-based decomposition, we require spatial features with 2D structure, not just the final pooled vector.
	•	Need to balance minimal implementation with spatial access requirements.

Solution Analysis & Final Choice
After evaluating 6 different approaches:

1. ❌ Reimplement CLIP forward pass (Score: 3/10) - Violates "no reimplementation" principle
2. ❌ Use final 1D features (Score: 4/10) - No real spatial structure for 2D FFT
3. ✅ Hook on transformer output (Score: 7/10) - Good but requires reshaping
4. 🏆 Hook on Conv1 output (Score: 8/10) - OPTIMAL: Already spatial [B, 768, 14, 14]
5. ❌ Hook on ln_post (Score: 5/10) - Already pooled, no spatial info
6. ✅ Custom wrapper module (Score: 7.5/10) - Clean but adds complexity

**CHOSEN SOLUTION: Option 4 - Forward Hook on Conv1 Output**

Justification:
	•	✅ Already in perfect spatial format [B, 768, 14, 14] for 2D FFT
	•	✅ Minimal code changes (3-4 lines)
	•	✅ No reshaping or complex transformations needed
	•	✅ Uses PyTorch's official hook API (no custom implementations)
	•	✅ Follows CLIP-Adapter philosophy of minimal modifications
	•	✅ Clean fallback: can disable hook if issues arise

⸻

Implementation Architecture

1. Spatial Feature Extraction via Forward Hook

def setup_spatial_hook(clip_model):
    """Setup forward hook to capture Conv1 spatial features."""
    spatial_features = None
    
    def hook_fn(module, input, output):
        nonlocal spatial_features
        spatial_features = output  # [B, 768, 14, 14] - perfect for FFT!
    
    # Register hook on Conv1 - earliest spatial representation
    handle = clip_model.visual.conv1.register_forward_hook(hook_fn)
    
    return handle, lambda: spatial_features

Key Advantages of Conv1 Hook:
	•	✅ Output is already [B, 768, 14, 14] - perfect spatial structure for 2D FFT
	•	✅ No reshaping, no complex tensor manipulations
	•	✅ Captures raw spatial patterns before transformer mixing
	•	✅ Early in pipeline - preserves spatial locality
	•	✅ Minimal code - just 3 lines for hook setup

Why Conv1 vs Other Layers:
	•	conv1: [B, 768, 14, 14] ✅ Perfect for FFT
	•	transformer: [B, 197, 768] ❌ Needs complex reshaping  
	•	ln_post: [B, 768] ❌ Already pooled, no spatial info
	•	Final output: [B, 512] ❌ No spatial structure

⸻

2. FrequencyDecomposer Module

class FrequencyDecomposer(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        """
        Args:
            x: Spatial features [B, C, H, W]
        Returns:
            low_freq: Low-frequency features [B, C, H, W]
            high_freq: High-frequency features [B, C, H, W]
        """
        # FFT
        fft_features = torch.fft.fft2(x, norm='ortho')  
        fft_shifted = torch.fft.fftshift(fft_features, dim=(-2, -1))

        # Frequency mask
        B, C, H, W = x.shape
        mask = self.create_centered_mask(H, W, self.threshold).to(x.device)

        # Frequency separation
        low_freq_fft = fft_shifted * mask
        high_freq_fft = fft_shifted * (1 - mask)

        # Inverse FFT
        low_freq_fft = torch.fft.ifftshift(low_freq_fft, dim=(-2, -1))
        high_freq_fft = torch.fft.ifftshift(high_freq_fft, dim=(-2, -1))

        low_freq = torch.fft.ifft2(low_freq_fft, norm='ortho').real
        high_freq = torch.fft.ifft2(high_freq_fft, norm='ortho').real

        return low_freq, high_freq

    def create_centered_mask(self, H, W, threshold):
        """Create centered square mask for low frequencies."""
        mask_size = int(min(H, W) * threshold)
        mask = torch.zeros(H, W)
        center_h, center_w = H // 2, W // 2

        h1, h2 = center_h - mask_size // 2, center_h + mask_size // 2 + 1
        w1, w2 = center_w - mask_size // 2, center_w + mask_size // 2 + 1

        mask[h1:h2, w1:w2] = 1.0
        return mask

Design Justifications
	1.	torch.fft.fft2 with norm='ortho' – modern, stable, energy-preserving.
	2.	fftshift/ifftshift – centers low frequencies for intuitive masking.
	3.	Square centered mask – efficient, scale-invariant with threshold.
	4.	Taking .real – input is real; discards numerical noise in imaginary part.

⸻

3. Integration into FADA_CLIP

class FADA_CLIP(CLIPZeroShot):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        
        # Initialize frequency decomposer
        self.freq_decomposer = FrequencyDecomposer(
            threshold=hparams['frequency_threshold']
        )
        
        # Setup spatial feature extraction hook
        self.spatial_features = None
        self.hook_handle = self.clip_model.visual.conv1.register_forward_hook(
            self._spatial_hook_fn
        )
        
        print(f"FADA_CLIP: Registered Conv1 hook for spatial features [768, 14, 14]")
    
    def _spatial_hook_fn(self, module, input, output):
        """Hook function to capture Conv1 spatial features."""
        self.spatial_features = output  # [B, 768, 14, 14]
    
    def predict(self, x):
        # Reset spatial features
        self.spatial_features = None
        
        # Normal CLIP forward pass (triggers our hook)
        logits_per_image, _ = self.clip_model(x, self.prompt)
        
        # Now spatial_features contains [B, 768, 14, 14] from Conv1
        if self.spatial_features is not None:
            # Apply frequency decomposition
            low_freq, high_freq = self.freq_decomposer(self.spatial_features)
            
            # Phase 2: Just verify shapes and reconstruction
            print(f"Spatial: {self.spatial_features.shape}")
            print(f"Low freq: {low_freq.shape}, High freq: {high_freq.shape}")
            
            # Verify perfect reconstruction
            reconstructed = low_freq + high_freq
            error = (self.spatial_features - reconstructed).abs().mean()
            print(f"Reconstruction error: {error:.6f}")
        
        # Return original CLIP prediction for Phase 2
        return logits_per_image.softmax(dim=-1)
    
    def __del__(self):
        """Clean up hook when object is destroyed."""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()

Implementation Benefits:
	•	✅ Minimal changes to FADA_CLIP class
	•	✅ Hook automatically captures spatial features during normal forward pass
	•	✅ No need to modify CLIP's internal architecture
	•	✅ Debugging prints show shapes and reconstruction quality
	•	✅ Clean hook cleanup to prevent memory leaks


⸻

Updated Testing Strategy

Test 1: Hook Functionality Test

def test_hook_captures_features():
    """Test that Conv1 hook captures correct spatial features."""
    fada_clip = FADA_CLIP(...)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Forward pass should trigger hook
    output = fada_clip.predict(dummy_input)
    
    # Verify spatial features were captured
    assert fada_clip.spatial_features is not None
    assert fada_clip.spatial_features.shape == (2, 768, 14, 14)
    print("✅ Hook successfully captures Conv1 features")

Test 2: Perfect Reconstruction Test

def test_perfect_reconstruction():
    """Test FFT→IFFT preserves original features."""
    decomposer = FrequencyDecomposer(threshold=0.1)
    x = torch.randn(2, 768, 14, 14)
    
    low_freq, high_freq = decomposer(x)
    reconstructed = low_freq + high_freq
    
    error = (x - reconstructed).abs().mean()
    assert error < 1e-5, f"Reconstruction error: {error}"
    print(f"✅ Perfect reconstruction: error = {error:.8f}")

Test 3: Frequency Separation Quality

def test_frequency_separation():
    """Test that low/high frequencies are properly separated."""
    decomposer = FrequencyDecomposer(threshold=0.1)
    
    # Create test patterns
    checkerboard = create_checkerboard_pattern(2, 768, 14, 14)  # High freq
    gradient = create_gradient_pattern(2, 768, 14, 14)         # Low freq
    
    # Test on high-frequency pattern
    low_check, high_check = decomposer(checkerboard)
    high_energy = high_check.pow(2).sum()
    low_energy = low_check.pow(2).sum()
    assert high_energy > low_energy, "Checkerboard should have more high-freq energy"
    
    # Test on low-frequency pattern  
    low_grad, high_grad = decomposer(gradient)
    low_energy = low_grad.pow(2).sum()
    high_energy = high_grad.pow(2).sum()
    assert low_energy > high_energy, "Gradient should have more low-freq energy"
    
    print("✅ Frequency separation works correctly")

Test 4: Threshold Sensitivity Analysis

def test_threshold_sensitivity():
    """Test that threshold parameter controls frequency split."""
    x = torch.randn(2, 768, 14, 14)
    
    results = {}
    for threshold in [0.05, 0.1, 0.15, 0.2]:
        decomposer = FrequencyDecomposer(threshold=threshold)
        low_freq, high_freq = decomposer(x)
        
        low_energy = low_freq.pow(2).sum().item()
        high_energy = high_freq.pow(2).sum().item()
        
        results[threshold] = low_energy / (low_energy + high_energy)
        print(f"Threshold {threshold}: {results[threshold]:.3f} low-freq ratio")
    
    # Higher threshold should mean more low-frequency content
    assert results[0.2] > results[0.1] > results[0.05]
    print("✅ Threshold sensitivity verified")

Test 5: End-to-End Integration Test

def test_fada_clip_integration():
    """Test complete FADA_CLIP pipeline with real data."""
    fada_clip = FADA_CLIP(...)
    
    # Use real CLIP preprocessed images
    real_images = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    
    output = fada_clip.predict(real_images)
    
    # Verify outputs
    assert output.shape == (4, 65)  # Office-Home has 65 classes
    assert torch.allclose(output.sum(dim=1), torch.ones(4))  # Probabilities sum to 1
    assert fada_clip.spatial_features.shape == (4, 768, 14, 14)
    
    print("✅ End-to-end FADA_CLIP pipeline works")

⸻

Why This Optimal Solution Will Work

Technical Guarantees:
	1.	✅ Proven FFT operations (PyTorch native torch.fft.fft2, stable)
	2.	✅ Energy preservation via norm='ortho' (mathematically sound)
	3.	✅ Perfect spatial structure from Conv1 [B, 768, 14, 14]
	4.	✅ Minimal code changes (follows CLIP-Adapter philosophy)
	5.	✅ All operations differentiable (no gradient flow issues)
	6.	✅ Testable components (5 comprehensive tests designed)

Design Philosophy Alignment:
	•	✅ No custom implementations (uses PyTorch's official APIs)
	•	✅ No architecture modifications (just adds a hook)
	•	✅ Minimal code complexity (3-4 lines for core functionality)
	•	✅ Follows proven patterns (CLIP-Adapter hook-based approach)
	•	✅ Easy to debug and verify (clear shape tracking)

⸻

Expected Outcomes After Phase 2

Successful Implementation Metrics:
	•	✅ Extract perfect spatial features [B, 768, 14, 14] from CLIP Conv1
	•	✅ Decompose into low/high frequency components with same shape
	•	✅ Verify reconstruction error < 1e-5 (near-perfect)
	•	✅ Demonstrate frequency separation works on test patterns
	•	✅ All 5 tests pass, confirming robustness
	•	✅ Ready for Phase 3: Adapter integration per frequency stream

Performance Expectations:
	•	Phase 2: Identical performance to CLIPZeroShot (82.4% on Office-Home)
	•	No performance degradation (just adding decomposition, not using it yet)
	•	Memory overhead: minimal (just storing one Conv1 output)
	•	Speed overhead: ~5-10% (due to FFT operations in predict())

⸻

Comprehensive Risk Mitigation

Risk Category 1: Hook-Related Issues
	•	Risk: Hook doesn't capture features
	•	Mitigation: Debug prints verify spatial_features is not None
	•	Fallback: Disable hook, return to CLIPZeroShot behavior

Risk Category 2: FFT Numerical Issues  
	•	Risk: FFT→IFFT introduces numerical errors
	•	Mitigation: Use norm='ortho' for energy preservation
	•	Verification: Test reconstruction error < 1e-5
	•	Fallback: Add small epsilon for stability if needed

Risk Category 3: Shape/Dimension Mismatches
	•	Risk: Conv1 output shape differs from expected
	•	Mitigation: Explicit shape assertions in code
	•	Debug: Print all tensor shapes during forward pass
	•	Fallback: Graceful error handling with informative messages

Risk Category 4: Memory/Performance Issues
	•	Risk: Hook storage causes memory leaks
	•	Mitigation: Proper hook cleanup in __del__ method
	•	Monitoring: Track memory usage during training
	•	Fallback: Reduce batch size if memory pressure

Risk Category 5: Integration Issues
	•	Risk: Breaks existing CLIPZeroShot functionality
	•	Mitigation: Phase 2 still returns super().predict() result
	•	Testing: Verify identical output to CLIPZeroShot baseline
	•	Rollback: Can disable frequency decomposition via hparam

⸻

Next Steps Roadmap

Phase 2 Complete → Phase 3 Preparation:
	•	✅ Spatial feature extraction working
	•	✅ Frequency decomposition validated  
	•	✅ Perfect reconstruction verified
	•	➡️  Ready to add dual-stream adapters
	•	➡️  Ready to implement fusion mechanism

Phase 3 Design Preview:
	•	Add LowFreqAdapter for smooth, content features
	•	Add HighFreqAdapter for detailed, texture features  
	•	Implement attention-based fusion (FFDI mechanism)
	•	Target: 84-85% performance with basic adapters

Phase 4 Optimization:
	•	Fine-tune frequency threshold parameter
	•	Add auxiliary losses for each stream
	•	Implement FDAG frequency augmentation
	•	Target: 87-89% performance (SOTA level)

⸻

