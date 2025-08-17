#!/usr/bin/env python3
"""
Standalone test script for FrequencyDecomposer module.
Tests reconstruction quality, frequency separation, and threshold sensitivity.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add DomainBed to path
sys.path.append('DomainBed')

# Import our FrequencyDecomposer
from domainbed.algorithms import FrequencyDecomposer

def test_perfect_reconstruction():
    """Test that FFT→IFFT preserves original features."""
    print("🧪 Test 1: Perfect Reconstruction")
    
    decomposer = FrequencyDecomposer(threshold=0.1)
    
    # Test with random features matching CLIP Conv1 output shape
    x = torch.randn(2, 768, 14, 14)
    
    # Decompose and reconstruct
    low_freq, high_freq = decomposer(x)
    reconstructed = low_freq + high_freq
    
    # Calculate reconstruction error
    error = (x - reconstructed).abs().mean()
    
    print(f"  Input shape: {x.shape}")
    print(f"  Low freq shape: {low_freq.shape}")
    print(f"  High freq shape: {high_freq.shape}")
    print(f"  Reconstruction error: {error:.8f}")
    
    # Test passes if error is very small
    assert error < 1e-5, f"Reconstruction error too large: {error}"
    print("  ✅ Perfect reconstruction verified!\n")
    
    return error

def test_frequency_separation():
    """Test that low/high frequencies are properly separated."""
    print("🧪 Test 2: Frequency Separation Quality")
    
    decomposer = FrequencyDecomposer(threshold=0.1)
    
    # Create test patterns
    def create_checkerboard_pattern(B, C, H, W):
        """High frequency checkerboard pattern."""
        pattern = torch.zeros(B, C, H, W)
        for i in range(H):
            for j in range(W):
                if (i + j) % 2 == 0:
                    pattern[:, :, i, j] = 1.0
        return pattern
    
    def create_gradient_pattern(B, C, H, W):
        """Low frequency gradient pattern."""
        pattern = torch.zeros(B, C, H, W)
        for i in range(H):
            for j in range(W):
                pattern[:, :, i, j] = i / H + j / W  # Smooth gradient
        return pattern
    
    # Test on high-frequency pattern (checkerboard)
    checkerboard = create_checkerboard_pattern(1, 768, 14, 14)
    low_check, high_check = decomposer(checkerboard)
    
    high_energy = high_check.pow(2).sum().item()
    low_energy = low_check.pow(2).sum().item()
    high_ratio = high_energy / (high_energy + low_energy)
    
    print(f"  Checkerboard pattern:")
    print(f"    High freq energy ratio: {high_ratio:.3f}")
    assert high_ratio > 0.6, f"Checkerboard should have more high-freq energy, got {high_ratio}"
    
    # Test on low-frequency pattern (gradient)
    gradient = create_gradient_pattern(1, 768, 14, 14)
    low_grad, high_grad = decomposer(gradient)
    
    low_energy = low_grad.pow(2).sum().item()
    high_energy = high_grad.pow(2).sum().item()
    low_ratio = low_energy / (low_energy + high_energy)
    
    print(f"  Gradient pattern:")
    print(f"    Low freq energy ratio: {low_ratio:.3f}")
    assert low_ratio > 0.6, f"Gradient should have more low-freq energy, got {low_ratio}"
    
    print("  ✅ Frequency separation works correctly!\n")

def test_threshold_sensitivity():
    """Test that threshold parameter controls frequency split."""
    print("🧪 Test 3: Threshold Sensitivity Analysis")
    
    x = torch.randn(1, 768, 14, 14)
    
    results = {}
    for threshold in [0.05, 0.1, 0.15, 0.2]:
        decomposer = FrequencyDecomposer(threshold=threshold)
        low_freq, high_freq = decomposer(x)
        
        low_energy = low_freq.pow(2).sum().item()
        high_energy = high_freq.pow(2).sum().item()
        
        low_ratio = low_energy / (low_energy + high_energy)
        results[threshold] = low_ratio
        print(f"  Threshold {threshold:.2f}: {low_ratio:.3f} low-freq ratio")
    
    # Higher threshold should mean more low-frequency content
    assert results[0.2] > results[0.1] > results[0.05], "Threshold should control frequency split"
    print("  ✅ Threshold sensitivity verified!\n")

def test_batch_processing():
    """Test that batched inputs work correctly."""
    print("🧪 Test 4: Batch Processing")
    
    decomposer = FrequencyDecomposer(threshold=0.1)
    
    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 768, 14, 14)
        low_freq, high_freq = decomposer(x)
        
        # Verify shapes
        assert low_freq.shape == x.shape, f"Low freq shape mismatch for batch {batch_size}"
        assert high_freq.shape == x.shape, f"High freq shape mismatch for batch {batch_size}"
        
        # Verify reconstruction
        reconstructed = low_freq + high_freq
        error = (x - reconstructed).abs().mean()
        assert error < 1e-5, f"Batch {batch_size} reconstruction error: {error}"
        
        print(f"  Batch size {batch_size}: ✅")
    
    print("  ✅ Batch processing works correctly!\n")

def test_device_compatibility():
    """Test CPU/GPU compatibility."""
    print("🧪 Test 5: Device Compatibility")
    
    decomposer = FrequencyDecomposer(threshold=0.1)
    
    # Test CPU
    x_cpu = torch.randn(1, 768, 14, 14)
    low_cpu, high_cpu = decomposer(x_cpu)
    error_cpu = (x_cpu - (low_cpu + high_cpu)).abs().mean()
    print(f"  CPU reconstruction error: {error_cpu:.8f}")
    assert error_cpu < 1e-5
    
    # Test GPU if available
    if torch.cuda.is_available():
        x_gpu = x_cpu.cuda()
        decomposer_gpu = decomposer.cuda()
        low_gpu, high_gpu = decomposer_gpu(x_gpu)
        error_gpu = (x_gpu - (low_gpu + high_gpu)).abs().mean()
        print(f"  GPU reconstruction error: {error_gpu:.8f}")
        assert error_gpu < 1e-5
        print("  ✅ GPU compatibility verified!")
    else:
        print("  ⚠️  GPU not available, skipping GPU test")
    
    print("  ✅ Device compatibility verified!\n")

def main():
    """Run all FrequencyDecomposer tests."""
    print("🚀 FrequencyDecomposer Standalone Tests")
    print("=" * 50)
    
    try:
        # Run all tests
        error = test_perfect_reconstruction()
        test_frequency_separation()
        test_threshold_sensitivity()
        test_batch_processing()
        test_device_compatibility()
        
        print("🎉 ALL TESTS PASSED!")
        print(f"FrequencyDecomposer is ready for Phase 2 integration.")
        print(f"Best reconstruction error achieved: {error:.8f}")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()