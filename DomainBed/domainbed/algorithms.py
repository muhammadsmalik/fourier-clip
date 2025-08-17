# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, ErmPlusPlusMovingAvg, l2_between_dicts, proj, Nonparametric,
            LARS,  SupConLossLambda
    )


ALGORITHMS = [
    'CLIPZeroShot',
    'FADA_CLIP',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class CLIPZeroShot(Algorithm):
    """CLIP Zero-Shot baseline for domain generalization"""
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIPZeroShot, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        
        import clip
        
        # Load CLIP model using hparams like reference paper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = clip.load(self.hparams['clip_backbone'])[0].float()
        
        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print('Set self.clip_model.parameters.requires_grad = False!')
        
        # embedding dim for image and text encoder
        self.EMBEDDING_DIM = 512
        
        # Get class names from hparams and create prompts like reference paper
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
    
    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}
    
    def predict(self, x):
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)


class FrequencyDecomposer(nn.Module):
    """
    Decomposes spatial features into low and high frequency components using Gaussian filtering.
    
    This approach uses proven computer vision techniques:
    - Low-pass: Gaussian blur (smooth, well-tested)
    - High-pass: Original - Low-pass (perfect reconstruction guaranteed)
    - No custom FFT masking (avoids reconstruction errors)
    """
    
    def __init__(self, sigma=1.0):
        super(FrequencyDecomposer, self).__init__()
        self.sigma = sigma  # Gaussian blur sigma (higher = more low-freq content)
        
    def forward(self, x):
        """
        Args:
            x: Spatial features [B, C, H, W]
        Returns:
            low_freq: Low frequency features [B, C, H, W] (smoothed)
            high_freq: High frequency features [B, C, H, W] (details)
        """
        # Use Gaussian blur for low-pass filtering
        # This is differentiable and mathematically sound
        low_freq = self._gaussian_blur_2d(x, self.sigma)
        
        # High-pass = Original - Low-pass (guarantees perfect reconstruction)
        high_freq = x - low_freq
        
        return low_freq, high_freq
    
    def _gaussian_blur_2d(self, x, sigma):
        """Apply Gaussian blur using PyTorch's built-in functions."""
        # Convert sigma to kernel size (rule of thumb: kernel_size = 6*sigma + 1)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Use PyTorch's GaussianBlur (available in torchvision transforms)
        # For now, implement using conv2d with Gaussian kernel
        return self._conv_gaussian_blur(x, kernel_size, sigma)
    
    def _conv_gaussian_blur(self, x, kernel_size, sigma):
        """Implement Gaussian blur using conv2d with Gaussian kernel."""
        B, C, H, W = x.shape
        
        # Create 2D Gaussian kernel
        kernel = self._get_gaussian_kernel_2d(kernel_size, sigma)
        kernel = kernel.to(x.device, x.dtype)
        
        # Expand kernel for all channels
        kernel = kernel.expand(C, 1, kernel_size, kernel_size)
        
        # Apply convolution with padding to maintain size
        padding = kernel_size // 2
        blurred = F.conv2d(x, kernel, padding=padding, groups=C)
        
        return blurred
    
    def _get_gaussian_kernel_2d(self, kernel_size, sigma):
        """Create 2D Gaussian kernel."""
        # Create 1D Gaussian
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - (kernel_size - 1) / 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # Create 2D Gaussian by outer product
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        
        return gaussian_2d.unsqueeze(0).unsqueeze(0)


class FADA_CLIP(CLIPZeroShot):
    """FADA-CLIP: Frequency-Aware Dual-Stream Adaptation for CLIP"""
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FADA_CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        # Initialize frequency decomposer with Gaussian blur
        self.freq_decomposer = FrequencyDecomposer(
            sigma=hparams.get('gaussian_sigma', 1.0)
        )
        
        # Setup spatial feature extraction hook on Conv1
        self.spatial_features = None
        self.hook_handle = self.clip_model.visual.conv1.register_forward_hook(
            self._spatial_hook_fn
        )
        
        print(f"FADA_CLIP: Registered Conv1 hook for spatial features [768, 14, 14]")
        print(f"FADA_CLIP: Gaussian frequency decomposer sigma={hparams.get('gaussian_sigma', 1.0)}")
    
    def _spatial_hook_fn(self, module, input, output):
        """Hook function to capture Conv1 spatial features."""
        self.spatial_features = output  # [B, 768, 14, 14]
    
    def update(self, minibatches, unlabeled=None):
        # Phase 2: Use CLIPZeroShot behavior (no training needed yet)
        return super().update(minibatches, unlabeled)
    
    def predict(self, x):
        # Reset spatial features for this forward pass
        self.spatial_features = None
        
        # Normal CLIP forward pass (triggers our Conv1 hook)
        logits_per_image, _ = self.clip_model(x, self.prompt)
        
        # Now spatial_features contains [B, 768, 14, 14] from Conv1
        if self.spatial_features is not None:
            # Apply frequency decomposition
            low_freq, high_freq = self.freq_decomposer(self.spatial_features)
            
            # Phase 2: Verify shapes and reconstruction quality
            print(f"Spatial features: {self.spatial_features.shape}")
            print(f"Low freq: {low_freq.shape}, High freq: {high_freq.shape}")
            
            # Verify perfect reconstruction
            reconstructed = low_freq + high_freq
            error = (self.spatial_features - reconstructed).abs().mean()
            print(f"Reconstruction error: {error:.6f}")
            
            # For Phase 2: Store decomposed features but don't use them yet
            self._low_freq_features = low_freq
            self._high_freq_features = high_freq
        else:
            print("Warning: Spatial features not captured by hook")
        
        # Return original CLIP prediction for Phase 2
        return logits_per_image.softmax(dim=-1)
    
    def __del__(self):
        """Clean up hook when object is destroyed."""
        if hasattr(self, 'hook_handle') and self.hook_handle is not None:
            self.hook_handle.remove()
