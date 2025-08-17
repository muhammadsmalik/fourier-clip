# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

try:
    import torchvision.transforms.functional as TF
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
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
    
    Uses proven library implementations for robust, bug-free frequency separation:
    - Low-pass: torchvision's gaussian_blur (if available) or fallback implementation
    - High-pass: Original - Low-pass (mathematically guaranteed perfect reconstruction)
    """
    
    def __init__(self, sigma=1.0):
        super(FrequencyDecomposer, self).__init__()
        self.sigma = sigma  # Gaussian blur sigma (higher = more blur = more low-freq content)
        self.kernel_size = self._get_kernel_size(sigma)
        
        if not TORCHVISION_AVAILABLE:
            print("Warning: torchvision not available, using fallback Gaussian implementation")
        
    def forward(self, x):
        """
        Args:
            x: Spatial features [B, C, H, W]
        Returns:
            low_freq: Low frequency features [B, C, H, W] (smoothed)
            high_freq: High frequency features [B, C, H, W] (details)
        """
        # Apply Gaussian blur for low-pass filtering
        if TORCHVISION_AVAILABLE:
            low_freq = self._torchvision_gaussian_blur(x)
        else:
            low_freq = self._fallback_gaussian_blur(x)
        
        # High-pass = Original - Low-pass (guarantees perfect reconstruction)
        high_freq = x - low_freq
        
        return low_freq, high_freq
    
    def _get_kernel_size(self, sigma):
        """Calculate appropriate kernel size for given sigma."""
        # Standard rule: kernel should cover 3 standard deviations on each side
        # This ensures 99.7% of the Gaussian is captured
        kernel_size = 2 * int(3 * sigma) + 1
        return max(3, kernel_size)  # Minimum 3x3 kernel
    
    def _torchvision_gaussian_blur(self, x):
        """Use torchvision's proven Gaussian blur implementation."""
        return TF.gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)
    
    def _fallback_gaussian_blur(self, x):
        """Fallback Gaussian implementation if torchvision unavailable."""
        B, C, H, W = x.shape
        
        # Create 2D Gaussian kernel
        kernel = self._create_gaussian_kernel()
        kernel = kernel.to(x.device, x.dtype)
        
        # Expand kernel for all channels (depthwise convolution)
        kernel = kernel.expand(C, 1, self.kernel_size, self.kernel_size)
        
        # Apply convolution with padding to maintain spatial dimensions
        padding = self.kernel_size // 2
        blurred = F.conv2d(x, kernel, padding=padding, groups=C)
        
        return blurred
    
    def _create_gaussian_kernel(self):
        """Create normalized 2D Gaussian kernel."""
        # Create coordinate grids
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - (self.kernel_size - 1) / 2.0  # Center at 0
        
        # Create 2D coordinate grids
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')
        
        # Calculate Gaussian values
        gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * self.sigma**2))
        
        # Normalize so sum equals 1
        gaussian = gaussian / gaussian.sum()
        
        # Add batch and channel dimensions
        return gaussian.unsqueeze(0).unsqueeze(0)


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
