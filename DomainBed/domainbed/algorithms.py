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


class ImageFrequencyDecomposer(nn.Module):
    """
    Image-level frequency decomposition using FDA (Fourier Domain Adaptation) approach.
    
    Applies frequency decomposition to input images [B, 3, 224, 224] before CLIP processing.
    Supports both FFT-based (FDA-style) and Gaussian-based approaches.
    """
    
    def __init__(self, method='fft', threshold=0.1, sigma=1.0):
        super(ImageFrequencyDecomposer, self).__init__()
        self.method = method  # 'fft' or 'gaussian'
        self.threshold = threshold  # for FFT: fraction of image size for low-freq mask
        self.sigma = sigma  # for Gaussian: blur strength
        
        if method == 'gaussian':
            self.kernel_size = self._get_kernel_size(sigma)
            if not TORCHVISION_AVAILABLE:
                print("Warning: torchvision not available, using fallback Gaussian implementation")
        
        print(f"ImageFrequencyDecomposer: method={method}, threshold={threshold}, sigma={sigma}")
    
    def forward(self, images):
        """
        Args:
            images: Input images [B, 3, 224, 224] (RGB, normalized for CLIP)
        Returns:
            low_freq_images: Low frequency images [B, 3, 224, 224]
            high_freq_images: High frequency images [B, 3, 224, 224]
        """
        if self.method == 'fft':
            return self._fft_decomposition(images)
        elif self.method == 'gaussian':
            return self._gaussian_decomposition(images)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _fft_decomposition(self, images):
        """FDA-style FFT-based frequency decomposition."""
        B, C, H, W = images.shape
        
        # Apply FFT to each channel
        fft_images = torch.fft.fft2(images, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_images, dim=(-2, -1))
        
        # Create centered low-frequency mask (FDA approach)
        mask = self._create_centered_mask(H, W, self.threshold).to(images.device)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        
        # Apply mask to get low-frequency components
        fft_low = fft_shifted * mask
        
        # Inverse transform to get low-frequency image
        fft_low_shifted = torch.fft.ifftshift(fft_low, dim=(-2, -1))
        low_freq_images = torch.fft.ifft2(fft_low_shifted, dim=(-2, -1)).real
        
        # High-frequency = Original - Low-frequency (perfect reconstruction)
        high_freq_images = images - low_freq_images
        
        return low_freq_images, high_freq_images
    
    def _gaussian_decomposition(self, images):
        """Gaussian blur-based frequency decomposition."""
        # Apply Gaussian blur for low-pass filtering
        if TORCHVISION_AVAILABLE:
            low_freq_images = TF.gaussian_blur(images, kernel_size=self.kernel_size, sigma=self.sigma)
        else:
            low_freq_images = self._fallback_gaussian_blur(images)
        
        # High-pass = Original - Low-pass (guarantees perfect reconstruction)
        high_freq_images = images - low_freq_images
        
        return low_freq_images, high_freq_images
    
    def _create_centered_mask(self, height, width, threshold):
        """Create centered frequency mask following FDA approach."""
        # Calculate mask radius based on threshold
        min_dim = min(height, width)
        radius = int(min_dim * threshold)
        
        # Create coordinate grids
        center_h, center_w = height // 2, width // 2
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate distance from center
        dist_from_center = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        
        # Create circular mask (1 for low frequencies, 0 for high frequencies)
        mask = (dist_from_center <= radius).float()
        
        return mask
    
    def _get_kernel_size(self, sigma):
        """Calculate appropriate kernel size for given sigma."""
        kernel_size = 2 * int(3 * sigma) + 1
        return max(3, kernel_size)
    
    def _fallback_gaussian_blur(self, x):
        """Fallback Gaussian implementation if torchvision unavailable."""
        B, C, H, W = x.shape
        
        # Create 2D Gaussian kernel
        kernel = self._create_gaussian_kernel()
        kernel = kernel.to(x.device, x.dtype)
        
        # Expand kernel for all channels
        kernel = kernel.expand(C, 1, self.kernel_size, self.kernel_size)
        
        # Apply convolution with padding
        padding = self.kernel_size // 2
        blurred = F.conv2d(x, kernel, padding=padding, groups=C)
        
        return blurred
    
    def _create_gaussian_kernel(self):
        """Create normalized 2D Gaussian kernel."""
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - (self.kernel_size - 1) / 2.0
        
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')
        gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * self.sigma**2))
        gaussian = gaussian / gaussian.sum()
        
        return gaussian.unsqueeze(0).unsqueeze(0)


class FusionModule(nn.Module):
    """
    FFDI-inspired fusion mechanism for combining low and high frequency features.
    
    Uses spatial attention mechanism where low-frequency features generate attention
    masks to guide high-frequency feature processing.
    """
    
    def __init__(self, feature_dim=512, fusion_weight=0.5):
        super(FusionModule, self).__init__()
        self.fusion_weight = fusion_weight
        
        # Spatial attention mechanism (simplified FFDI approach)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Feature projection layers
        self.low_proj = nn.Linear(feature_dim, feature_dim)
        self.high_proj = nn.Linear(feature_dim, feature_dim)
        self.fusion_proj = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, low_features, high_features):
        """
        Args:
            low_features: Low frequency features [B, 512]
            high_features: High frequency features [B, 512]
        Returns:
            fused_features: Combined features [B, 512]
        """
        # Simple feature-level fusion (since we're working with 1D features from CLIP)
        # Project features
        low_proj = self.low_proj(low_features)
        high_proj = self.high_proj(high_features)
        
        # Weighted fusion with learnable attention
        alpha = self.fusion_weight
        attention_weight = torch.sigmoid(torch.mean(low_proj, dim=-1, keepdim=True))
        
        # Adaptive fusion based on low-frequency content
        fused = alpha * low_proj + (1 - alpha) * high_proj * attention_weight
        
        # Final projection
        fused_features = self.fusion_proj(fused)
        
        return fused_features


class FADA_CLIP(CLIPZeroShot):
    """FADA-CLIP: Frequency-Aware Dual-Stream Adaptation for CLIP
    
    New Architecture (Post Phase 2 Discovery):
    1. Apply frequency decomposition to INPUT IMAGES (not Conv1 features)
    2. Process low/high freq images through dual CLIP encoders
    3. Fuse dual features using FFDI-inspired mechanism
    4. Final classification with text prompts
    """
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FADA_CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        # Image-level frequency decomposition (NEW APPROACH)
        freq_method = hparams.get('frequency_method', 'fft')  # 'fft' or 'gaussian'
        freq_threshold = hparams.get('frequency_threshold', 0.1)  # for FFT mask
        gaussian_sigma = hparams.get('gaussian_sigma', 1.0)  # for Gaussian blur
        
        self.image_freq_decomposer = ImageFrequencyDecomposer(
            method=freq_method,
            threshold=freq_threshold,
            sigma=gaussian_sigma
        )
        
        # Feature fusion mechanism (FFDI-inspired)
        fusion_weight = hparams.get('fusion_weight', 0.5)
        self.fusion_module = FusionModule(
            feature_dim=self.EMBEDDING_DIM,  # 512 for CLIP
            fusion_weight=fusion_weight
        )
        
        # Move modules to same device as CLIP model
        self.image_freq_decomposer = self.image_freq_decomposer.to(self.device)
        self.fusion_module = self.fusion_module.to(self.device)
        
        # Training mode: use fusion vs. just low-freq vs. just high-freq
        self.training_mode = hparams.get('training_mode', 'fusion')  # 'fusion', 'low_only', 'high_only'
        
        print(f"FADA_CLIP: Image frequency decomposition method={freq_method}")
        print(f"FADA_CLIP: Frequency threshold={freq_threshold}, gaussian_sigma={gaussian_sigma}")
        print(f"FADA_CLIP: Fusion weight={fusion_weight}, training_mode={self.training_mode}")
        print(f"FADA_CLIP: All modules moved to device: {self.device}")
    
    def update(self, minibatches, unlabeled=None):
        # Still zero-shot approach - no training needed
        return super().update(minibatches, unlabeled)
    
    def predict(self, x):
        """
        New prediction pipeline:
        1. Decompose input images into low/high frequency components
        2. Process both through CLIP visual encoder
        3. Fuse features using FFDI-inspired mechanism
        4. Classify against text prompts
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Step 1: Apply frequency decomposition to input images
        low_freq_images, high_freq_images = self.image_freq_decomposer(x)
        
        # Step 2: Dual CLIP processing - encode both frequency components
        with torch.no_grad():  # CLIP is frozen
            low_freq_features = self.clip_model.encode_image(low_freq_images)  # [B, 512]
            high_freq_features = self.clip_model.encode_image(high_freq_images)  # [B, 512]
        
        # Step 3: Feature fusion or selection based on training mode
        if self.training_mode == 'fusion':
            # Use FFDI-inspired fusion
            fused_features = self.fusion_module(low_freq_features, high_freq_features)
            image_features = fused_features
        elif self.training_mode == 'low_only':
            image_features = low_freq_features
        elif self.training_mode == 'high_only':
            image_features = high_freq_features
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")
        
        # Step 4: Compute similarity with text prompts (CLIP-style)
        # Get text features from prompts
        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.prompt)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image.softmax(dim=-1)
    
    def get_frequency_decomposition_info(self, x):
        """Utility function to inspect frequency decomposition (for debugging)."""
        # Ensure input is on correct device
        x = x.to(self.device)
        low_freq_images, high_freq_images = self.image_freq_decomposer(x)
        
        # Calculate energy ratios
        original_energy = x.pow(2).sum().item()
        low_energy = low_freq_images.pow(2).sum().item()
        high_energy = high_freq_images.pow(2).sum().item()
        total_energy = low_energy + high_energy
        
        reconstruction_error = (x - (low_freq_images + high_freq_images)).abs().mean().item()
        
        return {
            'original_energy': original_energy,
            'low_energy_ratio': low_energy / total_energy,
            'high_energy_ratio': high_energy / total_energy,
            'reconstruction_error': reconstruction_error,
            'energy_conservation': total_energy / original_energy
        }
