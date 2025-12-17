"""
Foreground Feature Averaging (FFA) Feature Extractor for Template Matching.

This module implements feature extraction using DINOv3 with Foreground Feature Averaging,
which focuses on foreground regions of objects for better instance-level embeddings.

Based on NIDS-Net's FFA implementation.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T
from typing import Optional, Tuple, Union

# Add project root to path to import models
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dinov3 import Dinov3ViT


class FFAFeatureExtractor:
    """
    Feature extractor using DINOv3 with Foreground Feature Averaging (FFA).

    FFA averages patch features weighted by foreground masks, focusing on
    object-specific features and reducing background noise.

    IMPORTANT: When providing masks to extract_embedding():
    - Masks should be in FULL IMAGE SIZE (H_full, W_full)
    - The extractor will automatically crop masks to match the bbox region
    - This ensures proper spatial alignment with DINOv3 features
    """

    def __init__(
        self,
        backbone: str = 'vitb16',
        device: str = 'cuda',
        image_size: int = 448,
        normalize_features: bool = True
    ):
        """
        Initialize the FFA feature extractor.

        Args:
            backbone: DINOv3 backbone type ('vits16', 'vitb16', 'vitl16')
            device: Device to run the model on
            image_size: Target image size (must be multiple of 16)
            normalize_features: Whether to L2-normalize extracted features
        """
        self.device = device
        self.image_size = image_size
        self.normalize_features = normalize_features
        self.patch_size = 16  # DINOv3 uses patch_size=16
        self.feature_map_size = image_size // self.patch_size  # e.g., 448/16 = 28

        # Validate image size
        if image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be multiple of patch_size ({self.patch_size})"
            )

        # Load DINOv3 model
        print(f"Loading DINOv3 {backbone} model...")
        self.extractor = Dinov3ViT(backbone_type=backbone, device=device)
        self.extractor.eval()

        # Get feature dimension
        self.feat_dim = self.extractor.n_feats
        print(f"Feature dimension: {self.feat_dim}")

        # Image preprocessing transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> torch.Tensor:
        """
        Preprocess image for DINOv3.

        Args:
            image: PIL Image or numpy array
            bbox: Optional bounding box [x0, y0, x1, y1] to crop

        Returns:
            Preprocessed image tensor [1, 3, image_size, image_size]
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Crop if bbox provided
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            image = image.crop((x0, y0, x1, y1))

        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]

        return image_tensor.to(self.device)

    def align_mask_to_patch_grid(
        self,
        mask: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Align mask to DINOv3 patch grid by downsampling.

        Args:
            mask: Binary mask [H, W] or [1, 1, H, W]

        Returns:
            Aligned mask [1, 1, feature_map_size, feature_map_size]
        """
        # Convert to tensor if numpy
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        # Ensure 4D tensor [1, 1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)

        # Downsample to patch grid size
        mask_aligned = F.interpolate(
            mask.to(self.device),
            size=(self.feature_map_size, self.feature_map_size),
            mode='bilinear',
            align_corners=False
        )

        # Binarize (threshold at 0.5)
        mask_aligned = (mask_aligned > 0.5).float()

        return mask_aligned

    def extract_features(
        self,
        image_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract DINOv3 patch features.

        Args:
            image_tensor: Preprocessed image [1, 3, H, W]

        Returns:
            Features [1, feature_map_size, feature_map_size, feat_dim]
        """
        with torch.no_grad():
            features = self.extractor(image_tensor)  # [1, H', W', C]

        return features

    def foreground_feature_averaging(
        self,
        features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform Foreground Feature Averaging (FFA).

        Args:
            features: Patch features [1, H, W, C]
            mask: Aligned mask [1, 1, H, W]

        Returns:
            Averaged embedding [1, C]
        """
        # Permute mask to [1, H, W, 1] for broadcasting
        mask_permuted = mask.permute(0, 2, 3, 1)  # [1, H, W, 1]

        # Weighted sum
        weighted_features = (features * mask_permuted).sum(dim=(1, 2))  # [1, C]

        # Normalize by mask area
        mask_area = mask.sum(dim=(1, 2, 3)).unsqueeze(-1)  # [1, 1]

        # Avoid division by zero
        mask_area = torch.clamp(mask_area, min=1.0)

        embedding = weighted_features / mask_area  # [1, C]

        return embedding

    def global_average_pooling(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform global average pooling (fallback if no mask).

        Args:
            features: Patch features [1, H, W, C]

        Returns:
            Averaged embedding [1, C]
        """
        embedding = features.mean(dim=(1, 2))  # [1, C]
        return embedding

    def extract_embedding(
        self,
        image: Union[Image.Image, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        use_ffa: bool = True
    ) -> np.ndarray:
        """
        Extract embedding for a single object instance.

        Args:
            image: Input image (PIL or numpy array)
            bbox: Bounding box [x0, y0, x1, y1] to crop the object
            mask: Optional foreground mask for FFA (full image size)
            use_ffa: Whether to use FFA (requires mask) or global pooling

        Returns:
            Feature embedding [feat_dim] as numpy array
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image, bbox)

        # Extract features
        features = self.extract_features(image_tensor)

        # Apply FFA or global pooling
        if use_ffa and mask is not None:
            # Handle 3D mask (C, H, W) where C=1
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]

            # IMPORTANT: Crop mask to bbox region to align with cropped image
            if bbox is not None:
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Ensure valid coordinates
                if isinstance(mask, np.ndarray):
                    h, w = mask.shape[:2]
                else:
                    h, w = mask.shape[-2:]

                x0 = max(0, min(x0, w))
                y0 = max(0, min(y0, h))
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))

                if x1 > x0 and y1 > y0:
                    mask_cropped = mask[y0:y1, x0:x1]
                else:
                    mask_cropped = mask
            else:
                mask_cropped = mask

            # Align cropped mask to patch grid
            mask_aligned = self.align_mask_to_patch_grid(mask_cropped)
            # FFA
            embedding = self.foreground_feature_averaging(features, mask_aligned)
        else:
            # Global average pooling
            embedding = self.global_average_pooling(features)

        # L2 normalization
        if self.normalize_features:
            embedding = F.normalize(embedding, p=2, dim=-1)

        # Convert to numpy and squeeze
        embedding_np = embedding.cpu().numpy().squeeze()  # [feat_dim]

        return embedding_np

    def extract_batch_embeddings(
        self,
        images: list,
        bboxes: list,
        masks: Optional[list] = None,
        use_ffa: bool = True
    ) -> np.ndarray:
        """
        Extract embeddings for multiple objects in batch.

        Args:
            images: List of PIL Images or numpy arrays
            bboxes: List of bounding boxes
            masks: Optional list of foreground masks
            use_ffa: Whether to use FFA

        Returns:
            Embeddings [N, feat_dim] as numpy array
        """
        embeddings = []

        for i, (image, bbox) in enumerate(zip(images, bboxes)):
            mask = masks[i] if masks is not None else None
            embedding = self.extract_embedding(image, bbox, mask, use_ffa)
            embeddings.append(embedding)

        return np.stack(embeddings, axis=0)  # [N, feat_dim]
