"""
Template Manager for handling template feature generation, storage, and sampling.

This module manages the template embeddings used for instance-level matching.
"""

import os
import pickle
import json
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from PIL import Image
import tqdm


class TemplateManager:
    """
    Manages template features for instance-level object matching.

    Handles generation, storage, loading, and sampling of template embeddings.
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template manager.

        Args:
            template_dir: Directory to save/load template features
        """
        self.template_dir = template_dir
        self.templates = {}  # {instance_id: [N_templates, feat_dim]}
        self.metadata = {}   # {instance_id: {...}}

        if template_dir and os.path.exists(template_dir):
            print(f"Template directory: {template_dir}")

    def generate_templates(
        self,
        dataset: list,
        detector,
        segmenter,
        extractor,
        prompt_text: str = "item held by hand",
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate template embeddings from support (handheld) images.

        Args:
            dataset: List of samples, each with 'image_path', 'instance_id', 'bbox', etc.
            detector: RexOmniDetector instance
            segmenter: SAM2Segmenter instance
            extractor: MGFAFeatureExtractor instance
            prompt_text: Text prompt for detection
            save_path: Optional path to save templates
            verbose: Whether to print progress

        Returns:
            Dictionary mapping instance_id to embeddings array [N_templates, feat_dim]
        """
        from .bbox_utils import bbox_iou

        templates_by_instance = defaultdict(list)
        metadata_by_instance = defaultdict(list)

        iterator = tqdm.tqdm(dataset, desc="Generating templates") if verbose else dataset

        for sample in iterator:
            instance_id = sample.get('instance_id') or sample.get('labels')
            image_path = sample['image_path']
            gt_bbox = sample.get('bounding_box')

            # Load image
            image = Image.open(image_path).convert('RGB')

            # Detect object using RexOmni
            detection_result = detector.detect(image, prompt_text)

            if not detection_result['success'] or detection_result['num_detections'] == 0:
                if verbose:
                    print(f"Warning: No detection for {image_path}")
                continue

            # Get best prediction (highest IoU with GT if available)
            if gt_bbox is not None:
                pred_bbox, best_iou = detector.get_best_prediction(
                    detection_result['predictions'], gt_bbox, bbox_iou
                )
            else:
                pred_bbox = detection_result['predictions'][0]
                best_iou = None

            # Segment object using SAM2
            seg_result = segmenter.segment_single(image, pred_bbox)

            if not seg_result['success']:
                if verbose:
                    print(f"Warning: Segmentation failed for {image_path}")
                continue

            mask = seg_result['mask']

            # Extract embedding using MGFA
            embedding = extractor.extract_embedding(
                image, bbox=pred_bbox, mask=mask, use_mgfa=True
            )

            # Store template
            templates_by_instance[str(instance_id)].append(embedding)

            # Store metadata
            metadata_by_instance[str(instance_id)].append({
                'image_path': image_path,
                'bbox': pred_bbox,
                'gt_bbox': gt_bbox,
                'iou': best_iou
            })

        # Convert lists to arrays
        self.templates = {
            inst_id: np.stack(embeddings, axis=0)
            for inst_id, embeddings in templates_by_instance.items()
        }

        self.metadata = dict(metadata_by_instance)

        if verbose:
            print(f"\nGenerated templates for {len(self.templates)} instances")
            for inst_id, templates in self.templates.items():
                print(f"  Instance {inst_id}: {len(templates)} templates")

        # Save if path provided
        if save_path:
            self.save(save_path)

        return self.templates

    def save(self, save_path: str):
        """
        Save templates and metadata to disk.

        Args:
            save_path: Base path for saving (without extension)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save templates (pickle for numpy arrays)
        templates_path = f"{save_path}_templates.pkl"
        with open(templates_path, 'wb') as f:
            pickle.dump(self.templates, f)

        # Save metadata (JSON for readability)
        metadata_path = f"{save_path}_metadata.json"
        metadata_json = {}
        for inst_id, meta_list in self.metadata.items():
            metadata_json[inst_id] = []
            for meta in meta_list:
                meta_copy = meta.copy()
                if 'bbox' in meta_copy and meta_copy['bbox'] is not None:
                    meta_copy['bbox'] = [float(x) for x in meta_copy['bbox']]
                if 'gt_bbox' in meta_copy and meta_copy['gt_bbox'] is not None:
                    meta_copy['gt_bbox'] = [float(x) for x in meta_copy['gt_bbox']]
                if 'iou' in meta_copy and meta_copy['iou'] is not None:
                    meta_copy['iou'] = float(meta_copy['iou'])
                metadata_json[inst_id].append(meta_copy)

        with open(metadata_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)

        print(f"Templates saved to {templates_path}")
        print(f"Metadata saved to {metadata_path}")

    def load(self, load_path: str):
        """
        Load templates and metadata from disk.

        Args:
            load_path: Base path for loading (without extension)
        """
        # Load templates
        templates_path = f"{load_path}_templates.pkl"
        with open(templates_path, 'rb') as f:
            self.templates = pickle.load(f)

        # Load metadata
        metadata_path = f"{load_path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

        print(f"Loaded templates for {len(self.templates)} instances")

    def sample_templates(
        self,
        n_shots: int,
        seed: Optional[int] = None,
        strategy: str = 'random'
    ) -> Dict[str, np.ndarray]:
        """
        Sample n_shots templates per instance for few-shot evaluation.

        Args:
            n_shots: Number of templates to sample per instance
            seed: Random seed for reproducibility
            strategy: Sampling strategy ('random', 'first', 'diverse')

        Returns:
            Sampled templates {instance_id: [n_shots, feat_dim]}
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        sampled_templates = {}

        for inst_id, templates in self.templates.items():
            n_available = len(templates)

            if n_available == 0:
                print(f"Warning: No templates for instance {inst_id}")
                continue

            if n_shots >= n_available:
                sampled_templates[inst_id] = templates
            else:
                if strategy == 'random':
                    indices = np.random.choice(n_available, n_shots, replace=False)
                    sampled_templates[inst_id] = templates[indices]
                elif strategy == 'first':
                    sampled_templates[inst_id] = templates[:n_shots]
                else:
                    indices = np.random.choice(n_available, n_shots, replace=False)
                    sampled_templates[inst_id] = templates[indices]

        return sampled_templates

    def get_all_templates(self) -> Dict[str, np.ndarray]:
        """Get all templates."""
        return self.templates

    def get_template_statistics(self) -> Dict:
        """Get statistics about templates."""
        stats = {
            'num_instances': len(self.templates),
            'templates_per_instance': {},
            'total_templates': 0,
            'feat_dim': None
        }

        for inst_id, templates in self.templates.items():
            n_templates = len(templates)
            stats['templates_per_instance'][inst_id] = n_templates
            stats['total_templates'] += n_templates

            if stats['feat_dim'] is None and n_templates > 0:
                stats['feat_dim'] = templates.shape[1]

        stats['avg_templates_per_instance'] = (
            stats['total_templates'] / stats['num_instances']
            if stats['num_instances'] > 0 else 0
        )

        return stats

    def add_template(self, instance_id: str, embedding: np.ndarray, metadata: dict = None):
        """
        Add a single template to the manager.

        Args:
            instance_id: Instance identifier
            embedding: Feature embedding [feat_dim]
            metadata: Optional metadata dict
        """
        if instance_id not in self.templates:
            self.templates[instance_id] = embedding.reshape(1, -1)
            self.metadata[instance_id] = [metadata] if metadata else []
        else:
            self.templates[instance_id] = np.vstack([
                self.templates[instance_id],
                embedding.reshape(1, -1)
            ])
            if metadata:
                self.metadata[instance_id].append(metadata)

    def clear(self):
        """Clear all templates."""
        self.templates.clear()
        self.metadata.clear()
