"""
Template Matcher for instance-level object matching.

This module implements similarity computation and matching logic for
template-based object detection and classification.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class TemplateMatcher:
    """
    Matches proposal embeddings to template embeddings using similarity metrics.

    Supports multiple aggregation strategies for multi-template matching.
    """

    def __init__(
        self,
        similarity_metric: str = 'cosine',
        aggregation: str = 'max',
        device: str = 'cuda'
    ):
        """
        Initialize template matcher.

        Args:
            similarity_metric: Similarity metric ('cosine', 'euclidean')
            aggregation: Multi-template aggregation strategy ('max', 'mean', 'avg_k')
            device: Device for computation
        """
        self.similarity_metric = similarity_metric
        self.aggregation = aggregation
        self.device = device

    def compute_similarity(
        self,
        proposal_embeddings: np.ndarray,
        template_embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute similarity between proposals and templates.

        Args:
            proposal_embeddings: Proposal embeddings [N_proposals, feat_dim]
            template_embeddings: Template embeddings {instance_id: [N_templates, feat_dim]}

        Returns:
            Similarity matrix [N_proposals, N_instances, N_templates]
            List of instance IDs (ordered)
        """
        # Convert to torch tensors
        proposals_tensor = torch.from_numpy(proposal_embeddings).float().to(self.device)

        # Build template tensor
        instance_ids = sorted(template_embeddings.keys())
        max_templates = max(len(templates) for templates in template_embeddings.values())

        # Initialize similarity matrix
        n_proposals = len(proposal_embeddings)
        n_instances = len(instance_ids)

        # Store all templates
        all_templates = []
        for inst_id in instance_ids:
            all_templates.append(template_embeddings[inst_id])

        # Compute similarity for each instance
        similarity_matrix = np.zeros((n_proposals, n_instances, max_templates))

        for inst_idx, templates in enumerate(all_templates):
            templates_tensor = torch.from_numpy(templates).float().to(self.device)
            n_templates = len(templates)

            # Compute similarity
            if self.similarity_metric == 'cosine':
                proposals_norm = F.normalize(proposals_tensor, p=2, dim=-1)
                templates_norm = F.normalize(templates_tensor, p=2, dim=-1)
                sim = torch.matmul(proposals_norm, templates_norm.T)
            elif self.similarity_metric == 'euclidean':
                proposals_expanded = proposals_tensor.unsqueeze(1)
                templates_expanded = templates_tensor.unsqueeze(0)
                dist = torch.sqrt(((proposals_expanded - templates_expanded) ** 2).sum(dim=-1))
                sim = -dist
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

            similarity_matrix[:, inst_idx, :n_templates] = sim.cpu().numpy()

        return similarity_matrix, instance_ids

    def aggregate_template_scores(
        self,
        similarity_matrix: np.ndarray,
        k: int = 5
    ) -> np.ndarray:
        """
        Aggregate multi-template similarities to instance-level scores.

        Args:
            similarity_matrix: Similarity matrix [N_proposals, N_instances, N_templates]
            k: Number of top templates for 'avg_k' aggregation

        Returns:
            Instance scores [N_proposals, N_instances]
        """
        if self.aggregation == 'max':
            scores = np.max(similarity_matrix, axis=-1)
        elif self.aggregation == 'mean':
            mask = similarity_matrix != 0
            sum_sim = np.sum(similarity_matrix, axis=-1)
            count_sim = np.sum(mask, axis=-1)
            scores = sum_sim / np.maximum(count_sim, 1)
        elif self.aggregation == 'avg_k':
            n_templates = similarity_matrix.shape[-1]
            k_actual = min(k, n_templates)
            sorted_sim = -np.sort(-similarity_matrix, axis=-1)
            top_k_sim = sorted_sim[:, :, :k_actual]
            scores = np.mean(top_k_sim, axis=-1)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")

        return scores

    def match_proposals(
        self,
        proposal_embeddings: np.ndarray,
        template_embeddings: Dict[str, np.ndarray],
        return_scores: bool = True,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Match proposals to template instances.

        Args:
            proposal_embeddings: Proposal embeddings [N_proposals, feat_dim]
            template_embeddings: Template embeddings {instance_id: [N_templates, feat_dim]}
            return_scores: Whether to return all instance scores
            top_k: Number of top matches to return

        Returns:
            List of predictions, one per proposal:
            [{
                'instance_id': str,
                'score': float,
                'top_k_matches': [(instance_id, score), ...],
            }, ...]
        """
        # Compute similarity
        similarity_matrix, instance_ids = self.compute_similarity(
            proposal_embeddings, template_embeddings
        )

        # Aggregate to instance scores
        instance_scores = self.aggregate_template_scores(similarity_matrix)

        # Match each proposal to best instance
        predictions = []

        for prop_idx in range(len(proposal_embeddings)):
            scores = instance_scores[prop_idx]

            best_inst_idx = np.argmax(scores)
            best_score = scores[best_inst_idx]
            best_instance_id = instance_ids[best_inst_idx]

            pred = {
                'instance_id': best_instance_id,
                'score': float(best_score),
            }

            if return_scores:
                sorted_indices = np.argsort(scores)[::-1]
                top_k_actual = min(top_k, len(instance_ids))

                top_k_matches = [
                    (instance_ids[idx], float(scores[idx]))
                    for idx in sorted_indices[:top_k_actual]
                ]

                pred['top_k_matches'] = top_k_matches

            predictions.append(pred)

        return predictions


def compute_matching_metrics(
    predictions: List[Dict],
    ground_truths: List[str],
    top_k_values: List[int] = [1, 3, 5]
) -> Dict:
    """
    Compute matching accuracy metrics.

    Args:
        predictions: List of predictions from matcher
        ground_truths: List of ground truth instance IDs
        top_k_values: List of k values for top-k accuracy

    Returns:
        Metrics dictionary
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    n_samples = len(predictions)
    if n_samples == 0:
        return {f'top{k}_accuracy': 0.0 for k in top_k_values}

    metrics = {}

    # Compute top-k accuracy
    for k in top_k_values:
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            if 'top_k_matches' in pred:
                top_k_ids = [inst_id for inst_id, _ in pred['top_k_matches'][:k]]
                if str(gt) in top_k_ids:
                    correct += 1
            else:
                if k == 1 and str(pred['instance_id']) == str(gt):
                    correct += 1

        metrics[f'top{k}_accuracy'] = correct / n_samples

    # Compute Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for pred, gt in zip(predictions, ground_truths):
        if 'top_k_matches' in pred:
            rank = None
            for idx, (inst_id, _) in enumerate(pred['top_k_matches']):
                if str(inst_id) == str(gt):
                    rank = idx + 1
                    break

            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        else:
            if str(pred['instance_id']) == str(gt):
                reciprocal_ranks.append(1.0)
            else:
                reciprocal_ranks.append(0.0)

    metrics['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return metrics
