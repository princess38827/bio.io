"""Evaluation metrics for PRISM2."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from typing import Optional, Tuple


def compute_auc(
    predictions: np.ndarray,
    targets: np.ndarray,
    multi_class: str = "ovr"
) -> float:
    """
    Compute Area Under the ROC Curve (AUC).
    
    Args:
        predictions: Predicted probabilities or scores
        targets: Ground truth labels
        multi_class: Multi-class strategy ('ovr' or 'ovo')
    
    Returns:
        AUC score
    """
    try:
        if len(np.unique(targets)) == 2:
            # Binary classification
            if predictions.ndim == 2:
                predictions = predictions[:, 1]  # Use positive class probs
            auc = roc_auc_score(targets, predictions)
        else:
            # Multi-class classification
            auc = roc_auc_score(
                targets,
                predictions,
                multi_class=multi_class,
                average="macro"
            )
        return float(auc)
    except ValueError as e:
        print(f"Error computing AUC: {e}")
        return 0.0


def compute_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    balanced: bool = False
) -> float:
    """
    Compute accuracy.
    
    Args:
        predictions: Predicted labels or probabilities
        targets: Ground truth labels
        balanced: Whether to use balanced accuracy
    
    Returns:
        Accuracy score
    """
    # Convert probabilities to labels if needed
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    
    if balanced:
        acc = balanced_accuracy_score(targets, predictions)
    else:
        acc = accuracy_score(targets, predictions)
    
    return float(acc)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    task: str = "binary"
) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted probabilities or scores
        targets: Ground truth labels
        task: Task type ('binary' or 'multiclass')
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = compute_accuracy(predictions, targets, balanced=False)
    metrics["balanced_accuracy"] = compute_accuracy(predictions, targets, balanced=True)
    
    # AUC
    if task == "binary":
        metrics["auc"] = compute_auc(predictions, targets)
    elif task == "multiclass":
        metrics["auc_ovr"] = compute_auc(predictions, targets, multi_class="ovr")
        metrics["auc_ovo"] = compute_auc(predictions, targets, multi_class="ovo")
    
    return metrics


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    from sklearn.metrics import confusion_matrix
    
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    return cm
