"""
Evaluation Metrics Module
Computes Precision, Recall, F1, Accuracy, and per-entity metrics.
"""

import numpy as np
from collections import defaultdict


def compute_metrics(true_tags: list, pred_tags: list, label_list: list = None):
    """
    Compute overall and per-entity NER metrics using span-level evaluation.
    
    Args:
        true_tags: List of lists of true BIO tags
        pred_tags: List of lists of predicted BIO tags
        label_list: List of all possible labels
    
    Returns:
        dict with overall and per-entity metrics
    """
    entity_types = ["DISEASE", "SYMPTOM", "DRUG", "PROCEDURE", "ANATOMY"]

    # Collect true and predicted spans
    true_spans = set(extract_spans(true_tags))
    pred_spans = set(extract_spans(pred_tags))

    # Overall metrics
    overall = compute_span_metrics(true_spans, pred_spans)

    # Per-entity metrics
    per_entity = {}
    for etype in entity_types:
        true_e = {s for s in true_spans if s[3] == etype}
        pred_e = {s for s in pred_spans if s[3] == etype}
        per_entity[etype] = compute_span_metrics(true_e, pred_e)

    # Token-level accuracy
    flat_true = [t for seq in true_tags for t in seq]
    flat_pred = [t for seq in pred_tags for t in seq]
    accuracy = sum(t == p for t, p in zip(flat_true, flat_pred)) / max(len(flat_true), 1)

    # Confusion matrix data (token level)
    if label_list is None:
        label_list = sorted(set(flat_true + flat_pred))

    confusion = build_confusion_matrix(flat_true, flat_pred, label_list)

    return {
        "overall": {**overall, "accuracy": accuracy},
        "per_entity": per_entity,
        "confusion_matrix": confusion,
        "label_list": label_list,
    }


def extract_spans(tag_sequences: list) -> list:
    """Extract entity spans as list of (sentence_idx, start, end, type) tuples."""
    spans = []
    for sent_idx, tags in enumerate(tag_sequences):
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                etype = tag[2:]
                start = i
                i += 1
                while i < len(tags) and tags[i] == f"I-{etype}":
                    i += 1
                spans.append((sent_idx, start, i, etype))
            else:
                i += 1
    return spans


def compute_span_metrics(true_spans: set, pred_spans: set) -> dict:
    """Compute precision, recall, F1 from span sets."""
    tp = len(true_spans & pred_spans)
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def build_confusion_matrix(true_labels: list, pred_labels: list, label_list: list) -> np.ndarray:
    """Build token-level confusion matrix."""
    label2id = {l: i for i, l in enumerate(label_list)}
    n = len(label_list)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t in label2id and p in label2id:
            matrix[label2id[t]][label2id[p]] += 1
    return matrix


def format_metrics_report(metrics: dict) -> str:
    """Format metrics as a readable string report."""
    lines = []
    overall = metrics["overall"]
    lines.append("=" * 50)
    lines.append("OVERALL METRICS")
    lines.append("=" * 50)
    lines.append(f"  Precision : {overall['precision']:.4f}")
    lines.append(f"  Recall    : {overall['recall']:.4f}")
    lines.append(f"  F1 Score  : {overall['f1']:.4f}")
    lines.append(f"  Accuracy  : {overall['accuracy']:.4f}")
    lines.append("")
    lines.append("PER-ENTITY METRICS")
    lines.append("-" * 50)
    for etype, m in metrics["per_entity"].items():
        lines.append(f"  {etype:<12} P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    lines.append("=" * 50)
    return "\n".join(lines)
