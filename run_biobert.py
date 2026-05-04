"""Fine-tune BioBERT (frozen base, top-2 layers only) — fast CPU training."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.biobert import BioBERTTrainer
from utils.metrics import compute_metrics, format_metrics_report
from utils.preprocessing import LABEL_LIST


def load_tagged(path):
    with open(path) as f:
        return [(item[0], item[1]) for item in json.load(f)]


def deep_convert(d):
    def _c(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    if isinstance(d, dict): return {k: deep_convert(v) for k, v in d.items()}
    if isinstance(d, list): return [deep_convert(i) for i in d]
    return _c(d)


if __name__ == "__main__":
    print("Loading splits...")
    train_tagged = load_tagged("outputs/splits/train_tagged.json")
    val_tagged   = load_tagged("outputs/splits/val_tagged.json")
    test_tagged  = load_tagged("outputs/splits/test_tagged.json")
    print(f"Train: {len(train_tagged)} | Val: {len(val_tagged)} | Test: {len(test_tagged)}")

    config = {
        "model_name":        "dmis-lab/biobert-base-cased-v1.2",
        "max_length":        64,    # shorter sequences = faster
        "batch_size":        32,    # larger batch = fewer steps
        "epochs":            3,     # 3 epochs sufficient with frozen base
        "lr":                3e-4,  # higher LR since only classifier trains
        "warmup_ratio":      0.1,
        "weight_decay":      0.01,
        "max_train_samples": 2000,  # subsample 2k sentences
        "freeze_base":       True,  # freeze all but last 2 encoder layers
    }

    trainer = BioBERTTrainer(config)
    trainer.train(train_tagged, val_tagged)

    os.makedirs("outputs/models", exist_ok=True)
    trainer.save("outputs/models/biobert_ner")

    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_tagged)
    metrics = compute_metrics(results["true_tags"], results["pred_tags"], LABEL_LIST)
    metrics["train_losses"] = trainer.train_losses
    metrics["val_losses"]   = trainer.val_losses
    metrics["model"]        = "BioBERT"

    print(format_metrics_report(metrics))

    # Demo
    print("\n--- Entity Extraction Demo ---")
    for s in [
        "Patient has diabetes and hypertension. Prescribed Metformin 500mg and Lisinopril.",
        "She presents with chest pain and shortness of breath. EKG and CBC were ordered.",
        "History of COPD and asthma. Currently on Albuterol inhaler and Prednisone.",
    ]:
        print(f"  Input   : {s}")
        print(f"  Entities: {trainer.extract_entities(s)}\n")

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/biobert_metrics.json", "w") as f:
        json.dump(deep_convert(metrics), f, indent=2)
    print("Saved outputs/results/biobert_metrics.json")
