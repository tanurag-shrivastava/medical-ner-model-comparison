"""Train BiLSTM-CRF model."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.bilstm_crf import BiLSTMCRFTrainer
from utils.metrics import compute_metrics, format_metrics_report
from utils.preprocessing import LABEL_LIST


def load_tagged(path):
    with open(path) as f:
        return [(item[0], item[1]) for item in json.load(f)]


def convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def deep_convert(d):
    if isinstance(d, dict): return {k: deep_convert(v) for k, v in d.items()}
    if isinstance(d, list): return [deep_convert(i) for i in d]
    return convert(d)


if __name__ == "__main__":
    train_tagged = load_tagged("outputs/splits/train_tagged.json")
    val_tagged   = load_tagged("outputs/splits/val_tagged.json")
    test_tagged  = load_tagged("outputs/splits/test_tagged.json")

    config = {
        "embedding_dim": 100, "hidden_dim": 256, "num_layers": 2,
        "dropout": 0.3, "lr": 0.001, "batch_size": 32, "epochs": 15,
        "glove_path": None,
    }

    trainer = BiLSTMCRFTrainer(config)
    trainer.train(train_tagged, val_tagged)

    os.makedirs("outputs/models", exist_ok=True)
    trainer.save("outputs/models/bilstm_crf.pt", "outputs/models/bilstm_vocab.json")

    results = trainer.evaluate(test_tagged)
    metrics = compute_metrics(results["true_tags"], results["pred_tags"], LABEL_LIST)
    metrics["train_losses"] = trainer.train_losses
    metrics["val_losses"] = trainer.val_losses
    print(format_metrics_report(metrics))

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/bilstm_metrics.json", "w") as f:
        json.dump(deep_convert(metrics), f, indent=2)
    print("Saved bilstm_metrics.json")
