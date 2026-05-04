"""
ROC / AUC Curves for Medical NER
Runs inference on the test set for all three models, collects per-class
probability scores, and plots one-vs-rest ROC curves per entity type.
"""

import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax

from utils.preprocessing import LABEL_LIST, LABEL2ID, ID2LABEL

# ── Config ────────────────────────────────────────────────────────────────────
ENTITY_TYPES  = ["DISEASE", "SYMPTOM", "DRUG", "PROCEDURE", "ANATOMY"]
ENTITY_COLORS = {
    "DISEASE":   "#FF6B6B",
    "SYMPTOM":   "#FFA94D",
    "DRUG":      "#51CF66",
    "PROCEDURE": "#339AF0",
    "ANATOMY":   "#CC5DE8",
}
OUTPUT_PATH = "outputs/results/roc_curves.png"

def load_tagged(path):
    with open(path) as f:
        return [(item[0], item[1]) for item in json.load(f)]

# ── Helpers ───────────────────────────────────────────────────────────────────

def tags_to_binary(tag_sequences, label_list):
    """Flatten tag sequences → integer label array."""
    flat = [label_list.index(t) if t in label_list else 0
            for seq in tag_sequences for t in seq]
    return np.array(flat)


def build_entity_binary(flat_true, flat_scores, entity_type, label_list):
    """
    One-vs-rest: collapse B-X and I-X into a single positive class.
    Returns (y_true_binary, y_score_positive).
    """
    b_idx = label_list.index(f"B-{entity_type}") if f"B-{entity_type}" in label_list else -1
    i_idx = label_list.index(f"I-{entity_type}") if f"I-{entity_type}" in label_list else -1

    y_true = ((flat_true == b_idx) | (flat_true == i_idx)).astype(int)

    # Positive score = max(P(B-X), P(I-X))
    scores = np.zeros(len(flat_scores))
    if b_idx >= 0:
        scores = np.maximum(scores, flat_scores[:, b_idx])
    if i_idx >= 0:
        scores = np.maximum(scores, flat_scores[:, i_idx])

    return y_true, scores


# ── Rule-Based scores ─────────────────────────────────────────────────────────

def get_rule_based_scores(test_tagged, label_list):
    """
    Rule-based has no probabilities — simulate hard scores:
    predicted class gets 0.95, others share 0.05.
    """
    from models.rule_based import RuleBasedNER
    model = RuleBasedNER()
    n_labels = len(label_list)
    all_true, all_scores = [], []

    for tokens, true_tags in test_tagged:
        bio = model.predict_tokens(tokens)
        pred_tags = [t for _, t in bio]
        for true_tag, pred_tag in zip(true_tags, pred_tags):
            true_idx = label_list.index(true_tag) if true_tag in label_list else 0
            pred_idx = label_list.index(pred_tag) if pred_tag in label_list else 0
            scores = np.full(n_labels, 0.05 / (n_labels - 1))
            scores[pred_idx] = 0.95
            all_true.append(true_idx)
            all_scores.append(scores)

    return np.array(all_true), np.array(all_scores)


# ── BiLSTM-CRF scores ─────────────────────────────────────────────────────────

def get_bilstm_scores(test_tagged, label_list):
    """Extract emission logits from BiLSTM (before CRF) as probability proxy."""
    import torch
    from models.bilstm_crf import BiLSTMCRFTrainer, NERDataset, collate_fn
    from torch.utils.data import DataLoader

    trainer = BiLSTMCRFTrainer()
    trainer.load("outputs/models/bilstm_crf.pt", "outputs/models/bilstm_vocab.json")
    model = trainer.model
    model.eval()
    device = trainer.device

    all_true, all_scores = [], []

    for tokens, true_tags in test_tagged:
        word_ids = [trainer.word2id.get(w.lower(), trainer.word2id["<UNK>"]) for w in tokens]
        words = torch.tensor([word_ids], dtype=torch.long).to(device)
        mask  = (words != 0)

        with torch.no_grad():
            emissions = model._get_emissions(words, mask)  # (1, seq_len, n_tags)
            probs = torch.softmax(emissions[0], dim=-1).cpu().numpy()  # (seq_len, n_tags)

        seq_len = min(len(tokens), probs.shape[0])
        for j in range(seq_len):
            true_idx = label_list.index(true_tags[j]) if true_tags[j] in label_list else 0
            all_true.append(true_idx)
            all_scores.append(probs[j])

    return np.array(all_true), np.array(all_scores)


# ── BioBERT scores ────────────────────────────────────────────────────────────

def get_biobert_scores(test_tagged, label_list):
    """Extract softmax probabilities from BioBERT logits."""
    import torch
    from models.biobert import BioBERTTrainer

    trainer = BioBERTTrainer()
    trainer.load("outputs/models/biobert_ner")
    model     = trainer.model
    tokenizer = trainer.tokenizer
    device    = trainer.device
    model.eval()

    all_true, all_scores = [], []

    for tokens, true_tags in test_tagged:
        encoding = tokenizer(
            tokens, is_split_into_words=True,
            max_length=trainer.config["max_length"],
            truncation=True, padding="max_length", return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)

        with torch.no_grad():
            logits = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
                token_type_ids=encoding.get(
                    "token_type_ids", torch.zeros_like(encoding["input_ids"])
                ).to(device),
            ).logits[0]  # (seq_len, n_tags)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Map back to original tokens (first subword only)
        seen = {}
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in seen:
                seen[wid] = probs[idx]

        for j in range(len(tokens)):
            if j in seen:
                true_idx = label_list.index(true_tags[j]) if true_tags[j] in label_list else 0
                all_true.append(true_idx)
                all_scores.append(seen[j])

    return np.array(all_true), np.array(all_scores)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_roc(ax, true_labels, score_matrix, label_list, model_name,
             model_color, linestyles):
    """Plot per-entity ROC curves on a single axes."""
    auc_scores = {}
    for etype, ls in zip(ENTITY_TYPES, linestyles):
        y_true, y_score = build_entity_binary(true_labels, score_matrix, etype, label_list)
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores[etype] = roc_auc
        ax.plot(fpr, tpr, color=ENTITY_COLORS[etype], lw=2, linestyle=ls,
                label=f"{etype}  (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(model_name, fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#fafafa")
    return auc_scores


def plot_macro_comparison(ax, all_auc):
    """Bar chart comparing macro-average AUC across models."""
    models = list(all_auc.keys())
    x = np.arange(len(ENTITY_TYPES))
    width = 0.22
    model_colors = {"Rule-Based": "#FF6B6B", "BiLSTM-CRF": "#339AF0", "BioBERT": "#51CF66"}

    for i, model in enumerate(models):
        vals = [all_auc[model].get(e, 0) for e in ENTITY_TYPES]
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=model_colors[model], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(ENTITY_TYPES, fontsize=10)
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("AUC Score", fontsize=10)
    ax.set_title("AUC Comparison by Entity Type", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_facecolor("#fafafa")


def plot_macro_roc_overlay(ax, model_data):
    """Overlay macro-average ROC curves for all three models."""
    model_colors = {"Rule-Based": "#FF6B6B", "BiLSTM-CRF": "#339AF0", "BioBERT": "#51CF66"}
    model_ls     = {"Rule-Based": "--",       "BiLSTM-CRF": "-",        "BioBERT": "-."}

    for model_name, (true_labels, score_matrix, label_list) in model_data.items():
        all_fpr = np.linspace(0, 1, 200)
        tprs = []
        for etype in ENTITY_TYPES:
            y_true, y_score = build_entity_binary(true_labels, score_matrix, etype, label_list)
            if y_true.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            tprs.append(np.interp(all_fpr, fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr,
                color=model_colors[model_name],
                lw=2.5, linestyle=model_ls[model_name],
                label=f"{model_name}  (Macro AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("Macro-Average ROC — All Models", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#fafafa")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading test set...")
    test_tagged = load_tagged("outputs/splits/test_tagged.json")
    print(f"Test sentences: {len(test_tagged)}")

    label_list = LABEL_LIST
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    print("\n[1/3] Collecting Rule-Based scores...")
    rb_true,  rb_scores  = get_rule_based_scores(test_tagged, label_list)
    print(f"      Tokens: {len(rb_true)}")

    print("[2/3] Collecting BiLSTM-CRF scores...")
    bl_true,  bl_scores  = get_bilstm_scores(test_tagged, label_list)
    print(f"      Tokens: {len(bl_true)}")

    print("[3/3] Collecting BioBERT scores...")
    bb_true,  bb_scores  = get_biobert_scores(test_tagged, label_list)
    print(f"      Tokens: {len(bb_true)}")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16), facecolor="white")
    fig.suptitle(
        "ROC / AUC Curves — Medical NER (General Medicine)\n"
        "Entity Types: DISEASE · SYMPTOM · DRUG · PROCEDURE · ANATOMY",
        fontsize=15, fontweight="bold", y=0.98
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                  top=0.92, bottom=0.06, left=0.06, right=0.97)

    ax_rb  = fig.add_subplot(gs[0, 0])
    ax_bl  = fig.add_subplot(gs[0, 1])
    ax_bb  = fig.add_subplot(gs[0, 2])
    ax_ov  = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1:])

    all_auc = {}

    all_auc["Rule-Based"] = plot_roc(
        ax_rb, rb_true, rb_scores, label_list, "Rule-Based", "#FF6B6B", linestyles)
    all_auc["BiLSTM-CRF"] = plot_roc(
        ax_bl, bl_true, bl_scores, label_list, "BiLSTM-CRF", "#339AF0", linestyles)
    all_auc["BioBERT"] = plot_roc(
        ax_bb, bb_true, bb_scores, label_list, "BioBERT", "#51CF66", linestyles)

    model_data = {
        "Rule-Based": (rb_true, rb_scores, label_list),
        "BiLSTM-CRF": (bl_true, bl_scores, label_list),
        "BioBERT":    (bb_true, bb_scores, label_list),
    }
    plot_macro_roc_overlay(ax_ov, model_data)
    plot_macro_comparison(ax_bar, all_auc)

    os.makedirs("outputs/results", exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUTPUT_PATH}")

    # Print AUC table
    print("\n" + "=" * 62)
    print(f"{'Entity':<12}", end="")
    for m in all_auc: print(f"  {m:>12}", end="")
    print()
    print("-" * 62)
    for etype in ENTITY_TYPES:
        print(f"{etype:<12}", end="")
        for m in all_auc:
            print(f"  {all_auc[m].get(etype, 0):>12.4f}", end="")
        print()
    print("=" * 62)
    macro = {m: np.mean(list(v.values())) for m, v in all_auc.items()}
    print(f"{'MACRO AVG':<12}", end="")
    for m in all_auc: print(f"  {macro[m]:>12.4f}", end="")
    print()
    print("=" * 62)
