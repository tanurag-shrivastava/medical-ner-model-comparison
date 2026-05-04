"""
Preprocessing Module
Handles data loading, cleaning, sentence splitting, and dataset splitting.
"""

import re
import os
import json
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.abbreviation import expand_abbreviations

RANDOM_STATE = 42
LABEL_LIST = ["O", "B-DISEASE", "I-DISEASE", "B-SYMPTOM", "I-SYMPTOM",
              "B-DRUG", "I-DRUG", "B-PROCEDURE", "I-PROCEDURE", "B-ANATOMY", "I-ANATOMY"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def load_general_medicine(data_path: str) -> pd.DataFrame:
    """Load and filter General Medicine records from mtsamples.csv."""
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    gm = df[df["medical_specialty"].str.strip() == "General Medicine"].copy()
    gm = gm.dropna(subset=["transcription"])
    gm["transcription"] = gm["transcription"].astype(str)
    gm = gm[gm["transcription"].str.len() > 50]
    gm = gm.reset_index(drop=True)
    print(f"[Preprocessing] Loaded {len(gm)} General Medicine records.")
    return gm


def clean_text(text: str) -> str:
    """Clean clinical text."""
    if not isinstance(text, str):
        return ""
    # Remove section headers like "SUBJECTIVE:," or "PLAN:,"
    text = re.sub(r'\b[A-Z][A-Z\s/\-]+:\s*,?\s*', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def split_into_sentences(text: str) -> list:
    """Split clinical text into sentences."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Also split on numbered list items
    result = []
    for sent in sentences:
        # Split on numbered items like "1. " or "2. "
        parts = re.split(r'\d+\.\s+', sent)
        result.extend([p.strip() for p in parts if len(p.strip()) > 10])
    return result


def tokenize_sentence(sentence: str) -> list:
    """Simple whitespace + punctuation tokenizer."""
    # Split on whitespace and punctuation, keeping punctuation as tokens
    tokens = re.findall(r"\w+(?:[-']\w+)*|[.,;:!?()]", sentence)
    return tokens


def prepare_sentences(df: pd.DataFrame, expand_abbrev: bool = True) -> list:
    """
    Convert dataframe to list of tokenized sentences.
    Returns list of token lists.
    """
    all_sentences = []
    for _, row in df.iterrows():
        text = row["transcription"]
        if expand_abbrev:
            text = expand_abbreviations(text)
        text = clean_text(text)
        sentences = split_into_sentences(text)
        for sent in sentences:
            tokens = tokenize_sentence(sent)
            if len(tokens) >= 3:
                all_sentences.append(tokens)
    print(f"[Preprocessing] Total sentences: {len(all_sentences)}")
    return all_sentences


def split_dataset(sentences: list, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split sentences into train/val/test sets at sentence level.
    Uses random_state=42 for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    indices = list(range(len(sentences)))
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=RANDOM_STATE
    )
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_size), random_state=RANDOM_STATE
    )

    train_sents = [sentences[i] for i in train_idx]
    val_sents = [sentences[i] for i in val_idx]
    test_sents = [sentences[i] for i in test_idx]

    print(f"[Split] Train: {len(train_sents)} | Val: {len(val_sents)} | Test: {len(test_sents)}")
    return train_sents, val_sents, test_sents


def save_splits(train, val, test, output_dir: str):
    """Save dataset splits to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_sentences.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(output_dir, "val_sentences.json"), "w") as f:
        json.dump(val, f)
    with open(os.path.join(output_dir, "test_sentences.json"), "w") as f:
        json.dump(test, f)
    print(f"[Split] Saved splits to {output_dir}")


def load_splits(output_dir: str):
    """Load saved dataset splits."""
    with open(os.path.join(output_dir, "train_sentences.json")) as f:
        train = json.load(f)
    with open(os.path.join(output_dir, "val_sentences.json")) as f:
        val = json.load(f)
    with open(os.path.join(output_dir, "test_sentences.json")) as f:
        test = json.load(f)
    return train, val, test
