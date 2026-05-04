"""
Data preparation script — run this once before training.
Saves BIO-tagged splits to outputs/splits/
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import (
    load_general_medicine, prepare_sentences,
    split_dataset, save_splits,
)
from utils.bio_converter import tag_sentences

DATA_PATH = "data/mtsamples.csv"
SPLITS_DIR = "outputs/splits"


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # 1. Load & preprocess
    df = load_general_medicine(DATA_PATH)
    sentences = prepare_sentences(df, expand_abbrev=True)

    # 2. Split
    train_sents, val_sents, test_sents = split_dataset(sentences)
    save_splits(train_sents, val_sents, test_sents, SPLITS_DIR)

    # 3. BIO-tag
    print("[BIO] Tagging train set...")
    train_tagged = tag_sentences(train_sents)
    print("[BIO] Tagging val set...")
    val_tagged = tag_sentences(val_sents)
    print("[BIO] Tagging test set...")
    test_tagged = tag_sentences(test_sents)

    # 4. Save tagged splits
    for name, data in [("train", train_tagged), ("val", val_tagged), ("test", test_tagged)]:
        path = os.path.join(SPLITS_DIR, f"{name}_tagged.json")
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[BIO] Saved {name}_tagged.json ({len(data)} sentences)")

    # 5. Entity distribution
    counts = {}
    for tokens, tags in train_tagged:
        for tag in tags:
            if tag != "O":
                e = tag.split("-")[1]
                counts[e] = counts.get(e, 0) + 1
    print(f"\n[Stats] Entity distribution in train: {counts}")
    print(f"[Stats] Sample: {train_tagged[0]}")
    print("\n[Done] Data preparation complete.")


if __name__ == "__main__":
    main()
