"""
BioBERT-style Model for NER
Uses dmis-lab/biobert-base-cased-v1.2 with frozen base layers for fast CPU training.
Only the top 2 encoder layers + classifier are fine-tuned.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

from utils.preprocessing import LABEL_LIST, LABEL2ID, ID2LABEL
from utils.abbreviation import expand_abbreviations
from utils.normalization import normalize_entity
from utils.bio_converter import bio_to_entities
from utils.preprocessing import tokenize_sentence, clean_text

BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


# ─── Dataset (lazy encoding) ──────────────────────────────────────────────────

class BioBERTNERDataset(Dataset):
    def __init__(self, tagged_sentences, tokenizer, label2id, max_length=128):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.data = [(tokens, tags) for tokens, tags in tagged_sentences if len(tokens) > 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        return self._encode(tokens, tags)

    def _encode(self, tokens, tags):
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                label = tags[word_id] if word_id < len(tags) else "O"
                aligned_labels.append(self.label2id.get(label, self.label2id["O"]))
            else:
                label = tags[word_id] if word_id < len(tags) else "O"
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                aligned_labels.append(self.label2id.get(label, self.label2id["O"]))
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros_like(encoding["input_ids"])
            ).squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "word_ids": word_ids,
            "tokens": tokens,
            "tags": tags,
        }


def collate_biobert(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


# ─── BioBERT Trainer ──────────────────────────────────────────────────────────

class BioBERTTrainer:
    def __init__(self, config: dict = None):
        self.config = config or {
            "model_name": BIOBERT_MODEL_NAME,
            "max_length": 128,
            "batch_size": 32,
            "epochs": 3,
            "lr": 3e-4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_train_samples": 2000,   # subsample for speed
            "freeze_base": True,         # freeze all but top 2 layers + classifier
        }
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []

    def build(self):
        model_name = self.config["model_name"]
        print(f"[BioBERT] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"[BioBERT] Loading model: {model_name}")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        # Freeze all base encoder layers except the last 2 + classifier
        if self.config.get("freeze_base", True):
            # Freeze embeddings
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze encoder layers 0-9, keep 10-11 trainable
            total_layers = len(self.model.bert.encoder.layer)
            freeze_up_to = max(0, total_layers - 2)
            for i, layer in enumerate(self.model.bert.encoder.layer):
                if i < freeze_up_to:
                    for param in layer.parameters():
                        param.requires_grad = False
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"[BioBERT] Trainable params: {trainable:,} / {total:,} "
                  f"({100*trainable/total:.1f}%) — frozen base for speed")

        print(f"[BioBERT] Device: {self.device}")

    def train(self, train_tagged, val_tagged=None):
        if self.model is None:
            self.build()

        # Subsample training data for speed
        max_samples = self.config.get("max_train_samples", len(train_tagged))
        if max_samples < len(train_tagged):
            import random
            random.seed(42)
            train_tagged = random.sample(train_tagged, max_samples)
            print(f"[BioBERT] Subsampled train to {len(train_tagged)} sentences for speed")

        train_dataset = BioBERTNERDataset(
            train_tagged, self.tokenizer, LABEL2ID, self.config["max_length"]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"],
            shuffle=True, collate_fn=collate_biobert, num_workers=0
        )

        # Only optimize trainable params
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        total_steps = len(train_loader) * self.config["epochs"]
        warmup_steps = int(total_steps * self.config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.config["epochs"]):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels         = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                outputs.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += outputs.loss.item()

            avg_train = total_loss / len(train_loader)
            self.train_losses.append(avg_train)

            if val_tagged:
                val_loss = self._compute_val_loss(val_tagged)
                self.val_losses.append(val_loss)
                print(f"[BioBERT] Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"[BioBERT] Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Train Loss: {avg_train:.4f}")

        print("[BioBERT] Training complete.")

    def _compute_val_loss(self, val_tagged):
        self.model.eval()
        # Use at most 500 val samples for speed
        import random
        samples = val_tagged[:500]
        val_dataset = BioBERTNERDataset(
            samples, self.tokenizer, LABEL2ID, self.config["max_length"]
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["batch_size"],
            shuffle=False, collate_fn=collate_biobert, num_workers=0
        )
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    token_type_ids=batch["token_type_ids"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                total_loss += outputs.loss.item()
        return total_loss / max(len(val_loader), 1)

    def predict(self, tokens: list) -> list:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        self.model.eval()

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.config["max_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
                token_type_ids=encoding.get(
                    "token_type_ids", torch.zeros_like(encoding["input_ids"])
                ).to(self.device),
            )

        pred_ids = outputs.logits[0].argmax(dim=-1).cpu().numpy()

        # Map subword predictions back to original tokens (first subword wins)
        token_preds = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in token_preds:
                token_preds[word_id] = ID2LABEL.get(int(pred_ids[idx]), "O")

        pred_tags = [token_preds.get(i, "O") for i in range(len(tokens))]
        return list(zip(tokens, pred_tags))

    def predict_sentence(self, sentence: str, expand_abbrev: bool = True) -> list:
        if expand_abbrev:
            sentence = expand_abbreviations(sentence)
        sentence = clean_text(sentence)
        tokens = tokenize_sentence(sentence)
        if not tokens:
            return []
        return self.predict(tokens)

    def extract_entities(self, sentence: str, expand_abbrev: bool = True) -> list:
        bio = self.predict_sentence(sentence, expand_abbrev)
        raw_entities = bio_to_entities(bio)
        return [(text, etype, normalize_entity(text, etype)) for text, etype in raw_entities]

    def evaluate(self, tagged_sentences: list) -> dict:
        true_tags_all, pred_tags_all = [], []
        for tokens, true_tags in tagged_sentences:
            bio = self.predict(tokens)
            pred_tags_all.append([tag for _, tag in bio])
            true_tags_all.append(true_tags)
        return {"true_tags": true_tags_all, "pred_tags": pred_tags_all}

    def save(self, model_dir: str):
        import shutil
        # Remove old files to release memory-mapped handles before overwriting
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        with open(os.path.join(model_dir, "training_meta.json"), "w") as f:
            json.dump({
                "config": self.config,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }, f, indent=2)
        print(f"[BioBERT] Saved to {model_dir}")

    def load(self, model_dir: str):
        print(f"[BioBERT] Loading from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir, id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(self.device)
        self.model.eval()
        meta_path = os.path.join(model_dir, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.config.update(meta.get("config", {}))
            self.train_losses = meta.get("train_losses", [])
            self.val_losses   = meta.get("val_losses", [])
        print(f"[BioBERT] Loaded.")
