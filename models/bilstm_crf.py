"""
BiLSTM-CRF Model for NER
Implemented from scratch using PyTorch.
No spaCy or prebuilt NER libraries used.
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from utils.preprocessing import LABEL_LIST, LABEL2ID, ID2LABEL
from utils.abbreviation import expand_abbreviations
from utils.normalization import normalize_entity
from utils.bio_converter import bio_to_entities
from utils.preprocessing import tokenize_sentence, clean_text

# ─── Constants ────────────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "O"
START_TAG = "<START>"
STOP_TAG = "<STOP>"


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NERDataset(Dataset):
    def __init__(self, tagged_sentences, word2id, label2id):
        self.samples = []
        for tokens, tags in tagged_sentences:
            word_ids = [word2id.get(w.lower(), word2id[UNK_TOKEN]) for w in tokens]
            tag_ids = [label2id.get(t, label2id["O"]) for t in tags]
            self.samples.append((word_ids, tag_ids, len(word_ids)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences in a batch."""
    word_ids, tag_ids, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    # Sort by length descending for pack_padded_sequence
    sorted_idx = torch.argsort(lengths, descending=True)
    lengths = lengths[sorted_idx]

    word_tensors = [torch.tensor(w, dtype=torch.long) for w in word_ids]
    tag_tensors = [torch.tensor(t, dtype=torch.long) for t in tag_ids]

    word_tensors = [word_tensors[i] for i in sorted_idx]
    tag_tensors = [tag_tensors[i] for i in sorted_idx]

    padded_words = pad_sequence(word_tensors, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tag_tensors, batch_first=True, padding_value=label2id_global["O"])

    return padded_words, padded_tags, lengths


label2id_global = LABEL2ID


# ─── CRF Layer ────────────────────────────────────────────────────────────────

class CRF(nn.Module):
    """Conditional Random Field layer."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        # Transition matrix: transitions[i][j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        # Start and stop transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)

        # Enforce BIO constraints
        self._init_bio_constraints()

    def _init_bio_constraints(self):
        """Initialize transition constraints for BIO tagging."""
        label_list = LABEL_LIST
        with torch.no_grad():
            for i, from_tag in enumerate(label_list):
                for j, to_tag in enumerate(label_list):
                    # I-X can only follow B-X or I-X
                    if to_tag.startswith("I-"):
                        entity = to_tag[2:]
                        if not (from_tag == f"B-{entity}" or from_tag == f"I-{entity}"):
                            self.transitions[i, j] = -10000.0
                    # B-X can follow O or any B/I tag
                    # O can follow anything

    def forward(self, emissions, tags, mask):
        """
        Compute negative log-likelihood loss.
        emissions: (batch, seq_len, num_tags)
        tags: (batch, seq_len)
        mask: (batch, seq_len) bool
        """
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def _compute_log_likelihood(self, emissions, tags, mask):
        batch_size, seq_len, num_tags = emissions.shape
        # Score of correct path
        score = self._score_sentence(emissions, tags, mask)
        # Log partition function
        partition = self._forward_algorithm(emissions, mask)
        return score - partition

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0, :].gather(1, tags[:, 0:1]).squeeze(1)

        for t in range(1, seq_len):
            m = mask[:, t]
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[:, t, :].gather(1, tags[:, t:t + 1]).squeeze(1)
            score += (trans + emit) * m

        # End transition
        last_tag_idx = mask.long().sum(1) - 1
        last_tags = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        # Initialize
        alpha = self.start_transitions + emissions[:, 0]  # (batch, num_tags)

        for t in range(1, seq_len):
            m = mask[:, t].unsqueeze(1)  # (batch, 1)
            # (batch, num_tags, 1) + (num_tags, num_tags) + (batch, 1, num_tags)
            emit = emissions[:, t].unsqueeze(1)  # (batch, 1, num_tags)
            trans = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            alpha_t = alpha.unsqueeze(2) + trans + emit  # (batch, num_tags, num_tags)
            alpha_t = torch.logsumexp(alpha_t, dim=1)  # (batch, num_tags)
            alpha = torch.where(m.bool(), alpha_t, alpha)

        alpha += self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        """Viterbi decoding. Returns list of tag sequences."""
        batch_size, seq_len, num_tags = emissions.shape
        lengths = mask.long().sum(1)

        # Viterbi
        viterbi_score = self.start_transitions + emissions[:, 0]
        backpointers = []

        for t in range(1, seq_len):
            # (batch, num_tags, 1) + (1, num_tags, num_tags)
            v_t = viterbi_score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_tags = v_t.max(dim=1)
            backpointers.append(best_tags)
            emit = emissions[:, t]
            viterbi_score = best_scores + emit

        viterbi_score += self.end_transitions
        _, best_last_tag = viterbi_score.max(dim=1)

        # Backtrack
        best_paths = []
        for b in range(batch_size):
            path = [best_last_tag[b].item()]
            for bp in reversed(backpointers):
                path.append(bp[b, path[-1]].item())
            path.reverse()
            best_paths.append(path[:lengths[b].item()])

        return best_paths


# ─── BiLSTM-CRF Model ─────────────────────────────────────────────────────────

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags,
                 num_layers=2, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings, dtype=torch.float))

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, words, tags, mask):
        """Compute CRF loss."""
        emissions = self._get_emissions(words, mask)
        loss = self.crf(emissions, tags, mask)
        return loss

    def _get_emissions(self, words, mask):
        embeds = self.dropout(self.embedding(words))
        lengths = mask.long().sum(1).cpu()
        packed = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        emissions = self.hidden2tag(self.dropout(lstm_out))
        return emissions

    def predict(self, words, mask):
        """Predict tag sequences using Viterbi decoding."""
        with torch.no_grad():
            emissions = self._get_emissions(words, mask)
            return self.crf.decode(emissions, mask)


# ─── Vocabulary Builder ───────────────────────────────────────────────────────

def build_vocab(tagged_sentences, min_freq=1):
    """Build word vocabulary from tagged sentences."""
    from collections import Counter
    counter = Counter()
    for tokens, _ in tagged_sentences:
        counter.update([t.lower() for t in tokens])

    word2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            word2id[word] = len(word2id)

    return word2id


def build_simple_embeddings(word2id, embedding_dim=100):
    """Build random embeddings (fallback when GloVe not available)."""
    vocab_size = len(word2id)
    embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    embeddings[0] = 0  # PAD = zeros
    return embeddings


def load_glove_embeddings(glove_path, word2id, embedding_dim=100):
    """Load GloVe embeddings if available."""
    vocab_size = len(word2id)
    embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    embeddings[0] = 0  # PAD

    if not os.path.exists(glove_path):
        print(f"[BiLSTM] GloVe not found at {glove_path}, using random embeddings.")
        return embeddings

    loaded = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                continue
            word = parts[0]
            if word in word2id:
                vec = np.array(parts[1:], dtype=np.float32)
                embeddings[word2id[word]] = vec
                loaded += 1

    print(f"[BiLSTM] Loaded {loaded}/{len(word2id)} GloVe vectors.")
    return embeddings


# ─── Trainer ──────────────────────────────────────────────────────────────────

class BiLSTMCRFTrainer:
    def __init__(self, config: dict = None):
        self.config = config or {
            "embedding_dim": 100,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 15,
            "glove_path": None,
        }
        self.model = None
        self.word2id = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []

    def build(self, train_tagged, val_tagged=None):
        """Build vocabulary and model."""
        self.word2id = build_vocab(train_tagged)
        vocab_size = len(self.word2id)
        num_tags = len(LABEL_LIST)

        glove_path = self.config.get("glove_path")
        if glove_path and os.path.exists(glove_path):
            embeddings = load_glove_embeddings(glove_path, self.word2id, self.config["embedding_dim"])
        else:
            embeddings = build_simple_embeddings(self.word2id, self.config["embedding_dim"])

        self.model = BiLSTMCRF(
            vocab_size=vocab_size,
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_tags=num_tags,
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            pretrained_embeddings=embeddings,
        ).to(self.device)

        print(f"[BiLSTM] Model built. Vocab: {vocab_size}, Tags: {num_tags}, Device: {self.device}")

    def train(self, train_tagged, val_tagged=None):
        """Train the model."""
        if self.model is None:
            self.build(train_tagged, val_tagged)

        train_dataset = NERDataset(train_tagged, self.word2id, LABEL2ID)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"],
            shuffle=True, collate_fn=collate_fn
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.config["epochs"]):
            self.model.train()
            total_loss = 0
            for words, tags, lengths in train_loader:
                words = words.to(self.device)
                tags = tags.to(self.device)
                mask = (words != 0).to(self.device)

                optimizer.zero_grad()
                loss = self.model(words, tags, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation
            val_loss = 0
            if val_tagged:
                val_loss = self._compute_val_loss(val_tagged)
                self.val_losses.append(val_loss)
                scheduler.step(val_loss)
                print(f"[BiLSTM] Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"[BiLSTM] Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Train Loss: {avg_train_loss:.4f}")

        print("[BiLSTM] Training complete.")

    def _compute_val_loss(self, val_tagged):
        """Compute validation loss."""
        self.model.eval()
        val_dataset = NERDataset(val_tagged, self.word2id, LABEL2ID)
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"],
                                shuffle=False, collate_fn=collate_fn)
        total_loss = 0
        with torch.no_grad():
            for words, tags, lengths in val_loader:
                words = words.to(self.device)
                tags = tags.to(self.device)
                mask = (words != 0).to(self.device)
                loss = self.model(words, tags, mask)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def predict(self, tokens: list) -> list:
        """Predict BIO tags for a list of tokens."""
        self.model.eval()
        word_ids = [self.word2id.get(w.lower(), self.word2id[UNK_TOKEN]) for w in tokens]
        words = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        mask = (words != 0)
        pred_ids = self.model.predict(words, mask)[0]
        pred_tags = [ID2LABEL.get(i, "O") for i in pred_ids]
        # Pad/trim to match token length
        pred_tags = pred_tags[:len(tokens)]
        while len(pred_tags) < len(tokens):
            pred_tags.append("O")
        return list(zip(tokens, pred_tags))

    def predict_sentence(self, sentence: str, expand_abbrev: bool = True) -> list:
        """Predict from raw sentence string."""
        if expand_abbrev:
            sentence = expand_abbreviations(sentence)
        sentence = clean_text(sentence)
        tokens = tokenize_sentence(sentence)
        if not tokens:
            return []
        return self.predict(tokens)

    def extract_entities(self, sentence: str, expand_abbrev: bool = True) -> list:
        """Extract entities with normalization."""
        bio = self.predict_sentence(sentence, expand_abbrev)
        raw_entities = bio_to_entities(bio)
        result = []
        for text, etype in raw_entities:
            normalized = normalize_entity(text, etype)
            result.append((text, etype, normalized))
        return result

    def evaluate(self, tagged_sentences: list) -> dict:
        """Evaluate on tagged sentences."""
        true_tags_all = []
        pred_tags_all = []
        for tokens, true_tags in tagged_sentences:
            bio = self.predict(tokens)
            pred_tags = [tag for _, tag in bio]
            true_tags_all.append(true_tags)
            pred_tags_all.append(pred_tags)
        return {"true_tags": true_tags_all, "pred_tags": pred_tags_all}

    def save(self, model_path: str, vocab_path: str):
        """Save model and vocabulary."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, model_path)
        with open(vocab_path, "w") as f:
            json.dump(self.word2id, f)
        print(f"[BiLSTM] Saved model to {model_path}")

    def load(self, model_path: str, vocab_path: str):
        """Load model and vocabulary."""
        with open(vocab_path) as f:
            self.word2id = json.load(f)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        vocab_size = len(self.word2id)
        num_tags = len(LABEL_LIST)
        self.model = BiLSTMCRF(
            vocab_size=vocab_size,
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_tags=num_tags,
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"[BiLSTM] Loaded model from {model_path}")
