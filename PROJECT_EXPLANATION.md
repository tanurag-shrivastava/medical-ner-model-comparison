# Medical Named Entity Recognition — Project Explanation

## What This Project Does

This project builds a system that reads clinical notes from the **General Medicine** domain and automatically identifies medical entities in the text — things like diseases, drugs, symptoms, procedures, and anatomical locations.

For example, given the sentence:

> *"Patient has diabetes and hypertension. Prescribed Metformin 500mg and Lisinopril."*

The system extracts:

| Entity | Type | Normalized Form |
|---|---|---|
| diabetes | DISEASE | Diabetes Mellitus |
| hypertension | DISEASE | Hypertension |
| Metformin | DRUG | Metformin |
| Lisinopril | DRUG | Lisinopril |

This task is called **Named Entity Recognition (NER)**. The project compares three different approaches to solve it, from simple rule-matching all the way to a state-of-the-art transformer model.

---

## The Dataset

**Source:** `mtsamples.csv` — a collection of real medical transcription samples across many specialties.

**Filter:** Only **General Medicine** records are used (258 documents, ~9,190 sentences after processing).

**Split (sentence-level, reproducible with `random_state=42`):**

| Split | Sentences | Purpose |
|---|---|---|
| Train | 6,433 (70%) | Model learning |
| Validation | 1,378 (15%) | Hyperparameter tuning |
| Test | 1,379 (15%) | Final evaluation only |

The same three splits are used across all models so comparisons are fair.

---

## The Five Entity Types

| Label | What It Captures | Examples |
|---|---|---|
| **DISEASE** | Diagnosed conditions | diabetes, hypertension, COPD, atrial fibrillation |
| **SYMPTOM** | Patient complaints / findings | chest pain, shortness of breath, fatigue, nausea |
| **DRUG** | Medications and treatments | Metformin, Aspirin, Albuterol, Warfarin |
| **PROCEDURE** | Tests and interventions | EKG, CBC, MRI, colonoscopy, cholecystectomy |
| **ANATOMY** | Body parts and structures | heart, lung, left ventricle, knee, abdomen |

---

## Preprocessing Pipeline

Every piece of text goes through the same pipeline before reaching any model:

### 1. Abbreviation Expansion (`utils/abbreviation.py`)
Medical text is full of shorthand. This step expands abbreviations before NER so models see full words.

```
HTN  →  Hypertension
DM   →  Diabetes Mellitus
BP   →  Blood Pressure
SOB  →  Shortness of Breath
CBC  →  Complete Blood Count
EKG  →  Electrocardiogram
```

Over 120 common medical abbreviations are covered.

### 2. Text Cleaning (`utils/preprocessing.py`)
- Removes section headers like `SUBJECTIVE:,` or `PLAN:,`
- Collapses extra whitespace
- Strips non-informative formatting

### 3. Sentence Splitting
Documents are split into individual sentences using punctuation boundaries and numbered list detection. This is important because the dataset is split at **sentence level**, not document level — preventing data leakage between train and test.

### 4. Tokenization
A simple regex-based tokenizer splits sentences into word tokens while preserving hyphenated terms like `non-insulin` and contractions.

### 5. BIO Tagging (`utils/bio_converter.py`)
Each token is assigned a BIO label:
- `B-TYPE` — Beginning of an entity
- `I-TYPE` — Inside (continuation of) an entity
- `O` — Outside any entity

Example:
```
chest   →  B-DISEASE
pain    →  I-DISEASE
and     →  O
Aspirin →  B-DRUG
```

---

## The Three Models

### Model 1 — Rule-Based System (`models/rule_based.py`)

**How it works:** No machine learning. Uses three layers of rules:

**Layer 1 — Medical Gazetteers (dictionaries)**
Large hand-crafted lists of known medical terms for each entity type. For example, the DRUG list contains 80+ drug names; the DISEASE list contains 100+ conditions. The system scans each sentence and matches tokens against these lists using longest-match-first.

**Layer 2 — Regex Patterns**
Catches patterns that gazetteers miss:
- Drug dosage patterns: `Metformin 500mg`
- Procedure codes: `\b(CBC|BMP|CMP|EKG)\b`
- Condition suffixes: words ending in `-itis`, `-osis`, `-emia`, `-oma`

**Layer 3 — Contextual Rules**
Uses the surrounding words as clues:
- After `"prescribed"` → the next capitalized word is likely a DRUG
- After `"diagnosed with"` → the following phrase is likely a DISEASE
- After `"ordered"` → likely a PROCEDURE

**Strengths:** Fast, interpretable, no training data needed, perfect recall on known terms.

**Weaknesses:** Cannot generalize beyond its dictionaries. Misses novel drug names or unusual phrasings.

---

### Model 2 — BiLSTM-CRF (`models/bilstm_crf.py`)

**How it works:** A neural sequence labeling model built from scratch in PyTorch. No spaCy or prebuilt NER libraries.

**Architecture:**

```
Input tokens
     ↓
Word Embeddings (100-dim, randomly initialized)
     ↓
Bidirectional LSTM (256 hidden units, 2 layers)
  — reads the sentence left-to-right AND right-to-left
  — captures context from both directions
     ↓
Linear projection → emission scores (one per BIO label)
     ↓
CRF (Conditional Random Field) decoding layer
  — enforces valid BIO sequences (e.g. I-DRUG can't follow B-DISEASE)
  — finds the globally optimal tag sequence via Viterbi algorithm
     ↓
BIO tag sequence
```

**Why BiLSTM?** A word's meaning depends on its neighbors. "pain" alone is ambiguous, but "chest pain" is clearly a symptom. The bidirectional LSTM captures this context.

**Why CRF?** Without it, the model might predict `B-DRUG I-DISEASE` which is an invalid BIO sequence. The CRF layer learns transition probabilities between tags and guarantees valid output.

**Training details:**
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 15
- Gradient clipping: 5.0
- Learning rate scheduler: ReduceLROnPlateau

**Training curve (loss per epoch):**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 5.18 | 1.68 |
| 3 | 0.94 | 0.58 |
| 5 | 0.37 | 0.25 |
| 8 | 0.12 | 0.16 |
| 11 | 0.05 | 0.15 |
| 15 | 0.03 | 0.15 |

The model converges well with no significant overfitting.

---

### Model 3 — BioBERT (`models/biobert.py`)

**How it works:** Fine-tunes a pretrained transformer model specifically trained on biomedical text.

**Base model:** `dmis-lab/biobert-base-cased-v1.2` from HuggingFace — BERT pre-trained on PubMed abstracts and PMC full-text articles. It already understands biomedical language before we even start training.

**Fine-tuning strategy (optimized for CPU speed):**
- The 12-layer BERT encoder is **partially frozen** — layers 0–9 are frozen, only layers 10–11 and the classification head are trained
- This reduces trainable parameters from 108M to ~14M (13.2%), making training ~5× faster
- Training data subsampled to 2,000 sentences for speed
- Only 3 epochs needed because the pretrained weights already encode medical knowledge

**Token alignment:** BERT uses WordPiece tokenization which splits words into subwords (e.g. `hypertension` → `hyper`, `##tension`). The model carefully aligns BIO labels back to original word boundaries — only the first subword of each word gets the label.

**Training details:**
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Batch size: 32, max sequence length: 64
- Warmup: 10% of total steps
- Epochs: 3

**Training curve:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.517 | 0.150 |
| 2 | 0.120 | 0.113 |
| 3 | 0.069 | 0.102 |

---

## Entity Normalization (`utils/normalization.py`)

After extraction, raw entity text is mapped to a canonical standard form. This handles spelling variants, brand names, and informal descriptions.

| Raw Extracted Text | Normalized Form |
|---|---|
| high blood sugar | Diabetes Mellitus |
| heart attack | Myocardial Infarction |
| Tylenol | Acetaminophen |
| Coumadin | Warfarin |
| Lasix | Furosemide |
| Crestor | Rosuvastatin |
| a-fib | Atrial Fibrillation |
| echo | Echocardiogram |

This is applied after extraction in all three models.

---

## Evaluation Results

All models are evaluated on the **same held-out test set** (1,379 sentences, never seen during training).

### Overall Performance

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Rule-Based | 0.7798 | 0.9651 | 0.8626 | 0.9558 |
| BiLSTM-CRF | **0.9885** | **0.9794** | **0.9839** | **0.9976** |
| BioBERT | 0.8523 | 0.7977 | 0.8241 | 0.9729 |

### Per-Entity F1 Scores

| Entity | Rule-Based | BiLSTM-CRF | BioBERT |
|---|---|---|---|
| DISEASE | 0.9365 | **0.9896** | 0.8301 |
| SYMPTOM | 0.9823 | **0.9928** | 0.7542 |
| DRUG | 0.4343 | **0.9677** | 0.7483 |
| PROCEDURE | 0.9212 | **0.9276** | 0.7372 |
| ANATOMY | 0.9947 | **0.9954** | 0.8760 |

---

## Understanding the Results

### Why does Rule-Based have high Recall but lower Precision?
The rule-based system uses large dictionaries — if a term is in the list, it always gets tagged. This means it rarely misses known entities (high recall = 0.97) but sometimes tags things incorrectly, like generic words that happen to appear in the dictionary (lower precision = 0.78). The DRUG category suffers most because drug names overlap with common words.

### Why does BiLSTM-CRF perform best?
The BiLSTM-CRF was trained on the same BIO-tagged data that was generated by the same gazetteers used for evaluation. This means it learned exactly the patterns the test set was labeled with, leading to near-perfect scores. In a real-world scenario with human-annotated data, the gap between models would be smaller.

### Why does BioBERT score lower than BiLSTM-CRF here?
Three reasons:
1. **Subsampling** — BioBERT only trained on 2,000 of the 6,433 training sentences for speed
2. **Partial freezing** — Only the top 2 encoder layers were fine-tuned
3. **Shorter sequences** — max_length=64 truncates some longer sentences

In a full fine-tuning setup with all data and a GPU, BioBERT would typically outperform BiLSTM-CRF on real-world annotated data. The current setup prioritizes speed over maximum accuracy.

### Trade-off Summary

| Dimension | Rule-Based | BiLSTM-CRF | BioBERT |
|---|---|---|---|
| Training time | None | ~15 minutes (CPU) | ~5 minutes (frozen, CPU) |
| Inference speed | Very fast | Fast | Moderate |
| Generalization | Low — only known terms | Medium — learned patterns | High — pretrained knowledge |
| Interpretability | High — rules are readable | Low — black box | Low — black box |
| Data required | None | Labeled sentences | Pretrained + fine-tune data |
| Handles new drugs | No | Partially | Yes (if seen in pretraining) |

---

## Project File Structure

```
project/
│
├── data/
│   └── mtsamples.csv              # Raw dataset (General Medicine filtered internally)
│
├── models/
│   ├── rule_based.py              # Gazetteer + regex + contextual rules NER
│   ├── bilstm_crf.py              # PyTorch BiLSTM-CRF from scratch
│   └── biobert.py                 # HuggingFace BioBERT fine-tuning
│
├── utils/
│   ├── preprocessing.py           # Data loading, cleaning, splitting
│   ├── bio_converter.py           # BIO tagging with medical gazetteers
│   ├── abbreviation.py            # 120+ medical abbreviation expansions
│   ├── normalization.py           # Entity → canonical form mapping
│   └── metrics.py                 # Precision, Recall, F1, confusion matrix
│
├── app/
│   └── streamlit_app.py           # Interactive web UI
│
├── outputs/
│   ├── models/
│   │   ├── bilstm_crf.pt          # Saved BiLSTM-CRF weights
│   │   ├── bilstm_vocab.json      # Word vocabulary
│   │   └── biobert_ner/           # Saved BioBERT fine-tuned model
│   ├── results/
│   │   ├── rule_based_metrics.json
│   │   ├── bilstm_metrics.json
│   │   ├── biobert_metrics.json
│   │   └── comparison.json        # Side-by-side comparison
│   └── splits/
│       ├── train_tagged.json      # 6,433 BIO-tagged sentences
│       ├── val_tagged.json        # 1,378 BIO-tagged sentences
│       └── test_tagged.json       # 1,379 BIO-tagged sentences
│
├── prepare_data.py                # Run once: loads, cleans, splits, tags data
├── run_bilstm.py                  # Train BiLSTM-CRF
├── run_biobert.py                 # Fine-tune BioBERT
├── eval_bilstm.py                 # Evaluate saved BiLSTM-CRF
├── make_comparison.py             # Generate comparison.json
├── verify.py                      # Check all outputs exist and models load
├── train.py                       # All-in-one training pipeline
├── requirements.txt
└── PROJECT_EXPLANATION.md         # This file
```

---

## How to Run the Project

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Prepare data (run once)
```bash
cd project
python prepare_data.py
```
This loads the CSV, filters General Medicine, expands abbreviations, splits into train/val/test, and saves BIO-tagged JSON files.

### Step 3 — Train models

Rule-based (no training needed, just evaluate):
```bash
python -c "
import sys, os, json, numpy as np
sys.path.insert(0, '.')
from models.rule_based import RuleBasedNER
model = RuleBasedNER()
model.save('outputs/models/rule_based_config.json')
"
```

BiLSTM-CRF (15 epochs, ~15 min on CPU):
```bash
python run_bilstm.py
```

BioBERT (3 epochs, frozen base, ~5 min on CPU):
```bash
python run_biobert.py
```

### Step 4 — Generate comparison
```bash
python make_comparison.py
```

### Step 5 — Launch the app
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## The Streamlit App

The app has three tabs:

**Tab 1 — Entity Extraction**
- Paste any clinical text or choose from three built-in samples
- Select a model (Rule-Based / BiLSTM-CRF / BioBERT)
- Toggle abbreviation expansion and entity normalization
- See color-coded highlighted entities in the text
- View a table of extracted entities with their normalized forms
- See a pie chart of entity type distribution

**Tab 2 — Model Comparison**
- Summary cards showing each model's F1 score
- Grouped bar chart: Precision / Recall / F1 / Accuracy across all models
- Per-entity F1 bar chart broken down by entity type
- Interactive confusion matrix heatmap
- Full metrics table

**Tab 3 — Training Curves**
- Line charts of train and validation loss for BiLSTM-CRF and BioBERT
- Trade-off summary table comparing all three models across practical dimensions

---

## Key Design Decisions

**Why sentence-level splitting?**
Splitting at the document level risks putting sentences from the same document in both train and test, which inflates scores. Sentence-level splitting is stricter and more realistic.

**Why the same BIO tags across all models?**
Using identical label schemas means evaluation is directly comparable — the same ground truth is used for all three models.

**Why abbreviation expansion before NER?**
Models see `Hypertension` instead of `HTN`, `Diabetes Mellitus` instead of `DM`. This dramatically improves gazetteer matching and helps the neural models too, since expanded forms appear more frequently in pretraining corpora.

**Why normalization after extraction?**
Normalization is a post-processing step that maps surface forms to standard names. Doing it after extraction keeps the models focused on finding entity boundaries, not on standardization.
