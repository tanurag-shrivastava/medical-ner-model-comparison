# Medical Named Entity Recognition — Model Comparison

A system that automatically identifies medical entities in clinical notes and compares three different NER approaches — from rule-based matching to a fine-tuned transformer.

---

## What It Does

Given a clinical note like:

> *"Patient has diabetes and hypertension. Prescribed Metformin 500mg and Lisinopril."*

The system extracts and normalizes:

| Entity | Type | Normalized Form |
|---|---|---|
| diabetes | DISEASE | Diabetes Mellitus |
| hypertension | DISEASE | Hypertension |
| Metformin | DRUG | Metformin |
| Lisinopril | DRUG | Lisinopril |

---

## Dataset

**Source:** `mtsamples.csv` — real medical transcription samples  
**Filter:** General Medicine specialty only (258 documents, ~9,190 sentences)

| Split | Sentences |
|---|---|
| Train | 6,433 (70%) |
| Validation | 1,378 (15%) |
| Test | 1,379 (15%) |

---

## Entity Types

| Label | Examples |
|---|---|
| **DISEASE** | diabetes, hypertension, COPD, atrial fibrillation |
| **SYMPTOM** | chest pain, shortness of breath, fatigue, nausea |
| **DRUG** | Metformin, Aspirin, Albuterol, Warfarin |
| **PROCEDURE** | EKG, CBC, MRI, colonoscopy |
| **ANATOMY** | heart, lung, left ventricle, knee, abdomen |

---

## The Three Models

### 1. Rule-Based
No machine learning. Uses three layers:
- **Gazetteers** — hand-crafted dictionaries of 100+ terms per entity type
- **Regex patterns** — catches dosage patterns, condition suffixes (`-itis`, `-osis`, `-emia`)
- **Contextual rules** — e.g. after *"prescribed"* → next capitalized word is likely a DRUG

### 2. BiLSTM-CRF
Custom PyTorch sequence labeler built from scratch:
- Bidirectional LSTM (256 hidden units, 2 layers) captures left and right context
- CRF decoding layer enforces valid BIO sequences via Viterbi algorithm
- Trained for 15 epochs with Adam optimizer

### 3. BioBERT
Fine-tuned transformer pre-trained on biomedical text:
- Base model: `dmis-lab/biobert-base-cased-v1.2`
- Layers 0–9 frozen, only top 2 encoder layers + classifier trained (fast CPU training)
- 3 epochs with AdamW optimizer

---

## Results

Evaluated on the same held-out test set (1,379 sentences) across all models.

### Overall Performance

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Rule-Based | 0.7780 | 1.0000 | 0.8751 | 0.8680 |
| **BiLSTM-CRF** | **0.9797** | **0.9797** | **0.9797** | **0.9565** |
| BioBERT | 0.8774 | 0.7988 | 0.8363 | 0.9151 |

### Per-Entity F1 Scores

| Entity | Rule-Based | BiLSTM-CRF | BioBERT |
|---|---|---|---|
| DISEASE | 0.8415 | 0.9504 | 0.8392 |
| SYMPTOM | 0.9947 | 0.9872 | 0.8369 |
| DRUG | 0.4708 | 0.9524 | 0.7128 |
| PROCEDURE | 0.9212 | 0.9536 | 0.7138 |
| ANATOMY | 0.9949 | 0.9956 | 0.8891 |

### Model Trade-offs

| Dimension | Rule-Based | BiLSTM-CRF | BioBERT |
|---|---|---|---|
| Training time | None | ~15 min (CPU) | ~5 min (frozen, CPU) |
| Inference speed | Very fast | Fast | Moderate |
| Generalization | Low | Medium | High |
| Interpretability | High | Low | Low |
| Handles new terms | No | Partially | Yes |

---

## Project Structure

```
project/
├── data/
│   └── mtsamples.csv              # Raw dataset
├── models/
│   ├── rule_based.py              # Gazetteer + regex + contextual rules
│   ├── bilstm_crf.py              # PyTorch BiLSTM-CRF from scratch
│   └── biobert.py                 # HuggingFace BioBERT fine-tuning
├── utils/
│   ├── preprocessing.py           # Data loading, cleaning, splitting
│   ├── bio_converter.py           # BIO tagging with medical gazetteers
│   ├── abbreviation.py            # 120+ medical abbreviation expansions
│   ├── normalization.py           # Entity → canonical form mapping
│   └── metrics.py                 # Precision, Recall, F1, confusion matrix
├── app/
│   └── streamlit_app.py           # Interactive web UI
├── outputs/
│   ├── models/                    # Saved model checkpoints
│   ├── results/                   # Evaluation metrics + ROC curves
│   └── splits/                    # Train/val/test BIO-tagged JSON files
├── prepare_data.py                # One-time data preparation
├── run_bilstm.py                  # Train BiLSTM-CRF
├── run_biobert.py                 # Fine-tune BioBERT
├── plot_roc.py                    # Generate ROC/AUC curves
└── requirements.txt
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data (run once)
```bash
python prepare_data.py
```
Filters General Medicine records, expands abbreviations, splits into train/val/test, and saves BIO-tagged JSON files.

### 3. Train models
```bash
# BiLSTM-CRF (~15 min on CPU)
python run_bilstm.py

# BioBERT (~5 min on CPU, frozen base layers)
python run_biobert.py
```

> **Note:** `outputs/models/biobert_ner/model.safetensors` (411MB) is excluded from this repo due to GitHub's file size limit. Run `python run_biobert.py` to regenerate it.

### 4. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## Streamlit App

The app has three tabs:

- **Entity Extraction** — paste clinical text, select a model, see color-coded highlighted entities and a table of normalized forms
- **Model Comparison** — side-by-side metrics, grouped bar charts, per-entity F1, confusion matrix heatmap
- **Training Curves** — loss curves for BiLSTM-CRF and BioBERT, trade-off summary table

---

## Tech Stack

- **PyTorch** — BiLSTM-CRF implementation
- **HuggingFace Transformers** — BioBERT fine-tuning
- **Streamlit + Plotly** — interactive web UI
- **scikit-learn** — evaluation metrics
- **pandas / numpy** — data processing
