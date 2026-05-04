"""
Streamlit UI for Medical NER
Interactive interface for entity extraction and model comparison.
"""

import os
import sys
import json
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.abbreviation import expand_abbreviations
from utils.normalization import normalize_entity

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical NER — General Medicine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Color Scheme ─────────────────────────────────────────────────────────────
ENTITY_COLORS = {
    "DISEASE":   "#FF6B6B",
    "SYMPTOM":   "#FFA94D",
    "DRUG":      "#69DB7C",
    "PROCEDURE": "#74C0FC",
    "ANATOMY":   "#DA77F2",
}

ENTITY_BG = {
    "DISEASE":   "#FFE3E3",
    "SYMPTOM":   "#FFF3BF",
    "DRUG":      "#EBFBEE",
    "PROCEDURE": "#E7F5FF",
    "ANATOMY":   "#F3D9FA",
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .entity-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #4dabf7;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 16px;
        font-size: 0.85rem;
    }
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ─────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "results")


@st.cache_resource(show_spinner="Loading Rule-Based model...")
def load_rule_based():
    from models.rule_based import RuleBasedNER
    return RuleBasedNER()


@st.cache_resource(show_spinner="Loading BiLSTM-CRF model...")
def load_bilstm():
    from models.bilstm_crf import BiLSTMCRFTrainer
    trainer = BiLSTMCRFTrainer()
    model_path = os.path.join(MODELS_DIR, "bilstm_crf.pt")
    vocab_path = os.path.join(MODELS_DIR, "bilstm_vocab.json")
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        trainer.load(model_path, vocab_path)
        return trainer
    return None


@st.cache_resource(show_spinner="Loading BioBERT model...")
def load_biobert():
    from models.biobert import BioBERTTrainer
    trainer = BioBERTTrainer()
    model_dir = os.path.join(MODELS_DIR, "biobert_ner")
    if os.path.exists(model_dir):
        trainer.load(model_dir)
        return trainer
    return None


def load_metrics(model_name: str):
    """Load saved evaluation metrics."""
    path_map = {
        "Rule-Based": "rule_based_metrics.json",
        "BiLSTM-CRF": "bilstm_metrics.json",
        "BioBERT": "biobert_metrics.json",
    }
    path = os.path.join(RESULTS_DIR, path_map.get(model_name, ""))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_comparison():
    """Load comparison metrics."""
    path = os.path.join(RESULTS_DIR, "comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ─── Entity Highlighting ──────────────────────────────────────────────────────

def highlight_entities(tokens_tags: list) -> str:
    """Generate HTML with color-coded entity highlights."""
    html_parts = []
    i = 0
    while i < len(tokens_tags):
        token, tag = tokens_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            color = ENTITY_COLORS.get(etype, "#ccc")
            bg = ENTITY_BG.get(etype, "#f0f0f0")
            entity_tokens = [token]
            i += 1
            while i < len(tokens_tags) and tokens_tags[i][1] == f"I-{etype}":
                entity_tokens.append(tokens_tags[i][0])
                i += 1
            entity_text = " ".join(entity_tokens)
            html_parts.append(
                f'<span style="background-color:{bg}; border: 1.5px solid {color}; '
                f'border-radius:4px; padding:1px 5px; margin:1px; display:inline-block;">'
                f'<span style="color:{color}; font-weight:600;">{entity_text}</span>'
                f'<sup style="color:{color}; font-size:0.65rem; margin-left:3px;">{etype}</sup>'
                f'</span> '
            )
        else:
            html_parts.append(f"{token} ")
            i += 1
    return "".join(html_parts)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    selected_model = st.selectbox(
        "Select NER Model",
        ["Rule-Based", "BiLSTM-CRF", "BioBERT"],
        help="Choose the model for entity extraction",
    )

    st.markdown("---")
    st.markdown("**Preprocessing Options**")
    expand_abbrev = st.checkbox("Expand Abbreviations", value=True,
                                help="Expand medical abbreviations before NER")
    normalize = st.checkbox("Normalize Entities", value=True,
                            help="Map entities to canonical forms")

    st.markdown("---")
    st.markdown("**Entity Types**")
    for etype, color in ENTITY_COLORS.items():
        st.markdown(
            f'<span style="background:{ENTITY_BG[etype]}; border:1.5px solid {color}; '
            f'border-radius:4px; padding:2px 8px; color:{color}; font-weight:600; '
            f'font-size:0.8rem;">{etype}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**About**")
    st.caption("Medical NER for General Medicine domain. Compares Rule-Based, BiLSTM-CRF, and BioBERT approaches.")


# ─── Main Header ──────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🏥 Medical Named Entity Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">General Medicine Domain · Clinical Notes NER · Comparative Analysis</div>', unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Entity Extraction", "📊 Model Comparison", "📈 Training Curves"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: ENTITY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 🧾 Clinical Text Input")

    sample_texts = {
        "Sample 1 — Chronic Conditions": (
            "The patient is a 65-year-old male with HTN, DM, and CAD. "
            "He presents with SOB and CP. BP was 160/95. HR was 88. "
            "He is currently on Metformin 500mg BID, Lisinopril 10mg QD, and ASA 81mg. "
            "EKG and CBC were ordered. Chest X-ray showed no acute findings."
        ),
        "Sample 2 — Respiratory": (
            "A 45-year-old female presents with cough, fever, and shortness of breath for 3 days. "
            "She has a history of asthma and COPD. SpO2 was 92% on room air. "
            "Prescribed Azithromycin 500mg and Albuterol inhaler. "
            "CXR showed bilateral infiltrates consistent with pneumonia."
        ),
        "Sample 3 — Neurological": (
            "Patient is a 72-year-old with history of CVA and HTN. "
            "Presents with headache, dizziness, and confusion. "
            "MRI of the brain was ordered. "
            "Current medications include Warfarin, Atorvastatin, and Metoprolol. "
            "Neurological examination revealed weakness in the left arm."
        ),
    }

    col_input, col_sample = st.columns([3, 1])
    with col_sample:
        sample_choice = st.selectbox("Load Sample", ["Custom"] + list(sample_texts.keys()))

    default_text = sample_texts.get(sample_choice, "") if sample_choice != "Custom" else ""

    with col_input:
        input_text = st.text_area(
            "Enter clinical text:",
            value=default_text,
            height=150,
            placeholder="Type or paste clinical notes here...",
        )

    run_btn = st.button("🔍 Extract Entities", type="primary", use_container_width=True)

    if run_btn and input_text.strip():
        with st.spinner(f"Running {selected_model}..."):
            try:
                # Load model
                if selected_model == "Rule-Based":
                    model = load_rule_based()
                    bio_result = model.predict_sentence(input_text, expand_abbrev=expand_abbrev)
                elif selected_model == "BiLSTM-CRF":
                    model = load_bilstm()
                    if model is None:
                        st.warning("⚠️ BiLSTM-CRF model not found. Run `python train.py --models bilstm` first.")
                        st.stop()
                    bio_result = model.predict_sentence(input_text, expand_abbrev=expand_abbrev)
                else:  # BioBERT
                    model = load_biobert()
                    if model is None:
                        st.warning("⚠️ BioBERT model not found. Run `python train.py --models biobert` first.")
                        st.stop()
                    bio_result = model.predict_sentence(input_text, expand_abbrev=expand_abbrev)

                # Show preprocessed text if abbreviations expanded
                if expand_abbrev:
                    expanded = expand_abbreviations(input_text)
                    if expanded != input_text:
                        with st.expander("📝 Text after abbreviation expansion"):
                            st.write(expanded)

                # ── Highlighted Output ──
                st.markdown("### 🎨 Highlighted Entities")
                highlighted_html = highlight_entities(bio_result)
                st.markdown(
                    f'<div style="background:#fafafa; border:1px solid #e0e0e0; border-radius:8px; '
                    f'padding:16px; line-height:2.2; font-size:1rem;">{highlighted_html}</div>',
                    unsafe_allow_html=True,
                )

                # ── Entity Table ──
                from utils.bio_converter import bio_to_entities
                raw_entities = bio_to_entities(bio_result)

                if raw_entities:
                    st.markdown("### 📋 Extracted Entities")
                    rows = []
                    for text, etype in raw_entities:
                        norm = normalize_entity(text, etype) if normalize else text
                        rows.append({
                            "Entity": text,
                            "Type": etype,
                            "Normalized Form": norm,
                        })
                    df = pd.DataFrame(rows)

                    # Color-code the Type column
                    def style_type(val):
                        color = ENTITY_COLORS.get(val, "#999")
                        bg = ENTITY_BG.get(val, "#f0f0f0")
                        return f"background-color: {bg}; color: {color}; font-weight: 600;"

                    styled = df.style.map(style_type, subset=["Type"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # ── Entity Distribution Pie ──
                    st.markdown("### 🥧 Entity Distribution")
                    type_counts = df["Type"].value_counts().reset_index()
                    type_counts.columns = ["Type", "Count"]
                    fig_pie = px.pie(
                        type_counts, names="Type", values="Count",
                        color="Type",
                        color_discrete_map=ENTITY_COLORS,
                        hole=0.4,
                    )
                    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                    fig_pie.update_layout(
                        showlegend=True, height=350,
                        margin=dict(t=20, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No entities found in the input text.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                import traceback
                st.code(traceback.format_exc())

    elif run_btn:
        st.warning("Please enter some clinical text first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 📊 Model Performance Comparison")

    comparison = load_comparison()

    if comparison is None:
        st.info("No evaluation results found. Run `python train.py` to train and evaluate all models.")
    else:
        models = list(comparison.keys())
        metrics_keys = ["precision", "recall", "f1", "accuracy"]

        # ── Summary Cards ──
        cols = st.columns(len(models))
        for i, (model_name, m) in enumerate(comparison.items()):
            with cols[i]:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h4 style="margin:0;color:#1a1a2e;">{model_name}</h4>'
                    f'<p style="font-size:1.8rem;font-weight:700;color:#4dabf7;margin:8px 0;">'
                    f'{m["f1"]:.3f}</p>'
                    f'<p style="color:#666;margin:0;font-size:0.85rem;">F1 Score</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Bar Chart ──
        st.markdown("#### 📊 Precision / Recall / F1 / Accuracy")
        fig_bar = go.Figure()
        colors_bar = ["#4dabf7", "#69db7c", "#ffa94d", "#da77f2"]
        for i, metric in enumerate(metrics_keys):
            fig_bar.add_trace(go.Bar(
                name=metric.capitalize(),
                x=models,
                y=[comparison[m][metric] for m in models],
                marker_color=colors_bar[i],
                text=[f"{comparison[m][metric]:.3f}" for m in models],
                textposition="outside",
            ))
        fig_bar.update_layout(
            barmode="group",
            height=420,
            yaxis=dict(range=[0, 1.1], title="Score"),
            xaxis_title="Model",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=40, b=40),
        )
        fig_bar.update_xaxes(showgrid=False)
        fig_bar.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Per-Entity Metrics ──
        st.markdown("#### 🏷️ Per-Entity F1 Scores")
        entity_types = ["DISEASE", "SYMPTOM", "DRUG", "PROCEDURE", "ANATOMY"]
        fig_entity = go.Figure()
        model_colors = {"Rule-Based": "#FF6B6B", "BiLSTM-CRF": "#4dabf7", "BioBERT": "#69db7c"}

        for model_name in models:
            m_data = load_metrics(model_name)
            if m_data and "per_entity" in m_data:
                f1_vals = [m_data["per_entity"].get(e, {}).get("f1", 0) for e in entity_types]
                fig_entity.add_trace(go.Bar(
                    name=model_name,
                    x=entity_types,
                    y=f1_vals,
                    marker_color=model_colors.get(model_name, "#ccc"),
                    text=[f"{v:.3f}" for v in f1_vals],
                    textposition="outside",
                ))

        fig_entity.update_layout(
            barmode="group",
            height=400,
            yaxis=dict(range=[0, 1.1], title="F1 Score"),
            xaxis_title="Entity Type",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=40, b=40),
        )
        fig_entity.update_xaxes(showgrid=False)
        fig_entity.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_entity, use_container_width=True)

        # ── Confusion Matrix ──
        st.markdown("#### 🔥 Confusion Matrix")
        cm_model = st.selectbox("Select model for confusion matrix", models, key="cm_model")
        m_data = load_metrics(cm_model)

        if m_data and "confusion_matrix" in m_data:
            cm = np.array(m_data["confusion_matrix"])
            labels = m_data.get("label_list", [])

            # Filter to non-zero rows/cols for readability
            nonzero = np.where((cm.sum(axis=1) > 0) | (cm.sum(axis=0) > 0))[0]
            if len(nonzero) > 0:
                cm_filtered = cm[np.ix_(nonzero, nonzero)]
                labels_filtered = [labels[i] for i in nonzero]
            else:
                cm_filtered = cm
                labels_filtered = labels

            fig_cm = px.imshow(
                cm_filtered,
                x=labels_filtered,
                y=labels_filtered,
                color_continuous_scale="Blues",
                text_auto=True,
                aspect="auto",
            )
            fig_cm.update_layout(
                height=500,
                xaxis_title="Predicted",
                yaxis_title="True",
                margin=dict(t=20, b=80, l=80, r=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Metrics Table ──
        st.markdown("#### 📋 Detailed Metrics Table")
        rows = []
        for model_name in models:
            m_data = load_metrics(model_name)
            if m_data:
                row = {"Model": model_name}
                row.update({k.capitalize(): f"{comparison[model_name][k]:.4f}"
                             for k in metrics_keys})
                rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── ROC / AUC Curves ──
        st.markdown("---")
        st.markdown("#### 📈 ROC / AUC Curves")
        roc_path = os.path.join(RESULTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC curves per entity type for all three models", use_container_width=True)
            st.caption(
                "Each curve shows the True Positive Rate vs False Positive Rate for one entity type "
                "(one-vs-rest). AUC closer to 1.0 = better discrimination. "
                "The bottom-left panel overlays macro-average ROC for all models. "
                "The bar chart compares per-entity AUC side-by-side."
            )
        else:
            st.info("ROC curves not generated yet. Run `python plot_roc.py` from the project folder.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 📉 Training Loss Curves")
    st.caption("Training and validation loss for BiLSTM-CRF and BioBERT models.")

    bilstm_data = load_metrics("BiLSTM-CRF")
    biobert_data = load_metrics("BioBERT")

    has_data = False

    if bilstm_data and bilstm_data.get("train_losses"):
        has_data = True
        st.markdown("#### BiLSTM-CRF Training Loss")
        train_l = bilstm_data["train_losses"]
        val_l = bilstm_data.get("val_losses", [])
        epochs = list(range(1, len(train_l) + 1))

        fig_bilstm = go.Figure()
        fig_bilstm.add_trace(go.Scatter(
            x=epochs, y=train_l, mode="lines+markers",
            name="Train Loss", line=dict(color="#4dabf7", width=2),
            marker=dict(size=6),
        ))
        if val_l:
            fig_bilstm.add_trace(go.Scatter(
                x=list(range(1, len(val_l) + 1)), y=val_l,
                mode="lines+markers", name="Val Loss",
                line=dict(color="#FF6B6B", width=2, dash="dash"),
                marker=dict(size=6),
            ))
        fig_bilstm.update_layout(
            height=350, xaxis_title="Epoch", yaxis_title="Loss",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h"),
            margin=dict(t=20, b=40),
        )
        fig_bilstm.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_bilstm, use_container_width=True)

    if biobert_data and biobert_data.get("train_losses"):
        has_data = True
        st.markdown("#### BioBERT Training Loss")
        train_l = biobert_data["train_losses"]
        val_l = biobert_data.get("val_losses", [])
        epochs = list(range(1, len(train_l) + 1))

        fig_bert = go.Figure()
        fig_bert.add_trace(go.Scatter(
            x=epochs, y=train_l, mode="lines+markers",
            name="Train Loss", line=dict(color="#69db7c", width=2),
            marker=dict(size=6),
        ))
        if val_l:
            fig_bert.add_trace(go.Scatter(
                x=list(range(1, len(val_l) + 1)), y=val_l,
                mode="lines+markers", name="Val Loss",
                line=dict(color="#ffa94d", width=2, dash="dash"),
                marker=dict(size=6),
            ))
        fig_bert.update_layout(
            height=350, xaxis_title="Epoch", yaxis_title="Loss",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h"),
            margin=dict(t=20, b=40),
        )
        fig_bert.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_bert, use_container_width=True)

    if not has_data:
        st.info("No training loss data found. Run `python train.py` to train the models first.")

    # ── Trade-off Summary ──
    st.markdown("---")
    st.markdown("### ⚖️ Model Trade-offs")
    tradeoff_data = {
        "Model": ["Rule-Based", "BiLSTM-CRF", "BioBERT"],
        "Training Time": ["None", "Minutes", "Hours"],
        "Inference Speed": ["Very Fast", "Fast", "Moderate"],
        "Generalization": ["Low", "Medium", "High"],
        "Interpretability": ["High", "Medium", "Low"],
        "Data Required": ["None", "Moderate", "Large"],
        "Computational Cost": ["Minimal", "Low-Medium", "High"],
    }
    st.dataframe(pd.DataFrame(tradeoff_data), use_container_width=True, hide_index=True)
