"""
Nigerian Social Media Sentiment Analysis — Streamlit Demo
-----------------------------------------------------------
Loads the model + vectoriser exported by the Kaggle notebook
(`sentiment_model.joblib`, `vectorizer.joblib`, `label_encoder.joblib`,
`model_metadata.json`) and serves real-time and batch sentiment predictions.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this folder (app.py, requirements.txt, model/) to a GitHub repo.
    2. Go to https://share.streamlit.io -> New app -> point to app.py.
"""

import os
import re
import json
import string

import joblib
import emoji
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Nigerian Social Media Sentiment Analyzer",
    page_icon="🇳🇬",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

LABEL_COLORS = {
    "positive": "#2e7d32",
    "negative": "#c62828",
    "neutral": "#616161",
}
LABEL_EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}


# ----------------------------------------------------------------------
# Preprocessing pipeline — mirrors Chapter 3.4 / the Kaggle notebook exactly
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_nltk_data():
    for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True


NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing",
    "neither", "nor", "nowhere", "without", "cannot", "cant", "can't",
}

PIDGIN_LEXICON = {
    "wahala", "abeg", "sha", "oyinbo", "japa", "sabi", "gist", "waka",
    "wetin", "dey", "una", "wahalaaaa", "naija", "omo", "shey", "kuku",
    "abi", "sef", "oga", "yarn", "vex", "gbege", "jara", "chop",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"\bRT\b", flags=re.IGNORECASE)
HTML_ENTITY_MAP = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"}
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
NUMBER_RE = re.compile(r"\b\d+\b")
SENTIMENT_PUNCT_KEEP = {"!", "?"}


@st.cache_resource(show_spinner=False)
def get_nlp_tools():
    ensure_nltk_data()
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")) - NEGATION_WORDS
    return tokenizer, lemmatizer, stop_words


def decode_html_entities(text):
    for k, v in HTML_ENTITY_MAP.items():
        text = text.replace(k, v)
    return text


def clean_text(text):
    text = str(text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = RT_RE.sub("", text)
    text = decode_html_entities(text)
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = text.replace("_", " ")
    return text


def normalise_text(text):
    text = text.lower()
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)
    kept = []
    for ch in text:
        if ch in SENTIMENT_PUNCT_KEEP:
            kept.append(ch)
        elif ch in string.punctuation:
            continue
        else:
            kept.append(ch)
    text = "".join(kept)
    text = NUMBER_RE.sub(" <num> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text):
    tokenizer, lemmatizer, stop_words = get_nlp_tools()
    text = clean_text(text)
    text = normalise_text(text)
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if (t not in stop_words) and (len(t) > 1 or t in SENTIMENT_PUNCT_KEEP)]
    out = []
    for t in tokens:
        if t.startswith("#") or t in PIDGIN_LEXICON or t in SENTIMENT_PUNCT_KEEP:
            out.append(t)
        else:
            out.append(lemmatizer.lemmatize(t))
    return " ".join(out)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    enc_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")

    missing = [p for p in [model_path, vec_path, enc_path] if not os.path.exists(p)]
    if missing:
        return None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    encoder = joblib.load(enc_path)
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    return {"model": model, "vectorizer": vectorizer, "encoder": encoder, "metadata": metadata}


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.atleast_2d(scores)
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    raise ValueError("Model exposes neither predict_proba nor decision_function")


def predict_classical(text, artifacts):
    cleaned = preprocess(text)
    X_vec = artifacts["vectorizer"].transform([cleaned])
    pred_idx = artifacts["model"].predict(X_vec)[0]
    label = artifacts["encoder"].inverse_transform([pred_idx])[0]
    proba = get_scores(artifacts["model"], X_vec)[0]
    proba_dict = dict(zip(artifacts["encoder"].classes_, proba))
    return label, proba_dict, cleaned


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🇳🇬 Nigerian Social Media Sentiment Analyzer")
st.caption(
    "Live demo of the classifier trained in the Kaggle notebook — classifies posts "
    "about Nigerian public issues (elections, insecurity, economy, police brutality, "
    "health) as **positive**, **negative**, or **neutral**."
)

artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "Model artifacts not found. Copy `sentiment_model.pkl`, `vectorizer.pkl`, "
        "`label_encoder.pkl` and `model_metadata.json` (produced on Kaggle notebook's "
        "final cells, downloaded from `/kaggle/working`) into the `model/` folder next to "
        "this app, then rerun."
    )
    st.stop()

tab_single, tab_batch, tab_metrics, tab_about = st.tabs(
    ["🔎 Single Post", "📄 Batch (CSV)", "📊 Model Performance", "ℹ️ About"]
)

# ---------------- Single post ----------------
with tab_single:
    st.subheader("Analyze a single post")

    examples = [
        "This fuel subsidy removal is causing serious hardship for ordinary Nigerians #FuelSubsidyRemoval",
        "Big up to INEC for a smooth process today, well done!",
        "There will be a press briefing on the health sector reforms tomorrow.",
        "Wahala dey much with this insecurity for our region, government must act now!",
    ]

    def set_sample_text(sample):
        st.session_state["single_post_text"] = sample

    cols = st.columns(len(examples))
    for i, (c, ex) in enumerate(zip(cols, examples)):
        c.button(
            ex[:28] + "…",
            use_container_width=True,
            key=f"sample_{i}",
            on_click=set_sample_text,
            args=(ex,),
        )

    text_input = st.text_area(
        "Enter a tweet / post text",
        height=120,
        placeholder="e.g. Tinubu's economic policies are hurting the common man o!",
        key="single_post_text",
    )

    def render_result(title, label, proba, cleaned, key):
        color = LABEL_COLORS.get(label, "#333")
        emoji_icon = LABEL_EMOJI.get(label, "")
        st.markdown(f"##### {title}")
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown(
                f"<div style='padding:1.2rem;border-radius:12px;background:{color}20;"
                f"border:2px solid {color};text-align:center;'>"
                f"<span style='font-size:2.2rem'>{emoji_icon}</span><br>"
                f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>"
                f"{label.upper()}</span><br>"
                f"<span style='color:#555'>confidence: {max(proba.values()):.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Text fed to the model"):
                st.code(cleaned or "(empty after cleaning)")
        with c2:
            proba_df = pd.DataFrame(
                {"sentiment": list(proba.keys()), "probability": [float(v) for v in proba.values()]}
            ).sort_values("probability", ascending=True)
            fig = px.bar(
                proba_df, x="probability", y="sentiment", orientation="h",
                color="sentiment", color_discrete_map=LABEL_COLORS,
                range_x=[0, 1], text_auto=".1%",
            )
            fig.update_layout(showlegend=False, height=260, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key=key)

    if st.button("Analyze Sentiment", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                label, proba, cleaned = predict_classical(text_input, artifacts)
            except Exception as e:
                st.error(f"Classical model failed: {e}")
            else:
                render_result("Classical ML model", label, proba, cleaned, key="proba_classical")

# ---------------- Batch ----------------
with tab_batch:
    st.subheader("Batch-analyze a CSV of posts")
    st.caption("CSV must contain a text column (default name: `text`).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    text_col = st.text_input("Text column name", value="text")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            batch_df = None

        if batch_df is not None:
            if text_col not in batch_df.columns:
                st.error(f"Column '{text_col}' not found. Available columns: {list(batch_df.columns)}")
            else:
                if st.button("Run batch prediction", type="primary"):
                    with st.spinner(f"Scoring {len(batch_df)} rows…"):
                        labels, confidences = [], []
                        for text in batch_df[text_col].astype(str):
                            label, proba, _ = predict_classical(text, artifacts)
                            labels.append(label)
                            confidences.append(float(max(proba.values())))
                        batch_df["predicted_sentiment"] = labels
                        batch_df["confidence"] = confidences

                    st.success("Done.")
                    st.dataframe(batch_df, use_container_width=True)

                    dist = batch_df["predicted_sentiment"].value_counts().reset_index()
                    dist.columns = ["sentiment", "count"]
                    fig = px.bar(
                        dist, x="sentiment", y="count", color="sentiment",
                        color_discrete_map=LABEL_COLORS, text_auto=True,
                    )
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=csv_bytes,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )

# ---------------- Metrics ----------------
with tab_metrics:
    st.subheader("Model performance (from training/evaluation on the held-out test set)")
    meta = artifacts["metadata"]
    if not meta:
        st.info("No `model_metadata.json` found — metrics unavailable. This does not affect predictions.")
    else:
        st.markdown(f"**Deployed model:** {meta.get('best_model', 'N/A')}  "
                    f"(&nbsp;features: {meta.get('feature_type', 'N/A').upper()}&nbsp;)")
        metrics_df = pd.DataFrame(meta.get("metrics", []))
        if "Features" in metrics_df.columns:
            metrics_df = metrics_df[
                metrics_df["Features"].astype(str).str.upper() != "TRANSFORMER"
            ]
        if "Model" in metrics_df.columns:
            metrics_df = metrics_df[
                ~metrics_df["Model"].astype(str).str.contains(
                    r"transformer|afriberta|afroxlmr", case=False, regex=True
                )
            ]
        if not metrics_df.empty:
            st.dataframe(
                metrics_df.style.format({c: "{:.3f}" for c in metrics_df.columns
                                         if pd.api.types.is_numeric_dtype(metrics_df[c])}, na_rep="—"),
                use_container_width=True,
            )
            melt_cols = [c for c in metrics_df.columns if c not in ("Model", "Features", "CV Macro-F1")]
            long_df = metrics_df.melt(id_vars=["Model"], value_vars=melt_cols,
                                       var_name="Metric", value_name="Score")
            fig = px.bar(long_df, x="Model", y="Score", color="Metric", barmode="group")
            fig.update_layout(yaxis_range=[0, 1], height=420)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- About ----------------
with tab_about:
    st.markdown(
        """
        ### About this app
        This app serves the machine-learning sentiment classifier trained on the
        methodology described in the documentation of the research *Sentiment Analysis of Nigerian
        Public Issues on Social Media*:

        - **Preprocessing:** URL/mention/RT stripping, HTML-entity decoding,
          emoji → text conversion, lowercasing, repeated-character normalisation,
          TweetTokenizer tokenisation, negation-aware stop-word removal, and
          WordNet lemmatisation with a Nigerian Pidgin/slang pass-through list.
        - **Features:** Bag-of-Words, word TF-IDF, and word + character n-gram
          TF-IDF (robust to Pidgin's inconsistent spelling).
        - **Classical models compared:** Multinomial Naive Bayes, Linear SVM,
          Logistic Regression, Random Forest — tuned with 5-fold cross-validation
          where the class-imbalance strategy (SMOTE applied *inside* each fold,
          or class weighting) is itself a tuned choice, and the winner is picked
          on a validation set, never the test set.
        - **Classes:** positive / negative / neutral.

        **Note:** predictions reflect the classical ML pipeline trained on the
        annotated corpus used at training time and are provided for research
        and demonstration purposes — not for high-stakes decision-making.
        """
    )
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 0.5rem; color: #777;">
        © 2026 Computer Science Project · Designed by Abdulaziz Ahmad Abdul
    </div>
    """,
    unsafe_allow_html=True,
)

