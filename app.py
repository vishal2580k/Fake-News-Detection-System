# ============================================================
# app/app.py
# News Sentinel — Streamlit Web Application
# ============================================================

import sys
import os
import time
import json
import logging
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="News Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Dark Intelligence Theme
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e2940 100%);
        border-right: 1px solid rgba(99,179,237,0.15);
    }

    /* ── Main Cards ── */
    .sentinel-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 12px 0;
        backdrop-filter: blur(12px);
        transition: border-color 0.3s ease;
    }
    .sentinel-card:hover {
        border-color: rgba(99,179,237,0.35);
    }

    /* ── Badge FAKE ── */
    .badge-fake {
        display: inline-block;
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 10px 28px;
        border-radius: 50px;
        letter-spacing: 2px;
        box-shadow: 0 0 25px rgba(255,68,68,0.45);
        text-align: center;
    }
    /* ── Badge REAL ── */
    .badge-real {
        display: inline-block;
        background: linear-gradient(135deg, #00cc66, #007744);
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 10px 28px;
        border-radius: 50px;
        letter-spacing: 2px;
        box-shadow: 0 0 25px rgba(0,204,102,0.45);
        text-align: center;
    }

    /* ── Sentiment Badges ── */
    .badge-positive { background: linear-gradient(135deg,#00cc66,#007744); color:white; padding:6px 18px; border-radius:50px; font-weight:600; }
    .badge-negative { background: linear-gradient(135deg,#ff4444,#cc0000); color:white; padding:6px 18px; border-radius:50px; font-weight:600; }
    .badge-neutral  { background: linear-gradient(135deg,#4a9eff,#2266cc); color:white; padding:6px 18px; border-radius:50px; font-weight:600; }

    /* ── Metric Cards ── */
    .metric-box {
        background: rgba(99,179,237,0.08);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #63b3ed; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

    /* ── Hero Title ── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #63b3ed, #4a9eff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: -8px;
    }

    /* ── Inputs ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(99,179,237,0.25) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(99,179,237,0.25) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #4a9eff, #2266cc) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px 32px !important;
        width: 100%;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(74,158,255,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 25px rgba(74,158,255,0.5) !important;
    }

    /* ── Dividers ── */
    hr { border-color: rgba(99,179,237,0.15) !important; }

    /* ── Progress bar ── */
    .stProgress > div > div > div { background: linear-gradient(90deg, #4a9eff, #a78bfa) !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        gap: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        border-radius: 8px !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(74,158,255,0.2) !important;
        color: #63b3ed !important;
    }

    /* ── Confidence bar ── */
    .conf-bar-container {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        height: 10px;
        margin: 6px 0;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    /* ── Alert boxes ── */
    .alert-warning {
        background: rgba(255,193,7,0.1);
        border: 1px solid rgba(255,193,7,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #ffc107;
    }
    .alert-success {
        background: rgba(0,204,102,0.1);
        border: 1px solid rgba(0,204,102,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #00cc66;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CACHED MODEL LOADER
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load models once and cache them for the session."""
    from src.prediction import get_predictor
    return get_predictor(models_dir="models/", auto_train=True)


@st.cache_data(show_spinner=False)
def load_training_summary():
    """Load training metrics from saved JSON."""
    path = Path("models/training_summary.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_processed_data():
    """Load processed dataset for dashboard."""
    path = Path("data/processed_data.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <div style="font-size:3rem">🛡️</div>
            <div style="font-size:1.3rem; font-weight:700; color:#63b3ed;">News Sentinel</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">AI-Powered News Analyzer</div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🔍 Analyze News", "📊 Dashboard", "📈 Model Metrics", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Model status
        tfidf_exists = (Path("models") / "tfidf_vectorizer.joblib").exists()
        status_color = "#00cc66" if tfidf_exists else "#ff4444"
        status_text = "Models Ready" if tfidf_exists else "Models Not Trained"
        st.markdown(f"""
        <div style="text-align:center; padding:10px;">
            <span style="color:{status_color}; font-size:0.85rem;">● {status_text}</span>
        </div>
        """, unsafe_allow_html=True)

        summary = load_training_summary()
        if summary:
            st.markdown(f"""
            <div style="color:#64748b; font-size:0.78rem; text-align:center;">
                Dataset: {summary.get('dataset_size', '—'):,} articles<br>
                Trained: {summary.get('trained_at', '—')[:10]}
            </div>
            """, unsafe_allow_html=True)

        return page


# ──────────────────────────────────────────────
# PAGE 1: ANALYZE NEWS
# ──────────────────────────────────────────────

def render_analyze_page(predictor):
    st.markdown("""
    <div class="hero-title">🛡️ News Sentinel</div>
    <div class="hero-subtitle">Detect fake news & analyze sentiment with AI — paste text or enter a URL</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_text, tab_url = st.tabs(["📝 Paste Text", "🔗 Analyze URL"])

    # ── TEXT TAB ──
    with tab_text:
        st.markdown('<div class="sentinel-card">', unsafe_allow_html=True)
        user_text = st.text_area(
            "Paste your news article here",
            height=220,
            placeholder="Paste the full news article text here...\n\nExample: 'Scientists at Harvard have published new research suggesting...'",
            label_visibility="collapsed",
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("🔍 Analyze News Article", key="btn_text")
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_btn:
            if not user_text.strip():
                st.warning("⚠️ Please paste some text to analyze.")
            else:
                with st.spinner("Analyzing article..."):
                    result = predictor.predict_text(user_text)
                render_results(result)

    # ── URL TAB ──
    with tab_url:
        st.markdown('<div class="sentinel-card">', unsafe_allow_html=True)
        url_input = st.text_input(
            "Enter news article URL",
            placeholder="https://www.example.com/news/article",
            label_visibility="collapsed",
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            url_btn = st.button("🔗 Fetch & Analyze URL", key="btn_url")
        st.markdown('</div>', unsafe_allow_html=True)

        if url_btn:
            if not url_input.strip():
                st.warning("⚠️ Please enter a URL.")
            else:
                with st.spinner(f"Fetching article from {url_input}..."):
                    result = predictor.predict_url(url_input)

                if result.error and "fetch" in result.error.lower():
                    st.error(f"❌ Could not fetch URL: {result.error}")
                    st.info("💡 Tip: Some websites block automated access. Try pasting the text directly.")
                else:
                    render_results(result)


def render_results(result):
    """Render the prediction results with visual styling."""
    if result.error and not result.fake_label:
        st.error(f"❌ Analysis failed: {result.error}")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Analysis Results")

    # ── Main verdict row ──
    col_fake, col_sent = st.columns(2)

    with col_fake:
        st.markdown('<div class="sentinel-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("**🔎 Fake News Detection**")
        badge_class = "badge-fake" if result.fake_label == "FAKE" else "badge-real"
        icon = "⚠️" if result.fake_label == "FAKE" else "✅"
        st.markdown(f'<div style="margin:16px 0;"><span class="{badge_class}">{icon} {result.fake_label}</span></div>', unsafe_allow_html=True)

        conf = result.fake_confidence * 100
        st.markdown(f"**Confidence: {conf:.1f}%**")
        st.progress(result.fake_confidence)

        # Probability breakdown
        if result.fake_proba:
            st.markdown("<small style='color:#64748b;'>Class Probabilities:</small>", unsafe_allow_html=True)
            for cls, prob in sorted(result.fake_proba.items(), key=lambda x: -x[1]):
                st.markdown(f"<small style='color:#94a3b8;'>{cls}: {prob*100:.1f}%</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sent:
        st.markdown('<div class="sentinel-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("**😊 Sentiment Analysis**")
        sent_colors = {"positive": "badge-positive", "negative": "badge-negative", "neutral": "badge-neutral"}
        sent_icons = {"positive": "😊", "negative": "😠", "neutral": "😐"}
        badge_cls = sent_colors.get(result.sentiment_label, "badge-neutral")
        s_icon = sent_icons.get(result.sentiment_label, "😐")
        st.markdown(f'<div style="margin:16px 0;"><span class="{badge_cls}">{s_icon} {result.sentiment_label.title()}</span></div>', unsafe_allow_html=True)

        s_conf = result.sentiment_confidence * 100
        st.markdown(f"**Confidence: {s_conf:.1f}%**")
        st.progress(result.sentiment_confidence)

        if result.sentiment_proba:
            st.markdown("<small style='color:#64748b;'>Class Probabilities:</small>", unsafe_allow_html=True)
            for cls, prob in sorted(result.sentiment_proba.items(), key=lambda x: -x[1]):
                st.markdown(f"<small style='color:#94a3b8;'>{cls}: {prob*100:.1f}%</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary stats ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{result.word_count}</div>
            <div class="metric-label">Words Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        verdict_icon = "🚨" if result.fake_label == "FAKE" else "✅"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{verdict_icon}</div>
            <div class="metric-label">Verdict: {result.fake_label}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        avg_conf = (result.fake_confidence + result.sentiment_confidence) / 2
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{avg_conf*100:.0f}%</div>
            <div class="metric-label">Avg. Confidence</div>
        </div>""", unsafe_allow_html=True)

    # ── Contextual warning ──
    st.markdown("<br>", unsafe_allow_html=True)
    if result.fake_label == "FAKE" and result.fake_confidence > 0.75:
        st.markdown("""
        <div class="alert-warning">
            ⚠️ <strong>High-confidence FAKE detection.</strong> This article shows strong indicators of misinformation.
            Cross-check with trusted news sources before sharing.
        </div>
        """, unsafe_allow_html=True)
    elif result.fake_label == "REAL" and result.fake_confidence > 0.75:
        st.markdown("""
        <div class="alert-success">
            ✅ <strong>High-confidence REAL detection.</strong> This article appears to be credible.
            Always verify with primary sources.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Moderate confidence. Use additional judgment and cross-reference with trusted sources.")

    # ── URL source ──
    if result.source_url:
        st.markdown(f"<small style='color:#64748b;'>Source: {result.source_url}</small>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE 2: DASHBOARD
# ──────────────────────────────────────────────

def render_dashboard():
    st.markdown('<div class="hero-title">📊 Dataset Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Exploratory data analysis of the training corpus</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    df = load_processed_data()
    if df is None:
        st.warning("No processed dataset found. Run the training pipeline first.")
        return

    # ── Top stats ──
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(df), "Total Articles"),
        (c2, (df["label"] == "FAKE").sum(), "Fake Articles"),
        (c3, (df["label"] == "REAL").sum(), "Real Articles"),
        (c4, df["sentiment"].nunique(), "Sentiment Classes"),
    ]:
        with col:
            col.markdown(f'<div class="metric-box"><div class="metric-value">{val:,}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    col_left, col_right = st.columns(2)

    with col_left:
        # Fake / Real donut
        counts = df["label"].value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.55,
            marker_colors=["#00cc66", "#ff4444"],
        ))
        fig.update_layout(
            title="Fake vs Real Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            showlegend=True,
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Sentiment bar
        sent_counts = df["sentiment"].value_counts()
        sent_colors = {"positive": "#00cc66", "neutral": "#4a9eff", "negative": "#ff4444"}
        colors = [sent_colors.get(s, "#9e9e9e") for s in sent_counts.index]
        fig = go.Figure(go.Bar(
            x=sent_counts.index.str.title().tolist(),
            y=sent_counts.values.tolist(),
            marker_color=colors,
        ))
        fig.update_layout(
            title="Sentiment Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Word frequency ──
    st.markdown("### 🔤 Top Keywords")
    from src.feature_engineering import get_word_frequencies_by_label

    freq_by_label = get_word_frequencies_by_label(df, text_col="clean_text", label_col="label", top_n=15)

    col_fake_wf, col_real_wf = st.columns(2)
    for col, lbl, color in [(col_fake_wf, "FAKE", "#ff4444"), (col_real_wf, "REAL", "#00cc66")]:
        with col:
            freq = freq_by_label.get(lbl, pd.DataFrame())
            if not freq.empty:
                fig = go.Figure(go.Bar(
                    x=freq["frequency"].tolist(),
                    y=freq["word"].tolist(),
                    orientation="h",
                    marker_color=color,
                ))
                fig.update_layout(
                    title=f"Top Words — {lbl} Articles",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    yaxis=dict(autorange="reversed"),
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Saved static charts ──
    static_path = Path("static")
    images = list(static_path.glob("*.png")) if static_path.exists() else []
    if images:
        st.markdown("### 🖼️ Generated Visualizations")
        img_cols = st.columns(min(3, len(images)))
        for i, img_path in enumerate(images[:6]):
            with img_cols[i % 3]:
                st.image(str(img_path), caption=img_path.stem.replace("_", " ").title(), use_container_width=True)


# ──────────────────────────────────────────────
# PAGE 3: MODEL METRICS
# ──────────────────────────────────────────────

def render_model_metrics():
    st.markdown('<div class="hero-title">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Training results across all classifiers</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    summary = load_training_summary()
    if not summary:
        st.warning("No training summary found. Run `python src/model_training.py` first.")
        return

    def render_metrics_table(results: dict, task: str):
        st.markdown(f"#### {task}")
        rows = []
        for model_name, metrics in results.items():
            row = {"Model": model_name.replace("_", " ").title()}
            row.update({k.replace("_", " ").title(): v for k, v in metrics.items() if k not in ("model_name", "confusion_matrix")})
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({col: "{:.4f}" for col in df.select_dtypes("float").columns})
                    .highlight_max(subset=df.select_dtypes("float").columns, color="#1a3a2a")
                    .set_properties(**{"background-color": "rgba(0,0,0,0)", "color": "#e2e8f0"}),
            use_container_width=True,
        )

    # ── Fake news metrics ──
    if "fake_news_results" in summary:
        render_metrics_table(summary["fake_news_results"], "🚨 Fake News Detection")
        best = summary.get("best_fake_model", "—")
        st.success(f"✅ Best model: **{best.replace('_', ' ').title()}**")

    st.markdown("---")

    # ── Sentiment metrics ──
    if "sentiment_results" in summary:
        render_metrics_table(summary["sentiment_results"], "😊 Sentiment Analysis")
        best = summary.get("best_sentiment_model", "—")
        st.success(f"✅ Best model: **{best.replace('_', ' ').title()}**")

    # ── Confusion matrices ──
    st.markdown("---")
    st.markdown("#### Confusion Matrices")
    cm_images = list(Path("static").glob("cm_*.png")) if Path("static").exists() else []
    if cm_images:
        cols = st.columns(min(3, len(cm_images)))
        for i, img in enumerate(cm_images[:6]):
            with cols[i % 3]:
                st.image(str(img), caption=img.stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.info("Confusion matrix images not found. They are generated during training.")


# ──────────────────────────────────────────────
# PAGE 4: ABOUT
# ──────────────────────────────────────────────

def render_about():
    st.markdown('<div class="hero-title">ℹ️ About News Sentinel</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sentinel-card">
        <h3 style="color:#63b3ed;">🛡️ What is News Sentinel?</h3>
        <p style="color:#94a3b8; line-height:1.8;">
            News Sentinel is an AI-powered system that combines <strong>NLP</strong>, 
            <strong>machine learning</strong>, and <strong>deep learning</strong> to analyze 
            news articles for two key dimensions:
        </p>
        <ul style="color:#94a3b8; line-height:2;">
            <li><strong style="color:#e2e8f0;">Fake News Detection</strong> — Binary classification (FAKE vs REAL)</li>
            <li><strong style="color:#e2e8f0;">Sentiment Analysis</strong> — 3-class classification (Positive / Neutral / Negative)</li>
        </ul>
    </div>

    <div class="sentinel-card">
        <h3 style="color:#63b3ed;">🧠 How It Works</h3>
        <ol style="color:#94a3b8; line-height:2.2;">
            <li><strong style="color:#e2e8f0;">Text Preprocessing</strong> — Lowercasing, URL/HTML removal, stopword filtering, lemmatization</li>
            <li><strong style="color:#e2e8f0;">Feature Engineering</strong> — TF-IDF vectorization with bigrams (50,000 features)</li>
            <li><strong style="color:#e2e8f0;">Classification</strong> — Logistic Regression, Naive Bayes, LinearSVC ensemble</li>
            <li><strong style="color:#e2e8f0;">Inference</strong> — Real-time prediction with confidence scores</li>
        </ol>
    </div>

    <div class="sentinel-card">
        <h3 style="color:#63b3ed;">⚙️ Tech Stack</h3>
        <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:10px;">
    """ + "".join([f'<span style="background:rgba(99,179,237,0.15);border:1px solid rgba(99,179,237,0.3);padding:4px 14px;border-radius:20px;font-size:0.85rem;color:#63b3ed;">{t}</span>' for t in [
        "Python 3.10+", "scikit-learn", "NLTK", "Transformers", "Streamlit",
        "Plotly", "Pandas", "NumPy", "Seaborn", "TextBlob", "Gensim", "BeautifulSoup",
    ]]) + """
        </div>
    </div>

    <div class="sentinel-card">
        <h3 style="color:#63b3ed;">⚠️ Disclaimer</h3>
        <p style="color:#94a3b8; line-height:1.8;">
            This tool uses machine learning models that are not 100% accurate. Predictions should 
            be used as a <strong>starting point for verification</strong>, not as definitive judgements. 
            Always cross-reference with trusted news sources.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "🔍 Analyze News":
        with st.spinner("Loading AI models..."):
            predictor = load_predictor()
        render_analyze_page(predictor)

    elif page == "📊 Dashboard":
        render_dashboard()

    elif page == "📈 Model Metrics":
        render_model_metrics()

    elif page == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()