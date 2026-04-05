# ============================================================
# src/model_training.py
# News Sentinel: Model Training, Evaluation & Persistence
# ============================================================

import logging
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


# ──────────────────────────────────────────────
# 1. EVALUATION UTILITIES
# ──────────────────────────────────────────────

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
    class_names: list = None,
) -> dict:
    """
    Comprehensive model evaluation.
    Returns metrics dict; logs classification report.
    """
    y_pred = model.predict(X_test)

    # Probability scores (if supported)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        y_prob = None
        auc = None

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc": round(auc, 4) if auc else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f" {model_name} — Evaluation Results")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "model_name"):
            logger.info(f"  {k:20s}: {v}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    return metrics


def plot_confusion_matrix(
    y_true, y_pred,
    class_names: list,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> plt.Figure:
    """Plot a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved → {save_path}")

    return fig


# ──────────────────────────────────────────────
# 2. FAKE NEWS DETECTION MODELS
# ──────────────────────────────────────────────

class FakeNewsTrainer:
    """
    Trains Logistic Regression, Naive Bayes, and LinearSVC
    for binary fake news detection (FAKE vs REAL).
    """

    MODELS = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "naive_bayes": ComplementNB(alpha=0.1),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, class_weight="balanced", random_state=42),
            cv=3,
        ),
    }

    def __init__(self, models_dir: str = "models/", static_dir: str = "static/"):
        self.models_dir = Path(models_dir)
        self.static_dir = Path(static_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models = {}
        self.all_metrics = []
        self.best_model_name = None

    def train_all(
        self,
        X_train, X_test,
        y_train, y_test,
        class_names: list = ["REAL", "FAKE"],
    ) -> dict:
        """
        Train all classifiers and evaluate them.
        Returns dict of model_name → metrics.
        """
        results = {}

        for name, model in self.MODELS.items():
            logger.info(f"\nTraining: {name.replace('_', ' ').title()}...")
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, model_name=name, class_names=class_names)
            results[name] = metrics
            self.trained_models[name] = model
            self.all_metrics.append(metrics)

            # Confusion matrix visualization
            y_pred = model.predict(X_test)
            plot_confusion_matrix(
                y_test, y_pred,
                class_names=class_names,
                title=f"Confusion Matrix — {name.replace('_', ' ').title()}",
                save_path=str(self.static_dir / f"cm_fake_{name}.png"),
            )
            plt.close("all")

        # Determine best model by F1-score
        self.best_model_name = max(results, key=lambda k: results[k]["f1_score"])
        logger.info(f"\n✓ Best fake news model: {self.best_model_name} (F1={results[self.best_model_name]['f1_score']})")

        return results

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """Run stratified cross-validation on all models."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_results = {}

        for name, model in self.MODELS.items():
            scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
            cv_results[name] = {
                "mean_f1": round(scores.mean(), 4),
                "std_f1": round(scores.std(), 4),
            }
            logger.info(f"CV {name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")

        return cv_results

    def save_models(self):
        """Persist all trained models to disk."""
        for name, model in self.trained_models.items():
            path = self.models_dir / f"fake_{name}.joblib"
            joblib.dump(model, path)
            logger.info(f"Model saved → {path}")

        # Save best model reference
        best_path = self.models_dir / "fake_best_model.joblib"
        joblib.dump(self.trained_models[self.best_model_name], best_path)
        logger.info(f"Best model saved → {best_path}")

    def plot_model_comparison(self, results: dict) -> plt.Figure:
        """Bar chart comparing all models across key metrics."""
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
        model_names = list(results.keys())

        x = np.arange(len(model_names))
        width = 0.2
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, metric in enumerate(metrics_to_plot):
            values = [results[m][metric] for m in model_names]
            bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(), color=colors[i], alpha=0.85)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8
                )

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison — Fake News Detection", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.replace("_", "\n").title() for m in model_names], fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        save_path = self.static_dir / "model_comparison_fake.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Model comparison chart saved → {save_path}")
        return fig


# ──────────────────────────────────────────────
# 3. SENTIMENT ANALYSIS MODELS
# ──────────────────────────────────────────────

class SentimentTrainer:
    """
    Trains classifiers for 3-class sentiment (positive/neutral/negative).
    """

    MODELS = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, multi_class="multinomial",
            solver="lbfgs", class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "naive_bayes": ComplementNB(alpha=0.3),
    }

    def __init__(self, models_dir: str = "models/", static_dir: str = "static/"):
        self.models_dir = Path(models_dir)
        self.static_dir = Path(static_dir)
        self.trained_models = {}
        self.best_model_name = None

    def train_all(
        self,
        X_train, X_test,
        y_train, y_test,
        class_names: list = ["negative", "neutral", "positive"],
    ) -> dict:
        results = {}

        for name, model in self.MODELS.items():
            logger.info(f"\nTraining sentiment model: {name}...")
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, model_name=f"sentiment_{name}", class_names=class_names)
            results[name] = metrics
            self.trained_models[name] = model

            y_pred = model.predict(X_test)
            plot_confusion_matrix(
                y_test, y_pred,
                class_names=class_names,
                title=f"Sentiment Confusion Matrix — {name.replace('_', ' ').title()}",
                save_path=str(self.static_dir / f"cm_sentiment_{name}.png"),
            )
            plt.close("all")

        self.best_model_name = max(results, key=lambda k: results[k]["f1_score"])
        logger.info(f"\n✓ Best sentiment model: {self.best_model_name}")
        return results

    def save_models(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.trained_models.items():
            path = self.models_dir / f"sentiment_{name}.joblib"
            joblib.dump(model, path)
            logger.info(f"Model saved → {path}")

        best_path = self.models_dir / "sentiment_best_model.joblib"
        joblib.dump(self.trained_models[self.best_model_name], best_path)
        logger.info(f"Best sentiment model saved → {best_path}")


# ──────────────────────────────────────────────
# 4. VISUALIZATIONS
# ──────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = "static/sentiment_distribution.png") -> plt.Figure:
    """Donut chart of sentiment class distribution."""
    counts = df["sentiment"].value_counts()
    colors = {"positive": "#4CAF50", "neutral": "#2196F3", "negative": "#F44336"}
    color_list = [colors.get(c, "#9E9E9E") for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index.str.title(),
        autopct="%1.1f%%",
        colors=color_list,
        startangle=90,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight("bold")
    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Sentiment distribution saved → {save_path}")
    return fig


def plot_fake_real_distribution(df: pd.DataFrame, save_path: str = "static/fake_real_distribution.png") -> plt.Figure:
    """Bar chart of fake vs real article counts."""
    counts = df["label"].value_counts()
    colors = {"REAL": "#4CAF50", "FAKE": "#F44336"}

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[colors.get(c, "#9E9E9E") for c in counts.index], width=0.5, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{val:,}", ha="center", fontsize=12, fontweight="bold")
    ax.set_title("Fake vs Real Articles", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlabel("Label", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Fake/Real distribution saved → {save_path}")
    return fig


def plot_word_frequency(freq_df: pd.DataFrame, title: str = "Top Words", save_path: str = "static/word_frequency.png") -> plt.Figure:
    """Horizontal bar chart of word frequencies."""
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("Blues_r", len(freq_df))
    bars = ax.barh(freq_df["word"][::-1], freq_df["frequency"][::-1], color=palette)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_ylabel("Word", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Word frequency chart saved → {save_path}")
    return fig


def plot_wordcloud(texts: pd.Series, title: str = "Word Cloud", save_path: str = "static/wordcloud.png") -> plt.Figure:
    """Generate a word cloud from texts."""
    try:
        from wordcloud import WordCloud

        text = " ".join(texts.dropna().astype(str))
        wc = WordCloud(
            width=900, height=450,
            background_color="white",
            colormap="Blues",
            max_words=150,
            collocations=False,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Word cloud saved → {save_path}")
        return fig
    except ImportError:
        logger.warning("wordcloud not installed — skipping word cloud generation.")
        return plt.figure()


# ──────────────────────────────────────────────
# 5. MASTER TRAINING PIPELINE
# ──────────────────────────────────────────────

def run_training_pipeline(data_dir: str = "data/", models_dir: str = "models/", static_dir: str = "static/") -> dict:
    """
    End-to-end training pipeline:
    1. Load processed data
    2. Build features
    3. Train fake news models
    4. Train sentiment models
    5. Save models + visualizations
    """
    import sys
    sys.path.insert(0, ".")
    from src.data_preprocessing import run_preprocessing_pipeline
    from src.feature_engineering import TFIDFFeatureBuilder, get_word_frequencies

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(static_dir).mkdir(parents=True, exist_ok=True)

    # ── Load Data ──
    processed_path = Path(data_dir) / "processed_data.csv"
    if processed_path.exists():
        logger.info(f"Loading pre-processed data from {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        df = run_preprocessing_pipeline(data_dir)

    # ── Visualizations (EDA) ──
    logger.info("Generating EDA visualizations...")
    plot_sentiment_distribution(df, save_path=f"{static_dir}/sentiment_distribution.png")
    plot_fake_real_distribution(df, save_path=f"{static_dir}/fake_real_distribution.png")
    plt.close("all")

    freq_df = get_word_frequencies(df["clean_text"], top_n=25)
    plot_word_frequency(freq_df, title="Top 25 Words (All Articles)", save_path=f"{static_dir}/word_frequency_all.png")
    plt.close("all")

    plot_wordcloud(df["clean_text"], title="Most Common Words", save_path=f"{static_dir}/wordcloud_all.png")
    plt.close("all")

    # ── Feature Engineering ──
    logger.info("\nBuilding TF-IDF features...")
    tfidf = TFIDFFeatureBuilder(max_features=50_000, ngram_range=(1, 2))

    X = tfidf.fit_transform(df["clean_text"])
    tfidf.save(f"{models_dir}/tfidf_vectorizer.joblib")

    # ── Fake News Split ──
    y_fake = df["label"].values
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X, y_fake, test_size=0.2, random_state=42, stratify=y_fake
    )

    # ── Sentiment Split ──
    y_sent = df["sentiment"].values
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )

    # ── Train Fake News Models ──
    logger.info("\n── Training Fake News Classifiers ──")
    fake_trainer = FakeNewsTrainer(models_dir=models_dir, static_dir=static_dir)
    fake_results = fake_trainer.train_all(
        X_train_f, X_test_f, y_train_f, y_test_f, class_names=["REAL", "FAKE"]
    )
    fake_trainer.save_models()
    fake_trainer.plot_model_comparison(fake_results)
    plt.close("all")

    # ── Train Sentiment Models ──
    logger.info("\n── Training Sentiment Classifiers ──")
    sent_trainer = SentimentTrainer(models_dir=models_dir, static_dir=static_dir)
    sent_results = sent_trainer.train_all(
        X_train_s, X_test_s, y_train_s, y_test_s,
        class_names=sorted(df["sentiment"].unique()),
    )
    sent_trainer.save_models()

    # ── Save Summary ──
    summary = {
        "trained_at": datetime.now().isoformat(),
        "dataset_size": len(df),
        "fake_news_results": {k: {m: v for m, v in res.items() if m != "confusion_matrix"} for k, res in fake_results.items()},
        "sentiment_results": {k: {m: v for m, v in res.items() if m != "confusion_matrix"} for k, res in sent_results.items()},
        "best_fake_model": fake_trainer.best_model_name,
        "best_sentiment_model": sent_trainer.best_model_name,
    }
    summary_path = Path(models_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nTraining summary saved → {summary_path}")
    logger.info("\n✓ Training pipeline complete!")

    return summary


# ──────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────

if __name__ == "__main__":
    summary = run_training_pipeline(
        data_dir="data/",
        models_dir="models/",
        static_dir="static/",
    )
    print("\n── Training Summary ──")
    print(json.dumps({k: v for k, v in summary.items() if "results" not in k}, indent=2))
