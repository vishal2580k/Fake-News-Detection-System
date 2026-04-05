# ============================================================
# src/prediction.py
# News Sentinel: Inference Engine for Text & URLs
# ============================================================

import logging
import re
import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. RESULT DATACLASS
# ──────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Structured container for all analysis outputs."""
    # Input
    input_text: str
    source_url: Optional[str] = None

    # Fake news prediction
    fake_label: str = ""          # "FAKE" or "REAL"
    fake_confidence: float = 0.0  # 0.0 – 1.0
    fake_proba: dict = field(default_factory=dict)

    # Sentiment prediction
    sentiment_label: str = ""          # "positive" / "neutral" / "negative"
    sentiment_confidence: float = 0.0
    sentiment_proba: dict = field(default_factory=dict)

    # Meta
    word_count: int = 0
    clean_text_preview: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "fake_label": self.fake_label,
            "fake_confidence": round(self.fake_confidence * 100, 1),
            "sentiment_label": self.sentiment_label,
            "sentiment_confidence": round(self.sentiment_confidence * 100, 1),
            "word_count": self.word_count,
            "source_url": self.source_url,
            "error": self.error,
        }


# ──────────────────────────────────────────────
# 2. URL ARTICLE FETCHER (BONUS feature)
# ──────────────────────────────────────────────

def fetch_article_from_url(url: str) -> tuple[str, str]:
    """
    Fetch and extract article text from a URL.
    Returns (title, body_text). Falls back to requests + BeautifulSoup
    if newspaper3k is unavailable.
    """
    # Validate URL
    if not re.match(r"https?://", url):
        raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")

    # Try newspaper3k first (best quality extraction)
    try:
        from newspaper import Article

        article = Article(url)
        article.download()
        article.parse()
        title = article.title or ""
        text = article.text or ""
        if text.strip():
            logger.info(f"Extracted article via newspaper3k: {len(text)} chars")
            return title, f"{title}\n\n{text}"
    except ImportError:
        logger.info("newspaper3k not available — falling back to BeautifulSoup")
    except Exception as e:
        logger.warning(f"newspaper3k failed: {e} — trying BeautifulSoup")

    # Fallback: requests + BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsSentinel/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove nav, footer, scripts, ads
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Get title
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Get article paragraphs
        paragraphs = soup.find_all("p")
        body = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)

        if not body:
            body = soup.get_text(separator=" ", strip=True)[:3000]

        logger.info(f"Extracted article via BeautifulSoup: {len(body)} chars")
        return title, f"{title}\n\n{body}"

    except Exception as e:
        raise RuntimeError(f"Could not fetch article from URL: {e}")


# ──────────────────────────────────────────────
# 3. PREDICTION ENGINE
# ──────────────────────────────────────────────

class NewsSentinelPredictor:
    """
    Loads trained models and performs inference on raw text or URLs.
    Handles preprocessing, feature extraction, and confidence scoring.
    """

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = Path(models_dir)
        self.tfidf = None
        self.fake_model = None
        self.sentiment_model = None
        self.preprocessor = None
        self._loaded = False

    def load(self) -> "NewsSentinelPredictor":
        """Load all models from disk. Call once before predicting."""
        from src.feature_engineering import TFIDFFeatureBuilder
        from src.data_preprocessing import TextPreprocessor

        logger.info(f"Loading models from: {self.models_dir}")

        # TF-IDF vectorizer
        tfidf_path = self.models_dir / "tfidf_vectorizer.joblib"
        if not tfidf_path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {tfidf_path}. "
                "Please run `python src/model_training.py` first."
            )
        self.tfidf = TFIDFFeatureBuilder.load(str(tfidf_path))

        # Fake news model
        fake_path = self.models_dir / "fake_best_model.joblib"
        if not fake_path.exists():
            raise FileNotFoundError(f"Fake news model not found at {fake_path}.")
        self.fake_model = joblib.load(fake_path)
        logger.info(f"Fake news model loaded ← {fake_path.name}")

        # Sentiment model
        sent_path = self.models_dir / "sentiment_best_model.joblib"
        if not sent_path.exists():
            raise FileNotFoundError(f"Sentiment model not found at {sent_path}.")
        self.sentiment_model = joblib.load(sent_path)
        logger.info(f"Sentiment model loaded ← {sent_path.name}")

        # Preprocessor
        self.preprocessor = TextPreprocessor()

        self._loaded = True
        logger.info("All models loaded and ready for inference.")
        return self

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call .load() first.")

    def _clean_text(self, text: str) -> str:
        """Preprocess raw text."""
        return self.preprocessor.preprocess(text)

    def _get_probabilities(self, model, features) -> dict:
        """Extract class probabilities from a model."""
        try:
            proba = model.predict_proba(features)[0]
            classes = model.classes_
            return {str(cls): round(float(p), 4) for cls, p in zip(classes, proba)}
        except Exception:
            return {}

    def predict_text(self, text: str) -> PredictionResult:
        """
        Run full analysis on a raw text string.
        Returns a PredictionResult with fake + sentiment predictions.
        """
        self._require_loaded()

        result = PredictionResult(input_text=text)

        if not text or not text.strip():
            result.error = "Empty text provided."
            return result

        try:
            # Preprocess
            clean = self._clean_text(text)
            result.clean_text_preview = clean[:200]
            result.word_count = len(text.split())

            if not clean.strip():
                result.error = "Text too short or contains no meaningful words after cleaning."
                return result

            # Vectorize
            features = self.tfidf.transform([clean])

            # ── Fake news prediction ──
            fake_pred = self.fake_model.predict(features)[0]
            fake_proba = self._get_probabilities(self.fake_model, features)
            result.fake_label = str(fake_pred)
            result.fake_proba = fake_proba
            # Confidence = probability of the predicted class
            result.fake_confidence = fake_proba.get(str(fake_pred), 0.5)

            # ── Sentiment prediction ──
            sent_pred = self.sentiment_model.predict(features)[0]
            sent_proba = self._get_probabilities(self.sentiment_model, features)
            result.sentiment_label = str(sent_pred)
            result.sentiment_proba = sent_proba
            result.sentiment_confidence = sent_proba.get(str(sent_pred), 0.5)

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            result.error = str(e)

        return result

    def predict_url(self, url: str) -> PredictionResult:
        """
        Fetch article from URL and analyze it.
        """
        self._require_loaded()

        result = PredictionResult(input_text="", source_url=url)

        try:
            title, text = fetch_article_from_url(url)
            result.input_text = text
            logger.info(f"Fetched {len(text)} chars from: {url}")
        except Exception as e:
            result.error = f"URL fetch failed: {e}"
            logger.error(result.error)
            return result

        # Re-use text prediction
        text_result = self.predict_text(text)
        text_result.source_url = url
        return text_result

    def batch_predict(self, texts: list[str]) -> list[PredictionResult]:
        """Run predictions on a list of texts (more efficient than one-by-one)."""
        self._require_loaded()
        results = []

        if not texts:
            return results

        # Preprocess all
        cleaned = [self._clean_text(t) for t in texts]
        valid_mask = [bool(c.strip()) for c in cleaned]
        valid_cleaned = [c for c, v in zip(cleaned, valid_mask) if v]

        if valid_cleaned:
            features = self.tfidf.transform(valid_cleaned)
            fake_preds = self.fake_model.predict(features)
            sent_preds = self.sentiment_model.predict(features)
            try:
                fake_probas = self.fake_model.predict_proba(features)
                sent_probas = self.sentiment_model.predict_proba(features)
            except Exception:
                fake_probas = np.zeros((len(valid_cleaned), 2))
                sent_probas = np.zeros((len(valid_cleaned), 3))

        valid_idx = 0
        for i, (text, is_valid) in enumerate(zip(texts, valid_mask)):
            r = PredictionResult(input_text=text, word_count=len(text.split()))
            if is_valid:
                r.fake_label = str(fake_preds[valid_idx])
                r.sentiment_label = str(sent_preds[valid_idx])
                r.fake_confidence = float(fake_probas[valid_idx].max())
                r.sentiment_confidence = float(sent_probas[valid_idx].max())
                valid_idx += 1
            else:
                r.error = "Empty or invalid text"
            results.append(r)

        return results


# ──────────────────────────────────────────────
# DEMO PREDICTOR (auto-trains if needed)
# ──────────────────────────────────────────────

def get_predictor(models_dir: str = "models/", auto_train: bool = True) -> NewsSentinelPredictor:
    """
    Load predictor, training models if they don't exist yet.
    Convenience function for the Streamlit app.
    """
    predictor = NewsSentinelPredictor(models_dir=models_dir)
    tfidf_path = Path(models_dir) / "tfidf_vectorizer.joblib"

    if not tfidf_path.exists() and auto_train:
        logger.info("No trained models found — running training pipeline...")
        from src.model_training import run_training_pipeline
        run_training_pipeline(data_dir="data/", models_dir=models_dir, static_dir="static/")

    predictor.load()
    return predictor


# ──────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    predictor = get_predictor(models_dir="models/", auto_train=True)

    test_articles = [
        "Scientists at MIT have published new research on climate change, showing measurable increases in global temperature over the past decade.",
        "SHOCKING: Government secretly admits vaccines contain microchips to track citizens — whistleblowers reveal the truth!",
        "The Federal Reserve announced a 0.25% interest rate cut following its latest policy meeting with broad committee support.",
    ]

    print("\n── News Sentinel Predictions ──\n")
    for article in test_articles:
        result = predictor.predict_text(article)
        print(f"Text: {article[:70]}...")
        print(f"  Fake/Real  : {result.fake_label} ({result.fake_confidence*100:.1f}% confidence)")
        print(f"  Sentiment  : {result.sentiment_label} ({result.sentiment_confidence*100:.1f}% confidence)")
        if result.error:
            print(f"  Error      : {result.error}")
        print()
