# ============================================================
# src/feature_engineering.py
# News Sentinel: Feature Engineering (TF-IDF + Word2Vec)
# ============================================================

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


# ──────────────────────────────────────────────
# 1. TF-IDF FEATURES
# ──────────────────────────────────────────────

class TFIDFFeatureBuilder:
    """
    Builds TF-IDF feature matrices for fake news and sentiment tasks.
    Fits on training data and transforms train/test sets consistently.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,   # unigrams + bigrams
            min_df=min_df,             # ignore very rare terms
            max_df=max_df,             # ignore very common terms
            sublinear_tf=sublinear_tf, # apply log normalization to TF
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-z]{2,}\b",
        )
        self.is_fitted = False

    def fit(self, texts: pd.Series) -> "TFIDFFeatureBuilder":
        """Fit the vectorizer on training texts."""
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} samples...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"TF-IDF vocabulary size: {vocab_size:,} terms")
        return self

    def transform(self, texts: pd.Series):
        """Transform texts to TF-IDF sparse matrix."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transforming.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: pd.Series):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def get_top_features(self, n: int = 30) -> list[str]:
        """Return the top N features by IDF score (most discriminative terms)."""
        if not self.is_fitted:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        idf_scores = self.vectorizer.idf_
        top_idx = np.argsort(idf_scores)[::-1][:n]
        return [feature_names[i] for i in top_idx]

    def save(self, path: str = "models/tfidf_vectorizer.joblib"):
        """Persist the fitted vectorizer to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info(f"TF-IDF vectorizer saved → {path}")

    @classmethod
    def load(cls, path: str = "models/tfidf_vectorizer.joblib") -> "TFIDFFeatureBuilder":
        """Load a previously saved vectorizer."""
        obj = cls()
        obj.vectorizer = joblib.load(path)
        obj.is_fitted = True
        logger.info(f"TF-IDF vectorizer loaded ← {path}")
        return obj


# ──────────────────────────────────────────────
# 2. WORD2VEC FEATURES (optional / advanced)
# ──────────────────────────────────────────────

class Word2VecFeatureBuilder:
    """
    Builds dense document embeddings by averaging Word2Vec word vectors.
    Falls back gracefully if gensim is unavailable.
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, workers: int = 4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def _tokenize(self, texts: pd.Series) -> list[list[str]]:
        """Simple whitespace tokenizer (expects pre-cleaned text)."""
        return [str(t).split() for t in texts]

    def fit(self, texts: pd.Series) -> "Word2VecFeatureBuilder":
        """Train a Word2Vec model on the corpus."""
        try:
            from gensim.models import Word2Vec
        except ImportError:
            logger.warning("gensim not installed — Word2Vec unavailable. Install with: pip install gensim")
            return self

        sentences = self._tokenize(texts)
        logger.info(f"Training Word2Vec on {len(sentences)} documents...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10,
            seed=42,
        )
        logger.info(f"Word2Vec trained | Vocabulary: {len(self.model.wv):,} words")
        return self

    def _doc_vector(self, tokens: list[str]) -> np.ndarray:
        """Average word vectors for a document; return zeros if no words found."""
        vectors = [
            self.model.wv[word]
            for word in tokens
            if word in self.model.wv
        ]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)

    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts to document-level dense vectors."""
        if self.model is None:
            logger.warning("Word2Vec model not trained — returning zero matrix.")
            return np.zeros((len(texts), self.vector_size))
        sentences = self._tokenize(texts)
        return np.vstack([self._doc_vector(tokens) for tokens in sentences])

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str = "models/word2vec.model"):
        if self.model:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info(f"Word2Vec model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/word2vec.model") -> "Word2VecFeatureBuilder":
        try:
            from gensim.models import Word2Vec
            obj = cls()
            obj.model = Word2Vec.load(path)
            logger.info(f"Word2Vec model loaded ← {path}")
            return obj
        except Exception as e:
            logger.error(f"Could not load Word2Vec model: {e}")
            return cls()


# ──────────────────────────────────────────────
# 3. LABEL ENCODERS
# ──────────────────────────────────────────────

def build_label_encoders(df: pd.DataFrame) -> dict:
    """
    Build and return LabelEncoders for fake-news and sentiment tasks.
    """
    encoders = {}

    # Fake/Real encoder
    le_fake = LabelEncoder()
    le_fake.fit(df["label"].unique())
    encoders["fake"] = le_fake
    logger.info(f"Fake news classes: {le_fake.classes_}")

    # Sentiment encoder
    if "sentiment" in df.columns:
        le_sent = LabelEncoder()
        le_sent.fit(df["sentiment"].unique())
        encoders["sentiment"] = le_sent
        logger.info(f"Sentiment classes: {le_sent.classes_}")

    return encoders


# ──────────────────────────────────────────────
# 4. WORD FREQUENCY ANALYSIS (for visualizations)
# ──────────────────────────────────────────────

def get_word_frequencies(texts: pd.Series, top_n: int = 30) -> pd.DataFrame:
    """
    Compute word frequency counts for visualization.
    Returns a DataFrame with columns: word, frequency.
    """
    from collections import Counter

    all_words = " ".join(texts.dropna().astype(str)).split()
    counter = Counter(all_words)
    most_common = counter.most_common(top_n)

    return pd.DataFrame(most_common, columns=["word", "frequency"])


def get_word_frequencies_by_label(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Compute word frequencies separately for FAKE and REAL articles.
    Returns a dict: {"FAKE": df_freq, "REAL": df_freq}
    """
    result = {}
    for label in df[label_col].unique():
        subset = df[df[label_col] == label][text_col]
        result[label] = get_word_frequencies(subset, top_n=top_n)
    return result


# ──────────────────────────────────────────────
# 5. PIPELINE BUILDER
# ──────────────────────────────────────────────

def build_feature_pipeline(
    train_texts: pd.Series,
    use_word2vec: bool = False,
) -> dict:
    """
    Build complete feature engineering pipeline.
    Returns dict with fitted objects and training matrices.
    """
    result = {}

    # TF-IDF (primary)
    tfidf_builder = TFIDFFeatureBuilder(max_features=50_000, ngram_range=(1, 2))
    X_tfidf = tfidf_builder.fit_transform(train_texts)
    result["tfidf_builder"] = tfidf_builder
    result["X_tfidf"] = X_tfidf

    # Word2Vec (optional)
    if use_word2vec:
        w2v_builder = Word2VecFeatureBuilder(vector_size=100)
        X_w2v = w2v_builder.fit_transform(train_texts)
        result["w2v_builder"] = w2v_builder
        result["X_w2v"] = X_w2v

    return result


# ──────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: load processed data and engineer features
    import sys
    sys.path.insert(0, ".")
    from src.data_preprocessing import run_preprocessing_pipeline
    from sklearn.model_selection import train_test_split

    df = run_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = build_feature_pipeline(X_train, use_word2vec=False)
    tfidf_builder = pipeline["tfidf_builder"]

    print(f"\nTF-IDF training matrix shape: {pipeline['X_tfidf'].shape}")
    print(f"Top 10 discriminative features: {tfidf_builder.get_top_features(10)}")

    # Word frequencies
    freq_df = get_word_frequencies(df["clean_text"], top_n=10)
    print(f"\nTop 10 words overall:\n{freq_df}")

    # Save vectorizer
    tfidf_builder.save("models/tfidf_vectorizer.joblib")
