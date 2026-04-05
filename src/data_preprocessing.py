# ============================================================
# src/data_preprocessing.py
# News Sentinel: Data Loading & NLP Preprocessing Pipeline
# ============================================================

import re
import string
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
def download_nltk_data():
    """Download all required NLTK resources."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_data()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. DATASET LOADING
# ──────────────────────────────────────────────

def generate_sample_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic sample dataset when public datasets are unavailable.
    Returns a DataFrame with columns: text, label (REAL/FAKE), sentiment (positive/neutral/negative).
    """
    np.random.seed(random_state)

    fake_templates = [
        "SHOCKING: {topic} exposed in massive government cover-up — insiders reveal all",
        "You won't BELIEVE what {topic} is hiding from the public right now!",
        "BREAKING: Secret documents prove {topic} has been lying for years",
        "Doctors HATE this: {topic} cures everything with one simple trick",
        "Billionaires panic as {topic} leaks their secret plan to control society",
        "URGENT: {topic} linked to dangerous conspiracy involving world leaders",
        "Mainstream media SILENCED after {topic} truth goes viral",
        "Scientists baffled as {topic} defies all logic — cover-up suspected",
        "Anonymous whistleblower exposes {topic} in shocking new report",
        "Deep state fears {topic} — authorities desperately trying to hide the truth",
    ]

    real_templates = [
        "Researchers at {institution} publish findings on {topic} in peer-reviewed journal",
        "Government announces new policy regarding {topic} following expert consultation",
        "{institution} reports steady progress in {topic} sector this quarter",
        "Study of 10,000 participants shows moderate link between {topic} and outcomes",
        "Officials confirm {topic} update after months of careful review",
        "International summit addresses {topic}, delegates reach consensus",
        "New data from {institution} sheds light on {topic} trends over past decade",
        "Health authorities update guidelines on {topic} based on latest evidence",
        "{institution} and partners collaborate on {topic} initiative for communities",
        "Report: {topic} shows gradual improvement according to national statistics",
    ]

    topics = [
        "climate change", "vaccine safety", "election results", "economic policy",
        "public health", "technology regulation", "immigration policy", "education reform",
        "renewable energy", "artificial intelligence", "cybersecurity", "global trade",
    ]
    institutions = [
        "Harvard University", "WHO", "CDC", "MIT", "Stanford",
        "the UN", "Oxford University", "NASA", "Reuters", "AP News",
    ]

    records = []
    for i in range(n_samples):
        topic = np.random.choice(topics)
        institution = np.random.choice(institutions)
        is_fake = np.random.random() < 0.48  # ~48% fake, 52% real

        if is_fake:
            template = np.random.choice(fake_templates)
            text = template.format(topic=topic, institution=institution)
            label = "FAKE"
            # Fake news tends toward negative sentiment
            sentiment = np.random.choice(
                ["negative", "neutral", "positive"], p=[0.60, 0.30, 0.10]
            )
        else:
            template = np.random.choice(real_templates)
            text = template.format(topic=topic, institution=institution)
            label = "REAL"
            sentiment = np.random.choice(
                ["positive", "neutral", "negative"], p=[0.35, 0.45, 0.20]
            )

        # Add some realistic body text
        body_words = np.random.randint(20, 80)
        filler = " ".join(
            np.random.choice(
                ["the", "and", "is", "was", "has", "have", "said", "reported",
                 "according", "officials", "experts", "sources", "new", "recent",
                 topic.split()[0], institution.split()[-1]],
                size=body_words,
            )
        )
        full_text = f"{text}. {filler}."

        records.append({
            "id": i,
            "title": text,
            "text": full_text,
            "label": label,
            "sentiment": sentiment,
        })

    df = pd.DataFrame(records)
    logger.info(f"Generated sample dataset: {len(df)} rows | Fake: {(df.label=='FAKE').sum()} | Real: {(df.label=='REAL').sum()}")
    return df


def load_dataset(data_dir: str = "data/") -> pd.DataFrame:
    """
    Load dataset from disk if available, otherwise generate sample data.
    Supports WELFake / ISOT / custom CSV formats.
    """
    data_path = Path(data_dir)

    # Try loading common public dataset files
    candidates = [
        data_path / "WELFake_Dataset.csv",
        data_path / "fake_news.csv",
        data_path / "news.csv",
        data_path / "dataset.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Loading dataset from: {candidate}")
            df = pd.read_csv(candidate)

            # Normalize column names for different dataset formats
            col_map = {}
            for col in df.columns:
                if col.lower() in ["label", "class", "target", "fake"]:
                    col_map[col] = "label"
                elif col.lower() in ["text", "body", "content", "article"]:
                    col_map[col] = "text"
                elif col.lower() in ["title", "headline"]:
                    col_map[col] = "title"
            df = df.rename(columns=col_map)

            # Ensure required columns exist
            if "text" not in df.columns and "title" in df.columns:
                df["text"] = df["title"]
            if "label" not in df.columns:
                logger.warning("No 'label' column found — generating labels randomly for demo")
                df["label"] = np.random.choice(["FAKE", "REAL"], size=len(df))

            # Normalize labels
            df["label"] = df["label"].astype(str).str.upper().str.strip()
            df["label"] = df["label"].replace({"0": "REAL", "1": "FAKE", "FALSE": "FAKE", "TRUE": "REAL"})

            logger.info(f"Loaded {len(df)} rows from {candidate.name}")
            return df.dropna(subset=["text"]).reset_index(drop=True)

    logger.info("No dataset file found — generating sample data...")
    return generate_sample_dataset()


# ──────────────────────────────────────────────
# 2. NLP PREPROCESSING PIPELINE
# ──────────────────────────────────────────────

class TextPreprocessor:
    """
    Full NLP preprocessing pipeline:
    Lowercasing → URL/HTML removal → Punctuation removal →
    Tokenization → Stopword removal → Lemmatization
    """

    def __init__(self, language: str = "english"):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        # Add domain-specific stopwords
        self.stop_words.update([
            "said", "says", "according", "reported", "new", "would",
            "could", "one", "also", "however", "may", "us",
        ])

    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def remove_urls(self, text: str) -> str:
        """Remove URLs and email addresses."""
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        return text

    def remove_html_tags(self, text: str) -> str:
        """Strip HTML tags."""
        return re.sub(r"<[^>]+>", "", text)

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation and special characters."""
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)  # remove numbers
        return text

    def remove_extra_whitespace(self, text: str) -> str:
        """Collapse multiple spaces into one."""
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords and short tokens."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize tokens to their base form."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text: str) -> str:
        """
        Full pipeline: raw text → cleaned text (string form).
        Use this for TF-IDF and classical ML models.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.lowercase(text)
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_punctuation(text)
        text = self.remove_extra_whitespace(text)

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)

        return " ".join(tokens)

    def preprocess_batch(self, texts: pd.Series, verbose: bool = True) -> pd.Series:
        """Process a full pandas Series of texts with progress logging."""
        total = len(texts)
        logger.info(f"Preprocessing {total} texts...")

        cleaned = []
        for i, text in enumerate(texts):
            cleaned.append(self.preprocess(str(text)))
            if verbose and (i + 1) % 200 == 0:
                logger.info(f"  Processed {i+1}/{total}")

        logger.info("Preprocessing complete.")
        return pd.Series(cleaned, index=texts.index)


# ──────────────────────────────────────────────
# 3. SENTIMENT LABELING (TextBlob fallback)
# ──────────────────────────────────────────────

def assign_sentiment_labels(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Add sentiment labels using TextBlob if not already present.
    Categories: positive | neutral | negative
    """
    if "sentiment" in df.columns and df["sentiment"].notna().sum() > len(df) * 0.5:
        logger.info("Sentiment labels already present — skipping TextBlob inference.")
        return df

    from textblob import TextBlob

    logger.info("Assigning sentiment labels via TextBlob...")

    def get_sentiment(text: str) -> str:
        try:
            polarity = TextBlob(str(text)).sentiment.polarity
            if polarity > 0.1:
                return "positive"
            elif polarity < -0.1:
                return "negative"
            else:
                return "neutral"
        except Exception:
            return "neutral"

    df["sentiment"] = df[text_col].apply(get_sentiment)
    counts = df["sentiment"].value_counts()
    logger.info(f"Sentiment distribution: {counts.to_dict()}")
    return df


# ──────────────────────────────────────────────
# 4. FULL PREPROCESSING PIPELINE
# ──────────────────────────────────────────────

def run_preprocessing_pipeline(data_dir: str = "data/") -> pd.DataFrame:
    """
    End-to-end preprocessing:
    1. Load data
    2. Assign sentiment labels
    3. Clean text
    4. Save processed data
    Returns the processed DataFrame.
    """
    # Step 1: Load
    df = load_dataset(data_dir)

    # Step 2: Sentiment labels
    df = assign_sentiment_labels(df)

    # Step 3: Preprocess text
    preprocessor = TextPreprocessor()
    df["clean_text"] = preprocessor.preprocess_batch(df["text"])

    # Step 4: Encode labels
    df["label_encoded"] = (df["label"] == "FAKE").astype(int)  # FAKE=1, REAL=0
    sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
    df["sentiment_encoded"] = df["sentiment"].map(sentiment_map).fillna(1).astype(int)

    # Step 5: Drop empty clean_text rows
    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} empty rows after cleaning.")

    # Step 6: Save
    output_path = Path(data_dir) / "processed_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to: {output_path}")

    return df


# ──────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = run_preprocessing_pipeline(data_dir="data/")
    print("\n── Dataset Summary ──")
    print(df[["label", "sentiment", "clean_text"]].head(5).to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nLabel counts:\n{df['label'].value_counts()}")
    print(f"\nSentiment counts:\n{df['sentiment'].value_counts()}")
