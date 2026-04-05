# 🛡️ News Sentinel
### AI-Powered News Sentiment Analysis & Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Project Overview

**News Sentinel** is a complete, production-grade NLP system that analyzes news articles across two dimensions:

| Task | Description | Output |
|------|-------------|--------|
| 🚨 **Fake News Detection** | Binary classification | FAKE / REAL + confidence % |
| 😊 **Sentiment Analysis** | 3-class classification | Positive / Neutral / Negative + confidence % |

The system includes a full ML pipeline (preprocessing → feature engineering → training → evaluation) deployed as a **Streamlit web application** with real-time article analysis and an optional **URL scraping** feature.

---

## 🗂️ Project Structure

```
news-sentinel/
│
├── data/                          # Raw & processed datasets
│   ├── README.txt                 # Dataset instructions
│   └── processed_data.csv         # Auto-generated after first run
│
├── notebooks/
│   └── exploration.ipynb          # Full EDA + step-by-step walkthrough
│
├── src/                           # Core ML modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # NLP pipeline (tokenize, lemmatize, etc.)
│   ├── feature_engineering.py     # TF-IDF + Word2Vec features
│   ├── model_training.py          # Train, evaluate, visualize models
│   └── prediction.py              # Inference engine + URL scraper
│
├── models/                        # Saved models (auto-created)
│   ├── tfidf_vectorizer.joblib
│   ├── fake_best_model.joblib
│   ├── sentiment_best_model.joblib
│   └── training_summary.json
│
├── app/
│   └── app.py                     # Streamlit web application
│
├── static/                        # Saved charts (auto-created)
│   ├── sentiment_distribution.png
│   ├── fake_real_distribution.png
│   ├── word_frequency_all.png
│   ├── wordcloud_all.png
│   └── cm_fake_*.png              # Confusion matrices
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Library |
|-----------|---------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, TextBlob, spaCy |
| Features | scikit-learn TF-IDF, Gensim Word2Vec |
| Models | scikit-learn (LR, NB, SVC), Transformers (BERT) |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| Web App | Streamlit |
| URL Scraping | newspaper3k, BeautifulSoup4 |
| Persistence | joblib |

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- pip

### Step 1: Clone & Setup

```bash
git clone https://github.com/yourname/news-sentinel.git
cd news-sentinel

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: (Optional) Add Your Dataset

Download the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) 
and place `WELFake_Dataset.csv` in the `data/` folder.

If no dataset is provided, the system auto-generates a realistic sample dataset.

### Step 3: Run Training Pipeline

```bash
# From project root
python src/model_training.py
```

This will:
- Load / generate data
- Preprocess text (NLP pipeline)
- Train all models
- Save models to `models/`
- Save visualizations to `static/`
- Print evaluation metrics

### Step 4: Launch the Web App

```bash
streamlit run app/app.py
```

Visit **http://localhost:8501** in your browser.

### Alternative: Run Individual Modules

```bash
# Preprocessing only
python src/data_preprocessing.py

# Feature engineering only
python src/feature_engineering.py

# Inference demo
python src/prediction.py

# Jupyter notebook
jupyter notebook notebooks/exploration.ipynb
```

---

## 🖥️ Application Features

### Analyze News Tab
- Paste any news article text for instant analysis
- View FAKE/REAL verdict with confidence score
- View sentiment (Positive/Neutral/Negative) with confidence
- See class probability breakdowns

### URL Analysis (Bonus)
- Enter any news URL
- Article is automatically fetched and analyzed
- Falls back to BeautifulSoup if newspaper3k fails

### Dashboard
- Live charts: sentiment distribution, fake/real balance
- Top keywords for FAKE vs REAL articles
- All saved visualizations

### Model Metrics
- Performance table: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix heatmaps
- Best model highlight

---

## 📊 Model Performance (Sample Data)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | ~0.87 | ~0.87 | ~0.87 | ~0.87 |
| Complement Naive Bayes | ~0.83 | ~0.83 | ~0.83 | ~0.83 |
| Linear SVC | ~0.88 | ~0.88 | ~0.88 | ~0.88 |

*Results vary by dataset. Performance on WELFake is significantly higher (~96% F1).*

---

## 🧠 NLP Pipeline

```
Raw Text
   ↓ Lowercase
   ↓ Remove URLs & HTML
   ↓ Remove Punctuation & Numbers
   ↓ Tokenization (NLTK)
   ↓ Stopword Removal
   ↓ Lemmatization (WordNet)
   ↓
Clean Text → TF-IDF (50k features, bigrams) → Classifier
```

---

## 📸 Screenshots

![UI](screenshots/Screenshot 2026-04-05 150115.png)

![Result](screenshots/Screenshot 2026-04-05 150031.png)

![Dashboard](screenshots/Screenshot 2026-04-05 145951.png)

![Metrics](screenshots/Screenshot 2026-04-05 145911.png)

---

## 🔮 Future Improvements

- [ ] **BERT Fine-tuning** — Replace TF-IDF with `bert-base-uncased` embeddings
- [ ] **Multi-language support** — Detect and handle non-English articles
- [ ] **Source credibility scoring** — Rate news sources by historical accuracy
- [ ] **Browser extension** — Highlight fake news while browsing
- [ ] **API endpoint** — FastAPI REST service for programmatic access
- [ ] **Active learning** — Allow users to label predictions to retrain
- [ ] **Explainability** — SHAP/LIME to highlight which words triggered prediction
- [ ] **Real-time news feed** — Monitor RSS feeds continuously

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

*Built with ❤️ using Python, scikit-learn, NLTK, and Streamlit*
