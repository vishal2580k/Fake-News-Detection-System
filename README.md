# 🛡️ News Sentinel
### AI-Powered News Sentiment Analysis & Fake News Detection System

<p align="center">
  <img src="https://raw.githubusercontent.com/vishal2580k/Fake-News-Detection-System/main/screenshots/banner.png" alt="News Sentinel Banner" width="100%"/>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/></a>
  <a href="https://scikit-learn.org"><img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/></a>
</p>

<p align="center">
  <b>Detect fake news and analyze sentiment in real time — paste text or drop a URL.</b>
</p>

---

## 📖 Overview

**News Sentinel** is a complete, production-grade NLP system that analyzes news articles across two dimensions simultaneously:

| Task | Description | Output |
|------|-------------|--------|
| 🚨 **Fake News Detection** | Binary classification using TF-IDF + classical ML | `FAKE` / `REAL` + confidence % |
| 😊 **Sentiment Analysis** | 3-class classification | `Positive` / `Neutral` / `Negative` + confidence % |

The system ships with a full ML pipeline (preprocessing → feature engineering → training → evaluation) and is deployed as a **Streamlit web app** with a dark intelligence theme. It optionally scrapes articles directly from URLs.

---

## 📸 Screenshots

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/vishal2580k/Fake-News-Detection-System/main/screenshots/analyze_tab.png" alt="Analyze News Tab" width="100%"/>
      <br/><sub><b>🔍 Analyze News — Text Input</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/vishal2580k/Fake-News-Detection-System/main/screenshots/results.png" alt="Prediction Results" width="100%"/>
      <br/><sub><b>📋 Prediction Results with Confidence</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/vishal2580k/Fake-News-Detection-System/main/screenshots/dashboard.png" alt="Dashboard" width="100%"/>
      <br/><sub><b>📊 Dataset Dashboard & EDA Charts</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/vishal2580k/Fake-News-Detection-System/main/screenshots/model_metrics.png" alt="Model Metrics" width="100%"/>
      <br/><sub><b>📈 Model Performance & Confusion Matrices</b></sub>
    </td>
  </tr>
</table>

---

## ✨ Features

- **Dual-task analysis** — fake news detection + sentiment in a single inference pass
- **Real-time article scraping** — paste a URL; the app fetches and analyzes the full text automatically
- **Confidence scoring** — class probabilities displayed for every prediction
- **Interactive dashboard** — live Plotly charts for sentiment distribution and fake/real balance
- **Top-keyword comparison** — word frequency charts for FAKE vs REAL articles
- **Model metrics page** — accuracy, precision, recall, F1, ROC-AUC, and confusion matrix heatmaps
- **Auto-training** — if no models are found on startup, the training pipeline runs automatically
- **Dark intelligence UI** — custom Space Grotesk theme with gradient badges and glassmorphism cards

---

## 🗂️ Project Structure

```
news-sentinel/
│
├── app.py                         # Streamlit web application (4-page layout)
│
├── src/                           # Core ML modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # NLP pipeline: tokenize, lemmatize, clean
│   ├── feature_engineering.py     # TF-IDF (50k features, bigrams) + Word2Vec
│   ├── model_training.py          # Train, evaluate, save, visualize models
│   └── prediction.py              # Inference engine + URL article scraper
│
├── data/
│   └── processed_data.csv         # Auto-generated after first run
│
├── models/                        # Saved models (auto-created on first run)
│   ├── tfidf_vectorizer.joblib
│   ├── fake_best_model.joblib
│   ├── fake_logistic_regression.joblib
│   ├── fake_naive_bayes.joblib
│   ├── fake_linear_svc.joblib
│   ├── sentiment_best_model.joblib
│   ├── sentiment_logistic_regression.joblib
│   ├── sentiment_naive_bayes.joblib
│   └── training_summary.json
│
├── static/                        # Generated charts (auto-created)
│   ├── sentiment_distribution.png
│   ├── fake_real_distribution.png
│   ├── word_frequency_all.png
│   ├── wordcloud_all.png
│   └── cm_fake_*.png
│
├── exploration.ipynb              # Full EDA + step-by-step walkthrough
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, TextBlob, spaCy |
| Feature Engineering | scikit-learn TF-IDF, Gensim Word2Vec |
| Models | Logistic Regression, Complement Naive Bayes, LinearSVC (calibrated) |
| Deep Learning (optional) | Transformers, PyTorch (BERT-ready) |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| Web App | Streamlit 1.32 |
| URL Scraping | newspaper3k, BeautifulSoup4, requests |
| Persistence | joblib |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/vishal2580k/Fake-News-Detection-System.git
cd Fake-News-Detection-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Add Your Dataset

Download the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place `WELFake_Dataset.csv` in the `data/` folder.

> **No dataset?** No problem — the system auto-generates a realistic 1,000-article sample dataset on first run.

### 5. Train the Models

```bash
python src/model_training.py
```

This will preprocess text, train all classifiers, save models to `models/`, and save visualizations to `static/`.

### 6. Launch the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ Application Pages

### 🔍 Analyze News

Paste any article text or enter a URL for instant dual-task analysis.

<p align="center">
  <img src="https://raw.githubusercontent.com/vishal2580k/Fake-News-Detection-System/main/screenshots/url_analysis.png" alt="URL Analysis" width="80%"/>
</p>

**Text tab** — paste full article text, click Analyze.  
**URL tab** — enter a news URL; the article is fetched and analyzed automatically. Falls back from `newspaper3k` to `BeautifulSoup` if the primary extractor fails.

Each result shows:
- **FAKE / REAL badge** with glow effect and confidence bar
- **Sentiment badge** (Positive / Neutral / Negative) with confidence bar
- Class probability breakdown for both tasks
- Word count and average confidence metric card
- Contextual alert (green for high-confidence REAL, amber for high-confidence FAKE)

### 📊 Dashboard

<p align="center">
  <img src="https://raw.githubusercontent.com/vishal2580k/Fake-News-Detection-System/main/screenshots/dashboard.png" alt="Dashboard Charts" width="80%"/>
</p>

Live dataset statistics with Plotly charts:
- Fake vs Real donut chart
- Sentiment distribution bar chart
- Top-15 keywords for FAKE articles vs REAL articles side-by-side
- All static EDA images (word cloud, confusion matrices)

### 📈 Model Metrics

Tabular performance summary across all trained classifiers, with best-model highlighting and confusion matrix heatmaps.

### ℹ️ About

Architecture overview, tech stack pill badges, and an ethical disclaimer.

---

## 🧠 NLP Pipeline

```
Raw Text
   ↓  Lowercase
   ↓  Remove URLs & HTML tags
   ↓  Remove punctuation & numbers
   ↓  NLTK tokenization
   ↓  Stopword removal (NLTK + domain-specific)
   ↓  WordNet lemmatization
   ↓
Clean Text ──► TF-IDF (50,000 features, unigrams + bigrams) ──► Classifier
```

Sentiment labels are assigned via **TextBlob polarity scoring** if not already present in the dataset (`> 0.1` → positive, `< -0.1` → negative, otherwise neutral).

---

## 📊 Model Performance

### Fake News Detection (WELFake dataset — ~96% F1)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~0.96 | ~0.96 | ~0.96 | ~0.96 | ~0.99 |
| Complement Naive Bayes | ~0.94 | ~0.94 | ~0.94 | ~0.94 | ~0.98 |
| LinearSVC (calibrated) | ~0.97 | ~0.97 | ~0.97 | ~0.97 | ~0.99 |

### Sentiment Analysis

| Model | Accuracy | F1 |
|-------|----------|----|
| Logistic Regression | ~0.58 | ~0.57 |
| Complement Naive Bayes | ~0.61 | ~0.60 |

> Results shown are on the auto-generated sample dataset. Real-world accuracy improves significantly with the WELFake or ISOT datasets.

---

## 🔧 Alternative Execution

Run individual modules directly:

```bash
# Preprocessing only
python src/data_preprocessing.py

# Feature engineering demo
python src/feature_engineering.py

# Inference demo (3 test articles)
python src/prediction.py

# Full Jupyter walkthrough
jupyter notebook exploration.ipynb
```

---

## 🔮 Roadmap

- [ ] **BERT fine-tuning** — replace TF-IDF with `bert-base-uncased` embeddings
- [ ] **Multi-language support** — detect and handle non-English articles
- [ ] **Source credibility scoring** — rate news sources by historical accuracy
- [ ] **Browser extension** — highlight fake news while browsing
- [ ] **FastAPI REST endpoint** — programmatic access via REST API
- [ ] **Active learning** — allow users to label predictions to retrain the model
- [ ] **SHAP/LIME explainability** — highlight which words triggered the prediction
- [ ] **Real-time news feed** — monitor RSS feeds continuously

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 👤 Author

**Vishal** — [@vishal2580k](https://github.com/vishal2580k)

---

<p align="center">
  Built with ❤️ using Python · scikit-learn · NLTK · Streamlit
</p>
