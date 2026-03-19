# VeritasAI: Fake News & Misinformation Detector

VeritasAI is an NLP-powered system designed to analyze claims, social media posts, and news articles to classify them as credible or potentially misleading. It utilizes a powerful ensemble scoring approach that combines state-of-the-art transformer models, linguistic subjectivity analysis, and domain reputation.

## ✨ Features

- **Raw Text Analysis**: Paste a custom claim, tweet, or statement directly into the intuitive UI for instant analysis.
- **URL Context Analysis**: Provide a link to an article, and the backend will silently scrape the main content, stripping away ads and boilerplate, to analyze the core journalism.
- **Ensemble Credibility Scoring**: The final score isn't a black box. It's a mathematically aggregated confidence score derived from three pillars:
    1. **Transformer Base Score**: Evaluates linguistic patterns associated with misinformation using a pre-trained `RoBERTa` model fine-tuned for fake news.
    2. **Subjectivity Penalty**: Analyzes the emotional charge and opinionated nature of the text using `TextBlob`. Highly subjective, emotionally manipulative text receives a penalty.
    3. **Domain Reputation**: Checks a curated list of known reliable sources (e.g., Reuters, AP News) giving them a credibility boost, and penalizes known unreliable or satire domains.
- **Extracted Context (NER)**: Automatically extracts key entities (people, organizations, locations) using `spaCy` to give you quick context on the topic.
- **Detailed Analysis Breakdown**: The UI transparently displays the exact factors that influenced your final score.

## 🛠 Tech Stack

- **Backend Framework**: `FastAPI` (Python)
- **NLP & Inference Pipeline**: Hugging Face `transformers` (`hamzab/roberta-fake-news-classification`), `PyTorch`
- **Linguistic Processing**: `spaCy`, `TextBlob`
- **Web Scraping**: `trafilatura`
- **Frontend**: Vanilla HTML/JavaScript, Tailwind CSS (via CDN)
- **Testing**: `pytest`

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher.
- `pip` package manager.

### Installation

1. Clone the repository or navigate to your project directory.
   ```bash
   cd fake_news_detector
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Download the required language processing data:
   ```bash
   python -m spacy download en_core_web_sm
   python -m textblob.download_corpora
   ```

## 💻 Running the Application

To start the VeritasAI server, run:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Dashboard UI**: Open your web browser and navigate to `http://localhost:8000`.
- **Interactive API Docs**: Navigate to `http://localhost:8000/docs`.

### Testing

To verify the system components and API endpoints, run the included test suite:

```bash
python -m pytest tests/test_api.py -v
```

## 🔄 Reinforcement Learning (RLHF) Loop

This system includes a basic implementation of Reinforcement Learning from Human Feedback (RLHF):
1. **Collect Feedback**: When users analyze an article on the dashboard, they can click "Thumbs Up" or "Thumbs Down" to indicate if the AI was correct.
2. **Store Data**: This feedback is silently securely saved to a local SQLite database (`feedback.db`).
3. **Fine-Tune Model**: Periodically, you can run the retraining script to update the transformer model's weights based on the collected human feedback.

To execute a training run on your collected feedback:
```bash
python retrain.py
```
*(Note: Fine-tuning requires significant system memory/VRAM).*

## 🏗 Project Structure

```text
fake_news_detector/
├── src/
│   ├── api/             # FastAPI backend implementation
│   │   └── main.py      # Main application and endpoints
│   ├── data/            # Extraction and Preprocessing utilities
│   │   ├── preprocess.py # Text cleaning and standardization
│   │   └── scraper.py   # Web scraping logic using trafilatura
│   ├── frontend/        # Vanilla JS & HTML UI
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   └── models/          # ML and Logic implementations
│       ├── features.py  # spaCy named entity recognition
│       ├── pipeline.py  # Hugging Face inference and ensemble scoring
│       └── reputation.py# Domain reliability lists and modifiers
├── tests/               # Pytest suite
│   └── test_api.py
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

## ⚠️ Limitations & Disclaimer

This system is an AI tool designed to detect **linguistic patterns** and structural markers common in misinformation. It does not possess "ground truth" knowledge and cannot independently verify factual claims like a human journalist using primary sources. 

Always use this tool as a supplementary signal alongside verified fact-checking organizations (e.g., Snopes, PolitiFact, Reuters Fact Check).
