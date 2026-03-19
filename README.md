# TrioGodzAI
Fake News and Misinformation Detection System using NLP, CNN, and Reinforcement Learning

## Overview

TrioGodzAI is a web-based system designed to analyze news articles, social media posts, and online claims to determine whether they are credible or potentially misleading. The system uses natural language processing (NLP), deep learning, and a reinforcement learning feedback mechanism to improve prediction quality over time.

The goal of this project is to provide users with a fast and understandable way to evaluate the reliability of information found online.

---

## Features

- Text Analysis: Analyze raw text such as claims, tweets, or statements  
- URL Analysis: Extract and analyze content from news article links  
- Ensemble Credibility Scoring:
  - Transformer-based fake news detection model  
  - Subjectivity analysis using TextBlob  
  - Domain reputation scoring  
- Named Entity Recognition (NER): Extract key people, organizations, and locations  
- Transparent Results: Displays how the final score is computed  
- Feedback System: Users can provide feedback to improve the model  

---

## How It Works

1. User inputs text or a URL  
2. System extracts and preprocesses the content  
3. NLP pipeline converts text into numerical features  
4. CNN / Transformer model predicts fake or real  
5. Ensemble scoring adjusts the final credibility score  
6. User feedback is stored for future model improvement  

---

## Tech Stack

- Backend: FastAPI (Python)  
- Machine Learning / NLP: Hugging Face Transformers, PyTorch  
- Libraries: spaCy, TextBlob  
- Web Scraping: trafilatura  
- Frontend: HTML, JavaScript, Tailwind CSS  
- Testing: pytest  

---

## Project Structure


fake_news_detector/
├── src/
│ ├── api/ # FastAPI backend
│ ├── data/ # Preprocessing and scraping
│ ├── frontend/ # UI files
│ └── models/ # ML models and scoring logic
├── tests/ # Test cases
├── docs/ # Reports and screenshots
├── requirements.txt
└── README.md


---

## Getting Started

### Requirements

- Python 3.10 or higher  
- pip  

### Installation

```bash
cd fake_news_detector
python -m venv venv

Activate the virtual environment:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Download required NLP data:

python -m spacy download en_core_web_sm
python -m textblob.download_corpora
Running the Application
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Access the app:

http://localhost:8000

http://localhost:8000/docs

Testing
python -m pytest tests/test_api.py -v
Reinforcement Learning (Feedback Loop)

The system includes a basic feedback mechanism:

Users can provide feedback (correct/incorrect prediction)

Feedback is stored in a local SQLite database

Data can be used to retrain the model

To retrain:

python retrain.py

Note: Retraining may require higher memory or GPU resources.

Current Progress (Week 2)

Dataset collected and cleaned

Exploratory Data Analysis (EDA) completed

Baseline models implemented (e.g., Logistic Regression, Naive Bayes)

CNN model initialized with initial results

NLP preprocessing pipeline completed

Reinforcement Learning component implemented (basic version)

Version

v0.1 – Proposal

v0.9 – Release Candidate

v1.0 – Final Version

Limitations

The system does not verify factual truth like human fact-checkers

Predictions are based on patterns and model training data

Misclassification is possible

Disclaimer

This system is intended as a support tool only. Users are encouraged to verify information using trusted fact-checking sources such as Snopes, PolitiFact, or Reuters Fact Check.

Authors

Dela Cruz, Kian S.

Pasion Jr, Allan C.

Sigua, Carl Jerome J.

License

This project is for academic purposes.
