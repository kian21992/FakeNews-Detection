import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

def clean_text(text: str) -> str:
    """
    Basic text cleaning pipeline:
    - Removes URLs
    - Removes extra whitespaces
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_for_feature_extraction(text: str) -> str:
    """
    Advanced preprocessing for linguistic feature extraction.
    Returns lemmas for standard ML or feature engineering without stopwords/punct.
    For transformer models, the raw basic cleaned text is often preferred.
    """
    text = clean_text(text)
    
    if nlp is None:
        return text
        
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)
