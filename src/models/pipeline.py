import os
import torch
from transformers import pipeline

from textblob import TextBlob

class FakeNewsDetector:
    def __init__(self, base_model_name="hamzab/roberta-fake-news-classification"):
        """
        Initializes the fake news detection pipeline using an ensemble approach.
        It automatically checks if a user-trained (RLHF) model exists in the workspace
        and loads it to improve accuracy over time.
        """
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Check if we have a custom retrained model from the feedback loop
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "finetuned_model")
        
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            print(f"🧠 Loading custom fine-tuned model from {local_model_path}!")
            model_to_use = local_model_path
            self.is_custom_model = True
        else:
            print(f"📖 Loading base pre-trained model ({base_model_name}). No custom fine-tuning found.")
            model_to_use = base_model_name
            self.is_custom_model = False
            
        self.classifier = pipeline("text-classification", model=model_to_use, device=self.device)
        
    def predict(self, text: str) -> dict:
        """
        Predicts if the text is reliable or fake using an ensemble of:
        - Transformer Classifier
        - Subjectivity / Sentiment Analysis
        """
        # Truncate text if it's too long (most models handle max 512 tokens)
        # Assuming avg 4 chars per token, we roughly slice by characters
        max_chars = 2000 
        truncated_text = text[:max_chars] if len(text) > max_chars else text
        
        try:
            # 1. Transformer Score
            result = self.classifier(truncated_text)[0]
            label = result['label'].upper()
            transformer_score = result['score']
            is_transformer_real = (label == "REAL" or label == "TRUE" or label == "RELIABLE")
            base_score = transformer_score if is_transformer_real else (1.0 - transformer_score)
            
            # 2. Subjectivity Analysis
            # High subjectivity (closer to 1.0) often correlates with op-eds or misleading emotional content
            blob = TextBlob(text)
            subjectivity = blob.sentiment.subjectivity
            
            # Penalize the score if the text is highly subjective
            subjectivity_penalty = 0.0
            if subjectivity > 0.6:
                subjectivity_penalty = (subjectivity - 0.6) * 0.5 # Max penalty of ~0.2
            
            # 3. Ensemble Score Calculation
            final_score = base_score - subjectivity_penalty
            
            # Apply a "Benefit of the Doubt" adjustment to reduce false positives
            # Real news often uses sensational language that AI trips on. We boost the final score by 15%.
            final_score += 0.15
            
            final_score = max(0.0, min(1.0, final_score)) # Clamp between 0 and 1
            
            is_reliable = final_score >= 0.5
            
            return {
                "is_reliable": is_reliable,
                "confidence_score": round(final_score, 4),
                "breakdown": {
                    "transformer_base_score": round(base_score, 4),
                    "original_label": label,
                    "subjectivity": round(subjectivity, 4)
                }
            }
        except Exception as e:
            return {
                "error": str(e)
            }

# Singleton instance for easy importing
detector_pipeline = FakeNewsDetector()
