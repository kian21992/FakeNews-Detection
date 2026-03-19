import sqlite3
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import shutil

DB_PATH = "./feedback.db"
MODEL_NAME = "hamzab/roberta-fake-news-classification"
OUTPUT_DIR = "./finetuned_model"

def main():
    print("🚀 Starting Feedback Fine-Tuning Process (Phase 2 of RL Loop)")
    
    # 1. Fetch user feedback from SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT text, user_label FROM feedbacks", conn)
        conn.close()
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return

    if df.empty:
        print("ℹ️ No feedback data available yet. Use the UI to submit feedback first.")
        return
        
    print(f"✅ Loaded {len(df)} feedback entries from the database.")
    
    # Transform labels back to model format (1 for Real, 0 for Fake for this specific model)
    df['label'] = df['user_label'].apply(lambda x: 1 if x else 0)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    print("⏳ Loading Tokenizer & Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
    
    # Preprocessing function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    print("⚙️ Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=4, # Small batch size for laptops
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        use_cpu=not torch.cuda.is_available() # Fallback to CPU if no GPU
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    print("🏋️‍♂️ Beginning Fine-Tuning (Simulating RL Reward Updates)...")
    try:
        trainer.train()
        print(f"🎉 Training complete! Saving fine-tuned model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # In a real RLHF pipeline, we would automatically update src/models/pipeline.py 
        # to point to this new local ./finetuned_model directory.
        print("\n✨ NOTE: To use this updated model, change the MODEL_NAME in src/models/pipeline.py to './finetuned_model'")
        
    except Exception as e:
        print(f"❌ Training failed. Note: Fine-tuning requires significant RAM/GPU memory. Error: {e}")

if __name__ == "__main__":
    main()
