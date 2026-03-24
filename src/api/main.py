from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
import uvicorn

import sys
from pathlib import Path

# Add src to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocess import clean_text
from src.models.pipeline import detector_pipeline
from src.models.features import extract_entities
from src.api.database import get_db, Feedback

app = FastAPI(title="Fake News Detection API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/analyze/text")
async def analyze_text(request: TextRequest):
    """
    Analyzes a raw text string for misinformation.
    """
    cleaned_text = clean_text(request.text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    result = detector_pipeline.predict(cleaned_text)
    entities = extract_entities(cleaned_text)
    
    return {
        "text": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
        "prediction": result,
        "entities": entities
    }

class FeedbackRequest(BaseModel):
    text: str
    is_reliable: bool
    model_score: float

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Saves user feedback to the database for future fine-tuning (RL Loop Phase 1)
    """
    db_feedback = Feedback(
        text=request.text,
        user_label=request.is_reliable,
        original_model_score=request.model_score
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return {"status": "success", "id": db_feedback.id}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Mount the static site for the UI at the end so it doesn't swallow API routes
app.mount("/", StaticFiles(directory=str(Path(__file__).parent.parent / "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
