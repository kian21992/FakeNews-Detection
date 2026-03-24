import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_text_endpoint():
    # A blatant example of text (doesn't matter if it's true or false, just testing the pipeline)
    payload = {
        "text": "The moon is made of cheese. Scientists discovered this in 2024 using a new telescope."
    }
    response = client.post("/analyze/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "is_reliable" in data["prediction"]
    assert "confidence_score" in data["prediction"]
    assert "entities" in data
