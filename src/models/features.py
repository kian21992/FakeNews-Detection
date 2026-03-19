import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

def extract_entities(text: str) -> list[dict]:
    """
    Extracts named entities from the text to provide context for the user.
    """
    if not text or nlp is None:
        return []
    
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "NORP", "FAC"]:
            if ent.text not in entities:
                entities[ent.text] = {
                    "text": ent.text,
                    "label": ent.label_,
                    "count": 1
                }
            else:
                entities[ent.text]["count"] += 1
                
    # Sort entities by count
    sorted_entities = sorted(entities.values(), key=lambda x: x["count"], reverse=True)
    return sorted_entities[:10]  # Return top 10 entities
