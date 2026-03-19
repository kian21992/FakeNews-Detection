from urllib.parse import urlparse

# A very simplified list for demonstration purposes
RELIABLE_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", 
    "pbs.org", "wsj.com", "nytimes.com", "washingtonpost.com", 
    "bloomberg.com", "theguardian.com", "nature.com", "science.org",
    "wikipedia.org"
]

UNRELIABLE_DOMAINS = [
    "infowars.com", "breitbart.com", "theonion.com", "babylonbee.com",
    "naturalnews.com", "worldnewsdailyreport.com", "empirenews.net"
]

def check_domain_reputation(url: str) -> dict:
    """
    Checks the reputation of the given URL's domain.
    Returns a dictionary with the status and a score modifier (-1.0 to 1.0).
    """
    if not url:
        return {"status": "unknown", "modifier": 0.0}
        
    try:
        parsed_uri = urlparse(url)
        domain = parsed_uri.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
            
        if domain in RELIABLE_DOMAINS:
            return {"status": "credible", "modifier": 0.2} # Boost score
        elif domain in UNRELIABLE_DOMAINS:
            return {"status": "unreliable_or_satire", "modifier": -0.4} # Heavily penalize score
        else:
            return {"status": "unknown", "modifier": 0.0}
    except Exception:
        return {"status": "error", "modifier": 0.0}
