import trafilatura
from urllib.parse import urlparse

def is_valid_url(url: str) -> bool:
    """
    Checks if a given string is a valid URL.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def scrape_article_text(url: str) -> str | None:
    """
    Scrapes the main text content from a given news article URL.
    Returns None if scraping fails.
    """
    if not is_valid_url(url):
        return None
        
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
        
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return text
