"""
Web Agent module for Nexus AI.
Provides capabilities to browse the web, extract content, and perform research.
"""

import requests
import logging
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
import urllib.parse

class WebAgent:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        })
        self.last_page_content = ""
        self.last_url = ""

    def visit(self, url: str) -> str:
        """Visit a URL and extract its main text content."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
                
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            self.last_page_content = clean_text
            self.last_url = url
            
            title = soup.title.string if soup.title else url
            return f"✅ Visited: {title}\n\nSummary ({len(clean_text)} chars):\n{clean_text[:500]}..."
            
        except Exception as e:
            logging.error(f"WebAgent error: {e}")
            return f"❌ Failed to visit {url}: {e}"

    def get_content(self) -> str:
        """Return the full content of the last visited page."""
        return self.last_page_content

    def research(self, topic: str) -> str:
        """Perform a quick research on a topic using DuckDuckGo (HTML)."""
        query = urllib.parse.quote_plus(topic)
        url = f"https://html.duckduckgo.com/html/?q={query}"
        
        try:
            return self.visit(url)
        except Exception as e:
            return f"❌ Research failed: {e}"
