"""
Web Search & Document Fetcher for AetherAI.

This module provides:
- Web searching using DuckDuckGo (no API key needed)
- URL content fetching with HTML-to-text conversion
- Documentation fetching from URLs
- Web page summarization
"""

import os
import re
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from datetime import datetime

# Try importing web-related packages
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    DDGS_AVAILABLE = False

try:
    import html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    html2text = None
    HTML2TEXT_AVAILABLE = False


@dataclass
class SearchResult:
    """Represents a web search result."""
    title: str
    url: str
    snippet: str
    source: str = ""


@dataclass
class WebDocument:
    """Represents fetched web document."""
    url: str
    title: str
    content: str
    content_type: str
    fetched_at: str
    word_count: int
    links: List[str] = None


class WebSearcher:
    """Web search and document fetching for AetherAI."""
    
    # Common documentation sites
    DOC_SITES = {
        'docs.python.org': 'Python',
        'developer.mozilla.org': 'MDN',
        'reactjs.org': 'React',
        'vuejs.org': 'Vue',
        'nextjs.org': 'Next.js',
        'fastapi.tiangolo.com': 'FastAPI',
        'flask.palletsprojects.com': 'Flask',
        'django-docs': 'Django',
        'rust-lang.org': 'Rust',
        'go.dev': 'Go',
        'learn.microsoft.com': 'Microsoft',
        'cloud.google.com': 'Google Cloud',
        'aws.amazon.com': 'AWS',
        'stackoverflow.com': 'Stack Overflow',
        'github.com': 'GitHub',
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize web searcher.
        
        Args:
            cache_dir: Directory to cache fetched documents.
        """
        self.cache_dir = cache_dir or os.path.join(
            os.getenv('HOME') or os.getenv('USERPROFILE') or os.path.expanduser('~'),
            '.nexus', 'web_cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Session for requests
        self.session = None
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (compatible; AetherAI/1.0; +https://github.com/aetherAI)'
            })
        
        # HTML to text converter
        self.h2t = None
        if HTML2TEXT_AVAILABLE:
            self.h2t = html2text.HTML2Text()
            self.h2t.ignore_links = False
            self.h2t.ignore_images = True
            self.h2t.body_width = 0  # No line wrapping
    
    def is_available(self) -> Dict[str, bool]:
        """Check which web features are available.
        
        Returns:
            Dictionary of feature availability.
        """
        return {
            "requests": REQUESTS_AVAILABLE,
            "beautifulsoup": BS4_AVAILABLE,
            "duckduckgo": DDGS_AVAILABLE,
            "html2text": HTML2TEXT_AVAILABLE,
            "web_search": DDGS_AVAILABLE,
            "url_fetch": REQUESTS_AVAILABLE,
        }
    
    # =========================================================================
    # Web Search
    # =========================================================================
    
    def search(self, query: str, max_results: int = 5, 
               region: str = "wt-wt") -> List[SearchResult]:
        """Search the web using DuckDuckGo.
        
        Args:
            query: Search query.
            max_results: Maximum results to return.
            region: Region code (wt-wt = worldwide).
            
        Returns:
            List of search results.
        """
        if not DDGS_AVAILABLE:
            return []
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region=region, max_results=max_results))
            
            return [
                SearchResult(
                    title=r.get('title', ''),
                    url=r.get('href', r.get('link', '')),
                    snippet=r.get('body', r.get('snippet', '')),
                    source=self._get_source_name(r.get('href', ''))
                )
                for r in results
            ]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def search_docs(self, query: str, technology: str = None, 
                    max_results: int = 5) -> List[SearchResult]:
        """Search for documentation specifically.
        
        Args:
            query: Search query.
            technology: Specific technology (python, react, etc.)
            max_results: Maximum results.
            
        Returns:
            List of search results.
        """
        # Enhance query for documentation
        doc_query = f"{query} documentation"
        if technology:
            doc_query = f"{technology} {query} docs"
        
        results = self.search(doc_query, max_results * 2)
        
        # Prioritize known doc sites
        scored_results = []
        for r in results:
            score = 0
            for site, name in self.DOC_SITES.items():
                if site in r.url:
                    score += 10
                    break
            if 'docs' in r.url.lower() or 'documentation' in r.url.lower():
                score += 5
            if 'tutorial' in r.url.lower():
                score += 3
            scored_results.append((score, r))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored_results[:max_results]]
    
    def format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for display.
        
        Args:
            results: List of search results.
            
        Returns:
            Formatted string.
        """
        if not results:
            return "❌ No results found"
        
        lines = ["🔍 **Search Results:**\n"]
        
        for i, r in enumerate(results, 1):
            source = f" [{r.source}]" if r.source else ""
            lines.append(f"**{i}. {r.title}**{source}")
            lines.append(f"   {r.url}")
            if r.snippet:
                snippet = r.snippet[:150] + "..." if len(r.snippet) > 150 else r.snippet
                lines.append(f"   _{snippet}_")
            lines.append("")
        
        lines.append("💡 Use `/fetch <number or url>` to read a result")
        
        return "\n".join(lines)
    
    def _get_source_name(self, url: str) -> str:
        """Get friendly source name from URL."""
        try:
            domain = urlparse(url).netloc.lower()
            for site, name in self.DOC_SITES.items():
                if site in domain:
                    return name
            # Return cleaned domain
            return domain.replace('www.', '').split('.')[0].capitalize()
        except Exception:
            return ""
    
    # =========================================================================
    # URL Fetching
    # =========================================================================
    
    def fetch_url(self, url: str, max_length: int = 15000) -> WebDocument:
        """Fetch and parse content from a URL.
        
        Args:
            url: URL to fetch.
            max_length: Maximum content length.
            
        Returns:
            WebDocument with content.
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available. Install with: pip install requests")
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Handle different content types
            if 'application/json' in content_type:
                content = json.dumps(response.json(), indent=2)
                title = "JSON Response"
            elif 'text/plain' in content_type:
                content = response.text
                title = "Plain Text"
            elif 'text/html' in content_type or 'html' in content_type:
                content, title = self._parse_html(response.text, url)
            else:
                content = response.text[:max_length]
                title = "Document"
            
            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "\n\n... [CONTENT TRUNCATED]"
            
            return WebDocument(
                url=url,
                title=title,
                content=content,
                content_type=content_type,
                fetched_at=datetime.now().isoformat(),
                word_count=len(content.split()),
                links=[]
            )
            
        except requests.exceptions.Timeout:
            raise RuntimeError("Request timed out")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch URL: {str(e)}")
    
    def _parse_html(self, html: str, url: str) -> Tuple[str, str]:
        """Parse HTML to clean text.
        
        Args:
            html: Raw HTML content.
            url: URL for context.
            
        Returns:
            Tuple of (content, title).
        """
        title = "Web Page"
        content = html
        
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                if soup.title:
                    title = soup.title.string or title
                
                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 
                               'aside', 'noscript', 'iframe', 'svg']):
                    tag.decompose()
                
                # Try to find main content
                main_content = soup.find('main') or soup.find('article') or soup.find(class_=re.compile(r'content|article|main|post'))
                
                if main_content:
                    target = main_content
                else:
                    target = soup.body or soup
                
                # Convert to text
                if HTML2TEXT_AVAILABLE and self.h2t:
                    content = self.h2t.handle(str(target))
                else:
                    content = target.get_text(separator='\n', strip=True)
                
                # Clean up multiple newlines
                content = re.sub(r'\n{3,}', '\n\n', content)
                
            except Exception as e:
                # Fallback to simple text extraction
                content = re.sub(r'<[^>]+>', '', html)
        else:
            # No BeautifulSoup, basic cleanup
            content = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', '', content)
            content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip(), title
    
    def format_document(self, doc: WebDocument) -> str:
        """Format fetched document for display.
        
        Args:
            doc: WebDocument to format.
            
        Returns:
            Formatted string.
        """
        lines = [
            f"📄 **{doc.title}**",
            f"🔗 {doc.url}",
            f"📊 {doc.word_count} words | Fetched: {doc.fetched_at[:10]}",
            "",
            "---",
            "",
            doc.content
        ]
        
        return "\n".join(lines)
    
    # =========================================================================
    # Documentation Helpers
    # =========================================================================
    
    def fetch_docs(self, query: str, technology: str = None) -> str:
        """Search for and fetch documentation.
        
        Args:
            query: What to search for.
            technology: Specific technology.
            
        Returns:
            Documentation content or error message.
        """
        # Search for docs
        results = self.search_docs(query, technology)
        
        if not results:
            return f"❌ No documentation found for: {query}"
        
        # Try to fetch top result
        try:
            doc = self.fetch_url(results[0].url)
            
            output = [
                f"📚 **{results[0].title}**",
                f"🔗 {results[0].url}",
                "",
                "---",
                "",
                doc.content[:10000]  # Limit for display
            ]
            
            if len(results) > 1:
                output.append("\n\n---\n\n**Other Results:**")
                for r in results[1:3]:
                    output.append(f"- [{r.title}]({r.url})")
            
            return "\n".join(output)
            
        except Exception as e:
            # Return search results if fetch fails
            return self.format_search_results(results) + f"\n\n⚠️ Could not fetch content: {str(e)}"
    
    def is_url(self, text: str) -> bool:
        """Check if text is a URL.
        
        Args:
            text: Text to check.
            
        Returns:
            True if URL.
        """
        try:
            result = urlparse(text)
            return all([result.scheme in ('http', 'https'), result.netloc])
        except Exception:
            return False
    
    # =========================================================================
    # Caching
    # =========================================================================
    
    def _get_cache_path(self, url: str) -> str:
        """Get cache file path for URL."""
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")
    
    def _cache_document(self, doc: WebDocument):
        """Cache a fetched document."""
        cache_path = self._get_cache_path(doc.url)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "url": doc.url,
                    "title": doc.title,
                    "content": doc.content,
                    "content_type": doc.content_type,
                    "fetched_at": doc.fetched_at,
                    "word_count": doc.word_count
                }, f)
        except Exception:
            pass
    
    def _get_cached_document(self, url: str) -> Optional[WebDocument]:
        """Get cached document if exists."""
        cache_path = self._get_cache_path(url)
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return WebDocument(**data)
        except Exception:
            pass
        return None
    
    def fetch_with_cache(self, url: str, max_age_hours: int = 24) -> WebDocument:
        """Fetch URL with caching.
        
        Args:
            url: URL to fetch.
            max_age_hours: Maximum cache age in hours.
            
        Returns:
            WebDocument with content.
        """
        # Check cache
        cached = self._get_cached_document(url)
        if cached:
            try:
                cached_time = datetime.fromisoformat(cached.fetched_at)
                age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                if age_hours < max_age_hours:
                    return cached
            except Exception:
                pass
        
        # Fetch fresh
        doc = self.fetch_url(url)
        self._cache_document(doc)
        return doc


# Keep track of last search results for /fetch <number>
_last_search_results: List[SearchResult] = []


def get_last_search_results() -> List[SearchResult]:
    """Get last search results."""
    return _last_search_results


def set_last_search_results(results: List[SearchResult]):
    """Set last search results."""
    global _last_search_results
    _last_search_results = results
