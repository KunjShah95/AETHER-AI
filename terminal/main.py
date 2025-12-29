from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
import yaml
import re
import logging
from datetime import datetime
import sys
import time
import requests
from typing import Dict, Optional
import shlex
import hashlib
import os
import json
import threading
import openai
import ollama
from groq import Groq
try:
    import google.generativeai as genai
    from PIL import Image as PILImage  # Rename to avoid conflict if any
except ImportError:
    genai = None
    PILImage = None
from dotenv import load_dotenv

load_dotenv()
# --- New Feature Imports ---
try:
    from terminal.history import HistoryManager
    from terminal.plugin_manager import PluginManager
    from terminal.voice import VoiceManager
    from terminal.rag import RAGManager
    from terminal.analytics import AnalyticsManager
except ImportError:
    # Fallback for direct execution or different path structure
    try:
        from history import HistoryManager
        from plugin_manager import PluginManager
        from voice import VoiceManager
        from rag import RAGManager
        from analytics import AnalyticsManager
    except ImportError:
        # Define dummy classes if imports fail to prevent crash
        class HistoryManager: save_session = lambda *a: None
        class PluginManager: load_plugins = lambda *a: None
        class VoiceManager: is_available = lambda: False
        class RAGManager: query = lambda *a: []
        class AnalyticsManager: log_usage = lambda *a: None

# Optional advanced feature imports are loaded lazily to improve startup time.

# Import new advanced modules
try:
    from terminal.context_aware_ai import ContextAwareAI
    from terminal.analytics_monitor import AnalyticsMonitor
    from terminal.games_learning import GamesLearning
    from terminal.creative_tools import CreativeTools
    from terminal.advanced_security import AdvancedSecurity
    from terminal.task_manager import TaskManager
    from terminal.theme_manager import ThemeManager
    from terminal.integration_hub import IntegrationHub
    from terminal.code_review_assistant import CodeReviewAssistant
    from terminal.docker_manager import DockerManager
    from terminal.snippet_manager import SnippetManager
    from terminal.persona_manager import PersonaManager
    from terminal.network_tools import NetworkTools
    from terminal.games_tui import GamesTUI
    from terminal.dashboard_tui import NexusDashboard
    from terminal.api_client import APIClient
    from terminal.database_manager import DatabaseManager
    from terminal.package_manager_integration import PackageManager
    from terminal.test_runner import TestRunner
    from terminal.file_watcher import FileWatcher
    from terminal.cloud_integration import CloudIntegration
    from terminal.blockchain import BlockchainManager
    from terminal.ml_ops import MLOpsManager
except ImportError:
    try:
        from context_aware_ai import ContextAwareAI
        from analytics_monitor import AnalyticsMonitor
        from games_learning import GamesLearning
        from creative_tools import CreativeTools
        from advanced_security import AdvancedSecurity
        from task_manager import TaskManager
        from theme_manager import ThemeManager
        from integration_hub import IntegrationHub
        from code_review_assistant import CodeReviewAssistant
        from docker_manager import DockerManager
        from snippet_manager import SnippetManager
        from persona_manager import PersonaManager
        from network_tools import NetworkTools
        from games_tui import GamesTUI
        from dashboard_tui import NexusDashboard
        from api_client import APIClient
        from database_manager import DatabaseManager
        from package_manager_integration import PackageManager
        from test_runner import TestRunner
        from file_watcher import FileWatcher
        from cloud_integration import CloudIntegration
        from blockchain import BlockchainManager
        from ml_ops import MLOpsManager
    except ImportError:
        ContextAwareAI = None
        AnalyticsMonitor = None
        GamesLearning = None
        CreativeTools = None
        AdvancedSecurity = None
        TaskManager = None
        ThemeManager = None
        CodeReviewAssistant = None
        IntegrationHub = None
        DockerManager = None
        SnippetManager = None
        PersonaManager = None
        NetworkTools = None
        GamesTUI = None
        NexusDashboard = None
        APIClient = None
        DatabaseManager = None
        PackageManager = None
        TestRunner = None
        FileWatcher = None
        CloudIntegration = None
        BlockchainManager = None
        MLOpsManager = None

# Import new feature modules (context engine, dev tools, mcp, streaming)
try:
    from terminal.context_engine import ContextEngine, create_default_templates
    from terminal.developer_tools import DeveloperTools
    from terminal.mcp_manager import MCPManager
    from terminal.streaming import StreamingHandler, PipeInputHandler
    from terminal.skills_manager import SkillsManager
    from terminal.skills_manager import SkillsManager
    from terminal.web_search import WebSearcher, set_last_search_results, get_last_search_results
    from terminal.web_agent import WebAgent
except ImportError:
    try:
        from context_engine import ContextEngine, create_default_templates
        from developer_tools import DeveloperTools
        from mcp_manager import MCPManager
        from streaming import StreamingHandler, PipeInputHandler
        from skills_manager import SkillsManager
        from web_search import WebSearcher, set_last_search_results, get_last_search_results
        from web_agent import WebAgent
    except ImportError:
        ContextEngine = None
        DeveloperTools = None
        MCPManager = None
        StreamingHandler = None
        PipeInputHandler = None
        create_default_templates = None
        SkillsManager = None
        WebSearcher = None
        set_last_search_results = None
        get_last_search_results = None

# Import advanced feature modules
try:
    from terminal.code_agent import CodeAgent
    from terminal.smart_rag import SmartRAG
    from terminal.workflow_engine import WorkflowEngine, StepType
    from terminal.pair_programmer import PairProgrammer
except ImportError:
    try:
        from code_agent import CodeAgent
        from smart_rag import SmartRAG
        from workflow_engine import WorkflowEngine, StepType
        from pair_programmer import PairProgrammer
    except ImportError:
        CodeAgent = None
        SmartRAG = None
        WorkflowEngine = None
        StepType = None
        PairProgrammer = None

# Cache for Ollama model list to avoid repeated expensive calls.
from functools import lru_cache
import subprocess

def run_cli_list() -> str:
    """Run 'ollama list' command via subprocess."""
    try:
        # Try running ollama list
        # Use shell=True on Windows if needed, but list is usually safe
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        logging.warning("Ollama CLI not found in PATH")
    except Exception as e:
        logging.error(f"Error running ollama list: {e}")
    return ""

@lru_cache(maxsize=1)
def _get_ollama_models_list() -> list[str]:
    """Retrieve Ollama models once per process run."""
    models = []
    
    # 1. Try Python client first
    try:
        res = ollama.list()
        if hasattr(res, 'models'):
            model_list = res.models
        elif isinstance(res, dict) and res.get('models'):
            model_list = res.get('models')
        elif isinstance(res, list):
            model_list = res
        else:
            model_list = []
        
        for m in model_list:
            if hasattr(m, 'model'):
                name = m.model
            elif hasattr(m, 'name'):
                name = m.name
            elif isinstance(m, dict):
                name = m.get('name') or m.get('model') or str(m)
            else:
                name = str(m)
            models.append(name)
            
        if models:
            return models
            
    except Exception as e:
        logging.warning(f"Ollama Python client failed: {e}")

    # 2. Fallback to CLI
    try:
        raw = run_cli_list()
        if raw:
            lines = raw.strip().splitlines()
            # Skip header if present
            if lines and lines[0].upper().startswith("NAME"):
                lines = lines[1:]
            
            for line in lines:
                parts = line.split()
                if parts:
                    models.append(parts[0])
                    
    except Exception as e:
        logging.error(f"Ollama CLI parsing failed: {e}")
    
    return models

# --- Configuration ---
load_dotenv()

# Initialize console with default theme
console = Console()

# Global theme manager reference (will be set when NexusAI initializes)
global_theme_manager = None

def update_console_theme(theme_manager=None):
    """Update the global console theme"""
    global console, global_theme_manager
    if theme_manager:
        global_theme_manager = theme_manager
        rich_theme = theme_manager.get_rich_theme()
        console = Console(theme=rich_theme)
    else:
        console = Console()

logging.basicConfig(
    filename='ai_assistant.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Settings - resolve at runtime to respect test env overrides (HOME) and Windows USERPROFILE
def _get_home_dir() -> str:
    # Prefer HOME (for tests), then USERPROFILE (Windows), then expanduser
    return os.getenv('HOME') or os.getenv('USERPROFILE') or os.path.expanduser('~')


def CONFIG_PATH() -> str:
    return os.path.join(_get_home_dir(), '.aetherai', 'config.yaml')


def USER_DB_PATH() -> str:
    return os.path.join(_get_home_dir(), '.aetherai', 'users.json')
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.5
VERSION = "1.0.0"
REQUEST_TIMEOUT = 30

# Enhanced Output Configuration
MAX_OUTPUT_TOKENS = 4000  # Increased from 1000
MAX_RESPONSE_LENGTH = 8000  # Increased from 2000
MAX_INPUT_LENGTH = 4000  # Increased from 1000
CONTEXT_HISTORY_SIZE = 10  # Number of previous messages to include

# Security settings
ALLOWED_DOMAINS = [
    "api.groq.com",
    "generativelanguage.googleapis.com",
    "api-inference.huggingface.co"
]

# --- Custom Exceptions ---
class SecurityError(Exception):
    pass

class APIError(Exception):
    pass

# --- Security Manager ---
class SecurityManager:
    def __init__(self):
        self.blocklist_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r"sudo\s", r"rm\s+-[rf]", r"chmod\s+777",
                r"wget\s", r"curl\s", r"\|\s*sh",
                r">\s*/dev", r"nohup", r"fork\(\)",
                r"eval\(", r"base64_decode", r"UNION\s+SELECT",
                r"DROP\s+TABLE", r"<script", r"javascript:"
            ]
        ]
        # Allowlist for commands and file extensions
        self.allowed_commands = {
            'ls', 'pwd', 'whoami', 'date', 'uptime', 'echo', 'cat', 'head', 'tail', 'df', 'du', 'free', 'uname', 'id', 'git'
        }
        self.allowed_file_extensions = {'.txt', '.log', '.md', '.csv', '.json', '.py', '.js', '.html', '.css'}
        self.violation_count = 0
        self.violation_threshold = 5

        # Common prompt-injection phrases to detect when user-supplied content
        # will be sent to an LLM. These are conservative heuristics.
        self.prompt_injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r"ignore (previous|all) instructions",
                r"disregard (previous|earlier) instructions",
                r"you are now",
                r"from now on",
                r"follow these instructions",
                r"do not follow the",
                r"respond only with",
            ]
        ]

    def sanitize(self, input_str: str) -> str:
        if not isinstance(input_str, str) or len(input_str) > 10000:
            self.log_violation('Input too long or not a string')
            raise SecurityError("Invalid input")
        if not self.is_printable(input_str):
            self.log_violation('Non-printable characters detected')
            raise SecurityError("Input contains non-printable characters")
        if self.has_suspicious_unicode(input_str):
            self.log_violation('Suspicious unicode detected')
            raise SecurityError("Input contains suspicious unicode characters")
        
        sanitized = input_str.strip().replace("\0", "")
        for pattern in self.blocklist_patterns:
            if pattern.search(sanitized):
                self.log_violation('Blocked pattern detected')
                raise SecurityError("Blocked dangerous pattern")
        return sanitized

    def validate_api_key(self, key: str, provider: str = "generic") -> bool:
        if not key or not isinstance(key, str):
            return False
        rules = {
            "gemini": {"min_length": 30, "prefixes": ["AI"]},
            "groq": {"min_length": 40, "prefixes": ["gsk_"]},
            "huggingface": {"min_length": 30, "prefixes": ["hf_"]},
            "generic": {"min_length": 20, "prefixes": []}
        }
        rule = rules.get(provider.lower(), rules["generic"])
        if len(key) < rule["min_length"]:
            return False
        if rule["prefixes"] and not any(key.startswith(p) for p in rule["prefixes"]):
            return False
        return re.match(r'^[a-zA-Z0-9_-]+$', key) is not None

    def is_command_allowed(self, cmd: str) -> bool:
        return cmd in self.allowed_commands

    def is_file_extension_allowed(self, filename: str) -> bool:
        return any(filename.endswith(ext) for ext in self.allowed_file_extensions)

    def is_printable(self, s: str) -> bool:
        return all(32 <= ord(c) <= 126 or c in '\n\r\t' for c in s)

    def has_suspicious_unicode(self, s: str) -> bool:
        # Block invisible, right-to-left, or control unicode chars
        suspicious = [
            '\u202e', '\u202d', '\u202a', '\u202b', '\u202c', '\u200b', '\ufeff', '\u2066', '\u2067', '\u2068', '\u2069'
        ]
        return any(code in s for code in suspicious)

    def log_violation(self, reason: str):
        self.violation_count += 1
        logging.warning(f"Security violation: {reason}")
        if self.violation_count >= self.violation_threshold:
            logging.critical("Repeated security violations detected!")
            # Optionally, trigger alert/lockout here

    def detect_prompt_injection(self, s: str) -> bool:
        """Basic heuristic detection for prompt-injection phrases."""
        try:
            if not isinstance(s, str):
                return False
            for pattern in self.prompt_injection_patterns:
                if pattern.search(s):
                    logging.warning("Prompt injection pattern detected")
                    return True
        except Exception:
            pass
        return False

    def validate_url(self, url: str) -> bool:
        # Only allow http(s) and block local addresses
        if not url.startswith(('http://', 'https://')):
            return False
        if re.search(r'(localhost|127\\.0\\.1|0\\.0\\.0\\.0|::1)', url):
            return False
        return True

# --- Prompt Cache ---
from collections import OrderedDict
class PromptCache:
    def __init__(self, ttl=5, maxsize=100):
        self.ttl = ttl
        self.maxsize = maxsize
        self.store = OrderedDict()
    def get(self, key):
        entry = self.store.get(key)
        if entry and time.time() - entry[1] < self.ttl:
            return entry[0]
        return None
    def set(self, key, value):
        if len(self.store) >= self.maxsize:
            self.store.popitem(last=False)
        self.store[key] = (value, time.time())

_prompt_cache = PromptCache()

# --- Thread Pool ---
from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=4)

def _run_in_background(fn, *args, **kwargs):
    return _executor.submit(fn, *args, **kwargs)

# --- AI Manager ---
class RateLimiter:
    """Simple in-memory rate limiter per user.

    Window-based counter: allows `limit` requests per `window_seconds`.
    """
    def __init__(self, limit: int = 10, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._stores = {}

    def allow(self, user_id: str) -> bool:
        now = time.time()
        wins = self._stores.setdefault(user_id, [])
        # Remove timestamps outside of the sliding window
        while wins and wins[0] <= now - self.window_seconds:
            wins.pop(0)
        if len(wins) >= self.limit:
            return False
        wins.append(now)
        return True

# --- AI Manager ---
class AIManager:
    def __init__(self):
        self.security = SecurityManager()
        self.gemini = None
        self.groq = None
        self.session = self._create_session()
        self.status = {
            "gemini": "Not configured",
            "groq": "Not configured",
            "ollama": "Ready" if self._check_ollama() else "Not installed",
            "huggingface": "Ready (requires token)",
            "chatgpt": "Not configured",
            "mcp": "Not configured"
        }
        self._init_services()
        
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        # Ensure TLS verification is enabled. Do not set a global timeout on Session;
        # timeouts should be applied per-request to avoid unexpected behavior.
        session.verify = True
        session.headers.update({
            'User-Agent': f'NexusAI/{VERSION}',
            'Accept': 'application/json'
        })
        return session
        
    def _init_services(self):
        # Cache env vars once
        _GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        _GROQ_KEY = os.getenv("GROQ_API_KEY")
        _OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        _MCP_KEY = os.getenv("MCP_API_KEY")

        # Gemini 2.0 Flash (Fixed API)
        gemini_key = _GEMINI_KEY
        if gemini_key and self.security.validate_api_key(gemini_key, "gemini"):
            try:
                genai.configure(api_key=gemini_key)
                self.gemini = genai.GenerativeModel("gemini-2.0-flash-exp")
                self.status["gemini"] = " Ready"
                logging.info("Gemini 2.0 Flash initialized successfully")
            except Exception as e:
                self.status["gemini"] = f" Error: {str(e)[:50]}..."
                logging.error(f"Gemini init failed: {str(e)}")
        elif gemini_key:
            self.status["gemini"] = " Invalid API key format"
        
        # Groq Cloud
        groq_key = _GROQ_KEY
        if groq_key and self.security.validate_api_key(groq_key, "groq"):
            try:
                self.groq = Groq(api_key=groq_key)
                self.status["groq"] = " Ready"
                logging.info("Groq service initialized")
            except Exception as e:
                self.status["groq"] = f" Error: {str(e)[:50]}..."
                logging.error(f"Groq init failed: {str(e)}")
        elif groq_key:
            self.status["groq"] = " Invalid API key format"
        
        # Ollama local
        if self.status["ollama"] == "Ready":
            try:
                models = self._get_ollama_models()
                self.status["ollama"] = f" Ready ({models})" if models != "Unknown" else " No models"
            except Exception as e:
                self.status["ollama"] = f" Error: {str(e)[:50]}..."
                logging.error(f"Ollama model check failed: {str(e)}")
        else:
            self.status["ollama"] = " Not installed"
        
        # ChatGPT (OpenAI)
        openai_key = _OPENAI_KEY
        if openai_key and self.security.validate_api_key(openai_key, "generic"):
            try:
                openai.api_key = openai_key
                self.status["chatgpt"] = " Ready"
                logging.info("ChatGPT (OpenAI) initialized successfully")
            except Exception as e:
                self.status["chatgpt"] = f" Error: {str(e)[:50]}..."
                logging.error(f"ChatGPT init failed: {str(e)}")
        elif openai_key:
            self.status["chatgpt"] = " Invalid API key format"
        
        # MCP (Model Context Protocol)
        mcp_key = _MCP_KEY
        if mcp_key and self.security.validate_api_key(mcp_key, "generic"):
            self.status["mcp"] = " Ready"
        elif mcp_key:
            self.status["mcp"] = " Invalid API key format"
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running and has models."""
        try:
            models = _get_ollama_models_list()
            return bool(models)
        except Exception:
            return False
    
    def _get_ollama_models(self) -> str:
        """Get a formatted string of available Ollama models."""
        try:
            models = _get_ollama_models_list()
            if models:
                return ", ".join(models[:3]) + ("..." if len(models) > 3 else "")
        except Exception as e:
            logging.warning(f"Error getting Ollama models: {str(e)}")
        return "Unknown"
    
    def _query_huggingface(self, prompt: str) -> str:
        try:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token or not self.security.validate_api_key(hf_token, "huggingface"):
                return " HuggingFace token not configured or invalid"
            
            response = self.session.post(
                "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"inputs": prompt[:MAX_INPUT_LENGTH]}
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "No response")[:MAX_RESPONSE_LENGTH]
            
            return f" HuggingFace API Error: {response.status_code}"
        except Exception as e:
            return f" HuggingFace unavailable: {str(e)[:50]}..."
    
    def query(self, model: str, prompt: str) -> str:
        if not prompt or len(prompt.strip()) == 0:
            return " Empty prompt provided"
        
        try:
            clean_prompt = self.security.sanitize(prompt)
        except SecurityError as e:
            return f" Security error: {str(e)}"
        
        # Check cache
        cache_key = (model, hashlib.sha256(clean_prompt.encode()).hexdigest())
        cached = _prompt_cache.get(cache_key)
        if cached:
            return cached
        
        for attempt in range(MAX_RETRIES):
            try:
                if model == "gemini" and self.gemini:
                    response = self.gemini.generate_content(
                        clean_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=MAX_OUTPUT_TOKENS
                        )
                    )
                    
                    if not response or not hasattr(response, 'text') or not response.text:
                        raise APIError("Invalid Gemini response")
                    result_text = response.text[:MAX_RESPONSE_LENGTH]
                    _prompt_cache.set(cache_key, result_text)
                    return result_text
                
                elif model == "groq" and self.groq:
                    response = self.groq.chat.completions.create(
                        messages=[{"role": "user", "content": clean_prompt}],
                        model="mixtral-8x7b-32768",
                        max_tokens=MAX_OUTPUT_TOKENS
                    )
                    
                    if not response or not response.choices or not response.choices[0].message.content:
                        raise APIError("Invalid Groq response")
                    
                    result_text = response.choices[0].message.content[:MAX_RESPONSE_LENGTH]
                    _prompt_cache.set(cache_key, result_text)
                    return result_text
                
                elif model == "ollama" or model.startswith("ollama:"):
                    if not self._check_ollama():
                        return " Ollama not available"
                    
                    # Extract specific model name if provided
                    if ":" in model:
                        ollama_model = model.split(":", 1)[1]
                    else:
                        ollama_model = "llama3"  # Default fallback
                    
                    try:
                        response = ollama.chat(
                            model=ollama_model,
                            messages=[{"role": "user", "content": clean_prompt}]
                        )
                        
                        # Handle both dict and ChatResponse object
                        if hasattr(response, 'message'):
                            # It's a ChatResponse object
                            if hasattr(response.message, 'content'):
                                content = response.message.content
                            else:
                                content = str(response.message)
                        elif isinstance(response, dict) and "message" in response:
                            # It's a dict
                            content = response["message"].get("content", "")
                        else:
                            raise APIError(f"Invalid Ollama response type: {type(response)}")
                        
                        if not content:
                            raise APIError("Empty Ollama response")
                        
                        result_text = content[:MAX_RESPONSE_LENGTH]
                        _prompt_cache.set(cache_key, result_text)
                        return result_text
                    except Exception as e:
                        return f" Error: {str(e)}"
                elif model == "huggingface":
                    result_text = self._query_huggingface(clean_prompt)
                    _prompt_cache.set(cache_key, result_text)
                    return result_text
                
                elif model == "chatgpt":
                    openai_key = os.getenv("OPENAI_API_KEY")
                    if not openai_key or not self.security.validate_api_key(openai_key, "generic"):
                        return " OpenAI API key not configured or invalid"
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": clean_prompt}],
                        max_tokens=MAX_OUTPUT_TOKENS
                    )
                    if not response or not response.choices or not response.choices[0].message.content:
                        raise APIError("Invalid ChatGPT response")
                    result_text = response.choices[0].message.content[:MAX_RESPONSE_LENGTH]
                    _prompt_cache.set(cache_key, result_text)
                    return result_text
                
                elif model == "mcp":
                    mcp_key = os.getenv("MCP_API_KEY")
                    mcp_url = os.getenv("MCP_URL", "http://localhost:8080/api/v1/completions")
                    if not mcp_key:
                        return " MCP API key not configured"
                    headers = {"Authorization": f"Bearer {mcp_key}", "Content-Type": "application/json"}
                    data = {"prompt": clean_prompt, "max_tokens": MAX_OUTPUT_TOKENS}
                    try:
                        resp = self.session.post(mcp_url, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
                        if resp.status_code == 200:
                            result = resp.json()
                            result_text = result.get("text", "No response")[:MAX_RESPONSE_LENGTH]
                            _prompt_cache.set(cache_key, result_text)
                            return result_text
                        return f" MCP API Error: {resp.status_code}"
                    except Exception as e:
                        return f" MCP unavailable: {str(e)[:50]}..."
                
                else:
                    return f" Model '{model}' not available"
                
            except APIError as e:
                if attempt == MAX_RETRIES - 1:
                    return f" {model} API error: {str(e)}"
                time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
                
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed for {model}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return f" {model} error: {str(e)[:50]}..."
                time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
        
        return f" {model} unavailable after {MAX_RETRIES} attempts"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Analyze an image using a vision-capable model (Gemini)."""
        if not self.gemini:
            return "❌ Gemini model not active (required for vision)"
        if not PILImage:
            return "❌ PIL/Pillow not installed (pip install pillow)"        
        if not os.path.exists(image_path):
            return f"❌ Image not found: {image_path}"
        
        try:
            img = PILImage.open(image_path)
            
            # Use specific vision model if needed, but 2.0 Flash supports it
            model = self.gemini
            
            response = model.generate_content([prompt, img])
            
            if not response or not hasattr(response, 'text'):
                return "❌ No response from vision model"
                
            return response.text
        except Exception as e:
            return f"❌ Vision analysis failed: {str(e)}"
# --- User Management ---
class UserManager:
    def __init__(self):
        self.user_db = self._load_users()
        self.current_user = None
        self.chat_history = {}
        self.admins = set(["admin"])  # Default admin user
        self.last_active = {}
        self.activity_log = {}
        self.audit_log = []
        self.session_timeout = 900  # 15 minutes
        self._start_timeout_thread()

    def _load_users(self):
        path = USER_DB_PATH()
        if os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    return json.load(f)
                except Exception:
                    return {}
        return {}

    def _save_users(self):
        path = USER_DB_PATH()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.user_db, f)
        try:
            os.chmod(path, 0o600)
        except Exception:
            # On Windows, os.chmod with those modes may fail; ignore
            pass

    def hash_password(self, password) -> str:
        """Hash a password using bcrypt (via passlib). If passlib isn't available,
        fall back to SHA256 for compatibility (but log a warning).
        New hashes will use bcrypt; legacy SHA256 hashes will be detected and migrated on login.
        """
        try:
            from passlib.hash import bcrypt
            try:
                # Use passlib bcrypt to create a salted hash
                return bcrypt.hash(password)
            except Exception as e:
                # If bcrypt backend fails at runtime (some systems have broken bcrypt),
                # fall back to SHA256 but log the error for diagnostics.
                logging.warning(f"bcrypt hashing failed, falling back to SHA256: {e}")
                return hashlib.sha256(password.encode()).hexdigest()
        except Exception:
            logging.warning("passlib not available; falling back to SHA256 (insecure)")
            return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a plaintext password against a stored hash.

        Supports passlib bcrypt hashes and legacy SHA256 hex digests.
        """
        if not stored_hash or not isinstance(stored_hash, str):
            return False
        # Try passlib bcrypt verification first
        try:
            from passlib.hash import bcrypt
            try:
                return bcrypt.verify(password, stored_hash)
            except Exception:
                # If verification fails due to backend issues, fall through to sha256 check
                pass
        except Exception:
            # passlib not available; fall back to SHA256
            pass

        # Legacy SHA256 hex digest
        try:
            return stored_hash == hashlib.sha256(password.encode()).hexdigest()
        except Exception:
            return False

    def signup(self, username, password, is_admin=False):
        if username in self.user_db:
            return False, "User already exists."
        self.user_db[username] = {
            "password": self.hash_password(password),
            "api_keys": {},
            "model": "gemini",
            "role": "admin" if is_admin else "user"
        }
        self._save_users()
        return True, "Signup successful."

    def login(self, username, password):
        user = self.user_db.get(username)
        if not user:
            return False, "Invalid username or password."

        stored = user.get("password")

        # Verify password against stored hash (supports bcrypt via passlib and legacy sha256)
        try:
            if self._verify_password(password, stored):
                # If stored was a legacy SHA256 and passlib is available, migrate to bcrypt
                try:
                    # Detect legacy sha256 by length (64 hex chars)
                    if isinstance(stored, str) and len(stored) == 64:
                        # Attempt to re-hash using hash_password (which will prefer bcrypt)
                        new_hash = self.hash_password(password)
                        self.user_db[username]["password"] = new_hash
                        self._save_users()
                        return True, f"Welcome, {username}! (password migrated)"
                except Exception:
                    # If migration fails, continue to login normally
                    pass

                self.current_user = username
                return True, f"Welcome, {username}!"
        except Exception:
            # Fall through to invalid login
            pass

        return False, "Invalid username or password."

    def set_api_key(self, provider, key):
        if not self.current_user:
            return False, "Not logged in."
        self.user_db[self.current_user]["api_keys"][provider] = key
        self._save_users()
        return True, f"API key for {provider} set."

    def get_api_key(self, provider):
        if not self.current_user:
            return None
        return self.user_db[self.current_user]["api_keys"].get(provider)

    def set_model(self, model):
        if not self.current_user:
            return False, "Not logged in."
        self.user_db[self.current_user]["model"] = model
        self._save_users()
        return True, f"Model set to {model}."

    def get_model(self):
        if not self.current_user:
            return "gemini"
        return self.user_db[self.current_user].get("model", "gemini")

    def is_admin(self, username=None):
        user = username or self.current_user
        if not user:
            return False
        return self.user_db.get(user, {}).get("role") == "admin"

    def reset_password(self, username, newpassword):
        if username not in self.user_db:
            return False, "User not found."
        self.user_db[username]["password"] = self.hash_password(newpassword)
        self._save_users()
        return True, "Password reset."

    def list_users(self):
        return list(self.user_db.keys())

    def add_history(self, username, message):
        if username not in self.chat_history:
            self.chat_history[username] = []
        self.chat_history[username].append(message)
        if len(self.chat_history[username]) > 50:
            self.chat_history[username] = self.chat_history[username][-50:]

    def get_history(self, username):
        return self.chat_history.get(username, [])

    def clear_history(self, username):
        self.chat_history[username] = []

    def _start_timeout_thread(self):
        def timeout_checker():
            while True:
                now = time.time()
                for user, last in list(self.last_active.items()):
                    if self.current_user == user and now - last > self.session_timeout:
                        self.current_user = None
                time.sleep(60)
        t = threading.Thread(target=timeout_checker, daemon=True)
        t.start()

    def update_activity(self, username, action):
        def _do_update():
            self.last_active[username] = time.time()
            if username not in self.activity_log:
                self.activity_log[username] = []
            self.activity_log[username].append((time.strftime('%Y-%m-%d %H:%M:%S'), action))
            if len(self.activity_log[username]) > 100:
                self.activity_log[username] = self.activity_log[username][-100:]
            self.audit_log.append((username, time.strftime('%Y-%m-%d %H:%M:%S'), action))
            if len(self.audit_log) > 500:
                self.audit_log = self.audit_log[-500:]
        _run_in_background(_do_update)

# --- Core Application ---
# --- Core Application ---
class AetherAI:
    def __init__(self, quiet: bool = False):
        self.user_manager = UserManager()

        self.ai = AIManager()
        self.security = SecurityManager()
        self.current_model = self._load_config()
        self.allowed_commands = [
            'ls', 'pwd', 'whoami', 'date', 'uptime', 'echo', 'cat', 'head', 'tail', 'df', 'du', 'free', 'uname', 'id', 'git'
        ]
        
        # Initialize new core features
        self.history_manager = HistoryManager()
        self.plugin_manager = PluginManager()
        self.voice_manager = VoiceManager()
        self.rag_manager = RAGManager()
        self.analytics_manager = AnalyticsManager()
        
        # Load plugins
        self.plugin_manager.load_plugins()
        
        # Initialize advanced modules
        self.context_ai = ContextAwareAI() if ContextAwareAI else None
        self.analytics = AnalyticsMonitor() if AnalyticsMonitor else None
        self.games = GamesLearning() if GamesLearning else None
        self.creative = CreativeTools() if CreativeTools else None
        self.adv_security = AdvancedSecurity() if AdvancedSecurity else None
        self.task_manager = TaskManager() if TaskManager else None
        self.theme_manager = ThemeManager() if ThemeManager else None
        self.code_reviewer = CodeReviewAssistant() if CodeReviewAssistant else None
        self.integration_hub = IntegrationHub() if IntegrationHub else None
        self.docker_manager = DockerManager() if DockerManager else None
        self.snippet_manager = SnippetManager() if SnippetManager else None
        self.persona_manager = PersonaManager() if PersonaManager else None
        self.persona_manager = PersonaManager() if PersonaManager else None
        self.network_tools = NetworkTools() if NetworkTools else None
        self.web_agent = WebAgent() if WebAgent else None
        self.cloud_integration = CloudIntegration() if CloudIntegration else None
        self.blockchain = BlockchainManager() if BlockchainManager else None
        self.ml_ops = MLOpsManager() if MLOpsManager else None
        
        # Initialize new feature modules
        self.context_engine = ContextEngine() if ContextEngine else None
        self.developer_tools = DeveloperTools(ai_query_func=self.ai.query if hasattr(self, 'ai') else None) if DeveloperTools else None
        self.mcp_manager = MCPManager() if MCPManager else None
        self.streaming_handler = StreamingHandler(console) if StreamingHandler else None
        self.pipe_handler = PipeInputHandler() if PipeInputHandler else None
        
        # Initialize skills manager and web searcher
        self.skills_manager = SkillsManager() if SkillsManager else None
        self.web_searcher = WebSearcher() if WebSearcher else None
        
        # Initialize advanced modules
        self.code_agent = CodeAgent(ai_query_func=self.ai.query if hasattr(self, 'ai') else None) if CodeAgent else None
        self.smart_rag = SmartRAG() if SmartRAG else None
        self.workflow_engine = WorkflowEngine(ai_query_func=self.ai.query if hasattr(self, 'ai') else None) if WorkflowEngine else None
        self.pair_programmer = PairProgrammer(ai_query_func=self.ai.query if hasattr(self, 'ai') else None) if PairProgrammer else None
        
        # Create default templates
        if self.context_engine and create_default_templates:
            try:
                create_default_templates(self.context_engine)
            except Exception:
                pass
        
        # Update console theme if theme manager is available
        if self.theme_manager:
            update_console_theme(self.theme_manager)

        # Descriptions for models (used by /models and /current-model handlers)
        self.model_descriptions = {
            "gemini": "Google's Gemini 2.0 Flash (Fixed API)",
            "groq": "Groq Cloud - Mixtral 8x7B",
            "ollama": "Local Ollama Models (Most Secure)",
            "huggingface": "HuggingFace Inference API",
            "chatgpt": "OpenAI's ChatGPT (API)",
            "mcp": "Model Context Protocol (API)"
        }

        if not quiet:
            self.show_banner()
    
    def _load_config(self) -> str:
        try:
            cfg = CONFIG_PATH()
            if os.path.exists(cfg):
                with open(cfg) as f:
                    config = yaml.safe_load(f)
                    if isinstance(config, dict):
                        model = config.get("default_model", "gemini")
                        if model in ["gemini", "groq", "ollama", "huggingface"]:
                            return model
        except:
            pass
        return "gemini"
    
    def _save_config(self):
        try:
            cfg = CONFIG_PATH()
            os.makedirs(os.path.dirname(cfg), exist_ok=True)
            with open(cfg, "w") as f:
                yaml.dump({"default_model": self.current_model}, f)
            # Restrict config file permissions (owner read/write only)
            try:
                os.chmod(cfg, 0o600)
            except Exception:
                pass
        except Exception as e:
            logging.error(f"Config save error: {str(e)}")
    
    def show_banner(self):
        """Display the AetherAI startup banner with ASCII art and status."""
        # ASCII Art Banner
        banner_art = r"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗  █████╗ ██╗   ║
    ║    ██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗██╔══██╗██║   ║
    ║    ███████║█████╗     ██║   ███████║█████╗  ██████╔╝███████║██║   ║
    ║    ██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗██╔══██║██║   ║
    ║    ██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║██║  ██║██║   ║
    ║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
"""
        
        # Print ASCII art with gradient effect
        lines = banner_art.strip().split('\n')
        gradient_styles = ['bold bright_cyan', 'bold cyan', 'bold blue', 'bold bright_blue', 
                          'bold magenta', 'bold bright_magenta', 'bold cyan', 'bold bright_cyan']
        
        console.print()
        for i, line in enumerate(lines):
            style = gradient_styles[i % len(gradient_styles)]
            console.print(Text(line, style=style))
        console.print()
        
        # Create info grid
        info_grid = Table.grid(padding=1)
        info_grid.add_column(justify="center", ratio=1)
        
        info_grid.add_row(Text("⚡ Production-Ready AI Terminal Assistant ⚡", style="bold white"))
        info_grid.add_row(Text(f"v{VERSION} • {self.current_model.upper()} • Secure", style="dim white"))
        
        console.print(Panel(
            info_grid,
            border_style="bright_blue",
            padding=(0, 2),
            title="[bold green]● Online[/bold green]",
            title_align="right"
        ))
        console.print()
        
        # AI Services Status Table with enhanced styling
        status_table = Table(
            title="🤖 AI Services Status",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            title_style="bold cyan"
        )
        status_table.add_column("Service", style="bold cyan", width=12)
        status_table.add_column("Status", width=12)
        status_table.add_column("Description", style="dim white", width=38)
        
        # Add status for each service
        for service, status in self.ai.status.items():
            if "Ready" in status:
                status_style = "[bold green]● Ready[/bold green]"
            elif "Error" in status:
                status_style = "[bold red]✗ Error[/bold red]"
            elif "Not configured" in status:
                status_style = "[yellow]○ Not Set[/yellow]"
            else:
                status_style = f"[dim]{status}[/dim]"
            
            desc = self.model_descriptions.get(service, "")
            status_table.add_row(service.upper(), status_style, desc)
        
        console.print(status_table)
        console.print()
        
        # Quick hints panel
        hints = Table.grid(padding=(0, 3))
        hints.add_column(style="bold cyan")
        hints.add_column(style="white")
        hints.add_column(style="bold cyan")
        hints.add_column(style="white")
        
        hints.add_row("/help", "All commands", "/models", "AI services")
        hints.add_row("/switch", "Change model", "/exit", "Exit")
        
        console.print(Panel(
            hints,
            border_style="dim",
            padding=(0, 1),
            title="[dim]Quick Commands[/dim]"
        ))
        console.print()
        
        console.print(f"[bold cyan]Current Model:[/bold cyan] [bold yellow]{self.current_model.upper()}[/bold yellow]")
        console.print("[dim]Type your question or /help for commands[/dim]")
        console.print()

    def execute_command(self, cmd: str) -> str:
        try:
            clean_cmd = self.security.sanitize(cmd)
            if len(clean_cmd) > 500:
                return "❌ Command too long (max 500 characters)"
            # Only allow commands in the allowlist
            parts = shlex.split(clean_cmd)
            if not parts or parts[0] not in self.allowed_commands:
                return f"❌ Command '{parts[0] if parts else ''}' not allowed. Allowed: {', '.join(self.allowed_commands)}"

            # Block wildcards, path traversal, shell metacharacters in arguments
            forbidden_patterns = [r"[><|;&]", r"\*", r"\.\.", r"/etc", r"/var", r"/root", r"\\"]
            for arg in parts[1:]:
                for pat in forbidden_patterns:
                    if re.search(pat, arg):
                        return "❌ Command arguments contain forbidden patterns."
            
            # Restrict file arguments for file commands to current directory only
            file_cmds = {"cat", "head", "tail"}
            if parts[0] in file_cmds and len(parts) > 1:
                for arg in parts[1:]:
                    if not os.path.abspath(arg).startswith(os.getcwd()):
                        return "❌ Only files in the current directory are allowed."
            
            # Log only the command name for audit (not arguments)
            logging.info(f"Command executed: {parts[0]}")
            
            result = subprocess.run(
                parts, capture_output=True,
                text=True, timeout=15
            )
            output = (result.stdout or "")[:2000]
            error = (result.stderr or "")[:2000]
            return output if output else error or "Command executed"
            
        except subprocess.TimeoutExpired:
            return "❌ Command timed out (15s limit)"
        except Exception as e:
            return f"❌ Error: {str(e)[:100]}..."
    
    def execute_git_command(self, git_cmd: str) -> str:
        """Execute Git commands with enhanced formatting and error handling"""
        try:
            # Check if we're in a Git repository
            if not os.path.exists('.git') and not os.path.exists('../.git'):
                return "❌ Not a Git repository. Initialize with 'git init' or navigate to a Git repository."
            
            clean_cmd = self.security.sanitize(git_cmd)
            if len(clean_cmd) > 500:
                return "❌ Git command too long (max 500 characters)"
            
            # Parse the command
            parts = shlex.split(clean_cmd)
            if not parts or parts[0] != 'git':
                return "❌ Invalid Git command format"
            
            # Execute the Git command
            result = subprocess.run(
                parts, capture_output=True,
                text=True, timeout=30,  # Git commands can take longer
                cwd=os.getcwd()
            )
            
            # Format the output based on command type
            if result.returncode == 0:
                output = result.stdout.strip()
                if not output:
                    return "✅ Git command executed successfully"
                
                # Format specific commands
                if 'status' in git_cmd:
                    return self._format_git_status(output)
                elif 'log' in git_cmd:
                    return self._format_git_log(output)
                elif 'diff' in git_cmd:
                    return self._format_git_diff(output)
                elif 'branch' in git_cmd and '--list' not in git_cmd:
                    return self._format_git_branch(output)
                else:
                    return output[:2000]  # Limit output size
            else:
                error = result.stderr.strip()
                return f"❌ Git error: {error[:500]}"
                
        except subprocess.TimeoutExpired:
            return "❌ Git command timed out (30s limit)"
        except Exception as e:
            return f"❌ Git command failed: {str(e)[:100]}"
    
    def _format_git_status(self, output: str) -> str:
        """Format git status output with colors and structure"""
        if not output:
            return "📁 Repository is clean - no changes to commit"
        
        lines = output.split('\n')
        formatted = "📊 Git Status:\n\n"
        
        staged = []
        unstaged = []
        untracked = []
        
        for line in lines:
            if line.startswith('M '):
                staged.append(f"📝 Modified: {line[3:]}")
            elif line.startswith('A '):
                staged.append(f"➕ Added: {line[3:]}")
            elif line.startswith('D '):
                staged.append(f"🗑️  Deleted: {line[3:]}")
            elif line.startswith('R '):
                staged.append(f"📋 Renamed: {line[3:]}")
            elif line.startswith('C '):
                staged.append(f"📄 Copied: {line[3:]}")
            elif line.startswith('?? '):
                untracked.append(f"❓ Untracked: {line[3:]}")
            elif line.startswith(' M'):
                unstaged.append(f"📝 Modified: {line[3:]}")
            elif line.startswith(' D'):
                unstaged.append(f"🗑️  Deleted: {line[3:]}")
        
        if staged:
            formatted += "✅ Staged Changes:\n" + "\n".join(staged) + "\n\n"
        if unstaged:
            formatted += "📝 Unstaged Changes:\n" + "\n".join(unstaged) + "\n\n"
        if untracked:
            formatted += "❓ Untracked Files:\n" + "\n".join(untracked) + "\n\n"
        
        return formatted
    
    def _format_git_log(self, output: str) -> str:
        """Format git log output with better readability"""
        if not output:
            return "📜 No commits found"
        
        lines = output.split('\n')
        formatted = "📜 Commit History:\n\n"
        
        for i, line in enumerate(lines[:20]):  # Limit to 20 commits
            if line.strip():
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    commit_hash = parts[0][:8]
                    message = parts[1]
                    formatted += f"🔗 {commit_hash}: {message}\n"
        
        if len(lines) > 20:
            formatted += f"\n... and {len(lines) - 20} more commits"
        
        return formatted
    
    def _format_git_diff(self, output: str) -> str:
        """Format git diff output with syntax highlighting"""
        if not output:
            return "📄 No differences found"
        
        # For now, just return the raw diff with a header
        # In a more advanced implementation, this could include syntax highlighting
        return f"📄 Changes:\n\n{output[:3000]}"  # Limit size
    
    def _format_git_branch(self, output: str) -> str:
        """Format git branch output"""
        if not output:
            return "🌿 No branches found"
        
        lines = output.split('\n')
        formatted = "🌿 Branches:\n\n"
        
        for line in lines:
            if line.strip():
                if line.startswith('*'):
                    formatted += f"🔥 Current: {line[2:]}\n"
                else:
                    formatted += f"🌿 Branch: {line.strip()}\n"
        
        return formatted
    
    def _handle_wake_word(self):
        """Called when wake word is detected."""
        console.print("\n[bold green]🎤 Wake Word Detected: Listening...[/bold green]")
        # We want to capture the command immediately after the wake word
        try:
             cmd_text = self.voice_manager.listen(timeout=5, phrase_time_limit=10)
             if cmd_text:
                 console.print(f"[bold cyan]You said: {cmd_text}[/bold cyan]")
                 response = self.process_input(cmd_text)
                 console.print(response)
             else:
                 console.print("[dim]No command detected.[/dim]")
        except Exception as e:
            console.print(f"[red]Error processing voice command: {e}[/red]")

    def process_input(self, user_input: str) -> str:
        try:
            clean_input = self.security.sanitize(user_input)
            if self.user_manager.current_user:
                self.user_manager.update_activity(self.user_manager.current_user, clean_input)
            
            # Track usage analytics
            if self.analytics:
                feature = "ai_chat" if not clean_input.startswith("/") else clean_input.split()[0][1:]
                self.analytics.track_usage(feature, self.user_manager.current_user or "anonymous")
            
            # Handle / commands
            if clean_input.startswith("/"):
                response = self.handle_command(clean_input)
                # Track response in context engine
                if self.context_engine:
                    self.context_engine.add_message("user", clean_input)
                    self.context_engine.add_message("assistant", response, {"type": "command"})
                return response
            
            # Shell command shortcut: !command
            if clean_input.startswith("!"):
                shell_cmd = clean_input[1:].strip()
                if shell_cmd:
                    if self.developer_tools:
                        return self.developer_tools.execute_shell_command(shell_cmd, safe_only=True)
                    else:
                        return self.execute_command(shell_cmd)
            
            # AI features for logged-in users
            if self.user_manager.current_user:
                self.user_manager.add_history(self.user_manager.current_user, clean_input)
                if clean_input.startswith("summarize "):
                    return self.ai.query(self.current_model, f"Summarize: {clean_input[10:]}")
                if clean_input.startswith("translate "):
                    return self.ai.query(self.current_model, f"Translate: {clean_input[10:]}")
                if clean_input.startswith("explain "):
                    return self.ai.query(self.current_model, f"Explain: {clean_input[8:]}")
            
            # Track user input in context
            if self.context_engine:
                self.context_engine.add_message("user", clean_input)
            
            # Build enhanced prompt with context
            enhanced_prompt = clean_input
            
            # Add skills and rules context if available
            if self.skills_manager:
                system_context = self.skills_manager.get_system_context()
                if system_context:
                    # Prepend system context for better AI awareness
                    enhanced_prompt = f"[System Context - Follow these guidelines]\n{system_context}\n\n[User Query]\n{clean_input}"
            
            # Add conversation context for multi-turn conversations
            if self.context_engine and len(self.context_engine.messages) > 1:
                # Get last few messages for context
                context_str = self.context_engine.get_context_string(max_messages=5)
                if context_str and len(context_str) < 3000:  # Limit context size
                    enhanced_prompt = f"[Previous Conversation]\n{context_str}\n\n[Current Query]\n{clean_input}"
            
            # Get AI response with enhanced context
            response = self.ai.query(self.current_model, enhanced_prompt)
            
            # Track response in context
            if self.context_engine:
                self.context_engine.add_message("assistant", response, {"model": self.current_model})
            
            # Voice output if enabled
            if self.voice_manager.enabled:
                self.voice_manager.speak(response)
            return response
        except SecurityError:
            # Track security errors
            if self.analytics:
                self.analytics.track_error("security_violation", "Input security violation")
            return "🔒 Security block: Input contains dangerous content"
        except Exception as e:
            # Track general errors
            if self.analytics:
                self.analytics.track_error("processing_error", str(e))
            logging.error(f"Processing error: {str(e)}")
            return "❌ System error - see logs for details"
    
    def handle_command(self, command: str) -> str:
        try:
            cmd = command[1:].strip().lower()
            # --- Session Timeout ---
            if cmd == "myactivity":
                if not self.user_manager.current_user:
                    return "Not logged in."
                log = self.user_manager.activity_log.get(self.user_manager.current_user, [])
                return "\n".join([f"{t}: {a}" for t, a in log]) or "No activity."
            if cmd == "auditlog":
                if not self.user_manager.is_admin():
                    return "Admin only."
                return "\n".join([f"{u} {t}: {a}" for u, t, a in self.user_manager.audit_log]) or "No audit log."
            # --- Advanced User Management ---
            if cmd.startswith("resetpw"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /resetpw [username] [newpassword]"
                if not self.user_manager.is_admin():
                    return "Admin only."
                ok, msg = self.user_manager.reset_password(parts[1], parts[2])
                return msg
            if cmd == "listusers":
                if not self.user_manager.is_admin():
                    return "Admin only."
                return "Users: " + ", ".join(self.user_manager.list_users())
            # --- New Features ---
            if cmd == "voice on":
                self.voice_manager.enabled = True
                return "🎙️ Voice enabled"
            if cmd == "voice off":
                self.voice_manager.enabled = False
                return "🔇 Voice disabled"
            
            if cmd == "voice wake on":
                if not self.voice_manager.is_available():
                    return "❌ Voice features not available"
                self.voice_manager.listen_for_wake_word("aether", self._handle_wake_word)
                return "🎤 Wake word 'Hey Aether' enabled! Just say it to command me."
            
            if cmd == "voice wake off":
                self.voice_manager.config.continuous_listen = False
                return "🔇 Wake word disabled."
            
            if cmd == "listen":
                if not self.voice_manager.is_available():
                    return "❌ Voice features not available (check requirements)"
                text = self.voice_manager.listen()
                if text:
                    return self.process_input(text)
                return "❌ Could not hear anything."
            
            if cmd.startswith("rag add "):
                text = command[9:]
                doc_id = f"doc_{int(time.time())}"
                if self.rag_manager.add_document(doc_id, text):
                    return f"✅ Added to knowledge base: {doc_id}"
                return "❌ Failed to add document"
            
            if cmd == "plugins list":
                plugins = self.plugin_manager.list_commands()
                if not plugins:
                    return "No plugins loaded."
                return "\n".join([f"🔌 {name}: {desc}" for name, desc in plugins.items()])
            
            if cmd == "analytics stats":
                stats = self.analytics_manager.get_stats()
                if not stats:
                    return "No analytics data."
                return "\n".join([f"📊 {k}: {v}" for k, v in stats.items()])
            
            if cmd == "dashboard start":
                try:
                    from terminal.dashboard_tui import NexusDashboard
                    app = NexusDashboard()
                    _run_in_background(app.run)
                    return "📊 Dashboard launched."
                except Exception as e:
                    return f"❌ Failed to launch dashboard: {e}"

            if cmd == "admin start":
                try:
                    from terminal.web_admin import start_server
                    _run_in_background(start_server)
                    return "🌐 Web Admin started at http://localhost:8000"
                except Exception as e:
                    return f"❌ Failed to start admin: {e}"

            if cmd.startswith("docker "):
                if not self.docker_manager or not self.docker_manager.enabled:
                    return "❌ Docker integration not available."
                sub = cmd[7:].strip()
                if sub == "list":
                    containers = self.docker_manager.list_containers()
                    if not containers: return "No containers found."
                    return "\n".join([f"🐳 {c['name']} ({c['status']}) - {c['image']}" for c in containers])
                if sub.startswith("start "):
                    return self.docker_manager.start_container(sub[6:])
                if sub.startswith("stop "):
                    return self.docker_manager.stop_container(sub[5:])
                if sub.startswith("logs "):
                    return self.docker_manager.get_logs(sub[5:])
                return "Usage: /docker [list|start <id>|stop <id>|logs <id>]"

            if cmd.startswith("persona "):
                if not self.persona_manager:
                    return "❌ Persona manager not available."
                parts = cmd.split(maxsplit=2)
                action = parts[1] if len(parts) > 1 else "list"
                if action == "list":
                    return self.persona_manager.list_personas()
                if action == "set" and len(parts) == 3:
                    return self.persona_manager.set_persona(parts[2])
                if action == "create" and len(parts) == 3:
                    # Interactive prompt creation could go here, but for now simple usage
                    return "Usage: /persona create <name> (prompt must be added manually in code for now or via file)"
                return "Usage: /persona [list|set <name>]"

            if cmd.startswith("net "):
                if not self.network_tools:
                    return "❌ Network tools not available."
                sub = cmd[4:].strip()
                if sub == "ip":
                    return self.network_tools.get_local_ip()
                if sub.startswith("ping "):
                    return self.network_tools.ping(sub[5:])
                if sub.startswith("scan "):
                    target = sub[5:] or "localhost"
                    return self.network_tools.scan_common_ports(target)
                return "Usage: /net [ip|ping <host>|scan <host>]"

            if cmd.startswith("snippet "):
                if not self.snippet_manager:
                    return "❌ Snippet manager not available."
                parts = cmd.split(maxsplit=2)
                action = parts[1] if len(parts) > 1 else "list"
                if action == "list":
                    return self.snippet_manager.list_snippets()
                if action == "save" and len(parts) == 3:
                    # Save last history item as snippet
                    hist = self.user_manager.get_history(self.user_manager.current_user)
                    if not hist: return "No history to save."
                    return self.snippet_manager.save_snippet(parts[2], hist[-1])
                if action == "get" and len(parts) == 3:
                    return self.snippet_manager.get_snippet(parts[2]) or "Snippet not found."
                return "Usage: /snippet [list|save <name>|get <name>]"

            if cmd.startswith("rag ingest "):
                target = cmd[11:].strip()
                if target.startswith("http"):
                    return self.rag_manager.ingest_url(target)
                else:
                    return self.rag_manager.ingest_file(target)
            
            if cmd == "save-session":
                if not self.user_manager.current_user:
                    return "Not logged in."
                hist = self.user_manager.get_history(self.user_manager.current_user)
                messages = [{"role": "user", "content": m} for m in hist]
                path = self.history_manager.save_session(self.user_manager.current_user, messages)
                if path:
                    return f"💾 Session saved to {path}"
                return "❌ Failed to save session"

            # --- Chat History ---
            if cmd == "history":
                if not self.user_manager.current_user:
                    return "Not logged in."
                hist = self.user_manager.get_history(self.user_manager.current_user)
                return "\n".join(hist) if hist else "No history."
            if cmd == "clearhistory":
                if not self.user_manager.current_user:
                    return "Not logged in."
                self.user_manager.clear_history(self.user_manager.current_user)
                return "History cleared."
            # --- Git Advanced ---
            if cmd.startswith("git create-branch"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /git create-branch [name]"
                return self.execute_git_command(f"git branch {parts[2]}")
            if cmd.startswith("git delete-branch"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /git delete-branch [name]"
                return self.execute_git_command(f"git branch -d {parts[2]}")
            
            # --- Advanced Git Commands ---
            if cmd == "git status":
                return self.execute_git_command("git status --porcelain")
            
            if cmd.startswith("git add"):
                parts = command.split(maxsplit=2)
                if len(parts) == 1:
                    return "Usage: /git add [files] or /git add . for all files"
                files = parts[1] if len(parts) > 1 else "."
                return self.execute_git_command(f"git add {files}")
            
            if cmd.startswith("git commit"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git commit [message] - Commit staged changes"
                message = parts[2]
                return self.execute_git_command(f"git commit -m \"{message}\"")
            
            if cmd == "git push":
                return self.execute_git_command("git push origin HEAD")
            
            if cmd == "git pull":
                return self.execute_git_command("git pull --rebase")
            
            if cmd.startswith("git log"):
                parts = command.split()
                limit = parts[1] if len(parts) > 1 and parts[1].isdigit() else "10"
                return self.execute_git_command(f"git log --oneline -{limit}")
            
            if cmd == "git diff":
                return self.execute_git_command("git diff")
            
            if cmd == "git diff --staged":
                return self.execute_git_command("git diff --staged")
            
            if cmd == "git branch":
                return self.execute_git_command("git branch -a")
            
            if cmd.startswith("git checkout"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git checkout [branch]"
                branch = parts[2]
                return self.execute_git_command(f"git checkout {branch}")
            
            if cmd.startswith("git merge"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git merge [branch]"
                branch = parts[2]
                return self.execute_git_command(f"git merge {branch}")
            
            if cmd == "git stash":
                return self.execute_git_command("git stash")
            
            if cmd == "git stash pop":
                return self.execute_git_command("git stash pop")
            
            if cmd.startswith("git reset"):
                parts = command.split(maxsplit=3)
                if len(parts) == 3 and parts[2] == "--hard":
                    return self.execute_git_command("git reset --hard HEAD")
                elif len(parts) >= 3:
                    file_path = parts[2]
                    return self.execute_git_command(f"git reset HEAD {file_path}")
                else:
                    return "Usage: /git reset [file] or /git reset --hard"
            
            if cmd == "git remote -v":
                return self.execute_git_command("git remote -v")
            
            if cmd.startswith("git blame"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git blame [file]"
                file_path = parts[2]
                return self.execute_git_command(f"git blame {file_path}")
            
            if cmd.startswith("git cherry-pick"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git cherry-pick [commit-hash]"
                commit_hash = parts[2]
                return self.execute_git_command(f"git cherry-pick {commit_hash}")
            
            if cmd.startswith("git rebase"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git rebase [branch]"
                branch = parts[2]
                return self.execute_git_command(f"git rebase {branch}")
            
            if cmd == "git bisect start":
                return self.execute_git_command("git bisect start")
            
            if cmd.startswith("git tag"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git tag [tag-name]"
                tag_name = parts[2]
                return self.execute_git_command(f"git tag {tag_name}")
            
            if cmd == "git reflog":
                return self.execute_git_command("git reflog --oneline -10")
            
            # --- Git Workflow Commands ---
            if cmd.startswith("git new-branch"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git new-branch [name] - Create and switch to new branch"
                branch_name = parts[2]
                # Create branch and switch to it
                create_result = self.execute_git_command(f"git checkout -b {branch_name}")
                return create_result
            
            if cmd == "git undo-last-commit":
                return self.execute_git_command("git reset --soft HEAD~1")
            
            if cmd.startswith("git amend"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git amend [message] - Amend last commit with new message"
                message = parts[2]
                return self.execute_git_command(f"git commit --amend -m \"{message}\"")
            
            if cmd == "git uncommit":
                return self.execute_git_command("git reset --soft HEAD~1")
            
            if cmd == "git discard":
                return self.execute_git_command("git checkout -- .")
            
            if cmd.startswith("git ignore"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git ignore [pattern] - Add pattern to .gitignore"
                pattern = parts[2]
                try:
                    with open('.gitignore', 'a') as f:
                        f.write(f"\n{pattern}")
                    return f"✅ Added '{pattern}' to .gitignore"
                except Exception as e:
                    return f"❌ Failed to update .gitignore: {str(e)}"
            
            if cmd == "git repo-info":
                # Get comprehensive repository information
                info = []
                
                # Basic repo info
                try:
                    result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        info.append(f"🌐 Remote URL: {result.stdout.strip()}")
                except:
                    pass
                
                # Current branch
                try:
                    result = subprocess.run(['git', 'branch', '--show-current'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        info.append(f"🌿 Current Branch: {result.stdout.strip()}")
                except:
                    pass
                
                # Last commit
                try:
                    result = subprocess.run(['git', 'log', '--oneline', '-1'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        info.append(f"📝 Last Commit: {result.stdout.strip()}")
                except:
                    pass
                
                # Status summary
                try:
                    result = subprocess.run(['git', 'status', '--porcelain'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        staged = len([l for l in lines if l and not l.startswith(' ')])
                        unstaged = len([l for l in lines if l and l.startswith(' ')])
                        untracked = len([l for l in lines if l and l.startswith('??')])
                        info.append(f"📊 Changes: {staged} staged, {unstaged} unstaged, {untracked} untracked")
                except:
                    pass
                
                if info:
                    return "📋 Repository Information:\n" + "\n".join(info)
                else:
                    return "❌ Could not retrieve repository information"
            
            # --- Additional Git Commands ---
            if cmd.startswith("git "):
                # Run git commands in background to keep UI responsive
                def _run_git():
                    res = self.execute_git_command(command) # Use 'command' here, not 'cmd'
                    console.print(f"\n{res}")
                    console.print(f"\n[{self.current_model.upper()}] 🚀 > ", end="")
                _run_in_background(_run_git)
                return "⏳ Git command running in background..."
            
            if cmd == "git init":
                return self.execute_git_command("git init")
            
            if cmd.startswith("git clone"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git clone [url] - Clone a repository"
                url = parts[2]
                return self.execute_git_command(f"git clone {url}")
            
            if cmd == "git fetch":
                return self.execute_git_command("git fetch --all")
            
            if cmd.startswith("git pull-request"):
                return "💡 To create a pull request, push your branch and use your Git hosting service (GitHub, GitLab, etc.)"
            
            if cmd == "git contributors":
                return self.execute_git_command("git shortlog -sn --no-merges")
            
            if cmd == "git file-history":
                return "Usage: /git file-history [filename] - Show history of a specific file"
            
            if cmd.startswith("git file-history"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /git file-history [filename] - Show history of a specific file"
                filename = parts[2]
                return self.execute_git_command(f"git log --follow --oneline {filename}")
            
            if cmd == "git clean":
                return self.execute_git_command("git clean -fd")
            
            if cmd == "git stats":
                # Get repository statistics
                stats = []
                try:
                    # Total commits
                    result = subprocess.run(['git', 'rev-list', '--count', 'HEAD'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        stats.append(f"📊 Total Commits: {result.stdout.strip()}")
                except:
                    pass
                
                try:
                    # Contributors count
                    result = subprocess.run(['git', 'shortlog', '-sn', '--no-merges'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        stats.append(f"👥 Contributors: {len(lines)}")
                except:
                    pass
                
                try:
                    # Repository size
                    result = subprocess.run(['git', 'count-objects', '-vH'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'size-pack:' in line:
                                stats.append(f"💾 Repository Size: {line.split(':')[1].strip()}")
                                break
                except:
                    pass
                
                if stats:
                    return "📈 Repository Statistics:\n" + "\n".join(stats)
                else:
                    return "❌ Could not retrieve repository statistics"
            
            # --- AI Code Review ---
            if cmd.startswith("codereview"):
                parts = command.split()
                if len(parts) != 2:
                    return "Usage: /codereview [filename]"
                try:
                    with open(parts[1], 'r') as f:
                        code = f.read(2000)
                    return self.ai.query(self.current_model, f"Review this code for bugs and improvements:\n{code}")
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            # --- AI File Summarization ---
            if cmd.startswith("summarizefile"):
                parts = command.split()
                if len(parts) != 2:
                    return "Usage: /summarizefile [filename]"
                try:
                    with open(parts[1], 'r') as f:
                        content = f.read(2000)
                    return self.ai.query(self.current_model, f"Summarize this file:\n{content}")
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            # --- AI File Search ---
            if cmd.startswith("aifind"):
                parts = command.split()
                if len(parts) != 2:
                    return "Usage: /aifind [keyword]"
                matches = []
                for root, dirs, files in os.walk(os.getcwd()):
                    for file in files:
                        try:
                            with open(os.path.join(root, file), 'r', errors='ignore') as f:
                                for i, line in enumerate(f):
                                    if parts[1] in line:
                                        matches.append(f"{file}:{i+1}: {line.strip()}")
                                        if len(matches) > 10:
                                            break
                        except Exception:
                            continue
                if not matches:
                    return "No matches found."
                context = "\n".join(matches[:10])
                return self.ai.query(self.current_model, f"Explain the context of these code lines:\n{context}")
            # --- AI Commit Message Generator ---
            if cmd.startswith("git commitmsg"):
                parts = command.split()
                if len(parts) == 3:
                    try:
                        with open(parts[2], 'r') as f:
                            diff = f.read(2000)
                        return self.ai.query(self.current_model, f"Write a git commit message for this diff or file:\n{diff}")
                    except Exception as e:
                        return f"Error reading file: {str(e)}"
                return "Usage: /git commitmsg [diff or file]"
            # --- AI Bug Finder ---
            if cmd.startswith("findbugs"):
                parts = command.split()
                if len(parts) != 2:
                    return "Usage: /findbugs [filename]"
                try:
                    with open(parts[1], 'r') as f:
                        code = f.read(2000)
                    return self.ai.query(self.current_model, f"Find bugs in this code:\n{code}")
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            # --- AI Refactor ---
            if cmd.startswith("refactor"):
                parts = command.split(maxsplit=2)
                if len(parts) != 3:
                    return "Usage: /refactor [filename] [instruction]"
                try:
                    with open(parts[1], 'r') as f:
                        code = f.read(2000)
                    return self.ai.query(self.current_model, f"Refactor this code as per instruction '{parts[2]}':\n{code}")
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            # --- Project TODO Extractor ---
            if cmd == "todos":
                todos = []
                for root, dirs, files in os.walk(os.getcwd()):
                    for file in files:
                        try:
                            with open(os.path.join(root, file), 'r', errors='ignore') as f:
                                for i, line in enumerate(f):
                                    if 'TODO' in line or 'FIXME' in line:
                                        todos.append(f"{file}:{i+1}: {line.strip()}")
                                        if len(todos) > 20:
                                            break
                        except Exception:
                            continue
                if not todos:
                    return "No TODOs/FIXMEs found."
                return self.ai.query(self.current_model, f"Summarize these TODOs/FIXMEs:\n" + "\n".join(todos[:20]))
            # --- AI Documentation Generator ---
            if cmd.startswith("gendoc"):
                parts = command.split()
                if len(parts) != 2:
                    return "Usage: /gendoc [filename]"
                try:
                    with open(parts[1], 'r') as f:
                        code = f.read(2000)
                    return self.ai.query(self.current_model, f"Generate docstrings and comments for this code:\n{code}")
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            # --- AI Test Generator ---
            if cmd.startswith("gentest"):
                parts = command.split()
                if len(parts) != 2:
                    return "Usage: /gentest [filename]"
                try:
                    with open(parts[1], 'r') as f:
                        code = f.read(2000)
                    return self.ai.query(self.current_model, f"Write unit tests for this code:\n{code}")
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            # --- Existing Commands ---
            if cmd == "help":
                try:
                    help_text = Text()
                    help_text.append(f"\n🚀 AetherAI Terminal v{VERSION} - COMMAND REFERENCE\n\n", style="bold cyan")
                    help_text.append("📋 BASIC COMMANDS:\n", style="bold yellow")
                    help_text.append("/help          - Show this help menu\n", style="white")
                    help_text.append("/status        - Show detailed service status\n", style="white")
                    help_text.append("/security      - Show security information\n", style="white")
                    help_text.append("/clear         - Clear the screen\n", style="white")
                    help_text.append("/exit          - Quit application\n\n", style="white")
                    help_text.append("🤖 AI MODEL COMMANDS:\n", style="bold yellow")
                    help_text.append("/switch gemini      - Switch to Gemini 2.0 Flash\n", style="white")
                    help_text.append("/switch groq        - Switch to Groq Mixtral\n", style="white")
                    help_text.append("/switch ollama      - Interactive model picker (select by number!)\n", style="white")
                    help_text.append("/switch ollama [model] - Switch to specific Ollama model\n", style="white")
                    help_text.append("/switch huggingface - Switch to HuggingFace\n", style="white")
                    help_text.append("/switch chatgpt     - Switch to ChatGPT (OpenAI)\n", style="white")
                    help_text.append("/switch mcp         - Switch to MCP (Model Context Protocol)\n\n", style="white")
                    help_text.append("⚙️ SYSTEM COMMANDS:\n", style="bold yellow")
                    help_text.append("/run [command]      - Execute safe system commands (ls, pwd, whoami, date, etc.)\n", style="white")
                    help_text.append("/models             - List available AI models\n", style="white")
                    help_text.append("/ollama-models      - Show detailed list of available Ollama models\n", style="white")
                    help_text.append("/ollama-select      - Quick interactive Ollama model picker\n", style="white")
                    help_text.append("/current-model      - Show currently active AI model\n", style="white")
                    help_text.append("/config             - Show configuration\n", style="white")
                    help_text.append("/sysinfo            - Display system information and resources\n", style="white")
                    help_text.append("/calc [expression]  - Calculate mathematical expressions\n", style="white")
                    help_text.append("/explore            - Interactive file explorer for current directory\n", style="white")
                    help_text.append("/websearch [query]  - Search the web using DuckDuckGo\n", style="white")
                    help_text.append("/weather [city]     - Get current weather information\n", style="white")
                    help_text.append("/note [text]        - Save a quick note\n", style="white")
                    help_text.append("/notes              - View your saved notes\n", style="white")
                    help_text.append("/timer [seconds]    - Start a countdown timer\n", style="white")
                    help_text.append("/convert [val] [from] [to] - Unit converter (temp, weight, data)\n", style="white")
                    help_text.append("/joke               - Get a random joke\n", style="white")
                    help_text.append("/password [length]  - Generate a secure password\n", style="white")
                    help_text.append("/tip                - Get a random productivity tip\n", style="white")
                    help_text.append("/clear              - Clear the screen\n\n", style="white")
                    
                    help_text.append("🔀 GIT VERSION CONTROL:\n", style="bold yellow")
                    help_text.append("/git status            - Show repository status\n", style="white")
                    help_text.append("/git add [files]       - Stage files for commit\n", style="white")
                    help_text.append("/git commit [message]  - Commit staged changes\n", style="white")
                    help_text.append("/git push              - Push commits to remote\n", style="white")
                    help_text.append("/git pull              - Pull changes from remote\n", style="white")
                    help_text.append("/git log [n]           - Show commit history\n", style="white")
                    help_text.append("/git diff              - Show unstaged changes\n", style="white")
                    help_text.append("/git diff --staged     - Show staged changes\n", style="white")
                    help_text.append("/git branch            - List branches\n", style="white")
                    help_text.append("/git checkout [branch] - Switch to branch\n", style="white")
                    help_text.append("/git merge [branch]    - Merge branch into current\n", style="white")
                    help_text.append("/git stash             - Stash current changes\n", style="white")
                    help_text.append("/git stash pop         - Apply stashed changes\n", style="white")
                    help_text.append("/git reset [file]      - Unstage file\n", style="white")
                    help_text.append("/git reset --hard      - Reset to last commit\n", style="white")
                    help_text.append("/git remote -v         - Show remote repositories\n", style="white")
                    help_text.append("/git blame [file]      - Show who changed each line\n", style="white")
                    help_text.append("/git cherry-pick [hash] - Apply specific commit\n", style="white")
                    help_text.append("/git rebase [branch]   - Rebase current branch\n", style="white")
                    help_text.append("/git bisect start      - Start binary search for bugs\n", style="white")
                    help_text.append("/git tag [name]        - Create a tag\n", style="white")
                    help_text.append("/git reflog            - Show reference log\n", style="white")
                    help_text.append("/git new-branch [name] - Create and switch to new branch\n", style="white")
                    help_text.append("/git undo-last-commit  - Undo last commit (keep changes)\n", style="white")
                    help_text.append("/git amend [message]   - Amend last commit message\n", style="white")
                    help_text.append("/git uncommit          - Uncommit but keep changes staged\n", style="white")
                    help_text.append("/git discard           - Discard all unstaged changes\n", style="white")
                    help_text.append("/git ignore [pattern]  - Add pattern to .gitignore\n", style="white")
                    help_text.append("/git repo-info         - Show comprehensive repo information\n", style="white")
                    help_text.append("/git init              - Initialize a new Git repository\n", style="white")
                    help_text.append("/git clone [url]       - Clone a repository\n", style="white")
                    help_text.append("/git fetch             - Fetch all branches from remote\n", style="white")
                    help_text.append("/git contributors      - Show contributors by commit count\n", style="white")
                    help_text.append("/git file-history [f]  - Show history of a specific file\n", style="white")
                    help_text.append("/git clean             - Remove untracked files\n", style="white")
                    help_text.append("/git stats             - Show repository statistics\n\n", style="white")

                    help_text.append("💡 EXAMPLES:\n", style="bold green")
                    help_text.append("• Write a Python function to sort a list\n", style="cyan")
                    help_text.append("• /switch ollama llama2:13b\n", style="cyan")
                    help_text.append("• /ollama-models\n", style="cyan")
                    help_text.append("• /run ls -la\n", style="cyan")
                    help_text.append("• Explain quantum computing\n\n", style="cyan")

                    # Add new advanced features section
                    help_text.append("🚀 ADVANCED FEATURES:\n", style="bold magenta")
                    
                    help_text.append("🤖 CONTEXT-AWARE AI:\n", style="bold yellow")
                    help_text.append("/learn [topic]          - Teach AI about technologies\n", style="white")
                    help_text.append("/remind [task]          - Set task reminders\n", style="white")
                    help_text.append("/reminders              - View active reminders\n", style="white")
                    help_text.append("/complete-reminder [n]  - Mark reminder as complete\n\n", style="white")

                    help_text.append("📊 ANALYTICS & MONITORING:\n", style="bold yellow")
                    help_text.append("/analytics              - View usage statistics\n", style="white")
                    help_text.append("/error-analytics        - View error analytics\n", style="white")
                    help_text.append("/start-monitoring       - Start system monitoring\n", style="white")
                    help_text.append("/stop-monitoring        - Stop system monitoring\n", style="white")
                    help_text.append("/net-diag               - Network diagnostics\n", style="white")
                    help_text.append("/analyze-logs           - Analyze log files\n", style="white")
                    help_text.append("/health                 - System health check\n\n", style="white")

                    help_text.append("🎮 GAMES & LEARNING:\n", style="bold yellow")
                    help_text.append("/challenge [difficulty] - Get coding challenge\n", style="white")
                    help_text.append("/submit-challenge [id] [pid] [code] - Submit solution\n", style="white")
                    help_text.append("/tutorial [topic]       - Start interactive tutorial\n", style="white")
                    help_text.append("/tutorial-section [id] [num] - Get tutorial section\n", style="white")
                    help_text.append("/quiz [topic]           - Take interactive quiz\n", style="white")
                    help_text.append("/answer-quiz [id] [num] - Answer quiz question\n", style="white")
                    help_text.append("/user-stats             - View learning statistics\n\n", style="white")

                    help_text.append("🎨 CREATIVE TOOLS:\n", style="bold yellow")
                    help_text.append("/ascii [text]           - Generate ASCII art\n", style="white")
                    help_text.append("/colors [type] [base]   - Generate color schemes\n", style="white")
                    help_text.append("/music [mood] [length]  - Generate music patterns\n", style="white")
                    help_text.append("/story [genre] [length] - Generate creative stories\n\n", style="white")

                    help_text.append("🔒 ADVANCED SECURITY:\n", style="bold yellow")
                    help_text.append("/encrypt [message]      - Encrypt messages\n", style="white")
                    help_text.append("/decrypt [message]      - Decrypt messages\n", style="white")
                    help_text.append("/rotate-key [service] [key] - Rotate API keys\n", style="white")
                    help_text.append("/biometric-auth [data]  - Biometric authentication\n", style="white")
                    help_text.append("/secure-password [len]  - Generate secure passwords\n", style="white")
                    help_text.append("/security-report        - View security report\n", style="white")
                    help_text.append("/threat-scan [text]     - Scan for security threats\n\n", style="white")

                    help_text.append("🎨 THEME MANAGEMENT:\n", style="bold yellow")
                    help_text.append("/themes                  - List all available themes\n", style="white")
                    help_text.append("/theme set [name]        - Switch to a theme\n", style="white")
                    help_text.append("/theme current           - Show current theme\n", style="white")
                    help_text.append("/theme preview [name]    - Preview a theme\n", style="white")
                    help_text.append("/theme create [name] [base] - Create custom theme\n", style="white")
                    help_text.append("/theme delete [name]     - Delete custom theme\n", style="white")
                    help_text.append("/theme export [name] [fmt] - Export theme (json/python)\n", style="white")
                    help_text.append("/theme stats             - Show theme statistics\n", style="white")
                    help_text.append("/theme reset             - Reset to default theme\n\n", style="white")

                    help_text.append("� CODE REVIEW ASSISTANT:\n", style="bold yellow")
                    help_text.append("/review analyze [file]   - Full code analysis\n", style="white")
                    help_text.append("/review security [file]  - Security analysis only\n", style="white")
                    help_text.append("/review performance [file] - Performance analysis only\n", style="white")
                    help_text.append("/review quality [file]   - Quality metrics only\n", style="white")
                    help_text.append("/review compare [f1] [f2] - Compare two files\n", style="white")
                    help_text.append("/review suggest [file]   - AI improvement suggestions\n", style="white")
                    help_text.append("/review language [file]  - Detect programming language\n", style="white")
                    help_text.append("/review history          - Recent review history\n", style="white")
                    help_text.append("/review stats            - Review statistics\n\n", style="white")

                    help_text.append("�📝 TASK MANAGEMENT:\n", style="bold yellow")
                    help_text.append("/task add [title]       - Add a new task\n", style="white")
                    help_text.append("/task create [title]    - Create a new task\n", style="white")
                    help_text.append("/tasks                  - List all pending tasks\n", style="white")
                    help_text.append("/task show [id]         - Show task details\n", style="white")
                    help_text.append("/task complete [id]     - Mark task as completed\n", style="white")
                    help_text.append("/task delete [id]       - Delete a task\n", style="white")
                    help_text.append("/task update [id] [field] [value] - Update task field\n", style="white")
                    help_text.append("/task priority [id] [priority] - Set priority (low/medium/high/urgent)\n", style="white")
                    help_text.append("/task category [id] [category] - Set task category\n", style="white")
                    help_text.append("/task due [id] [date]   - Set due date (YYYY-MM-DD)\n", style="white")
                    help_text.append("/task subtask add [task_id] [title] - Add subtask\n", style="white")
                    help_text.append("/task subtask complete [task_id] [subtask_id] - Complete subtask\n", style="white")
                    help_text.append("/task stats             - Show task statistics\n", style="white")
                    help_text.append("/task search [query]    - Search tasks\n", style="white")
                    help_text.append("/task overdue           - Show overdue tasks\n", style="white")
                    help_text.append("/task export [format]   - Export tasks (json/csv)\n\n", style="white")

                    help_text.append("🛠️ DEVELOPER TOOLS (NEW!):\n", style="bold magenta")
                    help_text.append("/analyze <file>         - AI-powered file analysis\n", style="white")
                    help_text.append("/commit-msg             - Generate commit message from staged changes\n", style="white")
                    help_text.append("/pr-desc [branch]       - Generate PR description\n", style="white")
                    help_text.append("/clip or /copy          - Copy last response to clipboard\n", style="white")
                    help_text.append("/paste                  - Paste from clipboard\n", style="white")
                    help_text.append("!command                - Shell shortcut (e.g., !ls, !pwd)\n", style="white")
                    help_text.append("/generate-tests <file>  - Generate unit tests\n", style="white")
                    help_text.append("/explain-error <error>  - Debug errors with AI\n", style="white")
                    help_text.append("/compare <prompt>       - Compare multiple AI models\n\n", style="white")

                    help_text.append("💬 SESSIONS & CONTEXT:\n", style="bold magenta")
                    help_text.append("/session new [name]     - Start new session\n", style="white")
                    help_text.append("/session save [name]    - Save current session\n", style="white")
                    help_text.append("/session load <name>    - Load a session\n", style="white")
                    help_text.append("/session list           - List all sessions\n", style="white")
                    help_text.append("/session delete <name>  - Delete a session\n", style="white")
                    help_text.append("/export <file> [fmt]    - Export to md/json/html/txt\n", style="white")
                    help_text.append("/save [name]            - Save response as favorite\n", style="white")
                    help_text.append("/favorites              - List saved favorites\n", style="white")
                    help_text.append("/templates              - List prompt templates\n", style="white")
                    help_text.append("/template <name>        - Use a template\n", style="white")
                    help_text.append("/project                - Detect project context\n\n", style="white")

                    help_text.append("🌐 WEB SEARCH & DOCS:\n", style="bold magenta")
                    help_text.append("/search <query>         - Search the web (DuckDuckGo)\n", style="white")
                    help_text.append("/docs <topic> [tech]    - Search documentation\n", style="white")
                    help_text.append("/fetch <url or number>  - Fetch URL content\n", style="white")
                    help_text.append("/read <url>             - Read URL (alias)\n\n", style="white")

                    help_text.append("📋 SKILLS & RULES (Claude-like):\n", style="bold magenta")
                    help_text.append("/skills                 - List loaded skills\n", style="white")
                    help_text.append("/skills reload          - Reload skills/rules files\n", style="white")
                    help_text.append("/skills create          - Create SKILLS.md template\n", style="white")
                    help_text.append("/rules                  - List active rules\n", style="white")
                    help_text.append("/rules create           - Create RULES.md template\n\n", style="white")

                    help_text.append("📡 MCP (Model Context Protocol):\n", style="bold magenta")
                    help_text.append("/mcp list               - List configured servers\n", style="white")
                    help_text.append("/mcp available          - Show installable servers\n", style="white")
                    help_text.append("/mcp add <name>         - Add a server\n", style="white")
                    help_text.append("/mcp start <name>       - Start a server\n", style="white")
                    help_text.append("/mcp stop <name>        - Stop a server\n\n", style="white")

                    help_text.append("🎤 VOICE COMMANDS:\n", style="bold magenta")
                    help_text.append("/voice on/off           - Enable/disable voice\n", style="white")
                    help_text.append("/voice status           - Voice system status\n", style="white")
                    help_text.append("/voices                 - List available voices\n", style="white")
                    help_text.append("/speak <text>           - Speak text aloud\n", style="white")
                    help_text.append("/listen                 - Listen for voice input\n\n", style="white")

                    help_text.append("👁️ VISION & IMAGE:\n", style="bold magenta")
                    help_text.append("/vision <image> <prompt> - Analyze an image\n", style="white")
                    help_text.append("/see [prompt]           - (Alias for /vision)\n\n", style="white")

                    help_text.append("🤖 CODE AGENT (Autonomous):\n", style="bold cyan")
                    help_text.append("/agent edit <file> <instruction> - AI code editing\n", style="white")
                    help_text.append("/agent analyze [dir]    - Analyze project\n", style="white")
                    help_text.append("/agent issues <file>    - Find code issues\n", style="white")
                    help_text.append("/agent fix <file>       - Auto-fix issues\n\n", style="white")

                    help_text.append("👥 PAIR PROGRAMMING:\n", style="bold cyan")
                    help_text.append("/pair start <file>      - Start pair session\n", style="white")
                    help_text.append("/pair status            - Session status\n", style="white")
                    help_text.append("/pair end               - End session\n", style="white")
                    help_text.append("/suggest                - Get code suggestions\n", style="white")
                    help_text.append("/refactor [file]        - Refactoring suggestions\n\n", style="white")

                    help_text.append("⚙️ WORKFLOWS:\n", style="bold cyan")
                    help_text.append("/workflows              - List all workflows\n", style="white")
                    help_text.append("/workflow create <name> - Create workflow\n", style="white")
                    help_text.append("/workflow run <id>      - Run workflow\n", style="white")
                    help_text.append("/workflow code-review   - Pre-built code review\n", style="white")
                    help_text.append("/workflow standup       - Pre-built standup\n\n", style="white")

                    help_text.append("📚 KNOWLEDGE BASE (RAG):\n", style="bold cyan")
                    help_text.append("/kb                     - Knowledge base status\n", style="white")
                    help_text.append("/kb add <file/dir>      - Add to knowledge base\n", style="white")
                    help_text.append("/kb search <query>      - Search knowledge base\n", style="white")
                    help_text.append("/kb ask <question>      - Answer with KB context\n", style="white")
                    help_text.append("/kb list                - List indexed documents\n\n", style="white")

                    console.print(Panel(help_text, border_style="bright_green", padding=(1, 2)))
                    return ""
                except Exception as e:
                    return f"❌ Error displaying help: {str(e)}"
            
            elif cmd.startswith("setkey"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /setkey [provider] [key]"
                ok, msg = self.user_manager.set_api_key(parts[1], parts[2])
                # Set environment variable for session
                if ok:
                    if parts[1] == "openai":
                        os.environ["OPENAI_API_KEY"] = parts[2]
                    if parts[1] == "mcp":
                        os.environ["MCP_API_KEY"] = parts[2]
                return msg
            
            elif cmd.startswith("switch"):
                parts = cmd.split()
                if len(parts) < 2:
                    return "❌ Usage: /switch [model] [optional: specific_model]\n   Examples: /switch ollama, /switch ollama llama2:13b, /switch gemini"
                
                new_model = parts[1]
                valid_models = ["gemini", "groq", "ollama", "huggingface", "chatgpt", "mcp"]
                
                if new_model == "ollama":
                    # Handle Ollama model switching
                    try:
                        # Get available models
                        ollama_response = ollama.list()
                        
                        # Handle ListResponse object or dict
                        if hasattr(ollama_response, 'models'):
                            ollama_models = ollama_response.models
                        elif isinstance(ollama_response, dict):
                            ollama_models = ollama_response.get("models", [])
                        else:
                            ollama_models = []
                        
                        if not ollama_models:
                            return "❌ No Ollama models found. Please pull a model first.\n   Example: ollama pull llama3"
                        
                        # Extract model names
                        model_names = []
                        for m in ollama_models:
                            if hasattr(m, 'model'):
                                model_names.append(m.model)
                            elif hasattr(m, 'name'):
                                model_names.append(m.name)
                            elif isinstance(m, dict):
                                model_names.append(m.get('name') or m.get('model', str(m)))
                            else:
                                model_names.append(str(m))
                        
                        if len(parts) >= 3:
                            # User specified a model name directly
                            specific_model = parts[2]
                            if specific_model in model_names:
                                self.current_model = f"ollama:{specific_model}"
                                self._save_config()
                                return f"✅ Switched to Ollama model: {specific_model}"
                            else:
                                # Try partial match
                                matches = [m for m in model_names if specific_model.lower() in m.lower()]
                                if len(matches) == 1:
                                    self.current_model = f"ollama:{matches[0]}"
                                    self._save_config()
                                    return f"✅ Switched to Ollama model: {matches[0]}"
                                elif len(matches) > 1:
                                    return f"❌ Multiple matches found: {', '.join(matches)}\n   Please be more specific."
                                else:
                                    return f"❌ Model '{specific_model}' not found.\n   Use /switch ollama to see available models."
                        
                        # Show interactive numbered list
                        console.print("\n[bold cyan]📦 Available Ollama Models:[/bold cyan]\n")
                        for i, model in enumerate(model_names, 1):
                            # Get model size if available
                            size_str = ""
                            for m in ollama_models:
                                name = getattr(m, 'model', None) or getattr(m, 'name', None) or (m.get('name') if isinstance(m, dict) else None)
                                if name == model:
                                    size = getattr(m, 'size', None) or (m.get('size') if isinstance(m, dict) else None)
                                    if size:
                                        if size >= 1024**3:
                                            size_str = f" ({size/(1024**3):.1f}GB)"
                                        elif size >= 1024**2:
                                            size_str = f" ({size/(1024**2):.0f}MB)"
                                    break
                            console.print(f"  [bold yellow]{i:2}[/bold yellow]) [green]{model}[/green]{size_str}")
                        
                        console.print(f"\n[dim]Enter number (1-{len(model_names)}) or model name, or 'q' to cancel:[/dim]")
                        
                        try:
                            selection = input("> ").strip()
                            
                            if selection.lower() in ('q', 'quit', 'cancel', ''):
                                return "❌ Model selection cancelled."
                            
                            # Check if it's a number
                            if selection.isdigit():
                                idx = int(selection)
                                if 1 <= idx <= len(model_names):
                                    selected_model = model_names[idx - 1]
                                    self.current_model = f"ollama:{selected_model}"
                                    self._save_config()
                                    return f"✅ Switched to Ollama model: [bold green]{selected_model}[/bold green]"
                                else:
                                    return f"❌ Invalid selection. Enter 1-{len(model_names)}"
                            else:
                                # Try to match by name
                                if selection in model_names:
                                    self.current_model = f"ollama:{selection}"
                                    self._save_config()
                                    return f"✅ Switched to Ollama model: {selection}"
                                else:
                                    # Partial match
                                    matches = [m for m in model_names if selection.lower() in m.lower()]
                                    if len(matches) == 1:
                                        self.current_model = f"ollama:{matches[0]}"
                                        self._save_config()
                                        return f"✅ Switched to Ollama model: {matches[0]}"
                                    elif len(matches) > 1:
                                        return f"❌ Multiple matches: {', '.join(matches)}"
                                    else:
                                        return f"❌ Model '{selection}' not found."
                        except EOFError:
                            return "❌ Model selection cancelled."
                        except KeyboardInterrupt:
                            return "❌ Model selection cancelled."
                            
                    except Exception as e:
                        return f"❌ Error checking Ollama models: {str(e)}\n   Make sure Ollama is running (ollama serve)"
                
                elif new_model in valid_models:
                    self.current_model = new_model
                    self._save_config()
                    return f"✅ Switched to {new_model.upper()}"
                
                return f"❌ Invalid model. Choose from: {', '.join(valid_models)}\n   For Ollama: /switch ollama [model_name]"

            # --- Core Utility Commands ---
            if cmd == "status":
                try:
                    rows = []
                    for service, status in self.ai.status.items():
                        desc = self.model_descriptions.get(service, "AI Service")
                        rows.append(f"{service.upper():<10} | {status:<20} | {desc}")
                    return "\n".join(["Service     | Status               | Description", "-"*60] + rows)
                except Exception as e:
                    return f"❌ Failed to gather status: {str(e)[:100]}"

            if cmd == "security":
                try:
                    details = [
                        "🔒 Security Info:",
                        f"Allowed commands: {', '.join(self.allowed_commands)}",
                        f"Config path: {CONFIG_PATH()}",
                        f"User DB path: {USER_DB_PATH()}"
                    ]
                    return "\n".join(details)
                except Exception as e:
                    return f"❌ Failed to fetch security info: {str(e)[:100]}"

            if cmd == "clear":
                try:
                    console.clear()
                    return ""
                except Exception:
                    return "\n" * 50

            if cmd == "exit":
                try:
                    return "👋 Exiting..."
                finally:
                    try:
                        sys.exit(0)
                    except SystemExit:
                        pass

            if cmd == "models":
                try:
                    lines = ["Available AI models:"]
                    for k, v in self.ai.status.items():
                        desc = self.model_descriptions.get(k, "AI Service")
                        lines.append(f"• {k.upper():<10} - {v} - {desc}")
                    lines.append("\nUse /switch [model] to change, e.g., /switch groq")
                    lines.append("For Ollama, you can target a specific model: /switch ollama llama3")
                    lines.append("Use /ollama-models to list local models")
                    return "\n".join(lines)
                except Exception as e:
                    return f"❌ Failed to list models: {str(e)[:100]}"

            if cmd == "ollama-models":
                try:
                    # Support detailed specs: /ollama-models [model_name]
                    parts = command.split(maxsplit=1)
                    try:
                        ollama_response = ollama.list()
                        # Handle ListResponse object or dict
                        if hasattr(ollama_response, 'models'):
                            models = ollama_response.models
                        elif isinstance(ollama_response, dict):
                            models = ollama_response.get("models", [])
                        else:
                            models = []
                    except Exception as e:
                        return f"❌ Ollama not available: {str(e)[:100]}"

                    if not models:
                        return "❌ No Ollama models found. Ensure Ollama is running and models are pulled."

                    def _fmt_size(sz: int) -> str:
                        try:
                            if sz >= 1024**3:
                                return f"{sz/(1024**3):.2f} GB"
                            if sz >= 1024**2:
                                return f"{sz/(1024**2):.2f} MB"
                            if sz >= 1024:
                                return f"{sz/1024:.2f} KB"
                            return f"{sz} B"
                        except Exception:
                            return str(sz)

                    # Detailed single-model view
                    if len(parts) == 2 and parts[1].strip():
                        target = parts[1].strip()
                        # Find the matching model entry (for size/modified)
                        meta = None
                        for m in models:
                            if hasattr(m, 'model') and m.model == target:
                                meta = m
                                break
                            elif hasattr(m, 'name') and m.name == target:
                                meta = m
                                break
                            elif isinstance(m, dict) and m.get("name") == target:
                                meta = m
                                break
                        
                        try:
                            info = ollama.show(target)
                        except Exception as e:
                            return f"❌ Could not fetch specs for '{target}': {str(e)[:100]}"

                        details = info.get("details", {}) or {}
                        
                        # Extract size and modified from meta
                        if meta:
                            if hasattr(meta, 'size'):
                                size_str = _fmt_size(meta.size)
                            elif isinstance(meta, dict):
                                size_str = _fmt_size(meta.get("size", 0))
                            else:
                                size_str = "Unknown"
                                
                            if hasattr(meta, 'modified_at'):
                                modified = str(meta.modified_at) or "Unknown"
                            elif isinstance(meta, dict):
                                modified = meta.get("modified_at", "Unknown")
                            else:
                                modified = "Unknown"
                        else:
                            size_str = "Unknown"
                            modified = "Unknown"

                        lines = [
                            f"🦙 Ollama Model Details: {target}",
                            f"Size: {size_str}",
                            f"Modified: {modified}",
                            f"Digest: {info.get('digest', info.get('model', 'Unknown'))}",
                            f"Family: {details.get('family') or (details.get('families') or ['Unknown'])[0] if isinstance(details.get('families'), list) else details.get('family', 'Unknown')}",
                            f"Format: {details.get('format', 'Unknown')}",
                            f"Parameters: {details.get('parameter_size', 'Unknown')}",
                            f"Quantization: {details.get('quantization_level', 'Unknown')}",
                        ]
                        if info.get('license'):
                            lines.append(f"License: {str(info.get('license'))[:200]}")
                        if info.get('parameters'):
                            lines.append(f"Params string: {str(info.get('parameters'))[:200]}")
                        return "\n".join(lines)

                    # Summary table view
                    header = f"{'NAME':<36}  {'SIZE':>10}  {'PARAMS':>8}  {'FAMILY':<12}  {'QUANT':<8}  MODIFIED"
                    sep = "-" * len(header)
                    rows = ["🦙 Installed Ollama Models:", header, sep]
                    for m in models:
                        # Handle both dict and Model object
                        if hasattr(m, 'model'):
                            # It's a Model object
                            name = m.model
                            size = _fmt_size(getattr(m, 'size', 0))
                            modified = str(getattr(m, 'modified_at', '')) or ''
                            # Get details from the model object if available
                            details = getattr(m, 'details', None)
                            if details:
                                params = getattr(details, 'parameter_size', '?') or '?'
                                family = getattr(details, 'family', '?') or '?'
                                quant = getattr(details, 'quantization_level', '?') or '?'
                            else:
                                params = family = quant = "?"
                        else:
                            # It's a dict (fallback)
                            name = m.get("name", "unknown")
                            size = _fmt_size(m.get("size", 0))
                            modified = m.get("modified_at", "") or ""
                            params = family = quant = "?"
                            # Try to enrich with details via ollama.show (best-effort)
                            try:
                                info = ollama.show(name)
                                d = (info or {}).get('details', {}) or {}
                                params = d.get('parameter_size', params) or params
                                family = d.get('family', family) or (d.get('families', [family])[0] if isinstance(d.get('families'), list) and d.get('families') else family)
                                quant = d.get('quantization_level', quant) or quant
                            except Exception:
                                pass
                        rows.append(f"{name:<36}  {size:>10}  {params:>8}  {family:<12}  {quant:<8}  {modified}")
                    rows.append("\n💡 Use '/ollama-models [model_name]' to see full specs for a single model")
                    rows.append("💡 Use '/switch ollama' for interactive model selection (pick by number!)")
                    return "\n".join(rows)
                except Exception as e:
                    return f"❌ Failed to fetch Ollama models: {str(e)[:100]}"

            # Quick alias for interactive ollama model selection
            if cmd == "ollama-select" or cmd == "ollama select":
                return self.handle_command("/switch ollama")

            if cmd == "current-model":
                try:
                    model = self.current_model
                    base = model.split(":", 1)[0]
                    desc = self.model_descriptions.get(base, "AI Service")
                    return f"🎯 Current model: {model.upper()}\n{desc}"
                except Exception as e:
                    return f"❌ Failed to read current model: {str(e)[:100]}"

            if cmd == "config":
                try:
                    cfg = CONFIG_PATH()
                    content = ""
                    if os.path.exists(cfg):
                        with open(cfg, "r", errors="ignore") as f:
                            content = f.read(800)
                    return f"⚙️ Config file: {cfg}\nDefault model: {self.current_model}\n\n{content}"
                except Exception as e:
                    return f"❌ Failed to read config: {str(e)[:100]}"

            if cmd == "sysinfo":
                try:
                    info = []
                    try:
                        import platform
                        info.extend([
                            f"OS: {platform.system()} {platform.release()} ({platform.version()})",
                            f"Machine: {platform.machine()}"
                        ])
                    except Exception:
                        pass
                    try:
                        import psutil
                        vm = psutil.virtual_memory()
                        cpu = psutil.cpu_percent(interval=0.3)
                        info.extend([
                            f"CPU: {cpu}%",
                            f"Memory: {round(vm.used/1e9,2)}/{round(vm.total/1e9,2)} GB ({vm.percent}%)",
                        ])
                        du = psutil.disk_usage('/')
                        info.append(f"Disk: {round(du.used/1e9,2)}/{round(du.total/1e9,2)} GB ({du.percent}%)")
                    except Exception:
                        pass
                    return "\n".join(["🖥️ System Info:"] + info) or "No system info available"
                except Exception as e:
                    return f"❌ Failed to get sysinfo: {str(e)[:100]}"

            if cmd.startswith("run "):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /run [command]"
                return self.execute_command(parts[1])

            if cmd.startswith("calc "):
                try:
                    expr = command.split(" ", 1)[1]
                    import math
                    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
                    allowed.update({})
                    # Very restricted eval
                    value = eval(expr, {"__builtins__": {}}, allowed)
                    return f"🧮 {expr} = {value}"
                except Exception as e:
                    return f"❌ Calculation error: {str(e)[:100]}"

            if cmd == "explore":
                try:
                    root = os.getcwd()
                    entries = []
                    for i, name in enumerate(sorted(os.listdir(root))[:100], 1):
                        path = os.path.join(root, name)
                        tag = "DIR" if os.path.isdir(path) else "FILE"
                        entries.append(f"{i:>2}. [{tag}] {name}")
                    return "\n".join([f"📁 {root}"] + entries)
                except Exception as e:
                    return f"❌ Explore failed: {str(e)[:100]}"

            if cmd.startswith("weather "):
                try:
                    city = command.split(" ", 1)[1].strip()
                    if not city:
                        return "Usage: /weather [city]"
                    url = f"https://wttr.in/{city}?format=j1"
                    resp = self.ai.session.get(url, timeout=10)
                    if resp.status_code != 200:
                        return f"❌ Weather error: {resp.status_code}"
                    data = resp.json()
                    cur = data.get("current_condition", [{}])[0]
                    tempC = cur.get("temp_C", "?")
                    desc = cur.get("weatherDesc", [{}])[0].get("value", "")
                    humid = cur.get("humidity", "?")
                    return f"🌤️ {city}: {tempC}°C, {desc}, humidity {humid}%"
                except Exception as e:
                    return f"❌ Weather failed: {str(e)[:100]}"

            if cmd.startswith("note "):
                try:
                    note = command.split(" ", 1)[1]
                    notes_path = os.path.join(_get_home_dir(), '.aetherai', 'notes.txt')
                    os.makedirs(os.path.dirname(notes_path), exist_ok=True)
                    with open(notes_path, 'a', encoding='utf-8') as f:
                        f.write(note + "\n")
                    return "📝 Note saved."
                except Exception as e:
                    return f"❌ Failed to save note: {str(e)[:100]}"

            if cmd == "notes":
                try:
                    notes_path = os.path.join(_get_home_dir(), '.nexus', 'notes.txt')
                    if not os.path.exists(notes_path):
                        return "No notes saved yet."
                    with open(notes_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                    return content or "No notes saved yet."
                except Exception as e:
                    return f"❌ Failed to read notes: {str(e)[:100]}"

            if cmd.startswith("timer "):
                try:
                    parts = command.split()
                    if len(parts) != 2 or not parts[1].isdigit():
                        return "Usage: /timer [seconds]"
                    secs = int(parts[1])
                    def _timer_thread(s):
                        try:
                            time.sleep(s)
                            console.print(f"⏰ Timer done: {s} seconds elapsed")
                        except Exception:
                            pass
                    threading.Thread(target=_timer_thread, args=(secs,), daemon=True).start()
                    return f"⏳ Timer started for {secs} seconds"
                except Exception as e:
                    return f"❌ Failed to start timer: {str(e)[:100]}"

            if cmd.startswith("convert "):
                try:
                    parts = command.split()
                    if len(parts) != 4:
                        return "Usage: /convert [val] [from] [to]"
                    val = float(parts[1])
                    src = parts[2].lower()
                    dst = parts[3].lower()
                    # Temperature
                    if src in ["c", "f", "k"] and dst in ["c", "f", "k"]:
                        c = val
                        if src == "f":
                            c = (val - 32) * 5/9
                        elif src == "k":
                            c = val - 273.15
                        out = c
                        if dst == "f":
                            out = c * 9/5 + 32
                        elif dst == "k":
                            out = c + 273.15
                        else:
                            out = c
                        return f"🌡️ {val}{src.upper()} = {round(out, 3)}{dst.upper()}"
                    # Weight
                    if src in ["kg", "lb"] and dst in ["kg", "lb"]:
                        kg = val if src == "kg" else val * 0.45359237
                        out = kg if dst == "kg" else kg / 0.45359237
                        return f"⚖️ {val}{src} = {round(out, 3)}{dst}"
                    # Data
                    if src in ["kb", "mb", "gb"] and dst in ["kb", "mb", "gb"]:
                        factor = {"kb": 1, "mb": 1024, "gb": 1024*1024}
                        kb = val * factor[src]
                        out = kb / factor[dst]
                        return f"💾 {val}{src.upper()} = {round(out, 3)}{dst.upper()}"
                    return "❌ Unsupported conversion"
                except Exception as e:
                    return f"❌ Conversion error: {str(e)[:100]}"

            if cmd == "joke":
                try:
                    import random
                    jokes = [
                        "Why do programmers prefer dark mode? Because light attracts bugs.",
                        "There are only 10 kinds of people in the world: those who understand binary and those who don’t.",
                        "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
                    ]
                    return random.choice(jokes)
                except Exception as e:
                    return f"❌ Joke error: {str(e)[:100]}"

            if cmd.startswith("password"):
                try:
                    import secrets, string
                    parts = command.split()
                    length = 16
                    if len(parts) == 2 and parts[1].isdigit():
                        length = max(8, min(128, int(parts[1])))
                    alphabet = string.ascii_letters + string.digits + string.punctuation
                    pwd = ''.join(secrets.choice(alphabet) for _ in range(length))
                    return pwd
                except Exception as e:
                    return f"❌ Password generation failed: {str(e)[:100]}"

            if cmd == "tip":
                try:
                    import random
                    tips = [
                        "Use meaningful commit messages.",
                        "Write tests before refactoring.",
                        "Keep functions small and focused.",
                        "Prefer composition over inheritance.",
                        "Automate repetitive tasks.",
                    ]
                    return random.choice(tips)
                except Exception as e:
                    return f"❌ Tip error: {str(e)[:100]}"

            # --- Web Search ---
            if cmd.startswith("websearch"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /websearch [query] - Search the web using DuckDuckGo"
                query = parts[1]
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]🔍 Searching the web..."),
                        transient=True
                    ) as progress:
                        progress.add_task("search", total=None)

                        resp = self.ai.session.get(
                            f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}",
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                            timeout=15
                        )

                    if resp.status_code == 200:
                        # Enhanced result extraction
                        import re
                        results = re.findall(r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>', resp.text)

                        if not results:
                            return " No search results found. Try different keywords."

                        # Create a nice table for results
                        search_table = Table(title=f" Web Search Results for: '{query}'", show_header=True, header_style="bold blue")
                        search_table.add_column("#", style="cyan", width=3)
                        search_table.add_column("Title", style="white", min_width=40)
                        search_table.add_column("URL", style="green", min_width=30)

                        for i, (url, title) in enumerate(results[:8]):  # Show more results
                            clean_title = re.sub('<.*?>', '', title).strip()
                            if len(clean_title) > 60:
                                clean_title = clean_title[:57] + "..."

                            # Clean up URL
                            if url.startswith('//'):
                                url = 'https:' + url
                            elif url.startswith('/'):
                                url = 'https://duckduckgo.com' + url

                            search_table.add_row(str(i+1), clean_title, url)

                        console.print(search_table)
                        console.print(f"\n Found {len(results)} results (showing top 8)")
                        console.print(" Click on URLs to visit the pages")
                        return ""

                    return f" Web search error: HTTP {resp.status_code}"

                except requests.exceptions.Timeout:
                    return " Web search timed out. Try again later."
                except requests.exceptions.ConnectionError:
                    return " No internet connection. Check your network."
                except Exception as e:
                    return f" Web search failed: {str(e)[:100]}"
            # --- Voice Input Command ---
            try:
                import speech_recognition as sr
            except ImportError:
                sr = None

            def listen_voice():
                if not sr:
                    return "SpeechRecognition not installed. Please install it to use voice input."
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    print(" Speak now...")
                    audio = recognizer.listen(source, timeout=5)
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    return text
                except Exception as e:
                    return f"Voice recognition failed: {str(e)}"

            if cmd.startswith("voice"):
                return listen_voice()

            # === NEW ADVANCED FEATURES ===

            # --- Context-Aware AI ---
            if cmd.startswith("learn"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /learn [topic] - Teach AI about a technology/framework"
                if not self.context_ai:
                    return " Context-Aware AI module not available"
                topic, content = parts[1], f"User is learning about {parts[1]}"
                return self.context_ai.learn_topic(topic, content)

            if cmd == "remind":
                return "Usage: /remind [task] - Set a reminder (optional: deadline)"

            if cmd.startswith("remind "):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /remind [task] - Set a reminder"
                if not self.context_ai:
                    return " Context-Aware AI module not available"
                return self.context_ai.remind_task(parts[1])

            if cmd == "reminders":
                if not self.context_ai:
                    return " Context-Aware AI module not available"
                reminders = self.context_ai.get_reminders()
                if not reminders:
                    return " No active reminders"
                output = " Your Reminders:\n"
                for i, reminder in enumerate(reminders):
                    output += f"{i+1}. {reminder['task']}"
                    if reminder.get('deadline'):
                        output += f" (Due: {reminder['deadline']})"
                    output += "\n"
                return output

            if cmd.startswith("complete-reminder"):
                parts = command.split()
                if len(parts) != 2:
                    return "Usage: /complete-reminder [number]"
                if not self.context_ai:
                    return "❌ Context-Aware AI module not available"
                try:
                    index = int(parts[1]) - 1
                    return self.context_ai.complete_reminder(index)
                except ValueError:
                    return "❌ Invalid reminder number"

            # --- Theme Management ---
            if cmd == "themes":
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                return self.theme_manager.list_themes()

            if cmd.startswith("theme set"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /theme set [theme_name] - Switch to a theme"
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                result = self.theme_manager.set_current_theme(parts[2])
                # Update console theme immediately
                update_console_theme(self.theme_manager)
                return result

            if cmd == "theme current":
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                theme = self.theme_manager.get_current_theme()
                return f"🎨 Current Theme: {theme['name']} - {theme['description']}"

            if cmd.startswith("theme preview"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /theme preview [theme_name] - Preview a theme"
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                return self.theme_manager.preview_theme(parts[2])

            if cmd.startswith("theme create"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /theme create [name] [base_theme] - Create custom theme"
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                return self.theme_manager.create_custom_theme(parts[2], parts[3] if len(parts) > 3 else "dark")

            if cmd.startswith("theme delete"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /theme delete [theme_name] - Delete custom theme"
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                return self.theme_manager.delete_custom_theme(parts[2])

            if cmd.startswith("theme export"):
                parts = command.split()
                format_type = parts[2] if len(parts) > 2 else "json"
                if len(parts) < 3:
                    return "Usage: /theme export [theme_name] [format] - Export theme (json/python)"
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                return self.theme_manager.export_theme(parts[2], format_type)

            if cmd == "theme stats":
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                return self.theme_manager.get_theme_stats()

            if cmd == "theme reset":
                if not self.theme_manager:
                    return "❌ Theme Manager module not available"
                result = self.theme_manager.reset_to_default()
                # Update console theme immediately
                update_console_theme(self.theme_manager)
                return result

            # --- Code Review Assistant ---
            if cmd.startswith("review analyze"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /review analyze [file] - Analyze code quality"
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                analysis = self.code_reviewer.analyze_file(parts[2])
                return self.code_reviewer.generate_review_report(analysis)

            if cmd.startswith("review security"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /review security [file] - Security analysis only"
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                analysis = self.code_reviewer.analyze_file(parts[2], "security")
                return self.code_reviewer.generate_review_report(analysis)

            if cmd.startswith("review performance"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /review performance [file] - Performance analysis only"
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                analysis = self.code_reviewer.analyze_file(parts[2], "performance")
                return self.code_reviewer.generate_review_report(analysis)

            if cmd.startswith("review quality"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /review quality [file] - Quality metrics only"
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                analysis = self.code_reviewer.analyze_file(parts[2], "quality")
                return self.code_reviewer.generate_review_report(analysis)

            if cmd.startswith("review compare"):
                parts = command.split()
                if len(parts) != 4:
                    return "Usage: /review compare [file1] [file2] - Compare two files"
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                comparison = self.code_reviewer.compare_files(parts[2], parts[3])
                if "error" in comparison:
                    return f"❌ {comparison['error']}"
                output = f"📊 File Comparison Results:\n\n"
                output += f"📁 File 1: {comparison['file1']}\n"
                output += f"📁 File 2: {comparison['file2']}\n"
                output += f"📏 Lines: {comparison['file1_lines']} → {comparison['file2_lines']}\n"
                output += f"📈 Difference: {comparison['line_difference']} lines\n"
                output += f"🔗 Similarity: {comparison['similarity_score']}%\n"
                return output

            if cmd == "review history":
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                history = self.code_reviewer.get_review_history(10)
                if not history:
                    return "📝 No review history found"
                output = "📋 Recent Code Reviews:\n\n"
                for i, review in enumerate(history[-10:], 1):
                    output += f"{i}. 📁 {review['file_path']}\n"
                    output += f"   🗣️ {review['language']} | 📅 {review['timestamp'][:10]}\n"
                    if "quality_metrics" in review and "quality_score" in review["quality_metrics"]:
                        score = review["quality_metrics"]["quality_score"]
                        output += f"   📊 Quality: {score}/100\n"
                    output += "\n"
                return output

            if cmd == "review stats":
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                stats = self.code_reviewer.get_review_stats()
                output = "📊 Code Review Statistics:\n\n"
                output += f"📝 Total Reviews: {stats['total_reviews']}\n"
                output += f"🗣️ Languages: {', '.join(stats.get('languages_reviewed', []))}\n"
                output += f"📊 Avg Quality Score: {stats.get('avg_quality_score', 0)}/100\n"
                output += f"🔒 Security Issues Found: {stats.get('security_issues_found', 0)}\n"
                output += f"⚡ Performance Suggestions: {stats.get('performance_suggestions', 0)}\n"
                return output

            if cmd.startswith("review suggest"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /review suggest [file] - Get AI-powered improvement suggestions"
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                if not os.path.exists(parts[1]):
                    return f"❌ File not found: {parts[1]}"
                try:
                    with open(parts[1], 'r') as f:
                        code = f.read(2000)  # Limit to 2000 chars for AI processing
                    return self.ai.query(self.current_model, f"Review this code and provide improvement suggestions:\n\n```{self.code_reviewer.detect_language(parts[1])}\n{code}\n```")
                except Exception as e:
                    return f"❌ Error reading file: {str(e)}"

            if cmd.startswith("review language"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /review language [file] - Detect programming language"
                if not self.code_reviewer:
                    return "❌ Code Review Assistant module not available"
                language = self.code_reviewer.detect_language(parts[2])
                if language:
                    return f"🗣️ Detected Language: {language.upper()}"
                else:
                    return f"❓ Could not detect language for: {parts[2]}"

            # --- Integration Hub ---
            if cmd == "integrate":
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                return "🔗 Integration Hub Commands:\n" \
                       "• /integrate list - List configured services\n" \
                       "• /integrate supported - List supported services\n" \
                       "• /integrate add [service] - Add a service\n" \
                       "• /integrate remove [service] - Remove a service\n" \
                       "• /integrate test [service] - Test connection\n" \
                       "• /integrate info [service] - Service information\n" \
                       "• /integrate action [service] [action] - Execute action"

            if cmd == "integrate list":
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                return self.integration_hub.list_services()

            if cmd == "integrate supported":
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                return self.integration_hub.list_supported_services()

            if cmd.startswith("integrate add"):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /integrate add [service] [config_json] - Add service integration"
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                try:
                    config = json.loads(parts[2])
                    return self.integration_hub.add_service(parts[1], config)
                except json.JSONDecodeError:
                    return "❌ Invalid JSON configuration"

            if cmd.startswith("integrate remove"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /integrate remove [service] - Remove service integration"
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                return self.integration_hub.remove_service(parts[2])

            if cmd.startswith("integrate test"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /integrate test [service] - Test service connection"
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                return self.integration_hub.test_connection(parts[2])

            if cmd.startswith("integrate info"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /integrate info [service] - Get service information"
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                return self.integration_hub.get_service_info(parts[2])

            if cmd.startswith("integrate action"):
                parts = command.split(maxsplit=3)
                if len(parts) < 4:
                    return "Usage: /integrate action [service] [action] [params_json] - Execute service action"
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                try:
                    params = json.loads(parts[3]) if len(parts) > 3 else {}
                    return self.integration_hub.execute_service_action(parts[1], parts[2], **params)
                except json.JSONDecodeError:
                    return "❌ Invalid JSON parameters"

            if cmd == "integrate stats":
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                stats = self.integration_hub.get_integration_stats()
                output = "📊 Integration Statistics:\n\n"
                output += f"🔗 Configured Services: {stats['total_services']}\n"
                output += f"🚀 Supported Services: {stats['supported_services']}\n"
                output += f"🟢 Connected Services: {stats['connected_services']}\n"
                output += f"🔴 Failed Connections: {stats['failed_connections']}\n"
                output += f"🪝 Total Webhooks: {stats['total_webhooks']}\n"
                output += f"✅ Active Webhooks: {stats['active_webhooks']}\n"
                return output

            if cmd == "webhooks":
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                return self.integration_hub.list_webhooks()

            if cmd.startswith("webhook add"):
                parts = command.split(maxsplit=3)
                if len(parts) < 4:
                    return "Usage: /webhook add [service] [url] [events_json] - Add webhook"
                if not self.integration_hub:
                    return "❌ Integration Hub module not available"
                try:
                    events = json.loads(parts[3])
                    return self.integration_hub.setup_webhook(parts[1], parts[2], events)
                except json.JSONDecodeError:
                    return "❌ Invalid JSON events list"

            # --- Task Management ---
            if cmd.startswith("task add"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /task add [title] - Add a new task"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                return self.task_manager.create_task(parts[1])

            if cmd.startswith("task create"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /task create [title] - Create a new task"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                return self.task_manager.create_task(parts[1])

            if cmd == "tasks":
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                tasks = self.task_manager.get_tasks(status="pending", limit=20)
                if not tasks:
                    return "📝 No pending tasks found"
                output = "📋 Your Tasks:\n\n"
                for i, task in enumerate(tasks, 1):
                    priority_icon = {"low": "🟢", "medium": "🟡", "high": "🔴", "urgent": "🟣"}.get(task["priority"], "⚪")
                    category_icon = self.task_manager.categories.get(task["category"], {}).get("icon", "📝")
                    output += f"{i}. {priority_icon} {category_icon} {task['title']}\n"
                    if task.get("due_date"):
                        due_date = datetime.fromtimestamp(float(task["due_date"])).strftime("%Y-%m-%d")
                        output += f"   📅 Due: {due_date}\n"
                    output += "\n"
                return output

            if cmd.startswith("task show"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /task show [id] - Show task details"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                task = self.task_manager.get_task_by_id(parts[2])
                if not task:
                    return f"❌ Task not found: {parts[2]}"
                output = f"📝 Task Details: {task['title']}\n\n"
                output += f"ID: {task['id']}\n"
                output += f"Status: {task['status'].upper()}\n"
                output += f"Priority: {task['priority'].upper()}\n"
                output += f"Category: {task['category'].title()}\n"
                if task.get("description"):
                    output += f"Description: {task['description']}\n"
                if task.get("due_date"):
                    due_date = datetime.fromtimestamp(float(task["due_date"])).strftime("%Y-%m-%d %H:%M")
                    output += f"Due Date: {due_date}\n"
                if task.get("tags"):
                    output += f"Tags: {', '.join(task['tags'])}\n"
                if task.get("subtasks"):
                    output += "Subtasks:\n"
                    for subtask in task["subtasks"]:
                        status = "✅" if subtask["completed"] else "❌"
                        output += f"  {status} {subtask['title']}\n"
                return output

            if cmd.startswith("task complete"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /task complete [id] - Mark task as completed"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                return self.task_manager.complete_task(parts[2])

            if cmd.startswith("task delete"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /task delete [id] - Delete a task"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                return self.task_manager.delete_task(parts[2])

            if cmd.startswith("task update"):
                parts = command.split(maxsplit=3)
                if len(parts) < 4:
                    return "Usage: /task update [id] [field] [value] - Update task field"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                task_id, field, value = parts[2], parts[3], " ".join(parts[4:])
                return self.task_manager.update_task(task_id, **{field: value})

            if cmd.startswith("task priority"):
                parts = command.split()
                if len(parts) != 4:
                    return "Usage: /task priority [id] [priority] - Set task priority (low/medium/high/urgent)"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                return self.task_manager.update_task(parts[2], priority=parts[3])

            if cmd.startswith("task category"):
                parts = command.split()
                if len(parts) != 4:
                    return "Usage: /task category [id] [category] - Set task category"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                return self.task_manager.update_task(parts[2], category=parts[3])

            if cmd.startswith("task due"):
                parts = command.split()
                if len(parts) != 4:
                    return "Usage: /task due [id] [date] - Set due date (YYYY-MM-DD)"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                try:
                    due_timestamp = datetime.strptime(parts[3], "%Y-%m-%d").timestamp()
                    return self.task_manager.update_task(parts[2], due_date=due_timestamp)
                except ValueError:
                    return "❌ Invalid date format. Use YYYY-MM-DD"

            if cmd.startswith("task subtask add"):
                parts = command.split(maxsplit=3)
                if len(parts) < 4:
                    return "Usage: /task subtask add [task_id] [title] - Add subtask"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                return self.task_manager.add_subtask(parts[3], " ".join(parts[4:]))

            if cmd.startswith("task subtask complete"):
                parts = command.split()
                if len(parts) != 5:
                    return "Usage: /task subtask complete [task_id] [subtask_id] - Complete subtask"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                return self.task_manager.complete_subtask(parts[3], parts[4])

            if cmd == "task stats":
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                stats = self.task_manager.get_task_stats()
                if "error" in stats:
                    return f"❌ {stats['error']}"
                output = "📊 Task Statistics:\n\n"
                output += f"Total Tasks: {stats['total_tasks']}\n"
                output += f"Completed: {stats['completed_tasks']}\n"
                output += f"Pending: {stats['pending_tasks']}\n"
                output += ".1f"
                output += "\nPriority Breakdown:\n"
                for priority, count in stats['priority_breakdown'].items():
                    output += f"  {priority.title()}: {count}\n"
                output += "\nCategory Breakdown:\n"
                for category, count in stats['category_breakdown'].items():
                    output += f"  {category.title()}: {count}\n"
                return output

            if cmd.startswith("task search"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /task search [query] - Search tasks"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                results = self.task_manager.search_tasks(parts[1])
                if not results:
                    return f"🔍 No tasks found matching: {parts[1]}"
                output = f"🔍 Search Results for '{parts[1]}':\n\n"
                for task in results[:10]:
                    output += f"📝 {task['title']} (ID: {task['id']})\n"
                    output += f"   Status: {task['status']} | Priority: {task['priority']}\n\n"
                return output

            if cmd == "task overdue":
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                overdue = self.task_manager.get_overdue_tasks()
                if not overdue:
                    return "✅ No overdue tasks!"
                output = "⚠️ Overdue Tasks:\n\n"
                for task in overdue:
                    due_date = datetime.fromtimestamp(float(task["due_date"])).strftime("%Y-%m-%d")
                    output += f"📝 {task['title']} (ID: {task['id']})\n"
                    output += f"   Was due: {due_date}\n\n"
                return output

            if cmd.startswith("task export"):
                parts = command.split()
                format_type = parts[1] if len(parts) > 1 else "json"
                if not self.task_manager:
                    return "❌ Task Manager module not available"
                export_data = self.task_manager.export_tasks(format_type)
                if format_type == "json":
                    return f"📄 JSON Export:\n{export_data}"
                else:
                    return f"📄 CSV Export:\n{export_data}"

            # --- Analytics & Monitoring ---
            if cmd == "analytics":
                if not self.analytics:
                    return "❌ Analytics module not available"
                stats = self.analytics.get_usage_stats()
                output = "📊 Usage Analytics:\n"
                output += f"Total Interactions: {stats['total_interactions']}\n"
                output += "Feature Usage:\n"
                for feature, count in stats['feature_usage'].items():
                    output += f"  {feature}: {count}\n"
                return output

            if cmd == "error-analytics":
                if not self.analytics:
                    return "❌ Analytics module not available"
                errors = self.analytics.get_error_analytics()
                output = "❌ Error Analytics:\n"
                output += f"Total Errors: {errors['total_errors']}\n"
                output += "Error Types:\n"
                for error_type, count in errors['error_types'].items():
                    output += f"  {error_type}: {count}\n"
                return output

            if cmd == "start-monitoring":
                if not self.analytics:
                    return "❌ Analytics module not available"
                return self.analytics.start_monitoring()

            if cmd == "stop-monitoring":
                if not self.analytics:
                    return "❌ Analytics module not available"
                return self.analytics.stop_monitoring()

            if cmd == "net-diag":
                if not self.analytics:
                    return "❌ Analytics module not available"
                diag = self.analytics.network_diagnostics()
                if "error" in diag:
                    return f"❌ {diag['error']}"
                output = "🌐 Network Diagnostics:\n"
                for service, status in diag['connectivity'].items():
                    output += f"{service}: {'[OK]' if status.get('status') == 'reachable' else '[FAIL]'} "
                    if 'latency_ms' in status:
                        output += f"({status['latency_ms']}ms)"
                    output += "\n"
                return output

            if cmd == "analyze-logs":
                if not self.analytics:
                    return "❌ Analytics module not available"
                analysis = self.analytics.analyze_logs()
                if "error" in analysis:
                    return f"❌ {analysis['error']}"
                output = "📋 Log Analysis:\n"
                output += f"Files Analyzed: {analysis['files_analyzed']}\n"
                output += f"Total Lines: {analysis['total_lines']}\n"
                output += f"Errors: {analysis['error_count']}\n"
                output += f"Warnings: {analysis['warning_count']}\n"
                return output

            if cmd == "health":
                if not self.analytics:
                    return "❌ Analytics module not available"
                health = self.analytics.health_check()
                if "error" in health:
                    return f"❌ {health['error']}"
                output = "🏥 System Health Check:\n"
                output += f"Overall Status: {health['overall_status'].upper()}\n"
                for component, check in health['checks'].items():
                    status_icon = "[OK]" if check['status'] == "good" else "[WARN]" if check['status'] == "warning" else "[FAIL]"
                    output += f"{component.title()}: {status_icon} {check['message']}\n"
                if health.get('recommendations'):
                    output += "\n💡 Recommendations:\n"
                    for rec in health['recommendations']:
                        output += f"• {rec}\n"
                return output

            # --- Games & Learning ---
            if cmd.startswith("challenge"):
                parts = command.split()
                difficulty = parts[1] if len(parts) > 1 else "easy"
                if not self.games:
                    return "❌ Games & Learning module not available"
                challenge = self.games.get_coding_challenge(difficulty)
                if "error" in challenge:
                    return f"❌ {challenge['error']}"
                output = f" {challenge['title']}\n"
                output += f"Difficulty: {challenge['difficulty'].upper()}\n"
                output += f"Problem: {challenge['problem']['question']}\n"
                output += f"Starter Code:\n{challenge['problem']['starter_code']}\n"
                return output

            if cmd.startswith("submit-challenge"):
                parts = command.split(maxsplit=3)
                if len(parts) < 4:
                    return "Usage: /submit-challenge [challenge_id] [problem_id] [solution]"
                if not self.games:
                    return "❌ Games & Learning module not available"
                result = self.games.submit_challenge_solution(parts[1], parts[2], parts[3])
                if "error" in result:
                    return f"❌ {result['error']}"
                output = f"📊 Challenge Result:\n"
                output += f"Score: {result['score']}%\n"
                output += f"Tests Passed: {result['passed']}/{result['total_tests']}\n"
                if result.get('achievements'):
                    output += f"Achievements: {', '.join(result['achievements'])}\n"
                return output

            if cmd.startswith("tutorial"):
                parts = command.split()
                tutorial_id = parts[1] if len(parts) > 1 else "python_intro"
                if not self.games:
                    return "❌ Games & Learning module not available"
                tutorial = self.games.start_tutorial(tutorial_id)
                if "error" in tutorial:
                    return f"❌ {tutorial['error']}"
                output = f"📚 {tutorial['title']}\n"
                output += f"Description: {tutorial['description']}\n"
                output += f"Sections: {tutorial['total_sections']}\n"
                return output

            if cmd.startswith("tutorial-section"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /tutorial-section [tutorial_id] [section_number]"
                if not self.games:
                    return "❌ Games & Learning module not available"
                try:
                    section_num = int(parts[2]) - 1
                    section = self.games.get_tutorial_section(parts[1], section_num)
                    if "error" in section:
                        return f"❌ {section['error']}"
                    output = f"📖 Section {section['section_index'] + 1}: {section['title']}\n\n"
                    output += f"{section['content']}\n\n"
                    output += "Examples:\n"
                    for example in section['examples']:
                        output += f"  {example}\n"
                    return output
                except ValueError:
                    return "❌ Invalid section number"

            if cmd.startswith("quiz"):
                parts = command.split()
                quiz_id = parts[1] if len(parts) > 1 else "python_fundamentals"
                if not self.games:
                    return "❌ Games & Learning module not available"
                quiz = self.games.take_quiz(quiz_id)
                if "error" in quiz:
                    return f"❌ {quiz['error']}"
                if quiz.get('completed'):
                    return f" Quiz Completed!\nFinal Score: {quiz['final_score']}%\nCorrect: {quiz['correct_answers']}/{quiz['total_questions']}"
                output = f"❓ Question {quiz['question_number']}/{quiz['total_questions']}\n"
                output += f"{quiz['question']}\n\n"
                for i, option in enumerate(quiz['options']):
                    output += f"{i+1}. {option}\n"
                return output

            if cmd.startswith("answer-quiz"):
                parts = command.split(maxsplit=2)
                if len(parts) != 3:
                    return "Usage: /answer-quiz [quiz_id] [answer_number]"
                if not self.games:
                    return "❌ Games & Learning module not available"
                try:
                    answer_num = int(parts[2]) - 1
                    if 0 <= answer_num < 4:  # Assuming 4 options max
                        answer = ["A", "B", "C", "D"][answer_num]
                        result = self.games.submit_quiz_answer(parts[1], answer)
                        if "error" in result:
                            return f"❌ {result['error']}"
                        correctness = " Correct!" if result['correct'] else " Incorrect"
                        output = f"{correctness}\n"
                        output += f"Explanation: {result['explanation']}\n"
                        if result.get('next_question'):
                            output += f"Next question ready. Use /quiz {parts[1]} to continue."
                        return output
                    else:
                        return "❌ Invalid answer number (1-4)"
                except ValueError:
                    return "❌ Invalid answer number"

            if cmd == "user-stats":
                if not self.games:
                    return "❌ Games & Learning module not available"
                stats = self.games.get_user_stats()
                if "error" in stats:
                    return f"❌ {stats['error']}"
                output = "🏆 Your Stats:\n"
                output += f"Challenges Completed: {stats['challenges_completed']}\n"
                output += f"Average Score: {stats['average_score']:.1f}%\n"
                output += f"Achievements: {stats['achievement_count']}\n"
                return output

            # --- Creative Tools ---
            if cmd.startswith("ascii"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /ascii [text] - Generate ASCII art"
                if not self.creative:
                    return "❌ Creative Tools module not available"
                art = self.creative.generate_ascii_art(parts[1])
                return f"🎨 ASCII Art:\n{art}"

            if cmd.startswith("colors"):
                parts = command.split()
                scheme_type = parts[1] if len(parts) > 1 else "complementary"
                base_color = parts[2] if len(parts) > 2 else None
                if not self.creative:
                    return "❌ Creative Tools module not available"
                scheme = self.creative.generate_color_scheme(base_color, scheme_type)
                if "error" in scheme:
                    return f"❌ {scheme['error']}"
                output = f"🎨 Color Scheme ({scheme['type']}):\n"
                for i, color in enumerate(scheme['colors']):
                    output += f"Color {i+1}: {color}\n"
                return output

            if cmd.startswith("music"):
                parts = command.split()
                mood = parts[1] if len(parts) > 1 else "happy"
                length = int(parts[2]) if len(parts) > 2 else 8
                if not self.creative:
                    return "❌ Creative Tools module not available"
                music = self.creative.generate_music(mood, length)
                if "error" in music:
                    return f"❌ {music['error']}"
                output = f"🎵 {music['description']}\n"
                output += f"Notes: {music['notation']}\n"
                return output

            if cmd.startswith("story"):
                parts = command.split()
                genre = parts[1] if len(parts) > 1 else "fantasy"
                length = parts[2] if len(parts) > 2 else "short"
                if not self.creative:
                    return "❌ Creative Tools module not available"
                story = self.creative.generate_story(genre, length)
                if "error" in story:
                    return f"❌ {story['error']}"
                output = f"📖 {story['title']}\n\n"
                output += f"{story['story']}\n"
                return output

            # --- Advanced Security ---
            if cmd.startswith("encrypt"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /encrypt [message] - Encrypt a message"
                if not self.adv_security:
                    return "❌ Advanced Security module not available"
                encrypted = self.adv_security.encrypt_message(parts[1])
                return f"🔐 Encrypted: {encrypted}"

            if cmd.startswith("decrypt"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /decrypt [encrypted_message] - Decrypt a message"
                if not self.adv_security:
                    return "❌ Advanced Security module not available"
                decrypted = self.adv_security.decrypt_message(parts[1])
                return f"🔓 Decrypted: {decrypted}"

            if cmd.startswith("rotate-key"):
                parts = command.split()
                if len(parts) != 3:
                    return "Usage: /rotate-key [service] [current_key] - Rotate API key"
                if not self.adv_security:
                    return "❌ Advanced Security module not available"
                result = self.adv_security.rotate_api_key(parts[1], parts[2])
                if "error" in result:
                    return f"❌ {result['error']}"
                return f"🔄 Key rotated for {result['service']}\nNew Key: {result['new_key']}"

            if cmd.startswith("biometric-auth"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /biometric-auth [biometric_data] - Authenticate with biometrics"
                if not self.adv_security:
                    return "❌ Advanced Security module not available"
                user = self.user_manager.current_user or "anonymous"
                result = self.adv_security.biometric_authenticate(user, parts[1])
                if "error" in result:
                    return f"❌ {result['error']}"
                status = "✅" if result['authenticated'] else "❌"
                return f"{status} {result['message']}"

            if cmd.startswith("secure-password"):
                parts = command.split()
                length = int(parts[1]) if len(parts) > 1 else 16
                if not self.adv_security:
                    return "❌ Advanced Security module not available"
                password = self.adv_security.generate_secure_password(length)
                return f"🔐 Secure Password: {password}"

            if cmd == "security-report":
                if not self.adv_security:
                    return "❌ Advanced Security module not available"
                report = self.adv_security.get_security_report()
                if "error" in report:
                    return f"❌ {report['error']}"
                output = "🔒 Security Report:\n"
                output += f"Period: {report['period_days']} days\n"
                output += f"Total Events: {report['total_events']}\n"
                output += f"Threats Detected: {report['threats_detected']}\n"
                output += f"Auth Attempts: {report['auth_attempts']}\n"
                return output

            if cmd == "threat-scan":
                return "Usage: /threat-scan [text] - Scan text for security threats"

            if cmd.startswith("threat-scan"):
                parts = command.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /threat-scan [text] - Scan text for security threats"
                if not self.adv_security:
                    return "❌ Advanced Security module not available"
                threats = self.adv_security.detect_threats(parts[1])
                if not threats:
                    return "✅ No threats detected"
                output = "🚨 Threats Detected:\n"
                for threat in threats:
                    output += f"Pattern: {threat['pattern'][:50]}...\n"
                    output += f"Severity: {threat['severity'].upper()}\n"
                    output += f"Matches: {len(threat['matches'])}\n\n"
                return output

            # --- System Commands ---
            if cmd == "models":
                """List all available AI models"""
                output = "🤖 Available AI Models:\n\n"
                for model, description in self.model_descriptions.items():
                    status = self.ai.status.get(model, "Unknown")
                    status_icon = "[OK]" if "[OK]" in status or "Ready" in status else "[FAIL]" if "[FAIL]" in status or "Error" in status else "[--]"
                    output += f"{status_icon} {model.upper()}: {description}\n"
                    output += f"   Status: {status}\n\n"
                output += "💡 Use '/switch [model]' to change models\n"
                output += "💡 Use '/ollama-models' for detailed Ollama model list"
                return output

            if cmd == "ollama-models":
                """Show detailed list of available Ollama models"""
                if not self.ai._check_ollama():
                    return "❌ Ollama is not running or not installed.\n   Please start Ollama first."
                
                try:
                    # Get models - handle both dict and ListResponse object
                    models_response = ollama.list()
                    if hasattr(models_response, 'models'):
                        models_data = models_response.models
                    else:
                        models_data = models_response.get("models", [])
                    
                    if not models_data:
                        return "❌ No Ollama models found.\n   Use 'ollama pull [model]' to download models."
                    
                    output = " Available Ollama Models:\n\n"
                    for model in models_data:
                        # Handle both dict and Model object
                        if hasattr(model, 'model'):
                            name = model.model
                            size = getattr(model, 'size', 0)
                            modified = getattr(model, 'modified_at', "Unknown")
                        else:
                            name = model.get("name", "Unknown")
                            size = model.get("size", 0)
                            modified = model.get("modified_at", "Unknown")
                        
                        # Format size
                        if size > 1024**3:  # GB
                            size_str = f"{size / (1024**3):.1f} GB"
                        elif size > 1024**2:  # MB
                            size_str = f"{size / (1024**2):.1f} MB"
                        else:
                            size_str = f"{size} bytes"
                        
                        # Format date
                        if modified != "Unknown":
                            try:
                                from datetime import datetime
                                date_obj = datetime.fromisoformat(modified.replace('Z', '+00:00'))
                                date_str = date_obj.strftime("%Y-%m-%d %H:%M")
                            except:
                                date_str = modified[:10]
                        else:
                            date_str = "Unknown"
                        
                        output += f" {name}\n"
                        output += f"   Size: {size_str}\n"
                        output += f"   Modified: {date_str}\n\n"
                    
                    output += f"💡 Use '/switch ollama [model_name]' to switch to a specific model\n"
                    # Get first model name correctly
                    first_model_name = models_data[0].model if hasattr(models_data[0], 'model') else models_data[0].get('name', 'model_name')
                    output += f"💡 Example: /switch ollama {first_model_name}"
                    return output
                    
                except Exception as e:
                    return f"❌ Error fetching Ollama models: {str(e)}\n   Make sure Ollama is running"

            if cmd == "current-model":
                """Show currently active AI model"""
                current = self.current_model or "None"
                status = self.ai.status.get(current.split(':')[0] if ':' in current else current, "Unknown")
                output = f" Current AI Model: {current.upper()}\n"
                output += f"📊 Status: {status}\n"
                if current.startswith("ollama:"):
                    model_name = current.split(":", 1)[1]
                    output += f" Ollama Model: {model_name}\n"
                description = self.model_descriptions.get(current.split(':')[0] if ':' in current else current, "No description available")
                output += f"📝 Description: {description}\n"
                return output

            if cmd == "run":
                return "Usage: /run [command] - Execute system commands like 'ls', 'pwd', 'whoami', etc."

            if cmd.startswith("run "):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /run [command]"
                return self.execute_command(parts[1])

            # Old dashboard command removed in favor of new subprocess-based one


            if cmd == "games":
                if GamesTUI:
                    try:
                        games_tui = GamesTUI(console)
                        games_tui.show_menu()
                        return " Games menu closed"
                    except Exception as e:
                        return f" Error running games: {e}"
                return " Games module not available"

            # ================================================================
            # NEW FEATURE COMMANDS - Developer Tools, Sessions, MCP
            # ================================================================
            
            # --- File Analysis ---
            if cmd.startswith("analyze "):
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                parts = command.split(maxsplit=1)
                if len(parts) < 2:
                    return "Usage: /analyze <file_path>"
                try:
                    # Update AI query function
                    self.developer_tools.ai_query = self.ai.query
                    analysis = self.developer_tools.analyze_file(parts[1].strip(), self.current_model)
                    return self.developer_tools.format_analysis(analysis)
                except Exception as e:
                    return f"❌ Analysis failed: {str(e)}"
            
            # --- Commit Message Generator ---
            if cmd == "commit-msg" or cmd == "commit-message":
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                try:
                    self.developer_tools.ai_query = self.ai.query
                    return self.developer_tools.generate_commit_message(self.current_model)
                except Exception as e:
                    return f"❌ Failed to generate commit message: {str(e)}"
            
            # --- PR Description Generator ---
            if cmd.startswith("pr-desc"):
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                parts = command.split()
                base_branch = parts[1] if len(parts) > 1 else "main"
                try:
                    self.developer_tools.ai_query = self.ai.query
                    return self.developer_tools.generate_pr_description(base_branch, self.current_model)
                except Exception as e:
                    return f"❌ Failed to generate PR description: {str(e)}"
            
            # --- Clipboard Commands ---
            if cmd == "clip" or cmd == "copy":
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                if not self.context_engine or not self.context_engine.last_response:
                    return "❌ No response to copy. Ask me something first!"
                success, msg = self.developer_tools.copy_to_clipboard(self.context_engine.last_response)
                return msg
            
            if cmd.startswith("clip ") or cmd.startswith("copy "):
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                text = command.split(maxsplit=1)[1] if len(command.split()) > 1 else ""
                if not text:
                    return "Usage: /clip <text to copy>"
                success, msg = self.developer_tools.copy_to_clipboard(text)
                return msg
            
            if cmd == "paste":
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                content, error = self.developer_tools.paste_from_clipboard()
                if error:
                    return f"❌ {error}"
                if not content:
                    return "📋 Clipboard is empty"
                return f"📋 Clipboard content:\n```\n{content[:2000]}\n```"
            
            # --- Session Management ---
            if cmd.startswith("session "):
                if not self.context_engine:
                    return "❌ Context engine not available"
                parts = command.split()
                action = parts[1] if len(parts) > 1 else "list"
                
                if action == "new":
                    name = parts[2] if len(parts) > 2 else None
                    session_id = self.context_engine.new_session(name)
                    return f"✅ New session started: {session_id}"
                
                elif action == "save":
                    name = parts[2] if len(parts) > 2 else None
                    try:
                        session_name = self.context_engine.save_session(name)
                        return f"✅ Session saved: {session_name}"
                    except Exception as e:
                        return f"❌ Failed to save session: {str(e)}"
                
                elif action == "load":
                    if len(parts) < 3:
                        return "Usage: /session load <name>"
                    try:
                        data = self.context_engine.load_session(parts[2])
                        return f"✅ Session loaded: {parts[2]} ({data.get('message_count', 0)} messages)"
                    except Exception as e:
                        return f"❌ Failed to load session: {str(e)}"
                
                elif action == "list":
                    sessions = self.context_engine.list_sessions()
                    if not sessions:
                        return "📁 No saved sessions"
                    lines = ["📁 **Saved Sessions:**\n"]
                    for i, s in enumerate(sessions[:10], 1):
                        lines.append(f"  {i}. **{s['name']}** - {s['message_count']} msgs ({s['created_at'][:10]})")
                    return "\n".join(lines)
                
                elif action == "delete":
                    if len(parts) < 3:
                        return "Usage: /session delete <name>"
                    if self.context_engine.delete_session(parts[2]):
                        return f"✅ Session deleted: {parts[2]}"
                    return f"❌ Session not found: {parts[2]}"
                
                elif action == "current":
                    current = self.context_engine.current_session or "No active session"
                    msg_count = len(self.context_engine.messages)
                    return f"📍 Current session: {current} ({msg_count} messages)"
                
                else:
                    return "Usage: /session [new|save|load|list|delete|current] [name]"
            
            # --- Export ---
            if cmd.startswith("export"):
                if not self.context_engine:
                    return "❌ Context engine not available"
                parts = command.split()
                
                if len(parts) < 2:
                    return "Usage: /export <filename> [format: md|json|html|txt]"
                
                filename = parts[1]
                format_type = parts[2] if len(parts) > 2 else None
                
                try:
                    result = self.context_engine.export_to_file(filename, format_type)
                    return f"✅ Exported to: {result}"
                except Exception as e:
                    return f"❌ Export failed: {str(e)}"
            
            # --- Favorites ---
            if cmd == "save" or cmd == "favorite":
                if not self.context_engine:
                    return "❌ Context engine not available"
                try:
                    fav_id = self.context_engine.save_favorite()
                    return f"⭐ Saved as favorite: {fav_id}"
                except Exception as e:
                    return f"❌ Failed to save: {str(e)}"
            
            if cmd.startswith("save "):
                if not self.context_engine:
                    return "❌ Context engine not available"
                name = command.split(maxsplit=1)[1].strip()
                try:
                    fav_id = self.context_engine.save_favorite(name)
                    return f"⭐ Saved as favorite: {fav_id}"
                except Exception as e:
                    return f"❌ Failed to save: {str(e)}"
            
            if cmd == "favorites" or cmd == "favs":
                if not self.context_engine:
                    return "❌ Context engine not available"
                favs = self.context_engine.list_favorites()
                if not favs:
                    return "⭐ No favorites saved\n   Use /save to save the last response"
                lines = ["⭐ **Your Favorites:**\n"]
                for i, fav in enumerate(favs[:10], 1):
                    query = fav.get('query', '')[:50]
                    lines.append(f"  {i}. **{fav['id']}** - {query}...")
                lines.append("\n💡 Use /favorite <number> to view")
                return "\n".join(lines)
            
            if cmd.startswith("favorite ") or cmd.startswith("fav "):
                if not self.context_engine:
                    return "❌ Context engine not available"
                parts = command.split()
                if len(parts) < 2:
                    return "Usage: /favorite <name_or_number>"
                fav = self.context_engine.get_favorite(parts[1])
                if fav:
                    return f"⭐ **{fav['id']}**\n\n**Query:** {fav.get('query', 'N/A')}\n\n**Response:**\n{fav['response']}"
                return f"❌ Favorite not found: {parts[1]}"
            
            # --- Templates ---
            if cmd == "templates" or cmd == "template list":
                if not self.context_engine:
                    return "❌ Context engine not available"
                templates = self.context_engine.list_templates()
                if not templates:
                    return "📝 No templates available"
                lines = ["📝 **Available Templates:**\n"]
                for t in templates:
                    vars_str = f" ({', '.join(t['variables'])})" if t['variables'] else ""
                    lines.append(f"  • **{t['name']}**{vars_str} - {t['description']}")
                lines.append("\n💡 Use /template <name> to apply")
                return "\n".join(lines)
            
            if cmd.startswith("template "):
                if not self.context_engine:
                    return "❌ Context engine not available"
                parts = command.split(maxsplit=1)[1].split()
                if not parts:
                    return "Usage: /template <name> [var1=value1 var2=value2]"
                
                template_name = parts[0]
                template = self.context_engine.get_template(template_name)
                if not template:
                    return f"❌ Template not found: {template_name}"
                
                # Parse variables
                kwargs = {}
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        kwargs[key] = value
                
                if template['variables'] and not kwargs:
                    return f"📝 Template '{template_name}' requires variables: {', '.join(template['variables'])}\n   Usage: /template {template_name} " + " ".join([f"{v}=<value>" for v in template['variables']])
                
                try:
                    prompt = self.context_engine.apply_template(template_name, **kwargs)
                    return f"📝 **Applied Template:** {template_name}\n\n{prompt}"
                except Exception as e:
                    return f"❌ Template error: {str(e)}"
            
            # --- Multi-Model Compare ---
            if cmd.startswith("compare "):
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                prompt = command.split(maxsplit=1)[1] if len(command.split()) > 1 else ""
                if not prompt:
                    return "Usage: /compare <prompt to send to multiple models>"
                
                console.print("[dim]Querying multiple models...[/dim]")
                self.developer_tools.ai_query = self.ai.query
                try:
                    results = self.developer_tools.compare_models(prompt)
                    return self.developer_tools.format_comparison(results)
                except Exception as e:
                    return f"❌ Comparison failed: {str(e)}"
            
            # --- Generate Tests ---
            if cmd.startswith("generate-tests ") or cmd.startswith("test-gen "):
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                parts = command.split()
                if len(parts) < 2:
                    return "Usage: /generate-tests <file_path> [test_framework]"
                filepath = parts[1]
                framework = parts[2] if len(parts) > 2 else "pytest"
                try:
                    self.developer_tools.ai_query = self.ai.query
                    return self.developer_tools.generate_tests(filepath, self.current_model, framework)
                except Exception as e:
                    return f"❌ Failed to generate tests: {str(e)}"
            
            # --- Error Explainer ---
            if cmd.startswith("explain-error ") or cmd.startswith("fix-error "):
                if not self.developer_tools:
                    return "❌ Developer tools module not available"
                error_text = command.split(maxsplit=1)[1] if len(command.split()) > 1 else ""
                if not error_text:
                    return "Usage: /explain-error <error message or paste your error>"
                try:
                    self.developer_tools.ai_query = self.ai.query
                    return self.developer_tools.explain_error(error_text, "", self.current_model)
                except Exception as e:
                    return f"❌ Error analysis failed: {str(e)}"
            
            # --- MCP Commands ---
            if cmd.startswith("mcp "):
                if not self.mcp_manager:
                    return "❌ MCP Manager not available"
                parts = command.split()
                action = parts[1] if len(parts) > 1 else "list"
                
                if action == "list":
                    return self.mcp_manager.format_server_list()
                
                elif action == "available":
                    servers = self.mcp_manager.list_available_servers()
                    lines = ["📡 **Available MCP Servers:**\n"]
                    for s in servers:
                        status = "✅ installed" if s['installed'] else "⚪ available"
                        lines.append(f"  {status} **{s['name']}** - {s['description']}")
                    return "\n".join(lines)
                
                elif action == "add":
                    if len(parts) < 3:
                        return "Usage: /mcp add <server_name>"
                    server_name = parts[2]
                    server = self.mcp_manager.add_default_server(server_name)
                    if server:
                        return f"✅ Added MCP server: {server_name}\n   Use /mcp start {server_name} to start it"
                    return f"❌ Unknown server: {server_name}\n   Use /mcp available to see options"
                
                elif action == "start":
                    if len(parts) < 3:
                        return "Usage: /mcp start <server_name>"
                    success, msg = self.mcp_manager.start_server(parts[2])
                    return f"{'✅' if success else '❌'} {msg}"
                
                elif action == "stop":
                    if len(parts) < 3:
                        return "Usage: /mcp stop <server_name>"
                    success, msg = self.mcp_manager.stop_server(parts[2])
                    return f"{'✅' if success else '❌'} {msg}"
                
                elif action == "status":
                    if len(parts) < 3:
                        return "Usage: /mcp status <server_name>"
                    status = self.mcp_manager.get_server_status(parts[2])
                    if "error" in status:
                        return f"❌ {status['error']}"
                    lines = [f"📡 **{status['name']}**"]
                    lines.append(f"   Status: {status['status']}")
                    lines.append(f"   Command: {status['command']}")
                    if status.get('pid'):
                        lines.append(f"   PID: {status['pid']}")
                    return "\n".join(lines)
                
                elif action == "remove":
                    if len(parts) < 3:
                        return "Usage: /mcp remove <server_name>"
                    if self.mcp_manager.remove_server(parts[2]):
                        return f"✅ Removed MCP server: {parts[2]}"
                    return f"❌ Server not found: {parts[2]}"
                
                else:
                    return "Usage: /mcp [list|available|add|start|stop|status|remove] <server_name>"
            
            # --- Enhanced Voice Commands ---
            if cmd == "voice status":
                return self.voice_manager.format_status() if hasattr(self.voice_manager, 'format_status') else "Voice: " + ("Enabled" if self.voice_manager.enabled else "Disabled")
            
            if cmd == "voices":
                if hasattr(self.voice_manager, 'list_voices'):
                    voices = self.voice_manager.list_voices()
                    if not voices:
                        return "❌ No voices available"
                    lines = ["🔊 **Available Voices:**\n"]
                    for i, v in enumerate(voices[:10], 1):
                        lines.append(f"  {i}. {v.get('name', 'Unknown')}")
                    lines.append("\n💡 Use /voice set <name> to change voice")
                    return "\n".join(lines)
                return "❌ Voice listing not available"
            
            if cmd.startswith("speak "):
                text = command.split(maxsplit=1)[1] if len(command.split()) > 1 else ""
                if not text:
                    return "Usage: /speak <text to speak>"
                if hasattr(self.voice_manager, 'speak'):
                    self.voice_manager.enabled = True  # Temporarily enable for this
                    self.voice_manager.speak(text)
                    return "🔊 Speaking..."
                return "❌ TTS not available"
            
            # --- Project Context ---
            if cmd == "project" or cmd == "context":
                if not self.context_engine:
                    return "❌ Context engine not available"
                try:
                    context = self.context_engine.detect_project_context()
                    lines = [f"📁 **Project Context:**\n"]
                    lines.append(f"  Name: {context.get('name', 'Unknown')}")
                    lines.append(f"  Type: {context.get('type', 'Unknown')}")
                    if context.get('languages'):
                        lines.append(f"  Languages: {', '.join(context['languages'])}")
                    if context.get('frameworks'):
                        lines.append(f"  Frameworks: {', '.join(context['frameworks'])}")
                    if context.get('vcs'):
                        lines.append(f"  VCS: {context['vcs']}")
                    if context.get('repository'):
                        lines.append(f"  Repository: {context['repository']}")
                    return "\n".join(lines)
                except Exception as e:
                    return f"❌ Failed to detect project: {str(e)}"

            # ================================================================
            # SKILLS & RULES MANAGEMENT (Claude-like)
            # ================================================================
            
            # --- Skills Commands ---
            if cmd == "skills":
                if not self.skills_manager:
                    return "❌ Skills manager not available"
                return self.skills_manager.format_skills_list()
            
            if cmd == "skills reload":
                if not self.skills_manager:
                    return "❌ Skills manager not available"
                self.skills_manager.reload()
                return "✅ Skills and rules reloaded"
            
            if cmd == "skills create":
                if not self.skills_manager:
                    return "❌ Skills manager not available"
                try:
                    filepath = self.skills_manager.create_skills_file()
                    return f"✅ Created skills file: {filepath}\n   Edit this file to customize your AI's skills."
                except Exception as e:
                    return f"❌ Failed to create skills file: {str(e)}"
            
            # --- Rules Commands ---
            if cmd == "rules":
                if not self.skills_manager:
                    return "❌ Skills manager not available"
                return self.skills_manager.format_rules_list()
            
            if cmd == "rules create":
                if not self.skills_manager:
                    return "❌ Skills manager not available"
                try:
                    filepath = self.skills_manager.create_rules_file()
                    return f"✅ Created rules file: {filepath}\n   Edit this file to add project-specific rules."
                except Exception as e:
                    return f"❌ Failed to create rules file: {str(e)}"
            
            # ================================================================
            # WEB SEARCH & DOCUMENT FETCHING
            # ================================================================
            
            # --- Web Search ---
            if cmd.startswith("search ") or cmd.startswith("websearch "):
                if not self.web_searcher:
                    return "❌ Web searcher not available. Install: pip install duckduckgo-search requests beautifulsoup4"
                
                query = command.split(maxsplit=1)[1] if len(command.split()) > 1 else ""
                if not query:
                    return "Usage: /search <query>"
                
                try:
                    availability = self.web_searcher.is_available()
                    if not availability.get('web_search'):
                        return "❌ Web search not available. Install: pip install duckduckgo-search"
                    
                    results = self.web_searcher.search(query)
                    if set_last_search_results:
                        set_last_search_results(results)
                    return self.web_searcher.format_search_results(results)
                except Exception as e:
                    return f"❌ Search failed: {str(e)}"
            
            # --- Documentation Search ---
            if cmd.startswith("docs "):
                if not self.web_searcher:
                    return "❌ Web searcher not available"
                
                parts = command.split(maxsplit=2)
                if len(parts) < 2:
                    return "Usage: /docs <query> [technology]"
                
                query = parts[1]
                technology = parts[2] if len(parts) > 2 else None
                
                try:
                    console.print("[dim]Searching documentation...[/dim]")
                    return self.web_searcher.fetch_docs(query, technology)
                except Exception as e:
                    return f"❌ Documentation fetch failed: {str(e)}"
            
            # --- Fetch URL ---
            if cmd.startswith("fetch "):
                if not self.web_searcher:
                    return "❌ Web searcher not available"
                
                target = command.split(maxsplit=1)[1] if len(command.split()) > 1 else ""
                if not target:
                    return "Usage: /fetch <url or search result number>"
                
                try:
                    # Check if it's a number (referring to last search results)
                    if target.isdigit():
                        results = get_last_search_results() if get_last_search_results else []
                        idx = int(target) - 1
                        if 0 <= idx < len(results):
                            url = results[idx].url
                        else:
                            return f"❌ Invalid result number. Last search had {len(results)} results."
                    elif self.web_searcher.is_url(target):
                        url = target
                    else:
                        return f"❌ '{target}' is not a valid URL. Use /search first, then /fetch <number>"
                    
                    console.print(f"[dim]Fetching {url}...[/dim]")
                    doc = self.web_searcher.fetch_with_cache(url)
                    return self.web_searcher.format_document(doc)
                    
                except Exception as e:
                    return f"❌ Fetch failed: {str(e)}"
            
            # --- Read URL (alias for fetch) ---
            if cmd.startswith("read ") and self.web_searcher and self.web_searcher.is_url(command.split()[1] if len(command.split()) > 1 else ""):
                url = command.split(maxsplit=1)[1]
                try:
                    console.print(f"[dim]Reading {url}...[/dim]")
                    doc = self.web_searcher.fetch_with_cache(url)
                    return self.web_searcher.format_document(doc)
                except Exception as e:
                    return f"❌ Read failed: {str(e)}"

            # ================================================================
            # UTILITY COMMANDS
            # ================================================================
            
            # --- Quick Start Guide ---
            if cmd == "quickstart" or cmd == "getting-started":
                lines = [
                    "🚀 **AetherAI Quick Start Guide**\n",
                    "━" * 40 + "\n",
                    "**1. Set up your API keys:**",
                    "   `/setkey gemini YOUR_API_KEY`",
                    "   `/setkey groq YOUR_API_KEY`\n",
                    "**2. Choose your AI model:**",
                    "   `/switch gemini`  - Google's Gemini",
                    "   `/switch groq`    - Fast Groq Cloud",
                    "   `/switch ollama`  - Local models (private)\n",
                    "**3. Start chatting:**",
                    "   Just type your question!\n",
                    "**4. Try these powerful features:**",
                    "   `/analyze main.py`   - Analyze code",
                    "   `/commit-msg`        - Generate commit message",
                    "   `/search python asyncio` - Search the web",
                    "   `/templates`         - View prompt templates\n",
                    "**5. Customize with Skills & Rules:**",
                    "   `/skills create`     - Create SKILLS.md",
                    "   `/rules create`      - Create RULES.md\n",
                    "📖 Type `/help` for full command list"
                ]
                return "\n".join(lines)
            
            # --- Feature Status ---
            if cmd == "status" or cmd == "features":
                lines = [
                    "📊 **AetherAI Feature Status**\n",
                    "━" * 40 + "\n",
                    "**Core:**",
                    f"  ✅ AI Models: {', '.join(self.ai.status.keys())}",
                    f"  ✅ Current Model: {self.current_model}\n",
                    "**Modules:**"
                ]
                
                # Check module availability
                modules = [
                    ("Context Engine", self.context_engine),
                    ("Developer Tools", self.developer_tools),
                    ("Skills Manager", self.skills_manager),
                    ("Web Searcher", self.web_searcher),
                    ("MCP Manager", self.mcp_manager),
                    ("Voice Manager", self.voice_manager),
                    ("Analytics", self.analytics),
                    ("Theme Manager", self.theme_manager),
                ]
                
                for name, module in modules:
                    status = "✅" if module else "❌"
                    lines.append(f"  {status} {name}")
                
                # Check optional dependencies
                lines.append("\n**Optional Dependencies:**")
                try:
                    import pyperclip
                    lines.append("  ✅ Clipboard (pyperclip)")
                except ImportError:
                    lines.append("  ❌ Clipboard - `pip install pyperclip`")
                
                if self.web_searcher:
                    avail = self.web_searcher.is_available()
                    lines.append(f"  {'✅' if avail.get('duckduckgo') else '❌'} Web Search - `pip install duckduckgo-search`")
                    lines.append(f"  {'✅' if avail.get('html2text') else '❌'} HTML2Text - `pip install html2text`")
                
                if hasattr(self.voice_manager, 'is_stt_available'):
                    lines.append(f"  {'✅' if self.voice_manager.is_stt_available() else '❌'} Speech-to-Text - `pip install SpeechRecognition pyaudio`")
                    lines.append(f"  {'✅' if self.voice_manager.is_tts_available() else '❌'} Text-to-Speech - `pip install pyttsx3`")
                
                # Show loaded skills/rules
                if self.skills_manager:
                    skill_count = len(self.skills_manager.list_skills())
                    rule_count = len(self.skills_manager.list_rules())
                    lines.append(f"\n**Configuration:**")
                    lines.append(f"  📋 Skills loaded: {skill_count}")
                    lines.append(f"  📜 Rules loaded: {rule_count}")
                
                return "\n".join(lines)
            
            # --- Cheatsheet ---
            if cmd == "cheat" or cmd == "cheatsheet":
                lines = [
                    "📖 **AetherAI Cheatsheet**\n",
                    "━" * 40 + "\n",
                    "**Quick Actions:**",
                    "  `!ls`              - List files (shell shortcut)",
                    "  `/analyze <file>`  - Analyze code",
                    "  `/commit-msg`      - Generate commit message",
                    "  `/clip`            - Copy last response\n",
                    "**AI Models:**",
                    "  `/switch gemini`   - Use Google Gemini",
                    "  `/switch groq`     - Use Groq (fast)",
                    "  `/switch ollama`   - Use local models\n",
                    "**Web:**",
                    "  `/search <query>`  - Search the web",
                    "  `/fetch <url>`     - Read a webpage",
                    "  `/docs <topic>`    - Search docs\n",
                    "**Sessions:**",
                    "  `/session save`    - Save conversation",
                    "  `/session list`    - List sessions",
                    "  `/export chat.md`  - Export to file\n",
                    "**Customization:**",
                    "  `/skills create`   - Create SKILLS.md",
                    "  `/rules create`    - Create RULES.md",
                    "  `/templates`       - View templates\n",
                    "**Help:**",
                    "  `/help`            - Full command list",
                    "  `/quickstart`      - Quick start guide"
                ]
                return "\n".join(lines)
            
            # --- Summarize URL ---
            if cmd.startswith("summarize-url ") or cmd.startswith("sum-url "):
                if not self.web_searcher:
                    return "❌ Web searcher not available"
                
                url = command.split(maxsplit=1)[1] if len(command.split()) > 1 else ""
                if not url or not self.web_searcher.is_url(url):
                    return "Usage: /summarize-url <url>"
                
                try:
                    console.print(f"[dim]Fetching and summarizing {url}...[/dim]")
                    doc = self.web_searcher.fetch_with_cache(url)
                    
                    # Use AI to summarize
                    summary_prompt = f"Please summarize the following web page content in 3-5 bullet points:\n\n{doc.content[:8000]}"
                    summary = self.ai.query(self.current_model, summary_prompt)
                    
                    return f"📄 **Summary of:** {doc.title}\n🔗 {url}\n\n{summary}"
                except Exception as e:
                    return f"❌ Summarization failed: {str(e)}"
            
            # --- Ask with URL Context ---
            if cmd.startswith("ask-url "):
                if not self.web_searcher:
                    return "❌ Web searcher not available"
                
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /ask-url <url> <question>"
                
                url = parts[1]
                question = parts[2]
                
                if not self.web_searcher.is_url(url):
                    return "❌ Invalid URL"
                
                try:
                    console.print(f"[dim]Fetching {url}...[/dim]")
                    doc = self.web_searcher.fetch_with_cache(url)
                    
                    # Use AI to answer with context
                    context_prompt = f"Based on the following web page content, answer this question: {question}\n\nWeb page content:\n{doc.content[:10000]}"
                    answer = self.ai.query(self.current_model, context_prompt)
                    
                    return f"📄 **Source:** {doc.title}\n❓ **Question:** {question}\n\n{answer}"
                except Exception as e:
                    return f"❌ Failed: {str(e)}"
            
            # --- Clear Context ---
            if cmd == "clear-context" or cmd == "reset-context":
                if self.context_engine:
                    self.context_engine.messages = []
                    return "✅ Conversation context cleared"
                return "❌ Context engine not available"
            
            # --- Show Context ---
            if cmd == "show-context":
                if not self.context_engine:
                    return "❌ Context engine not available"
                
                messages = self.context_engine.messages[-10:]  # Last 10 messages
                if not messages:
                    return "📭 No conversation context (start chatting!)"
                
                lines = ["📝 **Recent Conversation Context:**\n"]
                for msg in messages:
                    role = msg.get('role', 'unknown').capitalize()
                    content = msg.get('content', '')[:100]
                    if len(msg.get('content', '')) > 100:
                        content += "..."
                    lines.append(f"**{role}:** {content}")
                
                lines.append(f"\n_Total messages: {len(self.context_engine.messages)}_")
                return "\n".join(lines)

            # --- Vision Command ---
            if cmd.startswith("vision ") or cmd.startswith("see "):
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                     # Check if there is a default prompt or just image
                     if len(parts) == 2:
                         image_path = parts[1]
                         prompt = "Describe this image in detail."
                     else:
                        return "Usage: /vision <image_path> [prompt]"
                else:
                    image_path = parts[1]
                    prompt = parts[2]
                
                console.print(f"[dim]Analyzing image {image_path}...[/dim]")
                return self.ai.analyze_image(image_path, prompt)

            # ================================================================
            # ADVANCED FEATURES - CODE AGENT
            # ================================================================
            
            # --- Code Agent: Edit with AI ---
            if cmd.startswith("agent edit "):
                if not self.code_agent:
                    return "❌ Code agent not available"
                
                parts = command.split(maxsplit=3)
                if len(parts) < 4:
                    return "Usage: /agent edit <file> <instruction>"
                
                filepath = parts[2]
                instruction = parts[3]
                
                try:
                    console.print(f"[dim]Creating edit for {filepath}...[/dim]")
                    edit = self.code_agent.create_edit_from_ai(filepath, instruction, self.current_model)
                    
                    if not edit:
                        return "❌ Failed to generate edit"
                    
                    # Show diff preview
                    diff = self.code_agent.generate_diff(edit.original_content, edit.new_content, filepath)
                    return f"📝 **Proposed Edit:** {instruction}\n\n```diff\n{diff[:3000]}\n```\n\n💡 Use `/agent apply` to apply this change"
                    
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Code Agent: Analyze Project ---
            if cmd.startswith("agent analyze") or cmd == "analyze-project":
                if not self.code_agent:
                    return "❌ Code agent not available"
                
                directory = command.split(maxsplit=2)[2] if len(command.split()) > 2 else "."
                
                try:
                    console.print("[dim]Analyzing project...[/dim]")
                    analysis = self.code_agent.analyze_project(directory)
                    return self.code_agent.format_project_analysis(analysis)
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Code Agent: Find Issues ---
            if cmd.startswith("agent issues ") or cmd.startswith("find-issues "):
                if not self.code_agent:
                    return "❌ Code agent not available"
                
                filepath = command.split(maxsplit=2)[2] if len(command.split()) > 2 else ""
                if not filepath:
                    return "Usage: /agent issues <file>"
                
                try:
                    issues = self.code_agent.find_issues(filepath)
                    if not issues:
                        return f"✅ No issues found in {filepath}"
                    
                    lines = [f"🔍 **Issues in {filepath}:**\n"]
                    for issue in issues:
                        icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(issue.get('severity', ''), "⚪")
                        lines.append(f"{icon} Line {issue['line']}: [{issue['type']}] {issue['message']}")
                    
                    return "\n".join(lines)
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Code Agent: Auto-fix ---
            if cmd.startswith("agent fix ") or cmd.startswith("auto-fix "):
                if not self.code_agent:
                    return "❌ Code agent not available"
                
                filepath = command.split(maxsplit=2)[2] if len(command.split()) > 2 else ""
                if not filepath:
                    return "Usage: /agent fix <file>"
                
                try:
                    console.print(f"[dim]Finding and fixing issues in {filepath}...[/dim]")
                    edits = self.code_agent.auto_fix_file(filepath, self.current_model)
                    
                    if not edits:
                        return f"✅ No fixable issues found in {filepath}"
                    
                    edit = edits[0]
                    diff = self.code_agent.generate_diff(edit.original_content, edit.new_content, filepath)
                    return f"🔧 **Auto-fix for {filepath}:**\n\n```diff\n{diff[:3000]}\n```"
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Code Agent: Execute Code ---
            if cmd.startswith("agent run ") or cmd.startswith("run-code "):
                if not self.code_agent:
                    return "❌ Code agent not available"
                
                parts = command.split(maxsplit=2)
                if len(parts) < 3:
                    return "Usage: /agent run <language>\nThen enter your code..."
                
                language = parts[2]
                # For now just show usage - actual code would come from input
                return f"📋 Ready to execute {language} code.\nPaste your code and type `/execute` to run it in sandbox."
            
            # --- AutoPilot ---
            if cmd.startswith("autopilot ") or cmd.startswith("agent task "):
                if not self.code_agent:
                    return "❌ Code agent not available"
                
                parts = command.split(maxsplit=2)
                description = parts[2] if len(parts) > 2 else ""
                if not description:
                    return "Usage: /autopilot <task description>"
                
                try:
                    console.print(f"[bold magenta]🚀 Starting AutoPilot Task: {description}[/bold magenta]")
                    task = self.code_agent.create_task(description)
                    result = self.code_agent.execute_task(task, self.current_model)
                    return f"✅ AutoPilot Complete\n\n{result}"
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # ================================================================
            # PAIR PROGRAMMING
            # ================================================================
            
            # --- Start Pair Session ---
            if cmd.startswith("pair start ") or cmd.startswith("pair "):
                if not self.pair_programmer:
                    return "❌ Pair programmer not available"
                
                filepath = command.split(maxsplit=2)[-1]
                if not filepath or filepath in ["start", "pair"]:
                    return "Usage: /pair start <file>"
                
                try:
                    session = self.pair_programmer.start_session(filepath)
                    return f"👥 **Pair Programming Session Started**\n📄 File: {session.filepath}\n🔤 Language: {session.language}\n\n💡 Commands:\n  `/suggest` - Get code suggestions\n  `/fix <error>` - Get fix suggestions\n  `/refactor` - Refactoring suggestions\n  `/explain` - Explain code\n  `/pair end` - End session"
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Pair Status ---
            if cmd == "pair status":
                if not self.pair_programmer:
                    return "❌ Pair programmer not available"
                return self.pair_programmer.format_session_status()
            
            # --- Pair End ---
            if cmd == "pair end":
                if not self.pair_programmer:
                    return "❌ Pair programmer not available"
                summary = self.pair_programmer.end_session()
                if "error" in summary:
                    return f"❌ {summary['error']}"
                return f"👋 **Session Ended**\nSuggestions made: {summary['suggestions_made']}\nAccepted: {summary['suggestions_accepted']}"
            
            # --- Suggest ---
            if cmd == "suggest" or cmd.startswith("suggest "):
                if not self.pair_programmer:
                    return "❌ Pair programmer not available"
                
                if not self.pair_programmer.current_session:
                    return "❌ No active pair programming session. Use `/pair start <file>`"
                
                try:
                    filepath = self.pair_programmer.current_session.filepath
                    content, error = self.code_agent.read_file(filepath) if self.code_agent else ("", "No code agent")
                    
                    if error:
                        return f"❌ {error}"
                    
                    console.print("[dim]Getting suggestions...[/dim]")
                    suggestions = self.pair_programmer.suggest_completion(content, 0, self.current_model)
                    
                    if not suggestions:
                        return "💭 No suggestions at this time"
                    
                    return self.pair_programmer.format_suggestion(suggestions[0])
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Refactor ---
            if cmd == "refactor" or cmd.startswith("refactor "):
                if not self.pair_programmer:
                    return "❌ Pair programmer not available"
                
                filepath = command.split(maxsplit=1)[1] if len(command.split()) > 1 else None
                
                if not filepath and self.pair_programmer.current_session:
                    filepath = self.pair_programmer.current_session.filepath
                
                if not filepath:
                    return "Usage: /refactor <file> or start a pair session first"
                
                try:
                    content, error = self.code_agent.read_file(filepath) if self.code_agent else ("", "No code agent")
                    if error:
                        return f"❌ {error}"
                    
                    console.print("[dim]Generating refactoring suggestions...[/dim]")
                    suggestions = self.pair_programmer.suggest_refactor(content, "", self.current_model)
                    
                    if not suggestions:
                        return "✅ Code looks good, no refactoring needed"
                    
                    return self.pair_programmer.format_suggestion(suggestions[0])
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # ================================================================
            # WORKFLOW ENGINE
            # ================================================================
            
            # --- List Workflows ---
            if cmd == "workflow" or cmd == "workflows" or cmd == "workflow list":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                return self.workflow_engine.format_list()
            
            # --- Create Workflow ---
            if cmd.startswith("workflow create "):
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                
                name = command.split(maxsplit=2)[2] if len(command.split()) > 2 else "New Workflow"
                workflow = self.workflow_engine.create_workflow(name)
                return f"✅ Created workflow: {workflow.name}\nID: {workflow.workflow_id}\n\n💡 Add steps with `/workflow add-step {workflow.workflow_id} <type> <name>`"
            
            # --- Run Workflow ---
            if cmd.startswith("workflow run "):
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                
                workflow_id = command.split()[2] if len(command.split()) > 2 else ""
                if not workflow_id:
                    return "Usage: /workflow run <workflow_id>"
                
                try:
                    console.print(f"[dim]Running workflow {workflow_id}...[/dim]")
                    result = self.workflow_engine.run_workflow(workflow_id)
                    
                    if "error" in result:
                        return f"❌ {result['error']}"
                    
                    return f"✅ Workflow {result.get('status', 'completed')}\n\nResults:\n" + "\n".join([
                        f"  • {r['step']}: {r.get('status', 'done')}"
                        for r in result.get('results', [])
                    ])
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Show Workflow ---
            if cmd.startswith("workflow show "):
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                
                workflow_id = command.split()[2] if len(command.split()) > 2 else ""
                return self.workflow_engine.format_workflow(workflow_id)
            
            # --- Pre-built Workflows ---
            if cmd == "workflow code-review":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_code_review_workflow()
                return f"✅ Created code review workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"
            
            if cmd == "workflow deploy":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_deploy_workflow()
                return f"✅ Created deployment workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"
            
            if cmd == "workflow standup":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_daily_standup_workflow()
                return f"✅ Created standup workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            if cmd == "workflow test":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_test_workflow()
                return f"✅ Created test workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            if cmd == "workflow docs":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_documentation_workflow()
                return f"✅ Created documentation workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            if cmd == "workflow security":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_security_audit_workflow()
                return f"✅ Created security audit workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            if cmd == "workflow release":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_release_workflow()
                return f"✅ Created release workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            if cmd == "workflow refactor":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_refactor_workflow()
                return f"✅ Created refactor analysis workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            if cmd == "workflow deps":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_dependency_check_workflow()
                return f"✅ Created dependency check workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            if cmd == "workflow changelog":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_changelog_workflow()
                return f"✅ Created changelog workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            if cmd == "workflow health":
                if not self.workflow_engine:
                    return "❌ Workflow engine not available"
                workflow = self.workflow_engine.create_health_check_workflow()
                return f"✅ Created health check workflow: {workflow.workflow_id}\nUse `/workflow run {workflow.workflow_id}` to execute"

            
            # ================================================================
            # KNOWLEDGE BASE (Smart RAG)
            # ================================================================
            
            # --- Knowledge Base Status ---
            if cmd == "kb" or cmd == "knowledge" or cmd == "kb status":
                if not self.smart_rag:
                    return "❌ Knowledge base not available (install: pip install sentence-transformers numpy)"
                return self.smart_rag.format_stats()
            
            # --- Add to Knowledge Base ---
            if cmd.startswith("kb add "):
                if not self.smart_rag:
                    return "❌ Knowledge base not available"
                
                target = command.split(maxsplit=2)[2] if len(command.split()) > 2 else ""
                if not target:
                    return "Usage: /kb add <file or directory>"
                
                try:
                    from pathlib import Path
                    path = Path(target)
                    
                    if path.is_file():
                        doc_id = self.smart_rag.add_file(target)
                        if doc_id:
                            return f"✅ Added to knowledge base: {doc_id}"
                        return "❌ Failed to add file"
                    
                    elif path.is_dir():
                        console.print(f"[dim]Indexing directory {target}...[/dim]")
                        added = self.smart_rag.add_directory(target)
                        return f"✅ Added {len(added)} files to knowledge base"
                    
                    else:
                        return f"❌ Path not found: {target}"
                        
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Search Knowledge Base ---
            if cmd.startswith("kb search ") or cmd.startswith("kb find "):
                if not self.smart_rag:
                    return "❌ Knowledge base not available"
                
                query = command.split(maxsplit=2)[2] if len(command.split()) > 2 else ""
                if not query:
                    return "Usage: /kb search <query>"
                
                try:
                    results = self.smart_rag.search(query, top_k=5)
                    
                    if not results:
                        return f"❌ No results found for: {query}"
                    
                    lines = [f"🔍 **Knowledge Base Results for:** _{query}_\n"]
                    for i, result in enumerate(results, 1):
                        source = result.metadata.get('filename', result.doc_id)
                        snippet = result.content[:150] + "..." if len(result.content) > 150 else result.content
                        lines.append(f"**{i}. {source}** (score: {result.score:.2f})")
                        lines.append(f"   _{snippet}_")
                        lines.append("")
                    
                    return "\n".join(lines)
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- List Documents ---
            if cmd == "kb list" or cmd == "kb docs":
                if not self.smart_rag:
                    return "❌ Knowledge base not available"
                
                docs = self.smart_rag.list_documents()
                if not docs:
                    return "📚 Knowledge base is empty. Use `/kb add <file>` to add documents."
                
                lines = ["📚 **Knowledge Base Documents:**\n"]
                for doc in docs[:20]:
                    lines.append(f"  • **{doc['doc_id']}** - {doc.get('filename', 'unknown')} ({doc['chunks']} chunks)")
                
                if len(docs) > 20:
                    lines.append(f"\n_...and {len(docs) - 20} more_")
                
                return "\n".join(lines)
            
            # --- Ask with Knowledge ---
            if cmd.startswith("kb ask ") or cmd.startswith("ask-kb "):
                if not self.smart_rag:
                    return "❌ Knowledge base not available"
                
                query = command.split(maxsplit=2)[2] if len(command.split()) > 2 else ""
                if not query:
                    return "Usage: /kb ask <question>"
                
                try:
                    # Get relevant context
                    context = self.smart_rag.get_context_for_query(query)
                    
                    if not context:
                        return "❌ No relevant context found in knowledge base"
                    
                    # Query AI with context
                    prompt = f"""Answer this question using ONLY the context provided below.
If the context doesn't contain the answer, say so.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
                    
                    console.print("[dim]Querying with knowledge context...[/dim]")
                    answer = self.ai.query(self.current_model, prompt)
                    
                    return f"🧠 **Answer from Knowledge Base:**\n\n{answer}"
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            # --- Clear Knowledge Base ---
            if cmd == "kb clear":
                if not self.smart_rag:
                    return "❌ Knowledge base not available"
                self.smart_rag.clear()
                return "✅ Knowledge base cleared"

            if cmd.startswith("cloud "):
                if not self.cloud_integration:
                    return "❌ Cloud Integration module not available"
                parts = command.split(maxsplit=2)
                action = parts[1] if len(parts) > 1 else "status"
                if action == "status":
                    return self.cloud_integration.get_status()
                if action == "connect" and len(parts) == 3:
                    # Usage: /cloud connect aws
                    return self.cloud_integration.connect(parts[2], "credentials.json")
                if action == "deploy" and len(parts) == 3:
                    return self.cloud_integration.deploy("app", parts[2])
                return "Usage: /cloud [status|connect <provider>|deploy <path>]"

            # ================================================================
            # MISSION CONTROL & TOOLS
            # ================================================================

            if cmd == "dashboard" or cmd == "mission-control":
                try:
                    import subprocess
                    import sys
                    import os
                    
                    # Get absolute path to dashboard script
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    dashboard_script = os.path.join(current_dir, "dashboard_tui.py")
                    
                    subprocess.Popen([sys.executable, dashboard_script], shell=True)
                    return "✅ Mission Control Dashboard launched in separate window!"
                except Exception as e:
                    return f"❌ Failed to launch dashboard: {e}"

            if cmd.startswith("web "):
                if not self.web_agent:
                    return "❌ Web Agent not available"
                parts = command.split(maxsplit=2)
                if len(parts) < 2: return "Usage: /web [visit <url>|research <topic>]"
                action = parts[1]
                if action == "visit" and len(parts) > 2:
                    return self.web_agent.visit(parts[2])
                if action == "research" and len(parts) > 2:
                    return self.web_agent.research(parts[2])
                return "Usage: /web [visit <url>|research <topic>]"

            if cmd.startswith("docker "):
                if not self.docker_manager:
                    return "❌ Docker Manager not available"
                parts = command.split(maxsplit=2)
                if len(parts) < 2: return "Usage: /docker [info|list|config <type>|start <id>|stop <id>]"
                action = parts[1]
                if action == "info":
                    return self.docker_manager.get_info()
                if action == "list":
                    containers = self.docker_manager.list_containers()
                    return f"🐳 Containers:\n{containers}"
                if action == "config" and len(parts) > 2:
                    return self.docker_manager.generate_config(parts[2])
                if action == "start" and len(parts) > 2:
                    return self.docker_manager.start_container(parts[2])
                if action == "stop" and len(parts) > 2:
                    return self.docker_manager.stop_container(parts[2])
                return "Usage: /docker [info|list|config <type>]"

            if cmd.startswith("persona"):
                if not self.persona_manager:
                     return "❌ Persona Manager not available"
                parts = command.split()
                if len(parts) == 1 or parts[1] == "list":
                    return self.persona_manager.list_personas()
                if parts[1] == "switch" and len(parts) > 2:
                    return self.persona_manager.set_persona(parts[2])
                if parts[1] == "active":
                    return f"🎭 Current persona: {self.persona_manager.current_persona or 'Default'}"
                return "Usage: /persona [list|switch <name>|active]"

            # --- Blockchain ---
            if cmd.startswith("web3 "):
                if not self.blockchain:
                    return "❌ Blockchain module not available"
                parts = command.split(maxsplit=2)
                action = parts[1] if len(parts) > 1 else "balance"
                if action == "create-wallet" and len(parts) == 3:
                    return self.blockchain.create_wallet(parts[2])
                if action == "balance" and len(parts) == 3:
                    return self.blockchain.get_balance(parts[2])
                if action == "deploy" and len(parts) == 3:
                    return self.blockchain.deploy_contract(parts[2])
                return "Usage: /web3 [create-wallet <name>|balance <addr>|deploy <file>]"

            # --- ML Ops ---
            if cmd.startswith("ml "):
                if not self.ml_ops:
                    return "❌ ML Ops module not available"
                parts = command.split(maxsplit=2)
                action = parts[1] if len(parts) > 1 else "list"
                if action == "list":
                    return self.ml_ops.list_models()
                if action == "train" and len(parts) == 3:
                    return self.ml_ops.train_model("new_model", parts[2])
                if action == "evaluate" and len(parts) == 3:
                    return self.ml_ops.evaluate_model(parts[2])
                return "Usage: /ml [list|train <data>|evaluate <model>]"

        except Exception as e:
            logging.error(f"Command handling error: {str(e)}")
            return " Command processing error"
# --- Main Loop ---
# --- Main Loop ---
def run_cli_mode() -> int:
    """Run in headless CLI mode for VS Code extension."""
    try:
        aether = AetherAI(quiet=True)
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                response = aether.process_input(line)
                print(response)
                sys.stdout.flush()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                sys.stdout.flush()
        return 0
    except Exception as e:
        print(f"Fatal CLI error: {e}")
        return 1

def run_interactive_mode() -> int:
    """Run interactive mode with prompt_toolkit"""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.styles import Style
        
        aether = AetherAI()
        
        # Setup history
        history_file = os.path.join(os.path.expanduser("~"), ".aether", "history.txt")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
        )
        
        style = Style.from_dict({
            'prompt': '#00aa00 bold',
        })
        
        while True:
            try:
                user_input = session.prompt(
                    [('class:prompt', f"[{aether.current_model.upper()}] > ")],
                    style=style
                )
                
                if not user_input.strip():
                    continue
                    
                if user_input.lower() in ('exit', 'quit', '/exit', '/quit'):
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                response = aether.process_input(user_input)
                console.print(response)
                
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                
        return 0
    except ImportError:
        print("prompt_toolkit not installed. Please run: pip install prompt_toolkit")
        return 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

def main() -> int:
    """Start the interactive AetherAI terminal.
    
    This is the main entry point for the CLI.
    Returns exit code (0 for success, non-zero for errors).
    """
    import argparse
    
    # ASCII Art for --version
    VERSION_BANNER = r"""
     _    _____ _____ _   _ _____ ____      _    ___ 
    / \  | ____|_   _| | | | ____|  _ \    / \  |_ _|
   / _ \ |  _|   | | | |_| |  _| | |_) |  / _ \  | | 
  / ___ \| |___  | | |  _  | |___|  _ <  / ___ \ | | 
 /_/   \_\_____| |_| |_| |_|_____|_| \_\/_/   \_\___|
"""
    
    parser = argparse.ArgumentParser(
        prog='aetherai',
        description='AetherAI - Production-ready, secure, multi-model AI terminal assistant',
        epilog='For more info: https://github.com/KunjShah95/NEXUS-AI.io',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-v', '--version',
        action='store_true',
        help='Show version information and exit'
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in headless CLI mode (for VS Code extension)'
    )
    
    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='Skip the startup banner'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['gemini', 'groq', 'ollama', 'huggingface', 'chatgpt', 'mcp'],
        help='Start with a specific AI model'
    )
    
    args = parser.parse_args()
    
    # Handle version flag
    if args.version:
        console.print(f"[bold cyan]{VERSION_BANNER}[/bold cyan]")
        console.print(f"[bold white]AetherAI[/bold white] v[bold green]{VERSION}[/bold green]")
        console.print(f"[dim]Author: Kunj Shah[/dim]")
        console.print(f"[dim]Python: {sys.version.split()[0]} | Platform: {sys.platform}[/dim]")
        console.print(f"[dim]Install: pip install aetherai[/dim]")
        return 0
    
    # Run in CLI mode (headless for VS Code extension)
    if args.cli:
        return run_cli_mode()
    
    # Run interactive mode
    return run_interactive_mode()


if __name__ == "__main__":
    sys.exit(main())
