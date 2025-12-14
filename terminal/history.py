import json
import os
from datetime import datetime
from typing import List, Dict, Any

class HistoryManager:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.expanduser("~"), ".aetherai", "history")
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_session(self, username: str, messages: List[Dict[str, str]]) -> str:
        """Save the current chat session to a JSON file."""
        if not messages:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_{timestamp}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        data = {
            "username": username,
            "timestamp": timestamp,
            "messages": messages
        }
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            print(f"Error saving history: {e}")
            return None

    def list_sessions(self, username: str) -> List[str]:
        """List available session files for a user."""
        try:
            files = [f for f in os.listdir(self.base_dir) if f.startswith(username) and f.endswith(".json")]
            return sorted(files, reverse=True)
        except Exception:
            return []

    def load_session(self, filename: str) -> Dict[str, Any]:
        """Load a specific session."""
        filepath = os.path.join(self.base_dir, filename)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
