"""
Context Engine for AetherAI - Manages conversation context, sessions, and memory.

This module provides:
- Chat session management (save/load/list)
- Conversation history with context windows
- Project context detection
- Smart context compression for long conversations
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class ContextEngine:
    """Manages conversation context and sessions for AetherAI."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the context engine.
        
        Args:
            base_dir: Base directory for storing sessions. Defaults to ~/.nexus/
        """
        self.base_dir = base_dir or os.path.join(
            os.getenv('HOME') or os.getenv('USERPROFILE') or os.path.expanduser('~'),
            '.nexus'
        )
        self.sessions_dir = os.path.join(self.base_dir, 'sessions')
        self.favorites_file = os.path.join(self.base_dir, 'favorites.json')
        self.templates_dir = os.path.join(self.base_dir, 'templates')
        
        # Create directories
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Current session state
        self.current_session: Optional[str] = None
        self.messages: List[Dict[str, Any]] = []
        self.last_response: str = ""
        self.context_window_size: int = 10  # Number of messages to keep in context
        self.project_context: Dict[str, Any] = {}
        
        # Load favorites
        self.favorites: List[Dict[str, Any]] = self._load_favorites()
        
    def _load_favorites(self) -> List[Dict[str, Any]]:
        """Load favorites from file."""
        try:
            if os.path.exists(self.favorites_file):
                with open(self.favorites_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def _save_favorites(self):
        """Save favorites to file."""
        try:
            with open(self.favorites_file, 'w', encoding='utf-8') as f:
                json.dump(self.favorites, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving favorites: {e}")
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def new_session(self, name: Optional[str] = None) -> str:
        """Create a new chat session.
        
        Args:
            name: Optional session name. Auto-generated if not provided.
            
        Returns:
            Session ID.
        """
        session_id = name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = session_id
        self.messages = []
        self.last_response = ""
        return session_id
    
    def save_session(self, name: Optional[str] = None) -> str:
        """Save current session to disk.
        
        Args:
            name: Optional name override.
            
        Returns:
            Session filename.
        """
        session_name = name or self.current_session or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = os.path.join(self.sessions_dir, f"{session_name}.json")
        
        session_data = {
            "id": session_name,
            "created_at": datetime.now().isoformat(),
            "messages": self.messages,
            "project_context": self.project_context,
            "message_count": len(self.messages)
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
            self.current_session = session_name
            return session_name
        except Exception as e:
            raise RuntimeError(f"Failed to save session: {e}")
    
    def load_session(self, name: str) -> Dict[str, Any]:
        """Load a session from disk.
        
        Args:
            name: Session name to load.
            
        Returns:
            Session data dictionary.
        """
        filename = os.path.join(self.sessions_dir, f"{name}.json")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Session '{name}' not found")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_session = name
            self.messages = data.get('messages', [])
            self.project_context = data.get('project_context', {})
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load session: {e}")
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions.
        
        Returns:
            List of session info dictionaries.
        """
        sessions = []
        try:
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.sessions_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        sessions.append({
                            "name": filename[:-5],  # Remove .json
                            "created_at": data.get('created_at', 'Unknown'),
                            "message_count": data.get('message_count', len(data.get('messages', []))),
                            "size": os.path.getsize(filepath)
                        })
                    except Exception:
                        sessions.append({
                            "name": filename[:-5],
                            "created_at": "Unknown",
                            "message_count": 0,
                            "size": os.path.getsize(filepath)
                        })
        except Exception:
            pass
        
        return sorted(sessions, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def delete_session(self, name: str) -> bool:
        """Delete a session.
        
        Args:
            name: Session name to delete.
            
        Returns:
            True if deleted successfully.
        """
        filename = os.path.join(self.sessions_dir, f"{name}.json")
        try:
            if os.path.exists(filename):
                os.remove(filename)
                if self.current_session == name:
                    self.current_session = None
                    self.messages = []
                return True
        except Exception:
            pass
        return False
    
    # =========================================================================
    # Message Management
    # =========================================================================
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the current session.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (model, timestamp, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        if role == "assistant":
            self.last_response = content
    
    def get_context_messages(self, max_messages: Optional[int] = None) -> List[Dict]:
        """Get recent messages for context.
        
        Args:
            max_messages: Maximum messages to return. Uses context_window_size if not specified.
            
        Returns:
            List of recent messages.
        """
        limit = max_messages or self.context_window_size
        return self.messages[-limit:] if self.messages else []
    
    def get_context_string(self, max_messages: Optional[int] = None) -> str:
        """Get context as a formatted string for LLM input.
        
        Args:
            max_messages: Maximum messages to include.
            
        Returns:
            Formatted context string.
        """
        messages = self.get_context_messages(max_messages)
        if not messages:
            return ""
        
        context_parts = []
        for msg in messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n\n".join(context_parts)
    
    def clear_context(self):
        """Clear current session messages."""
        self.messages = []
        self.last_response = ""
    
    # =========================================================================
    # Favorites Management
    # =========================================================================
    
    def save_favorite(self, name: Optional[str] = None, query: Optional[str] = None) -> str:
        """Save the last response as a favorite.
        
        Args:
            name: Optional name for the favorite.
            query: Optional query that generated the response.
            
        Returns:
            Favorite ID.
        """
        if not self.last_response:
            raise ValueError("No response to save")
        
        fav_id = name or f"fav_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get last user message as query if not provided
        if not query and self.messages:
            for msg in reversed(self.messages):
                if msg['role'] == 'user':
                    query = msg['content']
                    break
        
        favorite = {
            "id": fav_id,
            "query": query or "",
            "response": self.last_response,
            "created_at": datetime.now().isoformat(),
            "tags": []
        }
        
        self.favorites.append(favorite)
        self._save_favorites()
        return fav_id
    
    def list_favorites(self) -> List[Dict[str, Any]]:
        """List all favorites.
        
        Returns:
            List of favorite dictionaries.
        """
        return self.favorites
    
    def get_favorite(self, name_or_index: str) -> Optional[Dict[str, Any]]:
        """Get a favorite by name or index.
        
        Args:
            name_or_index: Favorite name or numeric index.
            
        Returns:
            Favorite dictionary or None.
        """
        # Try as index first
        try:
            idx = int(name_or_index) - 1
            if 0 <= idx < len(self.favorites):
                return self.favorites[idx]
        except ValueError:
            pass
        
        # Try as name
        for fav in self.favorites:
            if fav.get('id') == name_or_index:
                return fav
        
        return None
    
    def delete_favorite(self, name_or_index: str) -> bool:
        """Delete a favorite.
        
        Args:
            name_or_index: Favorite name or numeric index.
            
        Returns:
            True if deleted.
        """
        # Try as index first
        try:
            idx = int(name_or_index) - 1
            if 0 <= idx < len(self.favorites):
                self.favorites.pop(idx)
                self._save_favorites()
                return True
        except ValueError:
            pass
        
        # Try as name
        for i, fav in enumerate(self.favorites):
            if fav.get('id') == name_or_index:
                self.favorites.pop(i)
                self._save_favorites()
                return True
        
        return False
    
    # =========================================================================
    # Templates Management
    # =========================================================================
    
    def save_template(self, name: str, prompt: str, description: str = "") -> str:
        """Save a prompt template.
        
        Args:
            name: Template name.
            prompt: Template prompt (can include {placeholders}).
            description: Optional description.
            
        Returns:
            Template filename.
        """
        template = {
            "name": name,
            "prompt": prompt,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "variables": self._extract_variables(prompt)
        }
        
        filename = os.path.join(self.templates_dir, f"{name}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)
        
        return name
    
    def _extract_variables(self, prompt: str) -> List[str]:
        """Extract {variable} placeholders from a prompt."""
        import re
        return list(set(re.findall(r'\{(\w+)\}', prompt)))
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name.
        
        Args:
            name: Template name.
            
        Returns:
            Template dictionary or None.
        """
        filename = os.path.join(self.templates_dir, f"{name}.json")
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return None
    
    def apply_template(self, name: str, **kwargs) -> str:
        """Apply a template with variables.
        
        Args:
            name: Template name.
            **kwargs: Variable values.
            
        Returns:
            Filled template string.
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        prompt = template['prompt']
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return prompt
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates.
        
        Returns:
            List of template info dictionaries.
        """
        templates = []
        try:
            for filename in os.listdir(self.templates_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(self.templates_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        templates.append({
                            "name": data.get('name', filename[:-5]),
                            "description": data.get('description', ''),
                            "variables": data.get('variables', [])
                        })
                    except Exception:
                        templates.append({"name": filename[:-5], "description": "", "variables": []})
        except Exception:
            pass
        return templates
    
    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        filename = os.path.join(self.templates_dir, f"{name}.json")
        try:
            if os.path.exists(filename):
                os.remove(filename)
                return True
        except Exception:
            pass
        return False
    
    # =========================================================================
    # Project Context Detection
    # =========================================================================
    
    def detect_project_context(self, directory: str = ".") -> Dict[str, Any]:
        """Detect project type and gather context.
        
        Args:
            directory: Directory to analyze.
            
        Returns:
            Project context dictionary.
        """
        context = {
            "type": "unknown",
            "name": os.path.basename(os.path.abspath(directory)),
            "languages": [],
            "frameworks": [],
            "files": {},
            "detected_at": datetime.now().isoformat()
        }
        
        dir_path = Path(directory)
        
        # Detect project type
        if (dir_path / "package.json").exists():
            context["type"] = "nodejs"
            context["languages"].append("javascript")
            try:
                with open(dir_path / "package.json", 'r') as f:
                    pkg = json.load(f)
                    context["name"] = pkg.get("name", context["name"])
                    context["files"]["package.json"] = pkg
                    # Detect frameworks
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                    if "react" in deps:
                        context["frameworks"].append("react")
                    if "next" in deps:
                        context["frameworks"].append("nextjs")
                    if "vue" in deps:
                        context["frameworks"].append("vue")
                    if "express" in deps:
                        context["frameworks"].append("express")
                    if "typescript" in deps:
                        context["languages"].append("typescript")
            except Exception:
                pass
        
        elif (dir_path / "pyproject.toml").exists() or (dir_path / "setup.py").exists():
            context["type"] = "python"
            context["languages"].append("python")
            if (dir_path / "pyproject.toml").exists():
                try:
                    import tomllib
                    with open(dir_path / "pyproject.toml", 'rb') as f:
                        pyproject = tomllib.load(f)
                    context["files"]["pyproject.toml"] = pyproject
                    if "project" in pyproject:
                        context["name"] = pyproject["project"].get("name", context["name"])
                except Exception:
                    pass
            # Detect Python frameworks
            req_files = ["requirements.txt", "Pipfile", "poetry.lock"]
            for req_file in req_files:
                if (dir_path / req_file).exists():
                    try:
                        content = (dir_path / req_file).read_text()
                        if "django" in content.lower():
                            context["frameworks"].append("django")
                        if "flask" in content.lower():
                            context["frameworks"].append("flask")
                        if "fastapi" in content.lower():
                            context["frameworks"].append("fastapi")
                    except Exception:
                        pass
        
        elif (dir_path / "Cargo.toml").exists():
            context["type"] = "rust"
            context["languages"].append("rust")
        
        elif (dir_path / "go.mod").exists():
            context["type"] = "go"
            context["languages"].append("go")
        
        elif (dir_path / "pom.xml").exists() or (dir_path / "build.gradle").exists():
            context["type"] = "java"
            context["languages"].append("java")
        
        # Check for README
        readme_files = ["README.md", "readme.md", "README.txt", "README"]
        for readme in readme_files:
            if (dir_path / readme).exists():
                try:
                    content = (dir_path / readme).read_text(encoding='utf-8')[:2000]
                    context["files"]["README"] = content
                    break
                except Exception:
                    pass
        
        # Check for .git
        if (dir_path / ".git").exists():
            context["vcs"] = "git"
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    context["repository"] = result.stdout.strip()
            except Exception:
                pass
        
        self.project_context = context
        return context
    
    # =========================================================================
    # Export Functions
    # =========================================================================
    
    def export_session(self, format: str = "md") -> str:
        """Export current session to a file.
        
        Args:
            format: Export format ('md', 'json', 'html', 'txt')
            
        Returns:
            Exported content string.
        """
        if format == "json":
            return json.dumps({
                "session": self.current_session,
                "messages": self.messages,
                "exported_at": datetime.now().isoformat()
            }, indent=2, default=str)
        
        elif format == "html":
            html_parts = [
                "<!DOCTYPE html>",
                "<html><head><title>AetherAI Session</title>",
                "<style>body{font-family:system-ui;max-width:800px;margin:0 auto;padding:20px}",
                ".user{background:#e3f2fd;padding:10px;border-radius:8px;margin:10px 0}",
                ".assistant{background:#f5f5f5;padding:10px;border-radius:8px;margin:10px 0}",
                "pre{background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:4px;overflow-x:auto}",
                "</style></head><body>",
                f"<h1>AetherAI Session: {self.current_session or 'Unnamed'}</h1>",
            ]
            
            for msg in self.messages:
                css_class = msg['role']
                content = msg['content'].replace('\n', '<br>')
                html_parts.append(f'<div class="{css_class}"><strong>{msg["role"].title()}:</strong><br>{content}</div>')
            
            html_parts.append("</body></html>")
            return "\n".join(html_parts)
        
        elif format == "txt":
            lines = [f"AetherAI Session: {self.current_session or 'Unnamed'}", "=" * 50, ""]
            for msg in self.messages:
                lines.append(f"[{msg['role'].upper()}]")
                lines.append(msg['content'])
                lines.append("")
            return "\n".join(lines)
        
        else:  # Markdown (default)
            lines = [f"# AetherAI Session: {self.current_session or 'Unnamed'}", ""]
            for msg in self.messages:
                if msg['role'] == 'user':
                    lines.append(f"## 🧑 User")
                else:
                    lines.append(f"## 🤖 Assistant")
                lines.append("")
                lines.append(msg['content'])
                lines.append("")
            return "\n".join(lines)
    
    def export_to_file(self, filename: str, format: Optional[str] = None) -> str:
        """Export session to a file.
        
        Args:
            filename: Output filename.
            format: Format override.
            
        Returns:
            Filename written.
        """
        if not format:
            ext = os.path.splitext(filename)[1].lower()
            format = {"json": "json", ".html": "html", ".txt": "txt", ".md": "md"}.get(ext, "md")
        
        content = self.export_session(format)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filename


# Create default templates
def create_default_templates(context_engine: ContextEngine):
    """Create default prompt templates."""
    defaults = [
        {
            "name": "code-review",
            "prompt": "Please review the following code and provide feedback on:\n1. Code quality and style\n2. Potential bugs or issues\n3. Performance considerations\n4. Security concerns\n5. Suggestions for improvement\n\nCode:\n```{language}\n{code}\n```",
            "description": "Comprehensive code review template"
        },
        {
            "name": "explain-code",
            "prompt": "Please explain the following code in detail:\n- What does it do?\n- How does it work?\n- What are the key concepts used?\n\n```{language}\n{code}\n```",
            "description": "Explain code with detailed breakdown"
        },
        {
            "name": "refactor",
            "prompt": "Please refactor the following code to improve:\n- Readability\n- Performance\n- Maintainability\n\nProvide the refactored code with explanations.\n\n```{language}\n{code}\n```",
            "description": "Refactor code for improvements"
        },
        {
            "name": "generate-tests",
            "prompt": "Generate comprehensive unit tests for the following code using {test_framework}:\n\n```{language}\n{code}\n```\n\nInclude:\n- Happy path tests\n- Edge cases\n- Error handling tests",
            "description": "Generate unit tests for code"
        },
        {
            "name": "document",
            "prompt": "Generate comprehensive documentation for the following code:\n\n```{language}\n{code}\n```\n\nInclude:\n- Function/class docstrings\n- Parameter descriptions\n- Return value descriptions\n- Usage examples",
            "description": "Generate documentation for code"
        },
        {
            "name": "fix-error",
            "prompt": "I'm getting this error:\n\n```\n{error}\n```\n\nIn this code:\n\n```{language}\n{code}\n```\n\nPlease:\n1. Explain what's causing the error\n2. Provide a fix\n3. Explain how to prevent this in the future",
            "description": "Debug and fix code errors"
        },
        {
            "name": "commit-msg",
            "prompt": "Generate a conventional commit message for the following changes:\n\n{changes}\n\nFollow the format: type(scope): description\n\nTypes: feat, fix, docs, style, refactor, test, chore",
            "description": "Generate git commit messages"
        },
        {
            "name": "pr-description",
            "prompt": "Generate a pull request description for the following changes:\n\n{changes}\n\nInclude:\n- Summary of changes\n- Motivation/reason\n- Testing done\n- Checklist",
            "description": "Generate PR descriptions"
        }
    ]
    
    for template in defaults:
        if not context_engine.get_template(template["name"]):
            context_engine.save_template(
                template["name"],
                template["prompt"],
                template["description"]
            )
