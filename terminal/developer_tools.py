"""
Developer Tools for AetherAI - File analysis, code generation, and productivity tools.

This module provides:
- File analysis (/analyze <file>)
- Commit message generator (/commit-msg)
- PR description generator (/pr-desc)
- Clipboard integration (/clip)
- Shell command shortcuts (!)
- Multi-model comparison (/compare)
"""

import os
import subprocess
import sys
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

# Try to import clipboard module
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False


class DeveloperTools:
    """Developer productivity tools for AetherAI."""
    
    # Supported file extensions for analysis
    SUPPORTED_EXTENSIONS = {
        # Programming languages
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.jsx': 'jsx', '.tsx': 'tsx', '.java': 'java', '.kt': 'kotlin',
        '.go': 'go', '.rs': 'rust', '.c': 'c', '.cpp': 'cpp', '.h': 'c',
        '.cs': 'csharp', '.rb': 'ruby', '.php': 'php', '.swift': 'swift',
        '.scala': 'scala', '.r': 'r', '.sql': 'sql', '.sh': 'bash',
        '.ps1': 'powershell', '.lua': 'lua', '.dart': 'dart',
        # Config/Data
        '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml',
        '.xml': 'xml', '.ini': 'ini', '.env': 'dotenv', '.cfg': 'ini',
        # Web
        '.html': 'html', '.css': 'css', '.scss': 'scss', '.less': 'less',
        '.vue': 'vue', '.svelte': 'svelte',
        # Documentation
        '.md': 'markdown', '.rst': 'rst', '.txt': 'text',
        # Other
        '.dockerfile': 'dockerfile', '.makefile': 'makefile',
    }
    
    # Shell commands allowed for ! shortcut
    SAFE_SHELL_COMMANDS = {
        'ls', 'dir', 'pwd', 'cd', 'cat', 'head', 'tail', 'grep', 'find',
        'echo', 'date', 'whoami', 'hostname', 'uptime', 'df', 'du',
        'wc', 'sort', 'uniq', 'which', 'where', 'type', 'file',
        'git', 'npm', 'yarn', 'pnpm', 'pip', 'python', 'node',
        'docker', 'kubectl', 'curl', 'wget', 'tree',
    }
    
    def __init__(self, ai_query_func=None):
        """Initialize developer tools.
        
        Args:
            ai_query_func: Function to call for AI queries (model, prompt) -> response
        """
        self.ai_query = ai_query_func
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.comparison_results: List[Dict[str, Any]] = []
    
    # =========================================================================
    # File Analysis
    # =========================================================================
    
    def analyze_file(self, filepath: str, ai_model: str = "gemini") -> Dict[str, Any]:
        """Analyze a file and provide AI-powered insights.
        
        Args:
            filepath: Path to the file to analyze.
            ai_model: AI model to use for analysis.
            
        Returns:
            Analysis results dictionary.
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {filepath}")
        
        # Get file info
        stat = path.stat()
        ext = path.suffix.lower()
        language = self.SUPPORTED_EXTENSIONS.get(ext, 'text')
        
        # Read file content
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding='latin-1')
            except Exception:
                return {
                    "error": "Could not read file (binary or unsupported encoding)",
                    "filepath": str(path),
                    "size": stat.st_size
                }
        
        # Truncate large files
        max_chars = 15000
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars] + "\n\n... [TRUNCATED - file too large]"
        
        # Build analysis
        result = {
            "filepath": str(path.absolute()),
            "filename": path.name,
            "extension": ext,
            "language": language,
            "size_bytes": stat.st_size,
            "size_human": self._format_size(stat.st_size),
            "lines": content.count('\n') + 1,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "truncated": truncated,
            "content_preview": content[:500] + "..." if len(content) > 500 else content
        }
        
        # Get AI analysis if available
        if self.ai_query:
            prompt = f"""Analyze the following {language} file and provide:

1. **Summary**: What does this file do?
2. **Key Components**: Main functions/classes/modules
3. **Dependencies**: External imports/dependencies
4. **Code Quality**: Rating (1-10) with brief explanation
5. **Potential Issues**: Any bugs, security concerns, or improvements
6. **Suggestions**: Top 3 recommendations

File: {path.name}
```{language}
{content}
```"""
            
            try:
                ai_response = self.ai_query(ai_model, prompt)
                result["ai_analysis"] = ai_response
            except Exception as e:
                result["ai_analysis"] = f"AI analysis failed: {str(e)}"
        
        self.last_analysis = result
        return result
    
    def format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results for display.
        
        Args:
            analysis: Analysis dictionary.
            
        Returns:
            Formatted string.
        """
        if "error" in analysis:
            return f"❌ Error: {analysis['error']}"
        
        lines = [
            f"📄 **File Analysis: {analysis['filename']}**",
            "",
            f"📁 Path: `{analysis['filepath']}`",
            f"📏 Size: {analysis['size_human']} ({analysis['lines']} lines)",
            f"🔤 Language: {analysis['language']}",
            f"📅 Modified: {analysis['modified']}",
        ]
        
        if analysis.get('truncated'):
            lines.append("⚠️ File was truncated for analysis")
        
        if analysis.get('ai_analysis'):
            lines.extend(["", "---", "", analysis['ai_analysis']])
        
        return "\n".join(lines)
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    # =========================================================================
    # Git Integration
    # =========================================================================
    
    def get_git_diff(self, staged: bool = False) -> Tuple[str, str]:
        """Get git diff output.
        
        Args:
            staged: If True, get staged changes only.
            
        Returns:
            Tuple of (diff_output, error_message)
        """
        cmd = ["git", "diff", "--staged"] if staged else ["git", "diff"]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return "", result.stderr or "Git command failed"
            
            return result.stdout, ""
        except FileNotFoundError:
            return "", "Git not found. Please install git."
        except subprocess.TimeoutExpired:
            return "", "Git command timed out"
        except Exception as e:
            return "", str(e)
    
    def get_staged_files(self) -> List[str]:
        """Get list of staged files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--staged", "--name-only"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except Exception:
            pass
        return []
    
    def generate_commit_message(self, ai_model: str = "gemini") -> str:
        """Generate a commit message based on staged changes.
        
        Args:
            ai_model: AI model to use.
            
        Returns:
            Generated commit message.
        """
        diff, error = self.get_git_diff(staged=True)
        
        if error:
            return f"❌ Error: {error}"
        
        if not diff.strip():
            # Try unstaged diff
            diff, error = self.get_git_diff(staged=False)
            if not diff.strip():
                return "❌ No changes detected. Stage your changes with `git add` first."
        
        staged_files = self.get_staged_files()
        
        if not self.ai_query:
            return "❌ AI query function not available"
        
        # Truncate diff if too large
        max_diff_size = 8000
        if len(diff) > max_diff_size:
            diff = diff[:max_diff_size] + "\n\n... [DIFF TRUNCATED]"
        
        prompt = f"""Based on the following git diff, generate a conventional commit message.

Rules:
1. Use conventional commit format: type(scope): description
2. Types: feat, fix, docs, style, refactor, perf, test, chore, build, ci
3. Keep the subject line under 72 characters
4. Use imperative mood ("add" not "added")
5. Be specific but concise

Staged files: {', '.join(staged_files) if staged_files else 'Various files'}

Git Diff:
```diff
{diff}
```

Respond with ONLY the commit message, no explanation. If there are multiple logical changes, suggest the primary commit message first, then list alternatives."""
        
        try:
            response = self.ai_query(ai_model, prompt)
            return f"📝 **Suggested Commit Message:**\n\n{response}"
        except Exception as e:
            return f"❌ Failed to generate commit message: {str(e)}"
    
    def generate_pr_description(self, base_branch: str = "main", ai_model: str = "gemini") -> str:
        """Generate a pull request description.
        
        Args:
            base_branch: Base branch to compare against.
            ai_model: AI model to use.
            
        Returns:
            Generated PR description.
        """
        try:
            # Get commit log
            result = subprocess.run(
                ["git", "log", f"{base_branch}..HEAD", "--oneline"],
                capture_output=True,
                text=True,
                timeout=10
            )
            commits = result.stdout.strip() if result.returncode == 0 else ""
            
            # Get diff stats
            result = subprocess.run(
                ["git", "diff", f"{base_branch}..HEAD", "--stat"],
                capture_output=True,
                text=True,
                timeout=10
            )
            stats = result.stdout.strip() if result.returncode == 0 else ""
            
            # Get diff (truncated)
            result = subprocess.run(
                ["git", "diff", f"{base_branch}..HEAD"],
                capture_output=True,
                text=True,
                timeout=30
            )
            diff = result.stdout[:5000] if result.returncode == 0 else ""
            
        except Exception as e:
            return f"❌ Error getting git info: {str(e)}"
        
        if not commits:
            return "❌ No commits found. Make sure you have commits ahead of the base branch."
        
        if not self.ai_query:
            return "❌ AI query function not available"
        
        prompt = f"""Generate a comprehensive pull request description based on these changes.

Commits:
{commits}

Changed Files Summary:
{stats}

Sample Diff:
```diff
{diff}
```

Create a PR description with:
1. **Title** - A clear, descriptive PR title
2. **Summary** - What this PR does (2-3 sentences)
3. **Changes** - Bullet points of key changes
4. **Testing** - How to test these changes
5. **Checklist** - Standard PR checklist items

Format as Markdown."""
        
        try:
            response = self.ai_query(ai_model, prompt)
            return f"📋 **Pull Request Description:**\n\n{response}"
        except Exception as e:
            return f"❌ Failed to generate PR description: {str(e)}"
    
    # =========================================================================
    # Clipboard Integration
    # =========================================================================
    
    def copy_to_clipboard(self, text: str) -> Tuple[bool, str]:
        """Copy text to system clipboard.
        
        Args:
            text: Text to copy.
            
        Returns:
            Tuple of (success, message)
        """
        if not CLIPBOARD_AVAILABLE:
            # Try fallback methods
            return self._clipboard_fallback(text, copy=True)
        
        try:
            pyperclip.copy(text)
            char_count = len(text)
            line_count = text.count('\n') + 1
            return True, f"✅ Copied to clipboard ({char_count} chars, {line_count} lines)"
        except Exception as e:
            return False, f"❌ Failed to copy: {str(e)}"
    
    def paste_from_clipboard(self) -> Tuple[str, str]:
        """Get text from system clipboard.
        
        Returns:
            Tuple of (content, error_message)
        """
        if not CLIPBOARD_AVAILABLE:
            content, msg = self._clipboard_fallback("", copy=False)
            return content, msg if not content else ""
        
        try:
            content = pyperclip.paste()
            return content, ""
        except Exception as e:
            return "", f"Failed to paste: {str(e)}"
    
    def _clipboard_fallback(self, text: str, copy: bool) -> Tuple[Any, str]:
        """Fallback clipboard methods for different platforms."""
        import platform
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                if copy:
                    subprocess.run(["pbcopy"], input=text.encode(), check=True)
                    return True, "✅ Copied to clipboard (using pbcopy)"
                else:
                    result = subprocess.run(["pbpaste"], capture_output=True, check=True)
                    return result.stdout.decode(), ""
                    
            elif system == "Linux":
                # Try xclip first, then xsel
                for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard"]]:
                    try:
                        if copy:
                            subprocess.run(cmd + (["-i"] if "xclip" in cmd[0] else ["-i"]),
                                         input=text.encode(), check=True)
                            return True, f"✅ Copied to clipboard (using {cmd[0]})"
                        else:
                            result = subprocess.run(cmd + (["-o"] if "xclip" in cmd[0] else ["-o"]),
                                                   capture_output=True, check=True)
                            return result.stdout.decode(), ""
                    except FileNotFoundError:
                        continue
                return False, "❌ No clipboard tool found. Install xclip or xsel."
                
            elif system == "Windows":
                if copy:
                    subprocess.run(["clip"], input=text.encode(), check=True, shell=True)
                    return True, "✅ Copied to clipboard (using clip)"
                else:
                    # PowerShell for paste
                    result = subprocess.run(
                        ["powershell", "-command", "Get-Clipboard"],
                        capture_output=True, check=True
                    )
                    return result.stdout.decode(), ""
                    
        except Exception as e:
            return (False, f"❌ Clipboard error: {str(e)}") if copy else ("", f"Clipboard error: {str(e)}")
        
        return (False, "❌ Unsupported platform") if copy else ("", "Unsupported platform")
    
    # =========================================================================
    # Shell Command Execution
    # =========================================================================
    
    def execute_shell_command(self, command: str, safe_only: bool = True) -> str:
        """Execute a shell command.
        
        Args:
            command: Shell command to execute.
            safe_only: Only allow safe commands.
            
        Returns:
            Command output or error message.
        """
        if not command.strip():
            return "❌ No command provided"
        
        # Parse command
        parts = command.strip().split()
        base_cmd = parts[0].lower()
        
        if safe_only and base_cmd not in self.SAFE_SHELL_COMMANDS:
            return f"❌ Command '{base_cmd}' not in safe list.\n   Allowed: {', '.join(sorted(self.SAFE_SHELL_COMMANDS)[:15])}..."
        
        try:
            # Use shell=True on Windows for better compatibility
            use_shell = sys.platform == 'win32'
            
            result = subprocess.run(
                command if use_shell else parts,
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            
            output = result.stdout
            error = result.stderr
            
            if result.returncode != 0 and error:
                return f"⚠️ Command returned error:\n{error}"
            
            if output:
                # Truncate very long output
                if len(output) > 5000:
                    output = output[:5000] + "\n... [OUTPUT TRUNCATED]"
                return output
            elif error:
                return f"⚠️ {error}"
            else:
                return "✅ Command executed (no output)"
                
        except subprocess.TimeoutExpired:
            return "❌ Command timed out (30s limit)"
        except FileNotFoundError:
            return f"❌ Command not found: {base_cmd}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    # =========================================================================
    # Multi-Model Comparison
    # =========================================================================
    
    def compare_models(self, prompt: str, models: List[str] = None) -> List[Dict[str, Any]]:
        """Query multiple models with the same prompt and compare results.
        
        Args:
            prompt: Prompt to send to all models.
            models: List of model names to compare.
            
        Returns:
            List of results from each model.
        """
        if not self.ai_query:
            return [{"error": "AI query function not available"}]
        
        models = models or ["gemini", "groq", "ollama"]
        results = []
        
        for model in models:
            start_time = datetime.now()
            try:
                response = self.ai_query(model, prompt)
                elapsed = (datetime.now() - start_time).total_seconds()
                results.append({
                    "model": model,
                    "response": response,
                    "elapsed_seconds": elapsed,
                    "success": True,
                    "char_count": len(response)
                })
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                results.append({
                    "model": model,
                    "response": str(e),
                    "elapsed_seconds": elapsed,
                    "success": False,
                    "error": str(e)
                })
        
        self.comparison_results = results
        return results
    
    def format_comparison(self, results: List[Dict[str, Any]]) -> str:
        """Format comparison results for display.
        
        Args:
            results: Comparison results from compare_models.
            
        Returns:
            Formatted string.
        """
        lines = ["# 🔄 Multi-Model Comparison\n"]
        
        for i, result in enumerate(results, 1):
            model = result['model'].upper()
            elapsed = result.get('elapsed_seconds', 0)
            
            if result.get('success'):
                lines.append(f"## {i}. {model} ({elapsed:.2f}s)")
                lines.append("")
                lines.append(result['response'])
            else:
                lines.append(f"## {i}. {model} ❌ FAILED ({elapsed:.2f}s)")
                lines.append("")
                lines.append(f"Error: {result.get('error', 'Unknown error')}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Summary
        successful = [r for r in results if r.get('success')]
        if successful:
            fastest = min(successful, key=lambda x: x.get('elapsed_seconds', float('inf')))
            lines.append(f"**⚡ Fastest:** {fastest['model'].upper()} ({fastest['elapsed_seconds']:.2f}s)")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Code Generation Helpers
    # =========================================================================
    
    def generate_tests(self, filepath: str, ai_model: str = "gemini", 
                       test_framework: str = "pytest") -> str:
        """Generate unit tests for a file.
        
        Args:
            filepath: Path to the source file.
            ai_model: AI model to use.
            test_framework: Testing framework to use.
            
        Returns:
            Generated test code.
        """
        try:
            content = Path(filepath).read_text(encoding='utf-8')
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"
        
        ext = Path(filepath).suffix.lower()
        language = self.SUPPORTED_EXTENSIONS.get(ext, 'python')
        
        # Map test frameworks
        framework_map = {
            'python': 'pytest',
            'javascript': 'jest',
            'typescript': 'jest',
            'java': 'junit',
            'go': 'testing',
            'rust': 'built-in tests',
        }
        
        framework = test_framework or framework_map.get(language, 'unittest')
        
        if not self.ai_query:
            return "❌ AI query function not available"
        
        prompt = f"""Generate comprehensive unit tests for the following {language} code using {framework}.

Requirements:
1. Test all public functions/methods
2. Include happy path tests
3. Include edge cases
4. Include error handling tests
5. Use descriptive test names
6. Add comments explaining each test

Source code:
```{language}
{content[:10000]}
```

Provide only the test code, ready to run."""
        
        try:
            response = self.ai_query(ai_model, prompt)
            return f"🧪 **Generated Tests ({framework}):**\n\n{response}"
        except Exception as e:
            return f"❌ Failed to generate tests: {str(e)}"
    
    def explain_error(self, error_text: str, context: str = "", ai_model: str = "gemini") -> str:
        """Explain an error message and suggest fixes.
        
        Args:
            error_text: The error message/traceback.
            context: Optional code context.
            ai_model: AI model to use.
            
        Returns:
            Explanation and suggestions.
        """
        if not self.ai_query:
            return "❌ AI query function not available"
        
        prompt = f"""Analyze this error and help fix it:

Error:
```
{error_text[:5000]}
```
"""
        
        if context:
            prompt += f"""
Related code:
```
{context[:5000]}
```
"""
        
        prompt += """
Please provide:
1. **What happened**: Explain the error in simple terms
2. **Root cause**: Why this error occurred
3. **Fix**: How to fix it (with code if applicable)
4. **Prevention**: How to prevent this in the future"""
        
        try:
            response = self.ai_query(ai_model, prompt)
            return f"🔍 **Error Analysis:**\n\n{response}"
        except Exception as e:
            return f"❌ Failed to analyze error: {str(e)}"
