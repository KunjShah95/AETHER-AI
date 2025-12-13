"""
AI Code Agent for AetherAI - Autonomous code editing and generation.

This module provides:
- Autonomous code editing with diff preview
- Multi-file analysis and refactoring
- Code execution in sandbox
- Project-wide changes with safety checks
- Auto-fix capabilities
"""

import os
import re
import json
import difflib
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class CodeEdit:
    """Represents a single code edit."""
    file_path: str
    original_content: str
    new_content: str
    description: str
    line_start: int = 0
    line_end: int = 0
    status: str = "pending"  # pending, applied, rejected


@dataclass
class AgentTask:
    """Represents an agent task."""
    task_id: str
    description: str
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    edits: List[CodeEdit] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    result: Optional[str] = None


class CodeAgent:
    """AI-powered autonomous code agent."""
    
    # Safe file extensions for editing
    SAFE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
        '.html', '.css', '.scss', '.sass', '.less', '.json', '.yaml', '.yml',
        '.md', '.txt', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat',
        '.sql', '.graphql', '.toml', '.ini', '.cfg', '.conf', '.env',
        '.dockerfile', '.xml', '.r', '.R', '.jl', '.lua', '.vim'
    }
    
    # Dangerous patterns to avoid
    DANGEROUS_PATTERNS = [
        r'rm\s+-rf',
        r'del\s+/s',
        r'format\s+c:',
        r'sudo\s+rm',
        r':(){ :|:& };:',  # Fork bomb
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__',
        r'os\.system',
        r'subprocess\.call.*shell=True',
    ]
    
    def __init__(self, ai_query_func: Callable = None, workspace: str = "."):
        """Initialize the code agent.
        
        Args:
            ai_query_func: Function to query AI models.
            workspace: Workspace directory.
        """
        self.ai_query = ai_query_func
        self.workspace = Path(workspace).absolute()
        self.tasks: Dict[str, AgentTask] = {}
        self.history: List[AgentTask] = []
        self.max_file_size = 100000  # 100KB max file size
        self.dry_run = True  # Safety: preview changes by default
    
    # =========================================================================
    # File Operations
    # =========================================================================
    
    def read_file(self, filepath: str) -> Tuple[str, Optional[str]]:
        """Safely read a file.
        
        Args:
            filepath: Path to file.
            
        Returns:
            Tuple of (content, error).
        """
        try:
            path = self._resolve_path(filepath)
            if not path:
                return "", "Invalid path or outside workspace"
            
            if not path.exists():
                return "", f"File not found: {filepath}"
            
            if path.stat().st_size > self.max_file_size:
                return "", f"File too large (>{self.max_file_size} bytes)"
            
            content = path.read_text(encoding='utf-8')
            return content, None
        except Exception as e:
            return "", str(e)
    
    def write_file(self, filepath: str, content: str, 
                   create_backup: bool = True) -> Tuple[bool, str]:
        """Safely write to a file.
        
        Args:
            filepath: Path to file.
            content: Content to write.
            create_backup: Create backup before writing.
            
        Returns:
            Tuple of (success, message).
        """
        try:
            path = self._resolve_path(filepath)
            if not path:
                return False, "Invalid path or outside workspace"
            
            # Check extension
            if path.suffix.lower() not in self.SAFE_EXTENSIONS:
                return False, f"Unsafe file extension: {path.suffix}"
            
            # Check for dangerous patterns
            for pattern in self.DANGEROUS_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    return False, f"Dangerous pattern detected: {pattern}"
            
            # Create backup
            if create_backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.bak')
                backup_path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
            
            # Write file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            
            return True, f"Successfully wrote to {filepath}"
        except Exception as e:
            return False, str(e)
    
    def _resolve_path(self, filepath: str) -> Optional[Path]:
        """Resolve and validate file path.
        
        Args:
            filepath: Input path.
            
        Returns:
            Resolved path or None if invalid.
        """
        try:
            path = Path(filepath)
            if not path.is_absolute():
                path = self.workspace / path
            
            path = path.resolve()
            
            # Security: ensure path is within workspace
            try:
                path.relative_to(self.workspace)
            except ValueError:
                return None
            
            return path
        except Exception:
            return None
    
    # =========================================================================
    # Code Analysis
    # =========================================================================
    
    def analyze_project(self, directory: str = ".", 
                        extensions: List[str] = None) -> Dict[str, Any]:
        """Analyze entire project/directory.
        
        Args:
            directory: Directory to analyze.
            extensions: File extensions to include.
            
        Returns:
            Project analysis results.
        """
        extensions = extensions or ['.py', '.js', '.ts', '.java', '.go']
        path = self._resolve_path(directory)
        
        if not path or not path.is_dir():
            return {"error": "Invalid directory"}
        
        analysis = {
            "directory": str(path),
            "files": [],
            "total_lines": 0,
            "total_files": 0,
            "languages": {},
            "issues": [],
            "structure": {}
        }
        
        for ext in extensions:
            for filepath in path.rglob(f"*{ext}"):
                # Skip hidden and vendor directories
                if any(p.startswith('.') or p in ['node_modules', 'venv', '__pycache__', '.git'] 
                       for p in filepath.parts):
                    continue
                
                try:
                    content = filepath.read_text(encoding='utf-8')
                    lines = len(content.split('\n'))
                    
                    file_info = {
                        "path": str(filepath.relative_to(path)),
                        "lines": lines,
                        "size": filepath.stat().st_size,
                        "language": ext[1:],
                    }
                    
                    analysis["files"].append(file_info)
                    analysis["total_lines"] += lines
                    analysis["total_files"] += 1
                    
                    # Count by language
                    lang = ext[1:]
                    if lang not in analysis["languages"]:
                        analysis["languages"][lang] = {"files": 0, "lines": 0}
                    analysis["languages"][lang]["files"] += 1
                    analysis["languages"][lang]["lines"] += lines
                    
                except Exception:
                    pass
        
        return analysis
    
    def find_issues(self, filepath: str) -> List[Dict[str, Any]]:
        """Find potential issues in a file.
        
        Args:
            filepath: Path to file.
            
        Returns:
            List of issues found.
        """
        content, error = self.read_file(filepath)
        if error:
            return [{"type": "error", "message": error}]
        
        issues = []
        lines = content.split('\n')
        
        path = Path(filepath)
        ext = path.suffix.lower()
        
        for i, line in enumerate(lines, 1):
            # Check for TODO/FIXME
            if 'TODO' in line or 'FIXME' in line:
                issues.append({
                    "type": "todo",
                    "line": i,
                    "message": line.strip(),
                    "severity": "info"
                })
            
            # Check for hardcoded secrets (basic)
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', "Possible hardcoded password"),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "Possible hardcoded API key"),
                (r'secret\s*=\s*["\'][^"\']+["\']', "Possible hardcoded secret"),
                (r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "Possible hardcoded token"),
            ]
            
            for pattern, message in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "type": "security",
                        "line": i,
                        "message": message,
                        "severity": "high"
                    })
            
            # Python-specific checks
            if ext == '.py':
                if 'import *' in line:
                    issues.append({
                        "type": "style",
                        "line": i,
                        "message": "Avoid wildcard imports",
                        "severity": "medium"
                    })
                if re.search(r'except\s*:', line):
                    issues.append({
                        "type": "style",
                        "line": i,
                        "message": "Avoid bare except clauses",
                        "severity": "medium"
                    })
            
            # JavaScript-specific checks
            if ext in ['.js', '.ts', '.jsx', '.tsx']:
                if 'var ' in line:
                    issues.append({
                        "type": "style",
                        "line": i,
                        "message": "Use 'let' or 'const' instead of 'var'",
                        "severity": "low"
                    })
                if '==' in line and '===' not in line:
                    issues.append({
                        "type": "style",
                        "line": i,
                        "message": "Use '===' instead of '=='",
                        "severity": "low"
                    })
        
        return issues
    
    # =========================================================================
    # Code Editing
    # =========================================================================
    
    def generate_diff(self, original: str, new: str, filepath: str = "") -> str:
        """Generate unified diff between original and new content.
        
        Args:
            original: Original content.
            new: New content.
            filepath: File path for context.
            
        Returns:
            Unified diff string.
        """
        original_lines = original.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines, 
            new_lines, 
            fromfile=f"a/{filepath}", 
            tofile=f"b/{filepath}",
            lineterm=''
        )
        
        return ''.join(diff)
    
    def apply_edit(self, edit: CodeEdit, confirm: bool = True) -> Tuple[bool, str]:
        """Apply a code edit.
        
        Args:
            edit: CodeEdit to apply.
            confirm: Require confirmation.
            
        Returns:
            Tuple of (success, message).
        """
        if edit.status != "pending":
            return False, f"Edit already {edit.status}"
        
        # Show diff first
        diff = self.generate_diff(edit.original_content, edit.new_content, edit.file_path)
        
        if confirm:
            print(f"\n📝 Proposed changes to {edit.file_path}:")
            print(diff)
            # In real implementation, would prompt for confirmation
        
        if self.dry_run:
            edit.status = "previewed"
            return True, f"Preview only (dry run). Diff:\n{diff}"
        
        success, msg = self.write_file(edit.file_path, edit.new_content)
        if success:
            edit.status = "applied"
        else:
            edit.status = "failed"
        
        return success, msg
    
    def create_edit_from_ai(self, filepath: str, instruction: str, 
                            model: str = "gemini") -> Optional[CodeEdit]:
        """Use AI to generate a code edit.
        
        Args:
            filepath: File to edit.
            instruction: What to change.
            model: AI model to use.
            
        Returns:
            CodeEdit or None if failed.
        """
        if not self.ai_query:
            return None
        
        content, error = self.read_file(filepath)
        if error:
            return None
        
        prompt = f"""You are a code editor. Modify the following code according to the instruction.
Output ONLY the complete modified code, nothing else.

INSTRUCTION: {instruction}

ORIGINAL CODE:
```
{content}
```

MODIFIED CODE (output the complete modified file):"""
        
        try:
            new_content = self.ai_query(model, prompt)
            
            # Clean up AI response
            new_content = new_content.strip()
            # Remove markdown code blocks if present
            if new_content.startswith('```'):
                lines = new_content.split('\n')
                # Remove first line (```language) and last line (```)
                if lines[-1].strip() == '```':
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                new_content = '\n'.join(lines)
            
            edit = CodeEdit(
                file_path=filepath,
                original_content=content,
                new_content=new_content,
                description=instruction
            )
            
            return edit
        except Exception as e:
            return None
    
    # =========================================================================
    # Autonomous Tasks
    # =========================================================================
    
    def create_task(self, description: str) -> AgentTask:
        """Create a new agent task.
        
        Args:
            description: Task description.
            
        Returns:
            Created AgentTask.
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = AgentTask(task_id=task_id, description=description)
        self.tasks[task_id] = task
        return task
    
    def execute_task(self, task: AgentTask, model: str = "gemini") -> str:
        """Execute an agent task.
        
        Args:
            task: Task to execute.
            model: AI model to use.
            
        Returns:
            Task result.
        """
        if not self.ai_query:
            return "❌ AI query function not available"
        
        task.status = "running"
        task.logs.append(f"[{datetime.now().isoformat()}] Task started")
        
        try:
            # Analyze the task
            plan_prompt = f"""You are an AI code agent. Analyze this task and create a plan.

TASK: {task.description}

WORKSPACE: {self.workspace}

Respond in JSON format:
{{
    "understanding": "Your understanding of the task",
    "steps": ["step 1", "step 2", ...],
    "files_to_modify": ["file1.py", "file2.py"],
    "files_to_create": [],
    "risks": ["potential risk 1"],
    "estimated_complexity": "low|medium|high"
}}"""
            
            plan_response = self.ai_query(model, plan_prompt)
            task.logs.append(f"Plan: {plan_response[:500]}")
            
            # Try to parse the plan
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', plan_response)
                if json_match:
                    plan = json.loads(json_match.group())
                else:
                    plan = {"understanding": plan_response, "steps": [], "files_to_modify": []}
            except json.JSONDecodeError:
                plan = {"understanding": plan_response, "steps": [], "files_to_modify": []}
            
            # Execute each step
            results = []
            for step in plan.get("steps", []):
                task.logs.append(f"Executing: {step}")
                
                # Simple heuristic to determine action
                try:
                    # Ask AI how to execute this step
                    exec_prompt = f"""How should I execute this step?
STEP: {step}
WORKSPACE: {self.workspace}

Available Actions:
1. EDIT <file> <instruction>
2. CREATE <file> <content>
3. RUN <command>
4. ANALYZE <directory>

Respond with ONE line starting with the Action keyword."""
                    
                    action_response = self.ai_query(model, exec_prompt).strip()
                    
                    if action_response.startswith("EDIT"):
                        # Parse: EDITfile.py instruction
                        parts = action_response.split(maxsplit=2)
                        if len(parts) >= 3:
                            filepath, instr = parts[1], parts[2]
                            edit = self.create_edit_from_ai(filepath, instr, model)
                            if edit:
                                self.apply_edit(edit, confirm=False)
                                results.append(f"✓ Edited {filepath}")
                            else:
                                results.append(f"⚠ Failed to edit {filepath}")
                    
                    elif action_response.startswith("CREATE"):
                        parts = action_response.split(maxsplit=2)
                        if len(parts) >= 3:
                            filepath, content = parts[1], parts[2]
                            self.write_file(filepath, content)
                            results.append(f"✓ Created {filepath}")
                            
                    elif action_response.startswith("RUN"):
                        cmd = action_response[4:]
                        # Safety check! only allow safe commands or ask confirmation? 
                        # For autopilot, we'll strip dangerous ones
                        if not any(re.search(p, cmd) for p in self.DANGEROUS_PATTERNS):
                             subprocess.run(cmd, shell=True, cwd=self.workspace, capture_output=True)
                             results.append(f"✓ Ran: {cmd}")
                        else:
                             results.append(f"⚠ Skipped dangerous command: {cmd}")

                    else:
                        results.append(f"✓ Analyzed: {step}")
                        
                except Exception as e:
                    results.append(f"⚠ Step failed: {str(e)}")
            
            task.status = "completed"
            task.result = "\n".join(results) if results else plan.get("understanding", "Task analyzed")
            task.logs.append(f"[{datetime.now().isoformat()}] Task completed")
            
            self.history.append(task)
            return task.result
            
        except Exception as e:
            task.status = "failed"
            task.logs.append(f"Error: {str(e)}")
            return f"❌ Task failed: {str(e)}"
    
    # =========================================================================
    # Code Execution Sandbox
    # =========================================================================
    
    def execute_code_safe(self, code: str, language: str = "python",
                          timeout: int = 10) -> Tuple[str, str, int]:
        """Execute code in a safe sandbox.
        
        Args:
            code: Code to execute.
            language: Programming language.
            timeout: Execution timeout in seconds.
            
        Returns:
            Tuple of (stdout, stderr, return_code).
        """
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return "", f"Dangerous pattern detected: {pattern}", 1
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix=self._get_extension(language),
                                          delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            cmd = self._get_execute_command(language, temp_path)
            if not cmd:
                return "", f"Language '{language}' not supported", 1
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir()
            )
            
            return result.stdout, result.stderr, result.returncode
            
        except subprocess.TimeoutExpired:
            return "", f"Execution timed out after {timeout} seconds", 124
        except Exception as e:
            return "", str(e), 1
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def _get_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'java': '.java',
            'c': '.c',
            'cpp': '.cpp',
            'go': '.go',
            'rust': '.rs',
            'ruby': '.rb',
            'php': '.php',
            'bash': '.sh',
            'shell': '.sh',
        }
        return extensions.get(language.lower(), '.txt')
    
    def _get_execute_command(self, language: str, filepath: str) -> Optional[List[str]]:
        """Get execution command for language."""
        commands = {
            'python': ['python', filepath],
            'javascript': ['node', filepath],
            'typescript': ['ts-node', filepath],
            'bash': ['bash', filepath],
            'shell': ['bash', filepath],
            'ruby': ['ruby', filepath],
            'php': ['php', filepath],
            'go': ['go', 'run', filepath],
        }
        return commands.get(language.lower())
    
    # =========================================================================
    # Auto-Fix
    # =========================================================================
    
    def auto_fix_file(self, filepath: str, model: str = "gemini") -> List[CodeEdit]:
        """Automatically fix issues in a file.
        
        Args:
            filepath: File to fix.
            model: AI model to use.
            
        Returns:
            List of proposed edits.
        """
        issues = self.find_issues(filepath)
        if not issues:
            return []
        
        content, error = self.read_file(filepath)
        if error:
            return []
        
        # Build fix prompt
        issues_text = "\n".join([
            f"Line {i['line']}: [{i['type']}] {i['message']}" 
            for i in issues if i['severity'] in ['high', 'medium']
        ])
        
        if not issues_text:
            return []
        
        prompt = f"""Fix the following issues in this code. Output the complete fixed code.

ISSUES:
{issues_text}

CODE:
```
{content}
```

FIXED CODE:"""
        
        if not self.ai_query:
            return []
        
        try:
            fixed_content = self.ai_query(model, prompt)
            
            # Clean up response
            if '```' in fixed_content:
                lines = fixed_content.split('\n')
                in_code = False
                code_lines = []
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code = not in_code
                        continue
                    if in_code:
                        code_lines.append(line)
                fixed_content = '\n'.join(code_lines)
            
            edit = CodeEdit(
                file_path=filepath,
                original_content=content,
                new_content=fixed_content,
                description=f"Auto-fix {len(issues)} issues"
            )
            
            return [edit]
        except Exception:
            return []
    
    # =========================================================================
    # Formatting
    # =========================================================================
    
    def format_task_result(self, task: AgentTask) -> str:
        """Format task result for display.
        
        Args:
            task: Task to format.
            
        Returns:
            Formatted string.
        """
        lines = [
            f"📋 **Task:** {task.task_id}",
            f"📝 {task.description}",
            f"📊 Status: {task.status}",
            f"⏰ Created: {task.created_at}",
            ""
        ]
        
        if task.edits:
            lines.append(f"**Edits ({len(task.edits)}):**")
            for edit in task.edits:
                lines.append(f"  • {edit.file_path}: {edit.description} [{edit.status}]")
            lines.append("")
        
        if task.result:
            lines.append("**Result:**")
            lines.append(task.result)
        
        return "\n".join(lines)
    
    def format_project_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format project analysis for display.
        
        Args:
            analysis: Analysis results.
            
        Returns:
            Formatted string.
        """
        if "error" in analysis:
            return f"❌ {analysis['error']}"
        
        lines = [
            f"📁 **Project Analysis**",
            f"📍 {analysis['directory']}",
            "",
            f"📊 **Statistics:**",
            f"  • Files: {analysis['total_files']}",
            f"  • Lines: {analysis['total_lines']:,}",
            ""
        ]
        
        if analysis.get("languages"):
            lines.append("**Languages:**")
            for lang, stats in analysis["languages"].items():
                lines.append(f"  • {lang}: {stats['files']} files, {stats['lines']:,} lines")
            lines.append("")
        
        if analysis.get("issues"):
            lines.append(f"**Issues Found:** {len(analysis['issues'])}")
        
        return "\n".join(lines)
