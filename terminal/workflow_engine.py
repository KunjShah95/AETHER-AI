"""
Workflow Automation Engine for AetherAI.

This module provides:
- Custom workflow definitions
- Automated task pipelines
- Scheduled actions
- Event-driven triggers
- Integration with other AetherAI modules
"""

import os
import re
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepType(Enum):
    """Types of workflow steps."""
    COMMAND = "command"       # Shell command
    AI_QUERY = "ai_query"     # AI query
    FILE_OP = "file_op"       # File operation
    HTTP = "http"             # HTTP request
    WAIT = "wait"             # Wait/delay
    CONDITION = "condition"   # Conditional branch
    LOOP = "loop"             # Loop
    PARALLEL = "parallel"     # Parallel execution
    NOTIFY = "notify"         # Notification


@dataclass
class WorkflowStep:
    """Represents a single workflow step."""
    step_id: str
    step_type: StepType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    on_success: Optional[str] = None  # Next step on success
    on_failure: Optional[str] = None  # Next step on failure
    timeout: int = 60  # Seconds
    retries: int = 0
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class Workflow:
    """Represents a complete workflow."""
    workflow_id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    trigger: Optional[Dict] = None
    logs: List[str] = field(default_factory=list)


class WorkflowEngine:
    """Executes and manages workflows."""
    
    def __init__(self, base_dir: Optional[str] = None,
                 ai_query_func: Callable = None):
        """Initialize workflow engine.
        
        Args:
            base_dir: Directory for storing workflows.
            ai_query_func: Function to query AI models.
        """
        self.base_dir = base_dir or os.path.join(
            os.getenv('HOME') or os.getenv('USERPROFILE') or os.path.expanduser('~'),
            '.nexus', 'workflows'
        )
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.ai_query = ai_query_func
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, threading.Thread] = {}
        self.step_handlers: Dict[StepType, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
        
        # Load saved workflows
        self._load_workflows()
    
    def _register_default_handlers(self):
        """Register default step handlers."""
        self.step_handlers[StepType.COMMAND] = self._handle_command
        self.step_handlers[StepType.AI_QUERY] = self._handle_ai_query
        self.step_handlers[StepType.FILE_OP] = self._handle_file_op
        self.step_handlers[StepType.WAIT] = self._handle_wait
        self.step_handlers[StepType.NOTIFY] = self._handle_notify
    
    def _load_workflows(self):
        """Load workflows from disk."""
        try:
            for filename in os.listdir(self.base_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.base_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    workflow = self._dict_to_workflow(data)
                    self.workflows[workflow.workflow_id] = workflow
        except Exception as e:
            print(f"Warning: Could not load workflows: {e}")
    
    def _save_workflow(self, workflow: Workflow):
        """Save workflow to disk."""
        try:
            filepath = os.path.join(self.base_dir, f"{workflow.workflow_id}.json")
            data = self._workflow_to_dict(workflow)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save workflow: {e}")
    
    def _workflow_to_dict(self, workflow: Workflow) -> Dict:
        """Convert workflow to dictionary."""
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_type": s.step_type.value,
                    "name": s.name,
                    "config": s.config,
                    "on_success": s.on_success,
                    "on_failure": s.on_failure,
                    "timeout": s.timeout,
                    "retries": s.retries
                }
                for s in workflow.steps
            ],
            "variables": workflow.variables,
            "trigger": workflow.trigger,
            "created_at": workflow.created_at
        }
    
    def _dict_to_workflow(self, data: Dict) -> Workflow:
        """Convert dictionary to workflow."""
        steps = [
            WorkflowStep(
                step_id=s["step_id"],
                step_type=StepType(s["step_type"]),
                name=s["name"],
                config=s.get("config", {}),
                on_success=s.get("on_success"),
                on_failure=s.get("on_failure"),
                timeout=s.get("timeout", 60),
                retries=s.get("retries", 0)
            )
            for s in data.get("steps", [])
        ]
        
        return Workflow(
            workflow_id=data["workflow_id"],
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            variables=data.get("variables", {}),
            trigger=data.get("trigger"),
            created_at=data.get("created_at", datetime.now().isoformat())
        )
    
    # =========================================================================
    # Workflow Management
    # =========================================================================
    
    def create_workflow(self, name: str, description: str = "",
                        workflow_id: Optional[str] = None) -> Workflow:
        """Create a new workflow.
        
        Args:
            name: Workflow name.
            description: Workflow description.
            workflow_id: Optional custom ID.
            
        Returns:
            Created workflow.
        """
        workflow_id = workflow_id or f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description
        )
        
        self.workflows[workflow_id] = workflow
        self._save_workflow(workflow)
        
        return workflow
    
    def add_step(self, workflow_id: str, step_type: StepType, name: str,
                 config: Dict = None, **kwargs) -> Optional[WorkflowStep]:
        """Add a step to a workflow.
        
        Args:
            workflow_id: Workflow ID.
            step_type: Type of step.
            name: Step name.
            config: Step configuration.
            **kwargs: Additional step options.
            
        Returns:
            Created step or None.
        """
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        step_id = f"step_{len(workflow.steps) + 1}"
        
        step = WorkflowStep(
            step_id=step_id,
            step_type=step_type,
            name=name,
            config=config or {},
            **kwargs
        )
        
        workflow.steps.append(step)
        self._save_workflow(workflow)
        
        return step
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow.
        
        Args:
            workflow_id: Workflow ID.
            
        Returns:
            True if deleted.
        """
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            
            filepath = os.path.join(self.base_dir, f"{workflow_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return True
        return False
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows.
        
        Returns:
            List of workflow summaries.
        """
        return [
            {
                "workflow_id": wf.workflow_id,
                "name": wf.name,
                "description": wf.description,
                "steps": len(wf.steps),
                "status": wf.status.value,
                "created_at": wf.created_at
            }
            for wf in self.workflows.values()
        ]
    
    # =========================================================================
    # Workflow Execution
    # =========================================================================
    
    def run_workflow(self, workflow_id: str, variables: Dict = None,
                     async_run: bool = False) -> Dict[str, Any]:
        """Run a workflow.
        
        Args:
            workflow_id: Workflow ID.
            variables: Runtime variables.
            async_run: Run asynchronously.
            
        Returns:
            Execution result.
        """
        if workflow_id not in self.workflows:
            return {"error": f"Workflow '{workflow_id}' not found"}
        
        workflow = self.workflows[workflow_id]
        
        # Update variables
        if variables:
            workflow.variables.update(variables)
        
        if async_run:
            thread = threading.Thread(
                target=self._execute_workflow,
                args=(workflow,)
            )
            thread.start()
            self.running_workflows[workflow_id] = thread
            
            return {
                "status": "started",
                "workflow_id": workflow_id,
                "message": "Workflow running in background"
            }
        else:
            return self._execute_workflow(workflow)
    
    def _execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            workflow: Workflow to execute.
            
        Returns:
            Execution result.
        """
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now().isoformat()
        workflow.logs = []
        
        self._log(workflow, f"Starting workflow: {workflow.name}")
        
        results = []
        current_step_idx = 0
        
        while current_step_idx < len(workflow.steps):
            step = workflow.steps[current_step_idx]
            
            self._log(workflow, f"Executing step: {step.name}")
            step.status = WorkflowStatus.RUNNING
            step.started_at = datetime.now().isoformat()
            
            try:
                # Get handler
                handler = self.step_handlers.get(step.step_type)
                if not handler:
                    raise ValueError(f"No handler for step type: {step.step_type}")
                
                # Execute with variables substitution
                config = self._substitute_variables(step.config, workflow.variables)
                result = handler(config, workflow)
                
                step.result = result
                step.status = WorkflowStatus.COMPLETED
                step.completed_at = datetime.now().isoformat()
                
                self._log(workflow, f"Step completed: {step.name}")
                results.append({"step": step.name, "result": result, "status": "success"})
                
                # Determine next step
                if step.on_success and step.on_success != "next":
                    # Jump to specified step
                    next_idx = next(
                        (i for i, s in enumerate(workflow.steps) if s.step_id == step.on_success),
                        current_step_idx + 1
                    )
                    current_step_idx = next_idx
                else:
                    current_step_idx += 1
                    
            except Exception as e:
                step.error = str(e)
                step.status = WorkflowStatus.FAILED
                step.completed_at = datetime.now().isoformat()
                
                self._log(workflow, f"Step failed: {step.name} - {str(e)}")
                results.append({"step": step.name, "error": str(e), "status": "failed"})
                
                # Handle failure
                if step.on_failure:
                    next_idx = next(
                        (i for i, s in enumerate(workflow.steps) if s.step_id == step.on_failure),
                        None
                    )
                    if next_idx is not None:
                        current_step_idx = next_idx
                        continue
                
                # Abort workflow on failure
                workflow.status = WorkflowStatus.FAILED
                workflow.completed_at = datetime.now().isoformat()
                self._save_workflow(workflow)
                
                return {
                    "status": "failed",
                    "workflow_id": workflow.workflow_id,
                    "results": results,
                    "error": str(e)
                }
        
        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = datetime.now().isoformat()
        self._log(workflow, f"Workflow completed: {workflow.name}")
        self._save_workflow(workflow)
        
        return {
            "status": "completed",
            "workflow_id": workflow.workflow_id,
            "results": results
        }
    
    def _log(self, workflow: Workflow, message: str):
        """Add log entry to workflow."""
        entry = f"[{datetime.now().isoformat()}] {message}"
        workflow.logs.append(entry)
    
    def _substitute_variables(self, config: Dict, variables: Dict) -> Dict:
        """Substitute variables in config.
        
        Args:
            config: Configuration dict.
            variables: Variables to substitute.
            
        Returns:
            Config with variables substituted.
        """
        result = {}
        
        for key, value in config.items():
            if isinstance(value, str):
                # Replace ${var} patterns
                for var_name, var_value in variables.items():
                    value = value.replace(f"${{{var_name}}}", str(var_value))
                result[key] = value
            elif isinstance(value, dict):
                result[key] = self._substitute_variables(value, variables)
            else:
                result[key] = value
        
        return result
    
    # =========================================================================
    # Step Handlers
    # =========================================================================
    
    def _handle_command(self, config: Dict, workflow: Workflow) -> str:
        """Handle command step."""
        import subprocess
        
        command = config.get("command", "")
        cwd = config.get("cwd", ".")
        timeout = config.get("timeout", 60)
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")
        
        # Store output in workflow variables
        workflow.variables["last_output"] = result.stdout
        
        return result.stdout
    
    def _handle_ai_query(self, config: Dict, workflow: Workflow) -> str:
        """Handle AI query step."""
        if not self.ai_query:
            raise RuntimeError("AI query function not available")
        
        model = config.get("model", "gemini")
        prompt = config.get("prompt", "")
        
        response = self.ai_query(model, prompt)
        workflow.variables["last_ai_response"] = response
        
        return response
    
    def _handle_file_op(self, config: Dict, workflow: Workflow) -> str:
        """Handle file operation step."""
        operation = config.get("operation", "read")
        filepath = config.get("path", "")
        
        path = Path(filepath)
        
        if operation == "read":
            content = path.read_text(encoding='utf-8')
            workflow.variables["file_content"] = content
            return content
            
        elif operation == "write":
            content = config.get("content", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return f"Written to {filepath}"
            
        elif operation == "append":
            content = config.get("content", "")
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Appended to {filepath}"
            
        elif operation == "delete":
            if path.exists():
                path.unlink()
            return f"Deleted {filepath}"
            
        elif operation == "copy":
            dest = config.get("dest", "")
            import shutil
            shutil.copy2(filepath, dest)
            return f"Copied to {dest}"
        
        raise ValueError(f"Unknown file operation: {operation}")
    
    def _handle_wait(self, config: Dict, workflow: Workflow) -> str:
        """Handle wait step."""
        seconds = config.get("seconds", 1)
        time.sleep(seconds)
        return f"Waited {seconds} seconds"
    
    def _handle_notify(self, config: Dict, workflow: Workflow) -> str:
        """Handle notification step."""
        message = config.get("message", "")
        channel = config.get("channel", "console")
        
        if channel == "console":
            print(f"📢 WORKFLOW NOTIFICATION: {message}")
        
        # Could add email, webhook, etc.
        
        return f"Notification sent: {message}"
    
    # =========================================================================
    # Pre-built Workflows
    # =========================================================================
    
    def create_code_review_workflow(self) -> Workflow:
        """Create a code review workflow."""
        workflow = self.create_workflow(
            name="Code Review Pipeline",
            description="Automated code review with AI analysis"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Get Git Diff", {
            "command": "git diff --cached"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Analyze Changes", {
            "model": "gemini",
            "prompt": "Review this code diff for issues, security concerns, and improvements:\n\n${last_output}"
        })
        
        self.add_step(workflow.workflow_id, StepType.NOTIFY, "Notify Result", {
            "message": "Code review complete: ${last_ai_response}",
            "channel": "console"
        })
        
        return workflow
    
    def create_deploy_workflow(self) -> Workflow:
        """Create a deployment workflow."""
        workflow = self.create_workflow(
            name="Deployment Pipeline",
            description="Automated deployment workflow"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Run Tests", {
            "command": "pytest -v"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Build", {
            "command": "python -m build"
        })
        
        self.add_step(workflow.workflow_id, StepType.NOTIFY, "Notify Success", {
            "message": "Deployment completed successfully!",
            "channel": "console"
        })
        
        return workflow
    
    def create_daily_standup_workflow(self) -> Workflow:
        """Create a daily standup summary workflow."""
        workflow = self.create_workflow(
            name="Daily Standup Summary",
            description="Generate daily standup summary from git history"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Get Yesterday's Commits", {
            "command": "git log --since='1 day ago' --oneline"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Generate Summary", {
            "model": "gemini",
            "prompt": "Create a concise standup summary from these commits:\n\n${last_output}\n\nFormat:\n- What I did yesterday\n- What I'm doing today\n- Blockers"
        })
        
        return workflow
    
    def create_test_workflow(self) -> Workflow:
        """Create a testing workflow."""
        workflow = self.create_workflow(
            name="Test Pipeline",
            description="Run tests and generate coverage report"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Install Dependencies", {
            "command": "pip install pytest pytest-cov"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Run Tests with Coverage", {
            "command": "pytest --cov=. --cov-report=term-missing -v"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Analyze Results", {
            "model": "gemini",
            "prompt": "Analyze these test results and provide a summary:\n\n${last_output}\n\nInclude:\n- Test pass/fail summary\n- Coverage analysis\n- Recommendations for improvement"
        })
        
        self.add_step(workflow.workflow_id, StepType.NOTIFY, "Notify Results", {
            "message": "Test results: ${last_ai_response}",
            "channel": "console"
        })
        
        return workflow
    
    def create_documentation_workflow(self) -> Workflow:
        """Create a documentation generation workflow."""
        workflow = self.create_workflow(
            name="Documentation Generator",
            description="Generate and update project documentation"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "List Python Files", {
            "command": "find . -name '*.py' -not -path './.venv/*' -not -path './__pycache__/*' | head -20"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Generate README", {
            "model": "gemini",
            "prompt": "Based on these project files, generate a comprehensive README.md:\n\n${last_output}\n\nInclude:\n- Project title and description\n- Installation instructions\n- Usage examples\n- API documentation overview\n- Contributing guidelines"
        })
        
        self.add_step(workflow.workflow_id, StepType.FILE_OP, "Save README", {
            "operation": "write",
            "path": "README_GENERATED.md",
            "content": "${last_ai_response}"
        })
        
        self.add_step(workflow.workflow_id, StepType.NOTIFY, "Done", {
            "message": "Documentation generated: README_GENERATED.md",
            "channel": "console"
        })
        
        return workflow
    
    def create_security_audit_workflow(self) -> Workflow:
        """Create a security audit workflow."""
        workflow = self.create_workflow(
            name="Security Audit",
            description="Scan code for security vulnerabilities"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Find Sensitive Files", {
            "command": "grep -r -l 'password\\|secret\\|api_key\\|token' --include='*.py' --include='*.js' --include='*.env' . 2>/dev/null || echo 'No sensitive patterns found'"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Check for Hardcoded Secrets", {
            "command": "grep -r -n 'password.*=.*\"' --include='*.py' . 2>/dev/null || echo 'No hardcoded passwords found'"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Security Analysis", {
            "model": "gemini",
            "prompt": "Analyze these security scan results and provide a report:\n\n${last_output}\n\nInclude:\n- Critical issues found\n- Security recommendations\n- Best practices to implement\n- Severity ratings"
        })
        
        self.add_step(workflow.workflow_id, StepType.NOTIFY, "Security Report", {
            "message": "Security audit complete",
            "channel": "console"
        })
        
        return workflow
    
    def create_release_workflow(self) -> Workflow:
        """Create a release preparation workflow."""
        workflow = self.create_workflow(
            name="Release Preparation",
            description="Prepare a new release with version bump and changelog"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Get Current Version", {
            "command": "cat pyproject.toml | grep 'version = ' | head -1 || echo 'version = \"0.0.0\"'"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Get Recent Changes", {
            "command": "git log --oneline -20"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Generate Changelog Entry", {
            "model": "gemini",
            "prompt": "Generate a changelog entry for a new release based on these commits:\n\n${last_output}\n\nFormat using Keep a Changelog format with:\n- Added\n- Changed\n- Fixed\n- Removed"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Run Final Tests", {
            "command": "pytest -v || echo 'Tests completed'"
        })
        
        self.add_step(workflow.workflow_id, StepType.NOTIFY, "Release Ready", {
            "message": "Release preparation complete!\n${last_ai_response}",
            "channel": "console"
        })
        
        return workflow
    
    def create_refactor_workflow(self) -> Workflow:
        """Create a refactoring analysis workflow."""
        workflow = self.create_workflow(
            name="Refactoring Analysis",
            description="Analyze code for refactoring opportunities"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Count Lines per File", {
            "command": "find . -name '*.py' -not -path './.venv/*' -exec wc -l {} + | sort -n | tail -10"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Find Complex Functions", {
            "command": "grep -n 'def ' *.py 2>/dev/null | head -20 || echo 'No Python files in current directory'"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Refactoring Suggestions", {
            "model": "gemini",
            "prompt": "Based on this code analysis, suggest refactoring opportunities:\n\nLargest files:\n${last_output}\n\nProvide:\n- Files that should be split\n- Code duplication concerns\n- Complexity reduction suggestions\n- Architecture improvements"
        })
        
        return workflow
    
    def create_dependency_check_workflow(self) -> Workflow:
        """Create a dependency check workflow."""
        workflow = self.create_workflow(
            name="Dependency Check",
            description="Check for outdated and vulnerable dependencies"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "List Outdated Packages", {
            "command": "pip list --outdated 2>/dev/null || echo 'Could not check outdated packages'"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Check for Vulnerabilities", {
            "command": "pip-audit 2>/dev/null || echo 'pip-audit not installed. Install with: pip install pip-audit'"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Dependency Report", {
            "model": "gemini",
            "prompt": "Analyze these dependency scan results:\n\n${last_output}\n\nProvide:\n- Critical updates needed\n- Security vulnerabilities to address\n- Recommended update strategy\n- Compatibility considerations"
        })
        
        return workflow
    
    def create_changelog_workflow(self) -> Workflow:
        """Create a changelog generation workflow."""
        workflow = self.create_workflow(
            name="Changelog Generator",
            description="Generate changelog from git history"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Get All Tags", {
            "command": "git tag --sort=-creatordate | head -5 || echo 'No tags found'"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Get Commits Since Last Tag", {
            "command": "git log $(git describe --tags --abbrev=0 2>/dev/null || echo HEAD~20)..HEAD --oneline || git log --oneline -20"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Generate Changelog", {
            "model": "gemini",
            "prompt": "Generate a professional changelog from these commits:\n\n${last_output}\n\nUse conventional changelog format with these sections:\n## [Unreleased]\n### Added\n### Changed\n### Fixed\n### Removed\n\nGroup related changes together."
        })
        
        self.add_step(workflow.workflow_id, StepType.FILE_OP, "Save Changelog", {
            "operation": "write",
            "path": "CHANGELOG_GENERATED.md",
            "content": "${last_ai_response}"
        })
        
        return workflow
    
    def create_health_check_workflow(self) -> Workflow:
        """Create a project health check workflow."""
        workflow = self.create_workflow(
            name="Project Health Check",
            description="Comprehensive project health analysis"
        )
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Check Git Status", {
            "command": "git status --short"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Count Files by Type", {
            "command": "find . -type f -not -path './.git/*' -not -path './.venv/*' | sed 's/.*\\.//' | sort | uniq -c | sort -rn | head -10"
        })
        
        self.add_step(workflow.workflow_id, StepType.COMMAND, "Check TODO/FIXME", {
            "command": "grep -r 'TODO\\|FIXME\\|XXX\\|HACK' --include='*.py' --include='*.js' . 2>/dev/null | wc -l || echo '0'"
        })
        
        self.add_step(workflow.workflow_id, StepType.AI_QUERY, "Health Report", {
            "model": "gemini",
            "prompt": "Generate a project health report based on this analysis:\n\nGit Status:\n${last_output}\n\nProvide:\n- Overall health score (1-10)\n- Areas of concern\n- Immediate actions needed\n- Long-term recommendations"
        })
        
        self.add_step(workflow.workflow_id, StepType.NOTIFY, "Health Check Complete", {
            "message": "Project health check complete. Review the results.",
            "channel": "console"
        })
        
        return workflow
    
    # =========================================================================
    # Formatting
    # =========================================================================
    
    def format_workflow(self, workflow_id: str) -> str:
        """Format workflow for display.
        
        Args:
            workflow_id: Workflow ID.
            
        Returns:
            Formatted string.
        """
        if workflow_id not in self.workflows:
            return f"❌ Workflow '{workflow_id}' not found"
        
        wf = self.workflows[workflow_id]
        
        lines = [
            f"📋 **{wf.name}**",
            f"🆔 {wf.workflow_id}",
            f"📝 {wf.description}" if wf.description else "",
            f"📊 Status: {wf.status.value}",
            f"⏰ Created: {wf.created_at}",
            ""
        ]
        
        if wf.steps:
            lines.append(f"**Steps ({len(wf.steps)}):**")
            for i, step in enumerate(wf.steps, 1):
                status_icon = {
                    WorkflowStatus.PENDING: "⏳",
                    WorkflowStatus.RUNNING: "🔄",
                    WorkflowStatus.COMPLETED: "✅",
                    WorkflowStatus.FAILED: "❌"
                }.get(step.status, "⚪")
                
                lines.append(f"  {i}. {status_icon} {step.name} [{step.step_type.value}]")
        
        if wf.variables:
            lines.append("")
            lines.append(f"**Variables:** {', '.join(wf.variables.keys())}")
        
        return "\n".join(filter(None, lines))
    
    def format_list(self) -> str:
        """Format workflow list for display.
        
        Returns:
            Formatted string.
        """
        if not self.workflows:
            return "📋 No workflows defined. Use /workflow create <name> to create one."
        
        lines = ["📋 **Workflows:**\n"]
        
        for wf in self.workflows.values():
            status_icon = {
                WorkflowStatus.PENDING: "⏳",
                WorkflowStatus.RUNNING: "🔄",
                WorkflowStatus.COMPLETED: "✅",
                WorkflowStatus.FAILED: "❌"
            }.get(wf.status, "⚪")
            
            lines.append(f"{status_icon} **{wf.name}** ({wf.workflow_id})")
            lines.append(f"   {len(wf.steps)} steps | {wf.description[:50]}..." if wf.description else f"   {len(wf.steps)} steps")
        
        return "\n".join(lines)
    
    def list_templates(self) -> str:
        """List available pre-built workflow templates."""
        templates = [
            ("code-review", "Automated code review with AI analysis"),
            ("deploy", "Build and deployment pipeline"),
            ("standup", "Daily standup summary from git"),
            ("test", "Run tests with coverage report"),
            ("docs", "Generate project documentation"),
            ("security", "Security audit and vulnerability scan"),
            ("release", "Release preparation with changelog"),
            ("refactor", "Code refactoring analysis"),
            ("deps", "Dependency check and updates"),
            ("changelog", "Generate changelog from commits"),
            ("health", "Project health check"),
        ]
        
        lines = ["📋 **Available Workflow Templates:**\n"]
        for name, desc in templates:
            lines.append(f"  • `/workflow {name}` - {desc}")
        
        lines.append("\n💡 Run any template to create it as a workflow")
        return "\n".join(lines)

