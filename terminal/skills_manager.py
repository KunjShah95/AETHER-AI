"""
Skills & Rules Manager for AetherAI.

This module provides Claude-like skills and rules file support:
- SKILLS.md or skills.md - Define AI capabilities and behaviors
- RULES.md or .rules - Define project-specific rules and guidelines
- Custom instructions that the AI must follow

These files can be placed in:
- Project root (./SKILLS.md, ./RULES.md)
- Home directory (~/.aetherai/SKILLS.md, ~/.aetherai/RULES.md)
- Terminal directory (terminal/skills.md)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class Skill:
    """Represents a skill or capability."""
    name: str
    description: str
    commands: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class Rule:
    """Represents a rule or guideline."""
    name: str
    content: str
    priority: int = 1  # 1 = low, 2 = medium, 3 = high
    category: str = "general"
    enabled: bool = True


class SkillsManager:
    """Manages skills and rules files for AetherAI."""
    
    # File names to look for
    SKILLS_FILES = [
        "SKILLS.md", "skills.md", "AETHER_SKILLS.md",
        ".skills", ".skills.md", "AI_SKILLS.md"
    ]
    
    RULES_FILES = [
        "RULES.md", "rules.md", "AETHER_RULES.md",
        ".rules", ".rules.md", "AI_RULES.md",
        "CLAUDE.md", ".claude", "INSTRUCTIONS.md"
    ]
    
    def __init__(self, project_dir: str = "."):
        """Initialize skills manager.
        
        Args:
            project_dir: Project directory to scan for skills/rules.
        """
        self.project_dir = Path(project_dir).absolute()
        self.home_dir = Path(
            os.getenv('HOME') or os.getenv('USERPROFILE') or os.path.expanduser('~')
        )
        self.aetherai_dir = self.home_dir / '.aetherai'
        
        self.skills: Dict[str, Skill] = {}
        self.rules: List[Rule] = []
        self.raw_skills_content: str = ""
        self.raw_rules_content: str = ""
        
        # Load skills and rules
        self._load_skills()
        self._load_rules()
    
    def _find_file(self, filenames: List[str], search_dirs: List[Path] = None) -> Optional[Path]:
        """Find first matching file in directories.
        
        Args:
            filenames: List of filenames to search for.
            search_dirs: Directories to search.
            
        Returns:
            Path to found file or None.
        """
        search_dirs = search_dirs or [
            self.project_dir,
            self.aetherai_dir,
            Path(__file__).parent  # terminal directory
        ]
        
        for directory in search_dirs:
            if not directory.exists():
                continue
            for filename in filenames:
                filepath = directory / filename
                if filepath.exists() and filepath.is_file():
                    return filepath
        
        return None
    
    def _load_skills(self):
        """Load skills from skills file."""
        filepath = self._find_file(self.SKILLS_FILES)
        
        if not filepath:
            # Create default skills if none exist
            self._create_default_skills()
            return
        
        try:
            content = filepath.read_text(encoding='utf-8')
            self.raw_skills_content = content
            self._parse_skills(content)
        except Exception as e:
            print(f"Warning: Could not load skills from {filepath}: {e}")
    
    def _load_rules(self):
        """Load rules from rules file."""
        filepath = self._find_file(self.RULES_FILES)
        
        if not filepath:
            return
        
        try:
            content = filepath.read_text(encoding='utf-8')
            self.raw_rules_content = content
            self._parse_rules(content)
        except Exception as e:
            print(f"Warning: Could not load rules from {filepath}: {e}")
    
    def _parse_skills(self, content: str):
        """Parse skills from markdown content.
        
        Args:
            content: Markdown content to parse.
        """
        # Parse sections starting with ## or ###
        sections = re.split(r'\n(?=##\s)', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            # Extract skill name from header
            header = lines[0].strip()
            match = re.match(r'^##\s*(.+?)(?:\s*[-:].*)?$', header)
            if not match:
                continue
            
            skill_name = match.group(1).strip()
            
            # Extract description (first paragraph after header)
            description_lines = []
            commands = []
            examples = []
            current_section = "description"
            
            for line in lines[1:]:
                stripped = line.strip()
                
                # Check for subsections
                if stripped.lower().startswith('### commands') or stripped.lower().startswith('**commands'):
                    current_section = "commands"
                    continue
                elif stripped.lower().startswith('### examples') or stripped.lower().startswith('**examples'):
                    current_section = "examples"
                    continue
                elif stripped.startswith('### ') or stripped.startswith('## '):
                    break
                
                # Parse content
                if current_section == "description":
                    if stripped and not stripped.startswith('```'):
                        description_lines.append(stripped)
                elif current_section == "commands":
                    # Look for command patterns like /command or `command`
                    cmd_match = re.findall(r'[`/]([a-zA-Z0-9_-]+)[`\s]', stripped)
                    commands.extend(cmd_match)
                elif current_section == "examples":
                    if stripped.startswith('- ') or stripped.startswith('* '):
                        examples.append(stripped[2:])
                    elif stripped.startswith('`'):
                        examples.append(stripped.strip('`'))
            
            self.skills[skill_name.lower()] = Skill(
                name=skill_name,
                description=' '.join(description_lines[:3]),  # First 3 lines
                commands=commands,
                examples=examples
            )
    
    def _parse_rules(self, content: str):
        """Parse rules from markdown content.
        
        Args:
            content: Markdown content to parse.
        """
        self.rules = []
        
        # Parse numbered or bulleted rules
        lines = content.split('\n')
        current_rule = []
        current_category = "general"
        current_priority = 1
        
        for line in lines:
            stripped = line.strip()
            
            # Check for category headers
            if stripped.startswith('## '):
                current_category = stripped[3:].strip().lower()
                continue
            
            # Check for priority markers
            if '⚠️' in stripped or 'CRITICAL' in stripped.upper() or 'MUST' in stripped.upper():
                current_priority = 3
            elif 'SHOULD' in stripped.upper() or 'IMPORTANT' in stripped.upper():
                current_priority = 2
            else:
                current_priority = 1
            
            # Parse rules (numbered, bulleted, or plain)
            if re.match(r'^(\d+\.|[-*])\s+', stripped):
                # Save previous rule
                if current_rule:
                    rule_text = ' '.join(current_rule)
                    if len(rule_text) > 10:  # Minimum rule length
                        self.rules.append(Rule(
                            name=f"Rule {len(self.rules) + 1}",
                            content=rule_text,
                            priority=current_priority,
                            category=current_category
                        ))
                
                # Start new rule
                current_rule = [re.sub(r'^(\d+\.|[-*])\s+', '', stripped)]
            elif current_rule and stripped:
                # Continue current rule
                current_rule.append(stripped)
        
        # Don't forget last rule
        if current_rule:
            rule_text = ' '.join(current_rule)
            if len(rule_text) > 10:
                self.rules.append(Rule(
                    name=f"Rule {len(self.rules) + 1}",
                    content=rule_text,
                    priority=current_priority,
                    category=current_category
                ))
    
    def _create_default_skills(self):
        """Create default skills if none exist."""
        default_skills = {
            "code_review": Skill(
                name="Code Review",
                description="Analyze and review code for quality, bugs, and improvements",
                commands=["analyze", "review"],
                examples=["Review my Python code", "Check for bugs"]
            ),
            "git_operations": Skill(
                name="Git Operations",
                description="Execute git commands and generate commit messages",
                commands=["git", "commit-msg", "pr-desc"],
                examples=["Generate commit message", "Show git status"]
            ),
            "file_management": Skill(
                name="File Management",
                description="Read, analyze, and manage files",
                commands=["analyze", "cat", "explore"],
                examples=["Analyze main.py", "List files"]
            ),
            "ai_chat": Skill(
                name="AI Chat",
                description="Answer questions and have conversations",
                commands=["switch", "compare"],
                examples=["Explain quantum computing", "Write a poem"]
            ),
            "shell_commands": Skill(
                name="Shell Commands",
                description="Execute safe shell commands",
                commands=["run", "!"],
                examples=["!ls", "/run pwd"]
            )
        }
        
        self.skills = default_skills
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_skills_prompt(self) -> str:
        """Get skills as a system prompt addition.
        
        Returns:
            Formatted skills for inclusion in AI prompt.
        """
        if self.raw_skills_content:
            return f"## Your Skills\n\n{self.raw_skills_content}"
        
        lines = ["## Your Skills\n"]
        for skill in self.skills.values():
            if skill.enabled:
                lines.append(f"### {skill.name}")
                lines.append(f"{skill.description}")
                if skill.commands:
                    lines.append(f"Commands: {', '.join(skill.commands)}")
                lines.append("")
        
        return "\n".join(lines)
    
    def get_rules_prompt(self) -> str:
        """Get rules as a system prompt addition.
        
        Returns:
            Formatted rules for inclusion in AI prompt.
        """
        if self.raw_rules_content:
            return f"## Rules to Follow\n\n{self.raw_rules_content}"
        
        if not self.rules:
            return ""
        
        lines = ["## Rules to Follow\n"]
        
        # Sort by priority (high first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if rule.enabled:
                priority_marker = "⚠️ " if rule.priority == 3 else ""
                lines.append(f"- {priority_marker}{rule.content}")
        
        return "\n".join(lines)
    
    def get_system_context(self) -> str:
        """Get combined skills and rules as system context.
        
        Returns:
            Combined context string for AI prompts.
        """
        parts = []
        
        skills_prompt = self.get_skills_prompt()
        if skills_prompt:
            parts.append(skills_prompt)
        
        rules_prompt = self.get_rules_prompt()
        if rules_prompt:
            parts.append(rules_prompt)
        
        return "\n\n".join(parts)
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List all loaded skills.
        
        Returns:
            List of skill dictionaries.
        """
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "commands": skill.commands,
                "enabled": skill.enabled
            }
            for skill in self.skills.values()
        ]
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """List all loaded rules.
        
        Returns:
            List of rule dictionaries.
        """
        return [
            {
                "name": rule.name,
                "content": rule.content[:100] + "..." if len(rule.content) > 100 else rule.content,
                "priority": rule.priority,
                "category": rule.category,
                "enabled": rule.enabled
            }
            for rule in self.rules
        ]
    
    def format_skills_list(self) -> str:
        """Format skills list for display.
        
        Returns:
            Formatted string.
        """
        lines = ["🧠 **Loaded Skills:**\n"]
        
        for skill in self.skills.values():
            status = "✅" if skill.enabled else "❌"
            lines.append(f"{status} **{skill.name}**")
            lines.append(f"   {skill.description}")
            if skill.commands:
                lines.append(f"   Commands: `{', '.join(skill.commands)}`")
            lines.append("")
        
        lines.append(f"\n📁 Skills file: {self._find_file(self.SKILLS_FILES) or 'Default (create SKILLS.md to customize)'}")
        
        return "\n".join(lines)
    
    def format_rules_list(self) -> str:
        """Format rules list for display.
        
        Returns:
            Formatted string.
        """
        if not self.rules:
            return "📋 No custom rules loaded.\n   Create a RULES.md file in your project root to add rules."
        
        lines = ["📋 **Active Rules:**\n"]
        
        for i, rule in enumerate(self.rules, 1):
            priority_icon = {"1": "🔵", "2": "🟡", "3": "🔴"}.get(str(rule.priority), "⚪")
            content = rule.content[:80] + "..." if len(rule.content) > 80 else rule.content
            lines.append(f"{priority_icon} {i}. {content}")
        
        lines.append(f"\n📁 Rules file: {self._find_file(self.RULES_FILES) or 'Not found'}")
        
        return "\n".join(lines)
    
    def reload(self):
        """Reload skills and rules from files."""
        self.skills = {}
        self.rules = []
        self.raw_skills_content = ""
        self.raw_rules_content = ""
        self._load_skills()
        self._load_rules()
    
    def create_skills_file(self, filepath: Optional[str] = None) -> str:
        """Create a template skills file.
        
        Args:
            filepath: Path for new file. Defaults to ./SKILLS.md
            
        Returns:
            Path to created file.
        """
        filepath = filepath or str(self.project_dir / "SKILLS.md")
        
        template = '''# AetherAI Skills

This file defines the skills and capabilities for AetherAI in this project.
The AI will follow these skills when responding to your requests.

## Code Review

Analyze and review code for:
- Code quality and style
- Potential bugs and issues
- Performance optimizations
- Security vulnerabilities

### Commands
- `/analyze <file>` - Analyze a file
- `/review` - Review code

### Examples
- "Review my code for bugs"
- "Analyze the security of this function"

## Documentation

Generate and improve documentation:
- Function docstrings
- README files
- API documentation
- Code comments

### Commands
- `/docs <file>` - Generate documentation

### Examples
- "Write docstrings for this module"
- "Create a README for this project"

## Testing

Generate and improve tests:
- Unit tests
- Integration tests
- Edge cases

### Commands
- `/generate-tests <file>` - Generate tests

### Examples
- "Write tests for the auth module"
- "Add edge case tests"

## Refactoring

Suggest code improvements:
- Better naming
- Cleaner structure
- Performance optimization
- Modern patterns

### Commands
- `/refactor` - Suggest refactoring

### Examples
- "How can I improve this code?"
- "Make this more Pythonic"
'''
        
        Path(filepath).write_text(template, encoding='utf-8')
        return filepath
    
    def create_rules_file(self, filepath: Optional[str] = None) -> str:
        """Create a template rules file.
        
        Args:
            filepath: Path for new file. Defaults to ./RULES.md
            
        Returns:
            Path to created file.
        """
        filepath = filepath or str(self.project_dir / "RULES.md")
        
        template = '''# AetherAI Rules

This file defines rules and guidelines that AetherAI MUST follow in this project.
These rules take priority over general behavior.

## Code Style

1. Use 4 spaces for indentation, never tabs
2. Maximum line length is 100 characters
3. Use snake_case for functions and variables
4. Use PascalCase for class names
5. Always add type hints to function signatures

## Security

⚠️ CRITICAL: Never expose API keys or secrets in code or responses
⚠️ CRITICAL: Always validate user input before processing
- Use parameterized queries for database operations
- Sanitize file paths to prevent path traversal

## Documentation

- Every public function MUST have a docstring
- Use Google-style docstrings
- Include examples in docstrings where helpful

## Testing

- All new functions SHOULD have corresponding tests
- Aim for 80% code coverage minimum
- Test edge cases and error conditions

## Git

- Use conventional commit messages (feat:, fix:, docs:, etc.)
- Keep commits focused and atomic
- Write meaningful commit descriptions

## Project Specific

- This is a Python project using asyncio
- Use `rich` for console output
- Follow PEP 8 guidelines
'''
        
        Path(filepath).write_text(template, encoding='utf-8')
        return filepath
