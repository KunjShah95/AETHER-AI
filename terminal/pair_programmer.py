"""
AI Pair Programming Assistant for AetherAI.

This module provides:
- Real-time code suggestions
- Interactive coding sessions
- Code completion
- Error detection and fixes
- Refactoring suggestions
- Documentation generation
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class CodeSuggestion:
    """Represents a code suggestion."""
    suggestion_type: str  # completion, fix, refactor, doc
    original_code: str
    suggested_code: str
    description: str
    confidence: float = 0.8
    line_number: int = 0
    accepted: bool = False


@dataclass
class CodingSession:
    """Represents an interactive coding session."""
    session_id: str
    filepath: str
    language: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    suggestions: List[CodeSuggestion] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    auto_suggest: bool = True


class PairProgrammer:
    """AI-powered pair programming assistant."""
    
    # Language-specific completions
    LANGUAGE_PATTERNS = {
        'python': {
            'function': r'^def\s+(\w+)\s*\(',
            'class': r'^class\s+(\w+)',
            'import': r'^import\s+|^from\s+',
            'comment': r'^\s*#',
            'docstring': r'^\s*"""',
        },
        'javascript': {
            'function': r'function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*\(',
            'class': r'^class\s+(\w+)',
            'import': r'^import\s+|^const.*require',
            'comment': r'^\s*//',
        },
        'typescript': {
            'function': r'function\s+(\w+)|:\s*\([^)]*\)\s*=>',
            'class': r'^class\s+(\w+)',
            'interface': r'^interface\s+(\w+)',
            'import': r'^import\s+',
        }
    }
    
    # Common code patterns to suggest
    CODE_TEMPLATES = {
        'python': {
            'try_except': '''try:
    {code}
except Exception as e:
    logging.error(f"Error: {e}")
    raise''',
            'with_file': '''with open("{filename}", "r") as f:
    content = f.read()''',
            'class_init': '''def __init__(self, {params}):
    {body}''',
            'async_function': '''async def {name}({params}):
    """
    {docstring}
    """
    {body}''',
            'dataclass': '''@dataclass
class {name}:
    """
    {docstring}
    """
    {fields}''',
            'test_function': '''def test_{name}():
    """Test {description}."""
    # Arrange
    {arrange}
    
    # Act
    result = {act}
    
    # Assert
    assert {assertion}''',
        },
        'javascript': {
            'try_catch': '''try {
    {code}
} catch (error) {
    console.error('Error:', error);
    throw error;
}''',
            'async_function': '''async function {name}({params}) {
    try {
        {body}
    } catch (error) {
        console.error(error);
        throw error;
    }
}''',
            'arrow_function': '''const {name} = ({params}) => {
    {body}
};''',
            'react_component': '''const {name} = ({ {props} }) => {
    return (
        <div>
            {content}
        </div>
    );
};

export default {name};''',
        }
    }
    
    def __init__(self, ai_query_func: Callable = None):
        """Initialize pair programmer.
        
        Args:
            ai_query_func: Function to query AI models.
        """
        self.ai_query = ai_query_func
        self.sessions: Dict[str, CodingSession] = {}
        self.current_session: Optional[CodingSession] = None
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def start_session(self, filepath: str) -> CodingSession:
        """Start a new coding session.
        
        Args:
            filepath: File to work on.
            
        Returns:
            New CodingSession.
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        language = self._detect_language(filepath)
        
        session = CodingSession(
            session_id=session_id,
            filepath=filepath,
            language=language
        )
        
        self.sessions[session_id] = session
        self.current_session = session
        
        return session
    
    def end_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """End a coding session.
        
        Args:
            session_id: Session to end.
            
        Returns:
            Session summary.
        """
        session = self._get_session(session_id)
        if not session:
            return {"error": "No active session"}
        
        summary = {
            "session_id": session.session_id,
            "filepath": session.filepath,
            "duration": str(datetime.now() - datetime.fromisoformat(session.created_at)),
            "suggestions_made": len(session.suggestions),
            "suggestions_accepted": sum(1 for s in session.suggestions if s.accepted)
        }
        
        if session == self.current_session:
            self.current_session = None
        
        del self.sessions[session.session_id]
        
        return summary
    
    def _get_session(self, session_id: Optional[str] = None) -> Optional[CodingSession]:
        """Get a session by ID or current session."""
        if session_id:
            return self.sessions.get(session_id)
        return self.current_session
    
    def _detect_language(self, filepath: str) -> str:
        """Detect programming language from filepath."""
        ext = Path(filepath).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
        }
        return language_map.get(ext, 'unknown')
    
    # =========================================================================
    # Code Suggestions
    # =========================================================================
    
    def suggest_completion(self, code: str, cursor_line: int = 0,
                          model: str = "gemini") -> List[CodeSuggestion]:
        """Suggest code completions.
        
        Args:
            code: Current code.
            cursor_line: Line number of cursor.
            model: AI model to use.
            
        Returns:
            List of suggestions.
        """
        if not self.ai_query:
            return self._suggest_from_templates(code)
        
        session = self._get_session()
        language = session.language if session else self._guess_language(code)
        
        prompt = f"""You are a coding assistant. Suggest code completions for the following code.
The cursor is at line {cursor_line}.

Language: {language}

Code:
```{language}
{code}
```

Provide 1-3 completion suggestions. For each suggestion, give:
1. The code to insert
2. A brief description

Format each suggestion as:
SUGGESTION:
```
<code>
```
DESCRIPTION: <description>
"""
        
        try:
            response = self.ai_query(model, prompt)
            suggestions = self._parse_suggestions(response, code, "completion")
            
            if session:
                session.suggestions.extend(suggestions)
            
            return suggestions
        except Exception as e:
            return []
    
    def suggest_fix(self, code: str, error_message: str,
                   model: str = "gemini") -> List[CodeSuggestion]:
        """Suggest fixes for an error.
        
        Args:
            code: Code with error.
            error_message: Error message.
            model: AI model to use.
            
        Returns:
            Fix suggestions.
        """
        if not self.ai_query:
            return []
        
        session = self._get_session()
        language = session.language if session else self._guess_language(code)
        
        prompt = f"""Fix this {language} code error.

Error:
{error_message}

Code:
```{language}
{code}
```

Provide the fixed code and explain what was wrong.

Format:
FIXED CODE:
```{language}
<fixed code>
```
EXPLANATION: <what was wrong and how it was fixed>
"""
        
        try:
            response = self.ai_query(model, prompt)
            suggestions = self._parse_suggestions(response, code, "fix")
            
            if session:
                session.suggestions.extend(suggestions)
            
            return suggestions
        except Exception:
            return []
    
    def suggest_refactor(self, code: str, instruction: str = "",
                        model: str = "gemini") -> List[CodeSuggestion]:
        """Suggest code refactoring.
        
        Args:
            code: Code to refactor.
            instruction: Optional specific instruction.
            model: AI model to use.
            
        Returns:
            Refactoring suggestions.
        """
        if not self.ai_query:
            return []
        
        session = self._get_session()
        language = session.language if session else self._guess_language(code)
        
        instruction_text = instruction or "Improve code quality, readability, and performance"
        
        prompt = f"""Refactor this {language} code. {instruction_text}

Code:
```{language}
{code}
```

Provide the refactored code with explanations.

Format:
REFACTORED:
```{language}
<refactored code>
```
CHANGES:
- <change 1>
- <change 2>
"""
        
        try:
            response = self.ai_query(model, prompt)
            suggestions = self._parse_suggestions(response, code, "refactor")
            
            if session:
                session.suggestions.extend(suggestions)
            
            return suggestions
        except Exception:
            return []
    
    def generate_docstring(self, code: str, 
                           model: str = "gemini") -> CodeSuggestion:
        """Generate documentation for code.
        
        Args:
            code: Code to document.
            model: AI model to use.
            
        Returns:
            Documentation suggestion.
        """
        if not self.ai_query:
            return None
        
        session = self._get_session()
        language = session.language if session else self._guess_language(code)
        
        # Determine documentation style
        doc_style = "Google-style docstring" if language == "python" else "JSDoc"
        
        prompt = f"""Generate {doc_style} documentation for this {language} code:

```{language}
{code}
```

Output the complete code with documentation added."""
        
        try:
            response = self.ai_query(model, prompt)
            
            # Clean up response
            if '```' in response:
                lines = response.split('\n')
                in_code = False
                code_lines = []
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code = not in_code
                        continue
                    if in_code:
                        code_lines.append(line)
                documented_code = '\n'.join(code_lines)
            else:
                documented_code = response
            
            suggestion = CodeSuggestion(
                suggestion_type="doc",
                original_code=code,
                suggested_code=documented_code,
                description="Added documentation"
            )
            
            if session:
                session.suggestions.append(suggestion)
            
            return suggestion
        except Exception:
            return None
    
    def _suggest_from_templates(self, code: str) -> List[CodeSuggestion]:
        """Generate suggestions from built-in templates."""
        suggestions = []
        language = self._guess_language(code)
        
        templates = self.CODE_TEMPLATES.get(language, {})
        
        # Check what the user might be writing
        lines = code.strip().split('\n')
        last_line = lines[-1] if lines else ""
        
        # Suggest try/except if function has risk
        if 'open(' in code or 'request' in code.lower():
            if language == 'python' and 'try:' not in code:
                template = templates.get('try_except', '')
                if template:
                    suggestions.append(CodeSuggestion(
                        suggestion_type="completion",
                        original_code=code,
                        suggested_code=template.replace('{code}', '    # Your code here'),
                        description="Wrap in try/except for error handling"
                    ))
        
        return suggestions
    
    def _parse_suggestions(self, response: str, original_code: str,
                          suggestion_type: str) -> List[CodeSuggestion]:
        """Parse suggestions from AI response."""
        suggestions = []
        
        # Find code blocks
        code_blocks = re.findall(r'```[\w]*\n(.*?)```', response, re.DOTALL)
        descriptions = re.findall(r'(?:DESCRIPTION|EXPLANATION|CHANGES):\s*(.*?)(?=\n\n|$)', response, re.DOTALL)
        
        for i, code_block in enumerate(code_blocks):
            description = descriptions[i] if i < len(descriptions) else f"{suggestion_type.title()} suggestion"
            
            suggestions.append(CodeSuggestion(
                suggestion_type=suggestion_type,
                original_code=original_code,
                suggested_code=code_block.strip(),
                description=description.strip()
            ))
        
        return suggestions
    
    def _guess_language(self, code: str) -> str:
        """Guess programming language from code content."""
        if 'def ' in code and ':' in code:
            return 'python'
        if 'function ' in code or '=>' in code:
            return 'javascript'
        if 'interface ' in code or ': string' in code or ': number' in code:
            return 'typescript'
        if '#include' in code:
            return 'c'
        if 'func ' in code:
            return 'go'
        if 'fn ' in code:
            return 'rust'
        return 'unknown'
    
    # =========================================================================
    # Interactive Features
    # =========================================================================
    
    def explain_code(self, code: str, model: str = "gemini") -> str:
        """Explain what code does.
        
        Args:
            code: Code to explain.
            model: AI model.
            
        Returns:
            Explanation.
        """
        if not self.ai_query:
            return "AI query not available"
        
        language = self._guess_language(code)
        
        prompt = f"""Explain this {language} code in detail. Include:
1. What it does
2. How it works step by step
3. Key concepts used
4. Potential issues or improvements

Code:
```{language}
{code}
```"""
        
        return self.ai_query(model, prompt)
    
    def review_code(self, code: str, model: str = "gemini") -> Dict[str, Any]:
        """Review code for issues.
        
        Args:
            code: Code to review.
            model: AI model.
            
        Returns:
            Review results.
        """
        if not self.ai_query:
            return {"error": "AI query not available"}
        
        language = self._guess_language(code)
        
        prompt = f"""Review this {language} code and provide feedback on:
1. Code quality (1-10 score)
2. Bugs or issues found
3. Security concerns
4. Performance issues
5. Suggestions for improvement

Format your response as JSON:
{{
    "quality_score": 8,
    "bugs": ["bug 1", "bug 2"],
    "security_issues": ["issue 1"],
    "performance_issues": ["issue 1"],
    "suggestions": ["suggestion 1", "suggestion 2"]
}}

Code:
```{language}
{code}
```"""
        
        try:
            response = self.ai_query(model, prompt)
            
            # Try to parse JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                import json
                return json.loads(json_match.group())
            
            return {"raw_review": response}
        except Exception as e:
            return {"error": str(e)}
    
    def get_next_step(self, code: str, context: str = "",
                      model: str = "gemini") -> str:
        """Suggest what to code next.
        
        Args:
            code: Current code.
            context: Additional context.
            model: AI model.
            
        Returns:
            Next step suggestion.
        """
        if not self.ai_query:
            return "AI query not available"
        
        language = self._guess_language(code)
        
        prompt = f"""Based on this {language} code, suggest what to implement next.

Current code:
```{language}
{code}
```

{f'Context: {context}' if context else ''}

Suggest the next logical step in development and provide a code snippet to get started."""
        
        return self.ai_query(model, prompt)
    
    # =========================================================================
    # Formatting
    # =========================================================================
    
    def format_suggestion(self, suggestion: CodeSuggestion) -> str:
        """Format a suggestion for display.
        
        Args:
            suggestion: Suggestion to format.
            
        Returns:
            Formatted string.
        """
        icon = {
            "completion": "💡",
            "fix": "🔧",
            "refactor": "♻️",
            "doc": "📝"
        }.get(suggestion.suggestion_type, "💭")
        
        lines = [
            f"{icon} **{suggestion.suggestion_type.title()} Suggestion**",
            f"_{suggestion.description}_",
            "",
            "```",
            suggestion.suggested_code[:500] + ("..." if len(suggestion.suggested_code) > 500 else ""),
            "```"
        ]
        
        return "\n".join(lines)
    
    def format_session_status(self) -> str:
        """Format current session status.
        
        Returns:
            Formatted string.
        """
        session = self.current_session
        if not session:
            return "📝 No active coding session. Use `/pair start <file>` to begin."
        
        lines = [
            f"👥 **Pair Programming Session**",
            f"📄 File: {session.filepath}",
            f"🔤 Language: {session.language}",
            f"⏰ Started: {session.created_at}",
            f"💡 Suggestions made: {len(session.suggestions)}",
            f"✅ Accepted: {sum(1 for s in session.suggestions if s.accepted)}",
            "",
            f"🔄 Auto-suggest: {'Enabled' if session.auto_suggest else 'Disabled'}"
        ]
        
        return "\n".join(lines)
