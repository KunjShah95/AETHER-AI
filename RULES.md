# AetherAI Rules

This file defines rules and guidelines that AetherAI MUST follow in this project.
These rules take priority over general behavior.

## Code Style

1. Use 4 spaces for indentation, never tabs
2. Maximum line length is 100 characters
3. Use snake_case for functions and variables
4. Use PascalCase for class names
5. Always add type hints to function signatures
6. Use docstrings for all public functions and classes

## Security

⚠️ CRITICAL: Never expose API keys or secrets in code or responses
⚠️ CRITICAL: Always validate and sanitize user input before processing
⚠️ CRITICAL: Never execute arbitrary code without explicit user confirmation

- Use parameterized queries for database operations
- Sanitize file paths to prevent path traversal
- Log security-relevant events

## Documentation

- Every public function MUST have a docstring
- Use Google-style docstrings (Args, Returns, Raises)
- Include examples in docstrings where helpful
- Keep documentation up to date with code changes

## Error Handling

- Provide clear, actionable error messages
- Include suggestions for fixing errors
- Log errors with appropriate severity levels
- Fail gracefully, never crash silently

## Testing

- All new functions SHOULD have corresponding tests
- Aim for 80% code coverage minimum
- Test edge cases and error conditions
- Use descriptive test names

## Git

- Use conventional commit messages (feat:, fix:, docs:, refactor:, test:, chore:)
- Keep commits focused and atomic
- Write meaningful commit descriptions
- Never commit sensitive data

## Performance

- Profile before optimizing
- Prefer readability over premature optimization
- Use appropriate data structures
- Consider memory usage for large datasets

## Project Specific

- This is a Python project using asyncio patterns
- Use `rich` library for console output
- Follow PEP 8 guidelines strictly
- Support Python 3.9+ compatibility
