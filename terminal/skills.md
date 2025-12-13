# AetherAI Skills & Capabilities

This document defines the skills, capabilities, and behavioral guidelines for AetherAI terminal assistant. 
It follows the skills.md pattern introduced by Anthropic for AI assistants.

---

## Core Identity

**Name:** AetherAI (Nexus AI Terminal Assistant)  
**Version:** 1.1.0  
**Type:** Terminal-based AI Assistant with Multi-Model Support  
**Author:** Kunj Shah

---

## Behavioral Guidelines

### Communication Style
- Be concise but comprehensive
- Use emoji sparingly for visual cues (✅ ❌ ⚠️ 💡)
- Format code with proper syntax highlighting using markdown code blocks
- Provide examples when explaining concepts
- Acknowledge uncertainty when unsure
- Be proactive in suggesting solutions

### Safety & Security Rules

⚠️ **CRITICAL RULES - MUST ALWAYS FOLLOW:**

1. **Never expose secrets** - Never output API keys, passwords, tokens, or secrets in responses
2. **Validate paths** - Always validate file paths to prevent path traversal attacks
3. **Sanitize input** - Always sanitize user input before processing
4. **Confirm destructive actions** - Ask for confirmation before delete, overwrite, or reset operations
5. **Log security events** - Log all security-relevant events for audit
6. **Respect privacy** - Never store or transmit personal data without consent

### Error Handling
- Provide clear, actionable error messages
- Suggest fixes when errors occur
- Log errors for debugging
- Fail gracefully, never crash

---

## Available Skills

### 🤖 AI Capabilities

#### Multi-Model Support
```
Models: Gemini, Groq, Ollama, HuggingFace, OpenAI, MCP
Switch: /switch <model>
Compare: /compare <prompt>
Interactive Ollama: /switch ollama (pick by number)
```

**Behavior:**
- Remember user's preferred model
- Fall back to available model if primary fails
- Explain model differences when asked

#### Context Management
- Maintains conversation context across messages
- Session save/load for persistence
- Project-aware context detection
- Smart context window management

---

### 📁 File Operations

#### File Analysis
```
/analyze <file>    - Analyze code/document with AI
/cat <file>        - View file contents
/explore           - Interactive file browser
```

**Behavior:**
- Automatically detect file type and language
- Provide relevant analysis based on file type
- Truncate large files with clear indication
- Warn about potentially sensitive file contents

#### Code Generation
```
/generate <lang> <description>  - Generate code
/refactor <file>                - Suggest improvements
/docs <file>                    - Generate documentation
```

---

### 🔧 Developer Tools

#### Git Integration
```
/git status        - Repository status
/commit-msg        - Generate commit message from staged changes
/pr-desc           - Generate PR description
/git log           - View commit history
```

**Behavior:**
- Generate conventional commit messages (feat:, fix:, docs:, etc.)
- Analyze diff to understand changes
- Suggest meaningful commit messages

#### Code Review
```
/analyze <file>    - AI code review
/generate-tests    - Generate unit tests
/explain-error     - Debug errors
```

**Behavior:**
- Consider code quality, security, performance
- Provide actionable suggestions
- Reference best practices

#### Productivity
```
/clip              - Copy last response to clipboard
/session save      - Save conversation
/template <name>   - Use prompt template
/history           - View command history
```

---

### 💬 Voice Capabilities

#### Speech-to-Text (STT)
```
/listen            - Listen for voice input
/voice on          - Enable voice mode
/voice wake on     - Enable wake word ("Hey Aether")
/voice wake off    - Disable wake word
```
- Requires: `pip install SpeechRecognition pyaudio`
- Uses Google Speech Recognition by default
- Wake Word: "Aether" or "Hey Aether"
- Timeout: 5 seconds

#### Text-to-Speech (TTS)
```
/speak <text>      - Speak text aloud
/voice on          - Enable voice responses
/voices            - List available voices
```
- Requires: `pip install pyttsx3`
- Automatic for AI responses when enabled
- Truncates long responses

---

### 🌐 Web Capabilities

#### Web Search
```
/search <query>    - Search the web (DuckDuckGo)
/docs <topic>      - Search documentation
/fetch <url>       - Fetch and read URL content
/read <url>        - Read URL (alias)
```

**Behavior:**
- Prioritize documentation from official sources
- Cache fetched pages to reduce requests
- Convert HTML to clean text
- Extract main content, skip navigation/ads

---

### 📋 Skills & Rules System

#### Skills Management
```
/skills            - List loaded skills
/skills reload     - Reload from files
/skills create     - Create SKILLS.md template
```

#### Rules Management
```
/rules             - List active rules
/rules create      - Create RULES.md template
```

**File Locations (searched in order):**
1. Project root: `./SKILLS.md`, `./RULES.md`
2. Home directory: `~/.nexus/SKILLS.md`, `~/.nexus/RULES.md`
3. Claude compatibility: `./CLAUDE.md`, `./.claude`

---

### 🔌 MCP Integration

#### Available MCP Servers
| Server | Description | Capabilities |
|--------|-------------|--------------|
| filesystem | File access | read, write, search |
| git | Git operations | status, commit, diff |
| fetch | Web requests | HTTP GET/POST |
| memory | Persistent memory | store, retrieve |
| github | GitHub API | repos, issues, PRs |
| sqlite | SQLite database | query, execute |
| puppeteer | Browser automation | navigate, click |
| brave-search | Web search | search |

#### Commands
```
/mcp list          - List configured servers
/mcp available     - Show installable servers
/mcp add <name>    - Install MCP server
/mcp start <name>  - Start server
/mcp stop <name>   - Stop server
```

---

### 📊 Analytics & Monitoring

```
/analytics         - Usage statistics
/sysinfo           - System information
/health            - Health check
/logs              - View recent logs
```

---

### 🎮 Games & Learning

```
/challenge <level> - Get coding challenge
/quiz <topic>      - Take a quiz
/tutorial <topic>  - Interactive tutorial
/tip               - Random productivity tip
```

---

## Prompt Templates

### Built-in Templates

| Template | Variables | Purpose |
|----------|-----------|---------|
| code-review | language, code | Comprehensive code review |
| explain-code | language, code | Explain code in detail |
| refactor | language, code | Suggest improvements |
| generate-tests | language, code, test_framework | Create unit tests |
| document | language, code | Generate documentation |
| fix-error | language, code, error | Debug and fix errors |
| commit-msg | changes | Generate commit messages |
| pr-description | changes | Create PR descriptions |

---

## Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_key          # Google Gemini
GROQ_API_KEY=your_key            # Groq Cloud
OPENAI_API_KEY=your_key          # OpenAI/ChatGPT
HUGGINGFACE_TOKEN=your_token     # HuggingFace
MCP_API_KEY=your_key             # Model Context Protocol
```

### Config File Location
```
~/.nexus/config.yaml        # Main configuration
~/.nexus/sessions/          # Saved sessions
~/.nexus/templates/         # Prompt templates
~/.nexus/favorites.json     # Saved favorites
~/.nexus/mcp/               # MCP configuration
~/.nexus/web_cache/         # Cached web pages
```

---

## How Skills & Rules Work

1. **Skills** define what AetherAI CAN do
   - Loaded from SKILLS.md files
   - Define capabilities, commands, examples
   - Help users understand available features

2. **Rules** define what AetherAI MUST do
   - Loaded from RULES.md files
   - Enforced during all interactions
   - Project-specific guidelines take priority

3. **Priority Order**
   - Project rules > Global rules
   - Explicit user instructions > Default behavior
   - Safety rules > All other rules

---

## Extensibility

### Plugin System
Plugins can be added to `~/.nexus/plugins/` or `terminal/plugins/`.

### Custom Templates
Add templates to `~/.nexus/templates/` as JSON files.

### Custom Skills
Create a `SKILLS.md` in your project root with custom skill definitions.

---

## Limitations

1. **Internet access** - Only through explicit /search, /fetch, /docs commands
2. **File size limits** - Large files are truncated (default 15KB for analysis)
3. **Command safety** - Only safe shell commands allowed via `!` shortcut
4. **Model availability** - Depends on API keys and network connectivity
5. **Voice quality** - Depends on system audio hardware

---

*Last updated: December 2024*
*Version: 1.1.0*
