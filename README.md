# 🚀 AETHER AI Terminal Assistant

[![PyPI version](https://img.shields.io/pypi/v/aetherai?color=blue&logo=python)](https://pypi.org/project/aetherai/)
[![Python](https://img.shields.io/pypi/pyversions/aetherai)](https://pypi.org/project/aetherai/)
[![Downloads](https://img.shields.io/pypi/dm/aetherai)](https://pypi.org/project/aetherai/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://kunjshah95.github.io/AETHER-AI.io/)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Production‑ready, secure, multi‑model AI for your terminal. Switch between Google Gemini, Groq, Ollama (local), HuggingFace, ChatGPT (OpenAI), and MCP with one CLI, enriched by security controls, utilities, and an extensible modular architecture.

---

## Highlights

- Multi‑Model Switching: gemini, groq, ollama, huggingface, chatgpt, mcp
- Local AI via Ollama: list and select installed models with specs
- Secure by Default: input sanitization, safe command allowlist, file boundary checks
- Powerful Utilities: web search, sysinfo, notes, timers, unit conversions, quick weather
- Developer Tools: code review assistant, refactor helper, TODO extractor, Git helpers
- Productivity Suite: tasks manager, themes, reminders, analytics, games & learning
- Cross‑Platform: Windows, macOS, Linux

---

## Prerequisites

- Python 3.9+
- Internet access for cloud models (Gemini/Groq/HF/ChatGPT/MCP)
- Optional: [Ollama](https://ollama.ai) installed and running for local models

---

## Installation

### 🚀 Quick Install (Recommended)

```bash
# Using pip
pip install aetherai

# Using uv (faster!)
uv pip install aetherai
```

After installation, run the assistant:

```bash
aetherai
# or
aether-ai
```

### 📦 Installation Extras

Choose the features you need:

```bash
# Lightweight install (core only - ~100MB)
pip install aetherai

# Full experience (voice, RAG, ML, server - like original)
pip install "aetherai[full]"

# Specific features
pip install "aetherai[voice]"      # Voice input/output
pip install "aetherai[rag]"        # RAG knowledge base
pip install "aetherai[ml]"         # HuggingFace transformers
pip install "aetherai[server]"     # FastAPI web server
pip install "aetherai[db]"         # Database integrations

# Development
pip install "aetherai[dev]"        # Testing & linting tools
pip install "aetherai[all]"        # Everything including dev tools
```

### 🔧 Non-OS scripts (consolidated)

- We also provide platform installers under `dist/`:
  - Windows: `dist/install_windows.bat`
  - macOS: `dist/install_mac.sh`
  - Linux: `dist/install_linux.sh`

### 🛠️ From source (for development)

```bash
# Clone
git clone https://github.com/KunjShah95/AETHER-AI.io.git
cd AETHER-AI.io

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env   # set your API keys; see below

# Run
aetherai
# or directly
python terminal/main.py
```

---

## Configure API Keys

Create and edit a `.env` file at the project root with the following variables:

```env
GEMINI_API_KEY=...
GROQ_API_KEY=...
HUGGINGFACE_TOKEN=...
OPENAI_API_KEY=...
MCP_API_KEY=...
```

- Ollama does not require an API key but must be installed and the daemon running.
- Per‑user config and data files live in:
  - Windows: `%USERPROFILE%\.aether`
  - macOS/Linux: `~/.aether`

Files created:

- ~/.aether/config.yaml (default model)
- ~/.aether/users.json (local users)
- ~/.aether/notes.txt (quick notes)

---

## Quick Start

```bash
python terminal/main.py
```

- Type `/help` to see the full command catalog.
- Switch models anytime with `/switch [model]` (e.g., `/switch groq`).
- Use `/ollama-models` to list your local Ollama models with specifications.

---

## Core Usage

A compact subset of the most useful commands. Use `/help` for the complete list.

- Models & status
  - `/models` — list available AI backends and states
  - `/current-model` — show the active model and description
  - `/switch gemini|groq|ollama|huggingface|chatgpt|mcp` — change model
  - `/ollama-models` — summary table of installed Ollama models (name, size, params, family, quant, modified)
  - `/ollama-models <name>` — detailed specs for a single Ollama model
- System & utilities
  - `/run <safe-cmd>` — run allowlisted commands (ls, pwd, whoami, date, cat, head, tail, df, du, free, uname, id, git)
  - `/sysinfo` — OS, CPU, memory, disk stats
  - `/websearch <query>` — quick DuckDuckGo search
  - `/weather <city>` — current weather
  - `/note <text>` and `/notes` — quick note and list
  - `/timer <seconds>` — countdown with notification
  - `/calc <expr>` — evaluate math expressions safely
  - `/convert <val> <from> <to>` — quick unit conversions (temp/weight/data)
- Developer helpers
  - `/review ...`, `/codereview <file>`, `/refactor <file> <instruction>`
  - `/todos`, `/gendoc <file>`, `/gentest <file>`
  - Git: `/git status`, `/git add`, `/git commit`, `/git log`, `/git diff`, `/git branch`, `/git checkout`, `/git repo-info`, `/git stats` …
- Productive UX
  - Tasks: `/task add <title>`, `/tasks`, `/task complete <id>`, `/task stats` …
  - Themes: `/themes`, `/theme set <name>`, `/theme current`, `/theme reset`
  - Learning & games: `/challenge <difficulty>`, `/quiz <topic>`
  - Analytics & monitoring: `/analytics`, `/error-analytics`, `/start-monitoring`, `/health`

---

## Ollama Integration

- Install Ollama: <https://ollama.ai>
- Pull a model (examples): `ollama pull llama3`, `ollama pull mistral`
- In AETHER AI:
  - `/ollama-models` — lists installed models with size, family, params, quantization, modified date
  - `/ollama-models llama3:8b` — full specs for a single model
  - `/switch ollama llama3:8b` — use a specific local model

Troubleshooting:

- If you see “Ollama not available,” ensure the daemon is running (e.g., `ollama serve`) and models are pulled.

---

## Security Model

- Input sanitization blocks dangerous patterns and suspicious Unicode.
- Safe command allowlist for `/run` (no shells/pipes/redirection/wildcards; current dir boundary enforced for file args).
- API key format validation for major providers.
- Config and user DB stored per user in `~/.aether` with restricted permissions where supported.

---

## Packaging for Production

You can ship a single-file binary for operators who don’t want Python on the host.

Example with PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile --name aetherai terminal/main.py
# Output will be in the dist/ directory as 'aetherai' (or aetherai.exe on Windows)
```

Distribute the binary and provide instructions to set environment variables (or bundle an `.env`).

---

## Troubleshooting

- Python not found:
  - Install Python 3.9+ and ensure `python --version` works in your shell.
- Dependencies fail to install:
  - `pip install --upgrade pip`
  - `pip install -r requirements.txt --no-cache-dir`
- API key errors:
  - Ensure `.env` exists and keys are valid; try a different provider with `/switch`.
- Ollama issues:
  - Make sure the Ollama service is running and models are pulled. Test with `ollama list`.
- Voice input:
  - Requires optional `speech_recognition` (and OS-specific audio dependencies). If unavailable, `/voice` will show guidance.

---

## Project Structure

```text
AETHER-AI.io/
├── index.html                # Website homepage
├── app.js                    # Website functionality
├── styles.css                # Website styles
├── CHANGELOG.md              # Version history
├── terminal/                 # Main application (CLI)
│   ├── main.py               # Entry point
│   ├── requirements.txt      # Dependencies
│   ├── install_*.{bat,sh}    # OS installers (root also contains installers)
│   └── tests/                # Unit tests
├── advanced_features.py
├── analytics_monitor.py
├── code_review_assistant.py
├── context_aware_ai.py
├── creative_tools.py
├── games_learning.py
├── advanced_security.py
├── task_manager.py
├── theme_manager.py
├── integration_hub.py
└── README.md
```

---

## Release & Governance

- Versioning: Semantic versioning tracked in `CHANGELOG.md` (badge shows current).
- Security: See `SECURITY.md` for reporting and policies.
- Code of Conduct: See `CODE_OF_CONDUCT.md`.
- Contribution: See `CONTRIBUTING.md`.

---

## Contributing

We welcome feature ideas, bug reports, and PRs.

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Install deps: `pip install -r requirements.txt`
4. Commit and push: `git commit -m "feat: add your feature"`
5. Open a Pull Request with a clear description and context

---

## License

MIT License — see [LICENSE](LICENSE)

---

---

## Deployment

### Website Deployment

The AETHER AI website (frontend) is ready for deployment to Vercel:

- **Quick Deploy**: [![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/KunjShah95/AETHER-AI.io)
- **Detailed Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md) for complete instructions
- **Configuration**: All necessary files (`vercel.json`, optimized `vite.config.ts`) are included

The website features:

- Optimized production build with code splitting
- SEO-ready with proper meta tags
- SPA routing configured
- Static asset caching

---

## Support & Links

- Website: [Project website](https://kunjshah95.github.io/AETHER-AI.io/)
- Issues: [Issue tracker](https://github.com/KunjShah95/AETHER-AI.io/issues)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Security Policy: [SECURITY.md](SECURITY.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Contributing Guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Deployment Guide: [DEPLOYMENT.md](DEPLOYMENT.md)

If this project helps you, consider starring the repo. It helps others find it too.
