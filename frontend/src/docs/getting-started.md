# Getting Started

## Prerequisites

- **Python 3.9+**
- Internet access for cloud models (Gemini/Groq/HF/ChatGPT/MCP)
- Optional: [Ollama](https://ollama.ai) installed and running for local models

## Installation

Choose the option that fits your environment.

### Option A: One‑click OS scripts (Recommended)

We provide a single canonical installer per platform under `dist/`.

- **Windows**: `dist/install_windows.bat`
- **macOS**: `dist/install_mac.sh`
- **Linux**: `dist/install_linux.sh`

Before running an installer, verify the SHA256 checksum in `dist/SHA256SUMS.txt`.

### Option B: From source (For Development)

```bash
# Clone
git clone https://github.com/KunjShah95/AETHER-AI.io.git
cd AETHER-AI.io

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env   # set your API keys; see below

# Run
python terminal/main.py
```

## Configuration

Create and edit a `.env` file at the project root with the following variables:

```env
GEMINI_API_KEY=...
GROQ_API_KEY=...
HUGGINGFACE_TOKEN=...
OPENAI_API_KEY=...
MCP_API_KEY=...
```

!!! note "Ollama"
    Ollama does not require an API key but must be installed and the daemon running.

Per‑user config and data files live in:

- **Windows**: `%USERPROFILE%\.aether`
- **macOS/Linux**: `~/.aether`

## Quick Start

Launch the terminal:

```bash
python terminal/main.py
```

- Type `/help` to see the full command catalog.
- Switch models anytime with `/switch [model]` (e.g., `/switch groq`).
- Use `/ollama-models` to list your local Ollama models with specifications.
