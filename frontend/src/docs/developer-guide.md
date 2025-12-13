# Developer Guide

We welcome contributions to AETHER AI! This guide will help you set up your development environment and understand the project structure.

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Installation from Source

1. **Clone the repository**:

    ```bash
    git clone https://github.com/KunjShah95/NEXUS-AI.io.git
    cd NEXUS-AI.io
    ```

2. **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment**:
    Copy `.env.example` to `.env` and add your API keys.

    ```bash
    cp .env.example .env
    ```

5. **Run the application**:

    ```bash
    python terminal/main.py
    ```

## Project Structure

```text
NEXUS-AI.io/
├── index.html                # Website homepage
├── app.js                    # Website functionality
├── styles.css                # Website styles
├── CHANGELOG.md              # Version history
├── terminal/                 # Main application (CLI)
│   ├── main.py               # Entry point
│   ├── requirements.txt      # Dependencies
│   ├── install_*.{bat,sh}    # OS installers
│   └── tests/                # Unit tests
├── advanced_features.py      # Advanced features module
├── analytics_monitor.py      # Analytics module
├── code_review_assistant.py  # Code review module
├── context_aware_ai.py       # Context awareness module
├── creative_tools.py         # Creative tools module
├── games_learning.py         # Games & learning module
├── advanced_security.py      # Security module
├── task_manager.py           # Task management module
├── theme_manager.py          # Theme management module
├── integration_hub.py        # Integration module
└── README.md                 # Project README
```

## Contributing Workflow

1. **Fork the repo** on GitHub.
2. **Create a branch** for your feature or fix:

    ```bash
    git checkout -b feat/your-feature-name
    ```

3. **Make your changes**. Ensure code quality and add tests if applicable.
4. **Commit your changes**:

    ```bash
    git commit -m "feat: add new feature"
    ```

5. **Push to your fork**:

    ```bash
    git push origin feat/your-feature-name
    ```

6. **Open a Pull Request** on the main repository.

## Code Style

- Follow PEP 8 guidelines for Python code.
- Use meaningful variable and function names.
- Add comments and docstrings where necessary.

## Testing

Run unit tests to ensure your changes don't break existing functionality.

```bash
python -m pytest terminal/tests/
```
