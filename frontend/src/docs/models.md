# AI Models

AETHER AI supports a wide range of AI models, allowing you to choose the best tool for the job.

## Supported Models

| Model | Command | Best For |
|-------|---------|----------|
| **Gemini 2.0 Flash** | `/switch gemini` | General tasks, coding, analysis. Google's fastest and most capable model. |
| **Groq Mixtral** | `/switch groq` | Speed-critical tasks. Ultra-fast inference. |
| **Ollama Local** | `/switch ollama [model]` | Privacy, offline use. Run models locally. |
| **HuggingFace** | `/switch huggingface` | Research, experimentation. Access to open-source models. |
| **ChatGPT** | `/switch chatgpt` | Creative writing, analysis. OpenAI's conversational AI. |
| **MCP Protocol** | `/switch mcp` | Advanced integrations. Model Context Protocol. |

## Managing Models

### Switching Models

You can switch models at any time using the `/switch` command.

```bash
/switch gemini
/switch groq
/switch ollama llama3
```

### Checking Status

To see which model is currently active:

```bash
/current-model
```

To list all available models:

```bash
/models
```

## Ollama Integration

Ollama allows you to run large language models locally.

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai).
2. **Pull Models**: Run `ollama pull [model_name]` in your terminal (e.g., `ollama pull llama3`).
3. **Use in AETHER**:
    * List models: `/ollama-models`
    * View specs: `/ollama-models [name]`
    * Switch: `/switch ollama [name]`
