# Sentinel AI Usage Guide

This guide shows how to use Sentinel AI day to day.

## 1. Start Sentinel

Build the binaries first:

```bash
go build -o bin/sentinel.exe ./cmd/sentinel
go build -o bin/sentinel-server.exe ./cmd/server
```

Then launch the client:

```bash
./bin/sentinel.exe
```

Sentinel starts the server as a child process and opens the terminal UI.

## 2. Configure the app

Sentinel reads `~/.sentinel/config.yaml`.

Useful settings:

- `llm.provider` - `anthropic` or `ollama`
- `llm.model` - the model name
- `llm.api_key` - provider key if needed
- `llm.base_url` - custom provider URL
- `mcp` - list of MCP servers
- `project.work_dir` - working directory for tool operations

Example:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: your-api-key
  base_url: https://api.anthropic.com

mcp:
  - name: filesystem
    type: stdio
    command: my-mcp-server
    args:
      - --stdio

project:
  work_dir: .
```

## 3. Use the terminal UI

In the TUI you can:

- type messages and press Enter to send them
- use `Ctrl+C` or `Ctrl+D` to exit
- use `Ctrl+U` to clear the current input

The UI shows your message history, processing state, and the current input box.

## 4. Use built-in tools

Sentinel ships with these local tools:

- `read` - read a file from disk
- `write` - write or overwrite a file
- `glob` - find files using a glob pattern
- `grep` - search files for a pattern
- `bash` - run approved shell commands

Tool requests go to `POST /session/:id/tool`.

Example request:

```json
{
  "tool_name": "read",
  "input": {
    "path": "README.md"
  },
  "approved": true
}
```

### Tool permissions

- `allow` and `always` run immediately
- `ask` requires `approved: true`
- `deny` and `never` are blocked

If Sentinel asks for approval, resend the request with `approved: true`.

## 5. Work with sessions

Sessions are stored in SQLite and can be accessed through the HTTP API.

### Create a session

```bash
curl -X POST http://localhost:8080/session \
  -H "Content-Type: application/json" \
  -d '{"model_id":"default"}'
```

### Send chat to a session

```bash
curl -X POST http://localhost:8080/session/<session-id>/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Show me the files in internal/tool"}'
```

### Inspect a session

```bash
curl http://localhost:8080/session/<session-id>
```

## 6. Use skills

Skills are reusable markdown instructions stored on disk.

Sentinel scans for skills in:

- `~/.sentinel/skills`
- `./skills`
- `./.agents/skills`

A skill file can include front matter like this:

```md
---
name: graphify
description: Convert notes into a knowledge graph
trigger:
  - /graphify
applyTo:
  - markdown
---

Use this skill when the user wants to turn notes into a graph.
```

Loaded skills are available at `GET /skills`.

## 7. Use MCP tools

MCP servers are configured under `mcp:` in `config.yaml`.

When Sentinel starts, it loads tools from those servers and makes them available in the same tool flow as built-in tools.

If a remote tool does not show up, check that:

- the command exists on your machine
- the arguments are correct
- the server supports stdio transport

## 8. Common workflows

### Read a project file

1. Create or open a session.
2. Ask Sentinel to read the file.
3. Approve the tool if required.

### Make a code change

1. Ask Sentinel to inspect the relevant files.
2. Use `write` to update the target file.
3. Review the change with `read` or `grep`.

### Add a reusable skill

1. Create a folder for the skill.
2. Add `SKILL.md` with front matter.
3. Restart Sentinel or reload the server.
4. Confirm the skill appears under `GET /skills`.

## 9. Troubleshooting

- If `sentinel.exe` cannot find the server, make sure `sentinel-server.exe` is in the same folder.
- If configuration fails, check the YAML syntax in `~/.sentinel/config.yaml`.
- If a tool is denied, confirm the permission level and approval flag.
- If skills are missing, confirm the skill file is named `SKILL.md`.
- If MCP tools are missing, verify the server command can run from your shell.
