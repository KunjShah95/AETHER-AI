# Sentinel AI

Sentinel AI is a Go-based coding terminal that combines a Bubble Tea TUI, an HTTP server, session persistence, built-in filesystem tools, MCP tool bridges, and a local skills registry.

## What it does

- Launches a terminal UI for interactive coding workflows
- Stores sessions in SQLite for later inspection
- Supports built-in tools like `read`, `write`, `glob`, `grep`, and `bash`
- Can load external MCP tools from configured servers
- Loads skills from local `SKILL.md` files
- Exposes a lightweight HTTP API for sessions, tools, and skills

## Requirements

- Go 1.25 or later
- A supported shell on your machine
- Optional: MCP servers, local skills folders, and LLM provider credentials

## Configuration

Sentinel reads configuration from:

- `~/.sentinel/config.yaml`

If the file is missing, Sentinel starts with defaults.

Example configuration:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: your-api-key
  base_url: https://api.anthropic.com

mcp:
  - name: filesystem-tools
    type: stdio
    command: your-mcp-server
    args:
      - --stdio
    env:
      EXAMPLE_TOKEN: value

project:
  work_dir: .
```

## Build

Build both binaries from the repository root:

```bash
go build -o bin/sentinel.exe ./cmd/sentinel
go build -o bin/sentinel-server.exe ./cmd/server
```

On Windows, keep both binaries in the same directory because `sentinel.exe` launches `sentinel-server.exe` from its own folder.

## Run

Start the full experience by running the Sentinel client:

```bash
./bin/sentinel.exe
```

This starts the TUI and launches the server as a child process.

You can also run the server directly:

```bash
./bin/sentinel-server.exe
```

The server listens on `:8080` by default.

## Built-in tools

### `read`

Read a file from disk.

Input:

- `path` string

### `write`

Write or overwrite a file.

Input:

- `path` string
- `content` string

### `glob`

Find files using a glob pattern.

Input:

- `pattern` string

### `grep`

Search source files for a pattern.

Input:

- `pattern` string
- `path` string, optional

### `bash`

Run an approved shell command.

Input:

- `command` string
- `description` string, optional

## Skills

Sentinel loads skill files from these locations when they exist:

- `~/.sentinel/skills`
- `./skills`
- `./.agents/skills`

A skill is a markdown file with optional YAML front matter. Sentinel exposes loaded skills through the HTTP API at `GET /skills`.

## HTTP API

### `GET /health`

Returns server health.

### `POST /session`

Creates a new session.

### `GET /session/:id`

Fetches a session and its messages.

### `POST /session/:id/chat`

Adds a chat message to a session.

### `POST /session/:id/tool`

Runs a tool.

Request example:

```json
{
  "tool_name": "read",
  "input": {
    "path": "README.md"
  },
  "approved": true
}
```

### `GET /session/:id/stream`

Streams session output using Server-Sent Events.

### `GET /skills`

Lists loaded skills.

## Tool permissions

Tool execution honors a simple permission model:

- `allow` / `always` run immediately
- `ask` requires approval
- `deny` / `never` block execution

For approval-required tools, pass `approved: true` in the tool request.

## Project layout

- `cmd/sentinel` - TUI entry point
- `cmd/server` - HTTP server entry point
- `internal/config` - config loading and validation
- `internal/session` - SQLite session store and manager
- `internal/provider` - LLM provider abstraction
- `internal/tool` - tool registry and built-ins
- `internal/mcp` - MCP client and bridge
- `internal/skills` - skill registry and loader
- `internal/server` - HTTP handlers and router
- `internal/tui` - Bubble Tea TUI

## Notes

This project is under active development. Some endpoints currently return placeholder responses while later phases are implemented.
