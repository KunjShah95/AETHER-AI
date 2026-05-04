# Sentinel AI User Guide

This folder contains the user-facing documentation for Sentinel AI.

## Start here

- `../README.md` - quick start, build steps, and API overview
- `superpowers/specs/` - product and feature specs
- `superpowers/plans/` - implementation plans and phase details

## How to use Sentinel AI

### 1. Configure your environment

Create `~/.sentinel/config.yaml` if you want to customize the default provider, model, MCP servers, or working directory.

You can also store reusable skills on disk in one of these locations:

- `~/.sentinel/skills`
- `./skills`
- `./.agents/skills`

### 2. Build the binaries

Compile the client and server from the repository root:

```bash
go build -o bin/sentinel.exe ./cmd/sentinel
go build -o bin/sentinel-server.exe ./cmd/server
```

### 3. Launch Sentinel

Run the client binary:

```bash
./bin/sentinel.exe
```

The client launches the server automatically and opens the terminal UI.

### 4. Work with sessions

Sessions are stored in SQLite and can be fetched through the HTTP API.

- `POST /session` creates a session
- `GET /session/:id` retrieves it
- `POST /session/:id/chat` adds a message
- `GET /session/:id/stream` streams output

### 5. Use tools

Available built-in tools:

- `read` - read file contents
- `write` - create or update files
- `glob` - find files by pattern
- `grep` - search source code
- `bash` - run approved shell commands

Tool requests are sent to `POST /session/:id/tool`.

Example:

```json
{
  "tool_name": "glob",
  "input": {
    "pattern": "internal/**/*.go"
  },
  "approved": true
}
```

If a tool is marked `ask`, Sentinel returns an approval-required response until you resend the request with `approved: true`.

### 6. Use skills

Skills are markdown files that help Sentinel understand repeatable tasks and workflows.

- Create a folder for the skill
- Add a `SKILL.md` file
- Optionally include YAML front matter with `name`, `description`, `trigger`, and `applyTo`
- Restart Sentinel or reload the server to pick up new skills

Loaded skills are visible through `GET /skills`.

### 7. Connect MCP servers

Add MCP servers in `config.yaml` under the `mcp:` section.

Sentinel loads tools from configured MCP servers on startup and exposes them through the same tool execution flow.

## Example skill file

```md
---
name: graphify
description: Turn input into a knowledge graph
trigger:
  - /graphify
applyTo:
  - markdown
---

Use this skill when the user wants to convert notes into a graph.
```

## Troubleshooting

- If the client cannot find `sentinel-server.exe`, make sure both binaries live in the same directory.
- If the server fails to start, check `~/.sentinel/config.yaml` for invalid YAML.
- If MCP tools do not appear, verify the configured command is installed and runnable from your shell.
- If skills do not appear, confirm the skill folder contains a `SKILL.md` file.
