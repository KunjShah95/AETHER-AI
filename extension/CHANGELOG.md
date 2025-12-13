# Changelog

All notable changes to the "Nexus AI Terminal Assistant" extension will be documented in this file.

## [0.1.0] - 2024-12-13

### Added
- **Initial Release** - First public version of the Nexus AI Terminal Assistant
- **Start/Stop Assistant** - Easily control the AI assistant lifecycle from VS Code
- **Quick Commands Palette** - Access frequently used commands with a single click
  - `/help` - Show full command catalog
  - `/models` - Display configured AI models and health status
  - `/current-model` - Show the currently active model
  - `/switch gpt-5.1-codex-mini` - Enable GPT-5.1-Codex-Mini preview
  - `/status` - Detailed assistant health information
  - `/git status` - Repository staging information
  - `/plugins list` - View registered plugins
  - `/tasks` - List open tasks
  - `/todos` - Summarize TODO and FIXME entries
  - `/note` - Quick note capture
- **Assistant Panel** - Interactive webview with:
  - Command input field
  - Quick command buttons
  - Live log streaming
  - Status indicator
- **Output Channel** - All assistant logs available in the "Nexus AI Terminal" output channel
- **Configuration Options**:
  - `nexusAiAssistant.pythonPath` - Custom Python interpreter path
  - `nexusAiAssistant.preferredModel` - Default preferred AI model
  - `nexusAiAssistant.autoStart` - Auto-launch assistant on workspace open

### Technical
- TypeScript-based implementation with full type safety
- Webview Content Security Policy for secure script execution
- Streaming process output handling
- Graceful process termination with SIGINT

## [Unreleased]

### Planned
- Sidebar panel for persistent assistant access
- Status bar integration showing assistant state
- Code context awareness for smarter suggestions
- Multiple workspace support
- Custom quick command configuration
