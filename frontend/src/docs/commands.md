# Commands Reference

## System Commands

| Command | Description |
|---------|-------------|
| `/help` | Show comprehensive help menu |
| `/status` | Display current model and system status |
| `/clear` | Clear the terminal screen |
| `/exit` | Exit the AETHER AI terminal |
| `/sysinfo` | Show detailed system information |
| `/config` | Display current configuration |

## Utility Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/run [command]` | Execute safe system commands | `/run ls -la` |
| `/calc [expression]` | Calculate mathematical expressions | `/calc 2 + 2 * 3` |
| `/websearch [query]` | Search the web using DuckDuckGo | `/websearch python tutorials` |
| `/weather [city]` | Get current weather information | `/weather New York` |
| `/note [text]` | Save a quick note | `/note Buy milk` |
| `/notes` | View all saved notes | |
| `/timer [seconds]` | Start a countdown timer | `/timer 300` |
| `/convert [val] [from] [to]` | Unit converter | `/convert 100 celsius fahrenheit` |
| `/password [length]` | Generate secure password | `/password 16` |

## Developer Commands

### Code Analysis

| Command | Description |
|---------|-------------|
| `/codereview [filename]` | AI code review for bugs and improvements |
| `/summarizefile [filename]` | AI file summarization |
| `/findbugs [filename]` | Find bugs in code using AI |
| `/refactor [filename] [instruction]` | AI code refactoring |

### Code Generation

| Command | Description |
|---------|-------------|
| `/gendoc [filename]` | Generate documentation for code |
| `/gentest [filename]` | Generate unit tests for code |
| `/git commitmsg [diff]` | Generate git commit messages |
| `/todos` | Extract TODOs and FIXMEs from codebase |

## Git Advanced Commands

| Command | Description |
|---------|-------------|
| `/git create-branch [name]` | Create a new branch |
| `/git delete-branch [name]` | Delete a branch |
| `/aifind [keyword]` | AI-powered file search |
| `/explore` | Explore codebase |

## Advanced AI Features

| Command | Description |
|---------|-------------|
| `/persona [name]` | Switch to a specialized AI persona (e.g., 'security', 'qa') |
| `/personas` | List available AI personas |
| `/web [url]` | Instruct the Web Agent to visit and analyze a URL |
| `/research [topic]` | Perform deep web research on a topic |
| `/docker init` | Generate Dockerfile and compose for the current project |
| `/docker up` | Start the dev environment |
| `/vision [path]` | Analyze an image using the vision model |
| `/voice` | Toggle conversational voice mode |
| `/dashboard` | Launch the TUI Mission Control dashboard |
| `/autopilot [goal]` | Start an autonomous workflow to achieve a goal |
| `/workflow list` | List available workflows |

