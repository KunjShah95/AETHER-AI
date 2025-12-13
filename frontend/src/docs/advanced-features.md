# Advanced Features

## Context-Aware AI

- **Learn**: Teach AI about specific technologies or topics.
  - `/learn [topic]`
- **Reminders**: Set and manage reminders.
  - `/remind [task]`
  - `/reminders`
  - `/complete-reminder [n]`

## Analytics & Monitoring

Monitor system performance and usage.

- `/analytics`: View usage statistics
- `/error-analytics`: View error analytics
- `/start-monitoring`: Start system monitoring
- `/stop-monitoring`: Stop system monitoring
- `/net-diag`: Network diagnostics
- `/analyze-logs`: Analyze log files
- `/health`: System health check

## Games & Learning

Improve your coding skills or take a break.

- **Challenges**: `/challenge [difficulty]`
- **Tutorials**: `/tutorial [topic]`
- **Quizzes**: `/quiz [topic]`
- **Stats**: `/user-stats`

## Creative Tools

- `/ascii [text]`: Generate ASCII art
- `/colors [type] [base]`: Generate color schemes
- `/music [mood] [length]`: Generate music patterns
- `/story [genre] [length]`: Generate creative stories

## User Management

- `/setkey [provider] [key]`: Set API keys
- `/history`: View chat history
- `/clearhistory`: Clear chat history
- `/myactivity`: View activity log
- `/listusers`: List all users (admin)
- `/resetpw [user] [newpass]`: Reset password (admin)

## Autonomous Agents & Personas

AETHER AI includes a powerful multi-agent system.

- **Personas**: Switch between `security`, `frontend`, `backend`, `qa`, and `architect` personas to get specialized assistance.
  - `/persona security`: Focus on vulnerability analysis.
  - `/persona qa`: Focus on test generation and bug finding.

## Web Agent & Research

The built-in Web Agent can browse the internet to gather real-time information.

- **Browsing**: `/web https://example.com` - Read and summarize a specific page.
- **Research**: `/research "latest react patterns"` - Synthesize information from multiple sources.

## DevOps & Docker capabilities

Manage your development environment directly from the terminal.

- **Auto-Docker**: `/docker init` analyzes your project and creates the perfect `Dockerfile` and `docker-compose.yml`.
- **Management**: `/docker up`, `/docker down`, `/docker logs` to control containers.

## Workflow Automation

Use AutoPilot to autonomously plan and execute complex multi-step tasks.

- `/autopilot "Refactor the login page to use hooks"`: The AI will plan, edit files, and verify changes.

