# Task Management System

AETHER AI includes a built-in task management system to help you stay organized directly from the terminal.

## Basic Operations

| Command | Description | Example |
|---------|-------------|---------|
| `/task add [title] [description]` | Add a new task | `/task add "Fix bug" "Login issue"` |
| `/task list` | List all tasks | |
| `/task view [id]` | View task details | `/task view 1` |
| `/task update [id] [field] [value]` | Update task field | `/task update 1 status completed` |
| `/task delete [id]` | Delete a task | `/task delete 1` |
| `/task complete [id]` | Mark task as completed | `/task complete 1` |

## Advanced Features

- **Priority**: Set task priority (low/medium/high)
  - `/task priority [id] [level]`
- **Categories**: Organize tasks by category
  - `/task category [id] [category]`
- **Due Dates**: Set deadlines
  - `/task due [id] [YYYY-MM-DD]`
- **Subtasks**: Break down complex tasks
  - `/task subtasks [id] [subtask]`
- **Search & Filter**: Find what you need
  - `/task search [keyword]`
  - `/task filter [criteria]`

## Tools

- **Export/Import**: `/task export [json/csv]`, `/task import [file]`
- **Statistics**: `/task stats`
- **Categories List**: `/task categories`
- **Overdue/Today**: `/task overdue`, `/task today`
