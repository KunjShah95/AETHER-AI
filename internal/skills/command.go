package skills

import (
	"fmt"
	"strings"
)

type BuiltInCommand struct {
	Name        string
	Description string
	Handler     func(args string) string
}

var BuiltInCommands []BuiltInCommand

func init() {
	BuiltInCommands = []BuiltInCommand{
		{
			Name:        "help",
			Description: "Show available commands and usage guide",
			Handler:     handleHelp,
		},
		{
			Name:        "version",
			Description: "Show Sentinel AI version",
			Handler:     handleVersion,
		},
		{
			Name:        "status",
			Description: "Show current session status",
			Handler:     handleStatus,
		},
		{
			Name:        "skills",
			Description: "List all available skills",
			Handler:     handleSkillsList,
		},
	}
}

func handleHelp(args string) string {
	if args == "" {
		var sb strings.Builder
		sb.WriteString("Available commands:\n")
		sb.WriteString("  /help [command] - Show all commands or specific command help\n")
		sb.WriteString("  /version - Show Sentinel AI version\n")
		sb.WriteString("  /status - Show current session status\n")
		sb.WriteString("  /skills - List all available skills\n")
		return sb.String()
	}
	for _, bc := range BuiltInCommands {
		if bc.Name == args {
			return fmt.Sprintf("  /%s - %s", bc.Name, bc.Description)
		}
	}
	return fmt.Sprintf("Unknown command: %s", args)
}

func handleVersion(args string) string {
	return "Sentinel AI v0.1.0\nCommand System: enabled"
}

func handleStatus(args string) string {
	return "Session: active\nWorkflow: ready\nSkills: loaded"
}

func handleSkillsList(args string) string {
	return "Use GET /api/v1/commands to list all available skills"
}

type CommandInfo struct {
	Name        string
	Description string
	Score       int
}

func ParseCommand(input string) (command, args string, isCommand bool) {
	input = strings.TrimSpace(input)
	if input == "" || !strings.HasPrefix(input, "/") {
		return "", "", false
	}

	rest := strings.TrimPrefix(input, "/")
	parts := strings.Fields(rest)
	if len(parts) == 0 {
		return "", "", false
	}

	command = parts[0]
	if len(parts) > 1 {
		args = strings.Join(parts[1:], " ")
	}
	return command, args, true
}

func (r *Registry) Add(name string, skill *Skill) {
	if skill == nil || name == "" {
		return
	}
	r.skills[name] = skill
}

func (r *Registry) MatchSkill(query string) (string, int) {
	if query == "" {
		return "", 0
	}

	query = strings.ToLower(query)
	var bestMatch string
	var bestScore int

	for name, skill := range r.skills {
		score := calculateScore(query, name, skill.Description)
		if score > bestScore {
			bestScore = score
			bestMatch = name
		}
	}

	return bestMatch, bestScore
}

func calculateScore(query, name, description string) int {
	lowerName := strings.ToLower(name)
	lowerDesc := strings.ToLower(description)

	if lowerName == query {
		return 100
	}
	if strings.HasPrefix(lowerName, query) {
		return 95
	}
	if strings.Contains(lowerName, query) {
		return 90
	}
	if strings.Contains(lowerDesc, query) {
		return 70
	}
	return 0
}

func (r *Registry) ListCommands() []CommandInfo {
	commands := make([]CommandInfo, 0, len(r.skills))
	for name, skill := range r.skills {
		commands = append(commands, CommandInfo{
			Name:        name,
			Description: skill.Description,
			Score:       0,
		})
	}
	return commands
}
