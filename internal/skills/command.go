package skills

import (
	"strings"
)

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
