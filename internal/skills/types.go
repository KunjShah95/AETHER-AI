package skills

import "strings"

type Metadata struct {
	Name         string   `yaml:"name" json:"name"`
	Description  string   `yaml:"description" json:"description"`
	Trigger      []string `yaml:"trigger" json:"trigger"`
	ApplyTo      []string `yaml:"applyTo" json:"applyTo"`
	AllowedTools []string `yaml:"allowedTools" json:"allowedTools"`
}

type Skill struct {
	Metadata
	Path    string `json:"path"`
	Content string `json:"content"`
}

func (s *Skill) Matches(query string) bool {
	query = strings.ToLower(strings.TrimSpace(query))
	if query == "" {
		return false
	}

	fields := []string{s.Name, s.Description, s.Path}
	for _, field := range fields {
		if strings.Contains(strings.ToLower(field), query) {
			return true
		}
	}

	for _, trigger := range s.Trigger {
		if strings.Contains(strings.ToLower(trigger), query) {
			return true
		}
	}

	for _, apply := range s.ApplyTo {
		if strings.Contains(strings.ToLower(apply), query) {
			return true
		}
	}

	return false
}
