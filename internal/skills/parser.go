package skills

import (
	"bytes"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

var (
	sectionObjective        = regexp.MustCompile(`(?s)<objective>\s*(.*?)\s*</objective>`)
	sectionExecutionContext = regexp.MustCompile(`(?s)<execution_context>\s*(.*?)\s*</execution_context>`)
	sectionProcess          = regexp.MustCompile(`(?s)<process>\s*(.*?)\s*</process>`)
	flagPattern             = regexp.MustCompile(`--(\w+)=(\S+)`)
)

type rawMetadata struct {
	Name         string   `yaml:"name"`
	Description  string   `yaml:"description"`
	Trigger      []string `yaml:"trigger"`
	ApplyTo      []string `yaml:"applyTo"`
	AllowedTools []string `yaml:"allowed-tools"`
	ArgumentHint string   `yaml:"argument-hint"`
}

func ParseFile(path string, data []byte) (*Skill, error) {
	meta, body, err := splitFrontMatter(data)
	if err != nil {
		return nil, err
	}

	var raw rawMetadata
	if len(meta) > 0 {
		if err := yaml.Unmarshal(meta, &raw); err != nil {
			return nil, fmt.Errorf("parse skill metadata: %w", err)
		}
	}

	md := toMetadata(raw)
	if md.Name == "" {
		md.Name = inferNameFromPath(path)
	}
	if md.Description == "" {
		md.Description = strings.TrimSpace(firstLine(string(body)))
	}

	objective := extractSection(string(body), sectionObjective)
	execCtx := extractSection(string(body), sectionExecutionContext)
	process := extractSection(string(body), sectionProcess)

	return &Skill{
		Metadata:         md,
		Path:             path,
		Content:          strings.TrimSpace(string(body)),
		Objective:        objective,
		ExecutionContext: execCtx,
		Process:          process,
	}, nil
}

func ParseSkillContent(path string, content string) (*Skill, error) {
	return ParseFile(path, []byte(content))
}

func ExtractArguments(input string) (map[string]string, []string) {
	flags := make(map[string]string)
	var posArgs []string

	input = strings.TrimSpace(input)
	if input == "" {
		return flags, []string{}
	}

	matches := flagPattern.FindAllStringSubmatch(input, -1)
	for _, match := range matches {
		if len(match) == 3 {
			flags[match[1]] = match[2]
		}
	}

	remaining := flagPattern.ReplaceAllString(input, "")
	remaining = strings.TrimSpace(remaining)
	if remaining != "" {
		parts := strings.Fields(remaining)
		posArgs = parts
	}

	if posArgs == nil {
		posArgs = []string{}
	}

	return flags, posArgs
}

func toMetadata(raw rawMetadata) Metadata {
	return Metadata{
		Name:         raw.Name,
		Description:  raw.Description,
		Trigger:      raw.Trigger,
		ApplyTo:      raw.ApplyTo,
		AllowedTools: raw.AllowedTools,
		ArgumentHint: raw.ArgumentHint,
	}
}

func splitFrontMatter(data []byte) ([]byte, []byte, error) {
	trimmed := bytes.TrimSpace(data)
	if !bytes.HasPrefix(trimmed, []byte("---")) {
		return nil, trimmed, nil
	}

	parts := bytes.SplitN(trimmed, []byte("\n---"), 2)
	if len(parts) != 2 {
		return nil, nil, fmt.Errorf("invalid front matter")
	}

	meta := bytes.TrimPrefix(parts[0], []byte("---"))
	body := bytes.TrimSpace(bytes.TrimPrefix(parts[1], []byte("\n")))
	return bytes.TrimSpace(meta), body, nil
}

func extractSection(content string, re *regexp.Regexp) string {
	match := re.FindStringSubmatch(content)
	if len(match) > 1 {
		return strings.TrimSpace(match[1])
	}
	return ""
}

func inferNameFromPath(path string) string {
	base := filepath.Base(filepath.Dir(path))
	if base == "." || base == string(filepath.Separator) || base == "" {
		base = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}
	return base
}

func firstLine(s string) string {
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		return s[:idx]
	}
	return s
}
