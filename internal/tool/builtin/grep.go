package builtin

import (
	"context"
	"fmt"
	"os/exec"
)

type GrepTool struct{}

func NewGrepTool() interface{} {
	return &GrepTool{}
}

func (t *GrepTool) Name() string {
	return "grep"
}

func (t *GrepTool) Description() string {
	return "Search for pattern in files. Input: pattern (string), path (string, optional)"
}

func (t *GrepTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"pattern": map[string]interface{}{
				"type":        "string",
				"description": "Regex pattern to search",
			},
			"path": map[string]interface{}{
				"type":        "string",
				"description": "File or directory to search",
			},
		},
		"required": []string{"pattern"},
	}
}

func (t *GrepTool) Execute(ctx context.Context, input map[string]interface{}) (string, error) {
	pattern, ok := input["pattern"].(string)
	if !ok {
		return "", fmt.Errorf("pattern is required")
	}

	path := "."
	if p, ok := input["path"].(string); ok {
		path = p
	}

	cmd := exec.Command("grep", "-r", "--include=*.go", pattern, path)
	out, err := cmd.Output()
	if err != nil {
		return "No matches found", nil
	}

	return string(out), nil
}

func (t *GrepTool) PermissionLevel() int {
	return 0 // PermAllow
}

