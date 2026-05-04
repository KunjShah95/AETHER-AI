package builtin

import (
	"context"
	"fmt"
	"path/filepath"
)

type GlobTool struct{}

func NewGlobTool() interface{} {
	return &GlobTool{}
}

func (t *GlobTool) Name() string {
	return "glob"
}

func (t *GlobTool) Description() string {
	return "Find files matching a glob pattern. Input: pattern (string)"
}

func (t *GlobTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"pattern": map[string]interface{}{
				"type":        "string",
				"description": "Glob pattern (e.g., **/*.go)",
			},
		},
		"required": []string{"pattern"},
	}
}

func (t *GlobTool) Execute(ctx context.Context, input map[string]interface{}) (string, error) {
	pattern, ok := input["pattern"].(string)
	if !ok {
		return "", fmt.Errorf("pattern is required")
	}

	matches, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}

	if len(matches) == 0 {
		return "No matches found", nil
	}

	result := fmt.Sprintf("Found %d matches:\n", len(matches))
	for _, m := range matches {
		result += m + "\n"
	}

	return result, nil
}

func (t *GlobTool) PermissionLevel() int {
	return 0 // PermAllow
}
