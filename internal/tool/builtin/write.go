package builtin

import (
	"context"
	"fmt"
	"os"
)

type WriteTool struct{}

func NewWriteTool() interface{} {
	return &WriteTool{}
}

func (t *WriteTool) Name() string {
	return "write"
}

func (t *WriteTool) Description() string {
	return "Write content to a file. Creates or overwrites. Input: path (string), content (string)"
}

func (t *WriteTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "Path to write to",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "Content to write",
			},
		},
		"required": []string{"path", "content"},
	}
}

func (t *WriteTool) Execute(ctx context.Context, input map[string]interface{}) (string, error) {
	path, ok := input["path"].(string)
	if !ok {
		return "", fmt.Errorf("path is required")
	}
	content, ok := input["content"].(string)
	if !ok {
		return "", fmt.Errorf("content is required")
	}

	err := os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		return "", err
	}

	return "File written successfully", nil
}

func (t *WriteTool) PermissionLevel() int {
	return 1 // PermAsk
}
