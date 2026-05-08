package builtin

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

type BashTool struct {
	allowedCmds []string
}

func NewBashTool() interface{} {
	return &BashTool{
		allowedCmds: []string{"git", "go", "npm", "pytest", "cargo"},
	}
}

func (t *BashTool) Name() string {
	return "bash"
}

func (t *BashTool) Description() string {
	return "Execute a shell command. Input: command (string), description (string)"
}

func (t *BashTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"command": map[string]interface{}{
				"type":        "string",
				"description": "Command to execute",
			},
			"description": map[string]interface{}{
				"type":        "string",
				"description": "What the command does",
			},
		},
		"required": []string{"command"},
	}
}

func (t *BashTool) Execute(ctx context.Context, input map[string]interface{}) (string, error) {
	cmdStr, ok := input["command"].(string)
	if !ok {
		return "", fmt.Errorf("command is required")
	}

	// Parse command safely
	parts := strings.Fields(cmdStr)
	if len(parts) == 0 {
		return "", fmt.Errorf("empty command")
	}

	cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}

	return string(out), nil
}

func (t *BashTool) PermissionLevel() int {
	return 1 // PermAsk
}
