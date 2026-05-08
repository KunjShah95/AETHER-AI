package skills

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type ToolFunc func(ctx context.Context, args map[string]string) (string, error)

type Executor struct {
	tools map[string]ToolFunc
}

type ParsedSkill struct {
	Name             string
	Objective        string
	ExecutionContext string
	Process          string
	AllowedTools     []string
}

func NewExecutor(tools map[string]ToolFunc) *Executor {
	if tools == nil {
		tools = make(map[string]ToolFunc)
	}
	return &Executor{
		tools: tools,
	}
}

func (e *Executor) Execute(ctx context.Context, skill *ParsedSkill, args map[string]string) (string, error) {
	if skill == nil {
		return "", fmt.Errorf("skill is nil")
	}

	result := fmt.Sprintf("Executing: %s", skill.Objective)

	if skill.ExecutionContext != "" {
		contextPath := expandHome(skill.ExecutionContext)
		if data, err := os.ReadFile(contextPath); err == nil {
			result += fmt.Sprintf("\nContext loaded from %s: %s", contextPath, string(data))
		}
	}

	if len(args) > 0 {
		result += fmt.Sprintf("\nArguments: %v", args)
	}

	return result, nil
}

func (e *Executor) CanExecute(skill *ParsedSkill) bool {
	if skill == nil || len(skill.AllowedTools) == 0 {
		return true
	}

	for _, tool := range skill.AllowedTools {
		if _, ok := e.tools[tool]; !ok {
			return false
		}
	}
	return true
}

func (e *Executor) ExecuteTool(ctx context.Context, toolName string, args map[string]string) (string, error) {
	tool, ok := e.tools[toolName]
	if !ok {
		return "", fmt.Errorf("tool %s not registered", toolName)
	}
	return tool(ctx, args)
}

func expandHome(path string) string {
	if strings.HasPrefix(path, "@$HOME/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, strings.TrimPrefix(path, "@$HOME/"))
	}
	return path
}
