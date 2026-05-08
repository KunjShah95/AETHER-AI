package skills

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSkillExecutor(t *testing.T) {
	executor := NewExecutor(nil)

	skill := &ParsedSkill{
		Name:             "test-skill",
		Objective:        "Read a file",
		ExecutionContext: "",
		Process:          "1. Read test.go\n2. Return content",
		AllowedTools:     []string{"Read"},
	}

	result, err := executor.Execute(context.Background(), skill, map[string]string{})
	assert.NoError(t, err)
	assert.Contains(t, result, "Executing: Read a file")
}

func TestSkillExecutorWithContext(t *testing.T) {
	executor := NewExecutor(nil)

	skill := &ParsedSkill{
		Name:             "test-skill",
		Objective:        "Test",
		ExecutionContext: "",
		Process:          "",
		AllowedTools:     []string{},
	}

	result, err := executor.Execute(context.Background(), skill, map[string]string{})
	assert.NoError(t, err)
	assert.Contains(t, result, "Executing: Test")
}

func TestCanExecute(t *testing.T) {
	tools := map[string]ToolFunc{
		"Read":  nil,
		"Bash":  nil,
		"Write": nil,
	}
	executor := NewExecutor(tools)

	skill := &ParsedSkill{
		Name:         "test",
		AllowedTools: []string{"Read", "Bash"},
	}

	assert.True(t, executor.CanExecute(skill))

	skillWithUnknownTool := &ParsedSkill{
		Name:         "test",
		AllowedTools: []string{"Read", "UnknownTool"},
	}

	assert.False(t, executor.CanExecute(skillWithUnknownTool))
}

func TestExpandHome(t *testing.T) {
	path := expandHome("@$HOME/test/file.txt")
	assert.True(t, len(path) > len("@$HOME/test/file.txt"))
	assert.False(t, strings.HasPrefix(path, "@$HOME"))

	pathNoHome := expandHome("/absolute/path.txt")
	assert.Equal(t, "/absolute/path.txt", pathNoHome)
}
