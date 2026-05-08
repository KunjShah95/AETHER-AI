package skills

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseSkillContent(t *testing.T) {
	content := `---
name: test-skill
description: "Test skill"
allowed-tools:
  - Read
  - Bash
---

<objective>
Do something specific.
</objective>

<execution_context>
@$HOME/.test/context.md
</execution_context>

<process>
1. Read the context file
2. Execute the task
</process>`

	skill, err := ParseSkillContent("test-path", content)
	assert.NoError(t, err)
	assert.Equal(t, "test-skill", skill.Name)
	assert.Equal(t, "Do something specific.", skill.Objective)
	assert.Equal(t, "@$HOME/.test/context.md", skill.ExecutionContext)
	assert.Contains(t, skill.Process, "Read the context file")
	assert.Equal(t, []string{"Read", "Bash"}, skill.AllowedTools)
}

func TestExtractArguments(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]string
		posArgs  []string
	}{
		{
			name:     "no args",
			input:    "",
			expected: map[string]string{},
			posArgs:  []string{},
		},
		{
			name:     "single flag",
			input:    "--depth=quick",
			expected: map[string]string{"depth": "quick"},
			posArgs:  []string{},
		},
		{
			name:     "positional and flags",
			input:    "2 --depth=deep --files=a.go,b.go",
			expected: map[string]string{"depth": "deep", "files": "a.go,b.go"},
			posArgs:  []string{"2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags, posArgs := ExtractArguments(tt.input)
			assert.Equal(t, tt.expected, flags)
			assert.Equal(t, tt.posArgs, posArgs)
		})
	}
}
