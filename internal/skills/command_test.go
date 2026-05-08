package skills

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseCommand(t *testing.T) {
	tests := []struct {
		name  string
		input string
		cmd   string
		args  string
		isCmd bool
	}{
		{
			name:  "simple command",
			input: "/help",
			cmd:   "help",
			args:  "",
			isCmd: true,
		},
		{
			name:  "command with args",
			input: "/gsd-code-review 2 --depth=deep",
			cmd:   "gsd-code-review",
			args:  "2 --depth=deep",
			isCmd: true,
		},
		{
			name:  "not a command",
			input: "hello world",
			cmd:   "",
			args:  "",
			isCmd: false,
		},
		{
			name:  "command with equals args",
			input: "/azure-prepare --location=eastus",
			cmd:   "azure-prepare",
			args:  "--location=eastus",
			isCmd: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd, args, isCmd := ParseCommand(tt.input)
			assert.Equal(t, tt.cmd, cmd)
			assert.Equal(t, tt.args, args)
			assert.Equal(t, tt.isCmd, isCmd)
		})
	}
}

func TestMatchSkill(t *testing.T) {
	registry := &Registry{
		skills: map[string]*Skill{
			"gsd-help":      {Metadata: Metadata{Name: "gsd-help", Description: "Show help"}},
			"azure-prepare": {Metadata: Metadata{Name: "azure-prepare", Description: "Prepare Azure"}},
		},
	}

	tests := []struct {
		input string
		match string
		score int
	}{
		{input: "gsd-help", match: "gsd-help", score: 100},
		{input: "help", match: "gsd-help", score: 80},
		{input: "azure", match: "azure-prepare", score: 90},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			name, score := registry.MatchSkill(tt.input)
			assert.Equal(t, tt.match, name)
			assert.GreaterOrEqual(t, score, tt.score)
		})
	}
}
