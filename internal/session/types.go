package session

import (
	"time"
)

type Session struct {
	ID        string       `json:"id"`
	ProjectID string       `json:"project_id"`
	CreatedAt time.Time    `json:"created_at"`
	Messages  []Message    `json:"messages"`
	State     SessionState `json:"state"`
}

type Message struct {
	Role    string    `json:"role"`
	Parts   []Part    `json:"parts"`
	Created time.Time `json:"created"`
}

type Part struct {
	Type         string   `json:"type"`
	Content      string   `json:"content,omitempty"`
	ToolUse      *ToolUse `json:"tool_use,omitempty"`
	ToolResultID string   `json:"tool_result_id,omitempty"`
}

type ToolUse struct {
	Name  string                 `json:"name"`
	Input map[string]interface{} `json:"input"`
	ID    string                 `json:"id"`
}

type ToolResult struct {
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
	IsError   bool   `json:"is_error"`
}

type SessionState struct {
	Model            string `json:"model"`
	TokenCount       int    `json:"token_count"`
	CompletionTokens int    `json:"completion_tokens"`
	Summary          string `json:"summary,omitempty"`
}
