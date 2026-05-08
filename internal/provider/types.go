package provider

import (
	"context"
)

type Provider interface {
	Send(ctx context.Context, msgs []Message, tools []Tool) (*Response, error)
	Stream(ctx context.Context, msgs []Message, tools []Tool) (<-chan Event, error)
	Model() string
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
	Parts   []Part `json:"parts,omitempty"`
}

type Part struct {
	Type         string `json:"type"`
	Text         string `json:"text,omitempty"`
	ToolUse      *ToolUse `json:"tool_use,omitempty"`
	ToolResultID string `json:"tool_result_id,omitempty"`
}

type ToolUse struct {
	Name  string                 `json:"name"`
	Input map[string]interface{} `json:"input"`
	ID    string                 `json:"id"`
}

type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

type Response struct {
	Content string
	Parts   []Part
	Usage   Usage
}

type Usage struct {
	InputTokens  int
	OutputTokens int
}

type Event struct {
	Type    string
	Content string
	Part    *Part
}
