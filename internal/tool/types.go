package tool

import (
	"context"
)

type Tool interface {
	Name() string
	Description() string
	InputSchema() map[string]interface{}
	Execute(ctx context.Context, input map[string]interface{}) (string, error)
	PermissionLevel() int
}

type Permission int

const (
	PermAllow Permission = iota
	PermAsk
	PermDeny
	PermAlways
	PermNever
)

type PermissionError struct {
	ToolName string
	Level    Permission
	Message  string
}

func (e *PermissionError) Error() string {
	if e == nil {
		return ""
	}
	if e.Message != "" {
		return e.Message
	}
	return "permission denied"
}

func (p Permission) String() string {
	switch p {
	case PermAllow:
		return "allow"
	case PermAsk:
		return "ask"
	case PermDeny:
		return "deny"
	case PermAlways:
		return "always"
	case PermNever:
		return "never"
	default:
		return "unknown"
	}
}

func Authorize(toolName string, level int, approved bool) error {
	switch Permission(level) {
	case PermAllow, PermAlways:
		return nil
	case PermAsk:
		if approved {
			return nil
		}
		return &PermissionError{
			ToolName: toolName,
			Level:    PermAsk,
			Message:  "approval required",
		}
	case PermDeny, PermNever:
		return &PermissionError{
			ToolName: toolName,
			Level:    Permission(level),
			Message:  "tool execution denied",
		}
	default:
		return &PermissionError{
			ToolName: toolName,
			Level:    Permission(level),
			Message:  "unknown permission policy",
		}
	}
}

type Registry struct {
	tools map[string]Tool
}

func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

func (r *Registry) Register(t Tool) {
	r.tools[t.Name()] = t
}

func (r *Registry) Get(name string) Tool {
	return r.tools[name]
}

func (r *Registry) List() []Tool {
	var list []Tool
	for _, t := range r.tools {
		list = append(list, t)
	}
	return list
}
