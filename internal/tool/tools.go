package tool

import (
	"sentinel-ai/internal/tool/builtin"
)

// NewRegistryWithBuiltins creates a registry and registers all built-in tools
func NewRegistryWithBuiltins() *Registry {
	r := NewRegistry()
	r.Register(builtin.NewReadTool().(Tool))
	r.Register(builtin.NewWriteTool().(Tool))
	r.Register(builtin.NewGlobTool().(Tool))
	r.Register(builtin.NewGrepTool().(Tool))
	r.Register(builtin.NewBashTool().(Tool))
	return r
}
