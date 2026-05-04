package mcp

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"sentinel-ai/internal/config"
	"sentinel-ai/internal/tool"
)

type Bridge struct {
	mu      sync.Mutex
	clients map[string]*Client
	tools   []tool.Tool
}

func NewBridge() *Bridge {
	return &Bridge{
		clients: make(map[string]*Client),
		tools:   make([]tool.Tool, 0),
	}
}

func (b *Bridge) LoadFromConfig(ctx context.Context, servers []config.MCPServer) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for _, server := range servers {
		client := NewClient(server.Command, server.Args, server.Env)
		if err := client.Start(ctx); err != nil {
			return fmt.Errorf("start mcp server %s: %w", server.Name, err)
		}

		defs, err := client.ListTools(ctx)
		if err != nil {
			_ = client.Close()
			return fmt.Errorf("list tools for %s: %w", server.Name, err)
		}

		b.clients[server.Name] = client
		for _, def := range defs {
			remote := &RemoteTool{
				serverName: server.Name,
				definition: def,
				client:     client,
				permission: 1,
				toolName:   normalizedToolName(server.Name, def.Name),
			}
			b.tools = append(b.tools, remote)
		}
	}

	return nil
}

func (b *Bridge) Tools() []tool.Tool {
	b.mu.Lock()
	defer b.mu.Unlock()

	out := make([]tool.Tool, 0, len(b.tools))
	out = append(out, b.tools...)
	return out
}

func (b *Bridge) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	var errs []string
	for name, client := range b.clients {
		if err := client.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", name, err))
		}
	}
	b.clients = make(map[string]*Client)
	b.tools = nil
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

type RemoteTool struct {
	serverName string
	definition ToolDefinition
	client     *Client
	permission int
	toolName   string
}

func normalizedToolName(serverName, toolName string) string {
	serverName = strings.TrimSpace(serverName)
	toolName = strings.TrimSpace(toolName)
	if serverName == "" {
		return toolName
	}
	if toolName == "" {
		return serverName
	}
	return serverName + ":" + toolName
}

func (t *RemoteTool) Name() string {
	return t.toolName
}

func (t *RemoteTool) Description() string {
	if t.definition.Description != "" {
		return t.definition.Description
	}
	return fmt.Sprintf("Remote MCP tool from %s", t.serverName)
}

func (t *RemoteTool) InputSchema() map[string]interface{} {
	if t.definition.InputSchema != nil {
		return t.definition.InputSchema
	}
	return map[string]interface{}{}
}

func (t *RemoteTool) Execute(ctx context.Context, input map[string]interface{}) (string, error) {
	return t.client.CallTool(ctx, t.definition.Name, input)
}

func (t *RemoteTool) PermissionLevel() int {
	return t.permission
}
