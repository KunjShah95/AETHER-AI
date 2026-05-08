package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
)

type Client struct {
	cfg         string
	command     string
	args        []string
	env         map[string]string
	cmd         *exec.Cmd
	stdin       io.WriteCloser
	stdout      io.ReadCloser
	encoder     *json.Encoder
	decoder     *json.Decoder
	mu          sync.Mutex
	nextID      int64
	initialized bool
}

func NewClient(command string, args []string, env map[string]string) *Client {
	return &Client{
		command: command,
		args:    append([]string{}, args...),
		env:     env,
	}
}

func (c *Client) Start(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd != nil {
		return nil
	}
	if strings.TrimSpace(c.command) == "" {
		return fmt.Errorf("mcp command is required")
	}

	c.cmd = exec.CommandContext(ctx, c.command, c.args...)
	c.cmd.Env = os.Environ()
	for k, v := range c.env {
		c.cmd.Env = append(c.cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	stdin, err := c.cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := c.cmd.StdoutPipe()
	if err != nil {
		return err
	}
	c.cmd.Stderr = os.Stderr

	if err := c.cmd.Start(); err != nil {
		return err
	}

	c.stdin = stdin
	c.stdout = stdout
	c.encoder = json.NewEncoder(c.stdin)
	c.decoder = json.NewDecoder(bufio.NewReader(c.stdout))
	return nil
}

func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.stdin != nil {
		_ = c.stdin.Close()
	}
	if c.cmd != nil && c.cmd.Process != nil {
		_ = c.cmd.Process.Kill()
		_, _ = c.cmd.Process.Wait()
	}
	c.cmd = nil
	c.stdin = nil
	c.stdout = nil
	c.encoder = nil
	c.decoder = nil
	c.initialized = false
	return nil
}

func (c *Client) Call(ctx context.Context, method string, params interface{}, result interface{}) error {
	if err := c.Start(ctx); err != nil {
		return err
	}
	if err := c.ensureInitialized(ctx); err != nil {
		return err
	}

	return c.callRaw(ctx, method, params, result)
}

func (c *Client) callRaw(ctx context.Context, method string, params interface{}, result interface{}) error {

	c.mu.Lock()
	defer c.mu.Unlock()

	c.nextID++
	id := c.nextID
	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}
	if err := c.encoder.Encode(req); err != nil {
		return err
	}

	var resp JSONRPCResponse
	if err := c.decoder.Decode(&resp); err != nil {
		return err
	}
	if resp.Error != nil {
		return fmt.Errorf("mcp %s: %s", method, resp.Error.Message)
	}
	if result == nil || len(resp.Result) == 0 {
		return nil
	}
	return json.Unmarshal(resp.Result, result)
}

func (c *Client) ensureInitialized(ctx context.Context) error {
	c.mu.Lock()
	if c.initialized {
		c.mu.Unlock()
		return nil
	}
	c.mu.Unlock()

	var initResult map[string]interface{}
	if err := c.callRaw(ctx, "initialize", InitializeParams{
		ProtocolVersion: "2024-11-05",
		ClientInfo: map[string]any{
			"name":    "Sentinel AI",
			"version": "0.1.0",
		},
	}, &initResult); err != nil {
		return err
	}

	c.mu.Lock()
	c.initialized = true
	c.mu.Unlock()
	return nil
}

func (c *Client) ListTools(ctx context.Context) ([]ToolDefinition, error) {
	var result struct {
		Tools []ToolDefinition `json:"tools"`
	}
	if err := c.Call(ctx, "tools/list", map[string]interface{}{}, &result); err != nil {
		return nil, err
	}
	return result.Tools, nil
}

func (c *Client) CallTool(ctx context.Context, name string, input map[string]interface{}) (string, error) {
	var result CallToolResult
	if err := c.Call(ctx, "tools/call", CallToolParams{Name: name, Input: input}, &result); err != nil {
		return "", err
	}
	if len(result.Content) == 0 {
		return "", nil
	}
	var out strings.Builder
	for i, item := range result.Content {
		if item.Text == "" {
			continue
		}
		if i > 0 {
			out.WriteString("\n")
		}
		out.WriteString(item.Text)
	}
	return out.String(), nil
}
