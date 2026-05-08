package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type AnthropicProvider struct {
	apiKey string
	model  string
	client *http.Client
}

func NewAnthropic(apiKey, model string) *AnthropicProvider {
	return &AnthropicProvider{
		apiKey: apiKey,
		model:  model,
		client: &http.Client{},
	}
}

func (p *AnthropicProvider) Model() string {
	return p.model
}

func (p *AnthropicProvider) Send(ctx context.Context, msgs []Message, tools []Tool) (*Response, error) {
	reqBody := map[string]interface{}{
		"model":      p.model,
		"max_tokens": 4096,
		"messages":   convertMessages(msgs),
	}

	if len(tools) > 0 {
		reqBody["tools"] = convertTools(tools)
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		"https://api.anthropic.com/v1/messages",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}

	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API error: %s", string(respBody))
	}

	var result map[string]interface{}
	json.Unmarshal(respBody, &result)

	content := extractContent(result)
	return &Response{Content: content}, nil
}

func (p *AnthropicProvider) Stream(ctx context.Context, msgs []Message, tools []Tool) (<-chan Event, error) {
	ch := make(chan Event)
	// Simplified - full implementation uses SSE
	return ch, nil
}

func convertMessages(msgs []Message) []map[string]interface{} {
	var result []map[string]interface{}
	for _, m := range msgs {
		result = append(result, map[string]interface{}{
			"role":    m.Role,
			"content": m.Content,
		})
	}
	return result
}

func convertTools(tools []Tool) []map[string]interface{} {
	var result []map[string]interface{}
	for _, t := range tools {
		result = append(result, map[string]interface{}{
			"name":        t.Name,
			"description": t.Description,
			"input_schema": t.InputSchema,
		})
	}
	return result
}

func extractContent(resp map[string]interface{}) string {
	content, ok := resp["content"].([]interface{})
	if !ok || len(content) == 0 {
		return ""
	}
	contentMap, ok := content[0].(map[string]interface{})
	if !ok {
		return ""
	}
	text, ok := contentMap["text"].(string)
	if !ok {
		return ""
	}
	return text
}
