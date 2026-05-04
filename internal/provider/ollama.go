package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type OllamaProvider struct {
	baseURL string
	model   string
	client  *http.Client
}

func NewOllama(baseURL, model string) *OllamaProvider {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	return &OllamaProvider{
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{},
	}
}

func (p *OllamaProvider) Model() string {
	return p.model
}

func (p *OllamaProvider) Send(ctx context.Context, msgs []Message, tools []Tool) (*Response, error) {
	reqBody := map[string]interface{}{
		"model":    p.model,
		"messages": convertOllamaMessages(msgs),
		"stream":   false,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		p.baseURL+"/api/chat",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}

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
		return nil, fmt.Errorf("Ollama error: %s", string(respBody))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, err
	}

	messageMap, ok := result["message"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected response format")
	}

	content, ok := messageMap["content"].(string)
	if !ok {
		return nil, fmt.Errorf("missing content in response")
	}

	return &Response{Content: content}, nil
}

func (p *OllamaProvider) Stream(ctx context.Context, msgs []Message, tools []Tool) (<-chan Event, error) {
	ch := make(chan Event)
	return ch, nil
}

func convertOllamaMessages(msgs []Message) []map[string]interface{} {
	var result []map[string]interface{}
	for _, m := range msgs {
		result = append(result, map[string]interface{}{
			"role":    m.Role,
			"content": m.Content,
		})
	}
	return result
}
