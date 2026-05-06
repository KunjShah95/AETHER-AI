package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
)

type OllamaProvider struct {
	baseURL string
	model   string
	client  *http.Client
}

type ollamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
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

func (p *OllamaProvider) ListModels(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.baseURL, "/")+"/api/tags", nil)
	if err != nil {
		return nil, err
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama tags error: %s", string(body))
	}

	var decoded ollamaTagsResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, err
	}

	models := make([]string, 0, len(decoded.Models))
	for _, model := range decoded.Models {
		if strings.TrimSpace(model.Name) != "" {
			models = append(models, model.Name)
		}
	}
	sort.Strings(models)
	return models, nil
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
