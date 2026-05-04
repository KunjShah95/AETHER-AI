package provider

import (
	"context"
	"fmt"
	"os"
)

type Gateway struct {
	providers map[string]Provider
	active    string
}

func NewGateway() *Gateway {
	return &Gateway{
		providers: make(map[string]Provider),
	}
}

func (g *Gateway) Register(name string, p Provider) {
	g.providers[name] = p
	if g.active == "" {
		g.active = name
	}
}

func (g *Gateway) SetActive(name string) error {
	if _, ok := g.providers[name]; !ok {
		return fmt.Errorf("unknown provider: %s", name)
	}
	g.active = name
	return nil
}

func (g *Gateway) Get() Provider {
	return g.providers[g.active]
}

func (g *Gateway) Active() string {
	return g.active
}

func (g *Gateway) Send(ctx context.Context, msgs []Message, tools []Tool) (*Response, error) {
	return g.Get().Send(ctx, msgs, tools)
}

func (g *Gateway) Stream(ctx context.Context, msgs []Message, tools []Tool) (<-chan Event, error) {
	return g.Get().Stream(ctx, msgs, tools)
}

func (g *Gateway) Model() string {
	if g.Get() != nil {
		return g.Get().Model()
	}
	return ""
}

// InitializeFromConfig sets up providers based on configuration
func (g *Gateway) InitializeFromConfig(provider string, apiKey string) error {
	switch provider {
	case "anthropic":
		model := os.Getenv("ANTHROPIC_MODEL")
		if model == "" {
			model = "claude-sonnet-4-20250514"
		}
		g.Register("anthropic", NewAnthropic(apiKey, model))
	case "ollama":
		baseURL := os.Getenv("OLLAMA_BASE_URL")
		model := os.Getenv("OLLAMA_MODEL")
		if model == "" {
			model = "llama2"
		}
		g.Register("ollama", NewOllama(baseURL, model))
	default:
		return fmt.Errorf("unknown provider: %s", provider)
	}
	return nil
}
