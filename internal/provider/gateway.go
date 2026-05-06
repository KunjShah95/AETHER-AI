package provider

import (
	"context"
	"fmt"
	"os"
	"strings"

	"sentinel-ai/internal/config"
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

// InitializeFromLLMConfig sets up providers based on configuration.
func (g *Gateway) InitializeFromLLMConfig(cfg config.LLMConfig) error {
	provider := strings.ToLower(strings.TrimSpace(cfg.Provider))
	if provider == "" {
		provider = "ollama"
	}
	switch provider {
	case "anthropic":
		model := strings.TrimSpace(cfg.Model)
		if model == "" {
			model = os.Getenv("ANTHROPIC_MODEL")
		}
		if model == "" {
			model = "claude-sonnet-4-20250514"
		}
		g.Register("anthropic", NewAnthropic(cfg.APIKey, model))
	case "ollama":
		baseURL := strings.TrimSpace(cfg.BaseURL)
		if baseURL == "" {
			baseURL = os.Getenv("OLLAMA_BASE_URL")
		}
		model := strings.TrimSpace(cfg.Model)
		if model == "" {
			model = os.Getenv("OLLAMA_MODEL")
		}
		if model == "" {
			model = "llama3.2"
		}
		g.Register("ollama", NewOllama(baseURL, model))
	default:
		baseURL := strings.TrimSpace(cfg.BaseURL)
		if baseURL == "" {
			baseURL = os.Getenv("OLLAMA_BASE_URL")
		}
		model := strings.TrimSpace(cfg.Model)
		if model == "" {
			model = os.Getenv("OLLAMA_MODEL")
		}
		if model == "" {
			model = "llama3.2"
		}
		g.Register("ollama", NewOllama(baseURL, model))
		return fmt.Errorf("unknown provider: %s; defaulted to ollama", provider)
	}
	return nil
}

// InitializeFromConfig is kept for compatibility with older callers.
func (g *Gateway) InitializeFromConfig(provider string, apiKey string) error {
	return g.InitializeFromLLMConfig(config.LLMConfig{Provider: provider, APIKey: apiKey})
}
