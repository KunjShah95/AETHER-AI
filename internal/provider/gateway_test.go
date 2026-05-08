package provider

import (
	"testing"
)

func TestGatewayRegistration(t *testing.T) {
	gateway := NewGateway()

	// Register Anthropic provider
	gateway.Register("anthropic", NewAnthropic("test-key", "claude-3-sonnet"))

	if gateway.Active() != "anthropic" {
		t.Error("expected active provider to be anthropic")
	}

	// Get the provider
	provider := gateway.Get()
	if provider == nil {
		t.Error("expected provider to be non-nil")
	}

	if provider.Model() != "claude-3-sonnet" {
		t.Error("expected model to be claude-3-sonnet")
	}
}

func TestGatewaySetActive(t *testing.T) {
	gateway := NewGateway()

	gateway.Register("anthropic", NewAnthropic("test-key", "claude-3-sonnet"))
	gateway.Register("ollama", NewOllama("http://localhost:11434", "llama2"))

	if err := gateway.SetActive("ollama"); err != nil {
		t.Error(err)
	}

	if gateway.Active() != "ollama" {
		t.Error("expected active provider to be ollama")
	}

	if gateway.Get().Model() != "llama2" {
		t.Error("expected model to be llama2")
	}
}

func TestGatewaySetActiveError(t *testing.T) {
	gateway := NewGateway()

	if err := gateway.SetActive("nonexistent"); err == nil {
		t.Error("expected error for nonexistent provider")
	}
}

func TestGatewayInitializeFromConfig(t *testing.T) {
	gateway := NewGateway()

	if err := gateway.InitializeFromConfig("anthropic", "test-key"); err != nil {
		t.Error(err)
	}

	if gateway.Active() != "anthropic" {
		t.Error("expected active provider to be anthropic")
	}
}

func TestGatewayInitializeFromConfigError(t *testing.T) {
	gateway := NewGateway()

	if err := gateway.InitializeFromConfig("unknown", "test-key"); err == nil {
		t.Error("expected error for unknown provider")
	}
}
