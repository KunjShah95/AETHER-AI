package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"

	"sentinel-ai/internal/config"
	"sentinel-ai/internal/provider"
	"sentinel-ai/internal/session"
)

type mockProvider struct {
	mu     sync.Mutex
	msgs   []provider.Message
	model  string
	output string
	models []string
}

func (m *mockProvider) Send(ctx context.Context, msgs []provider.Message, tools []provider.Tool) (*provider.Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.msgs = append([]provider.Message(nil), msgs...)
	return &provider.Response{Content: m.output}, nil
}

func (m *mockProvider) Stream(ctx context.Context, msgs []provider.Message, tools []provider.Tool) (<-chan provider.Event, error) {
	ch := make(chan provider.Event)
	close(ch)
	return ch, nil
}

func (m *mockProvider) Model() string {
	if m.model == "" {
		return "mock-model"
	}
	return m.model
}

func (m *mockProvider) ListModels(ctx context.Context) ([]string, error) {
	if len(m.models) > 0 {
		return append([]string(nil), m.models...), nil
	}
	return []string{m.Model()}, nil
}

func TestChatHandlerUsesProviderResponse(t *testing.T) {
	t.Parallel()

	storePath := filepath.Join(t.TempDir(), "sessions.db")
	store, err := session.NewStore(storePath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	defer store.Close()

	srv := New(&config.Config{}, store)
	mock := &mockProvider{output: "mock assistant reply"}
	gateway := provider.NewGateway()
	gateway.Register("mock", mock)
	srv.gateway = gateway

	createReq, err := http.NewRequest(http.MethodPost, "/session", bytes.NewBufferString(`{}`))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	createRec := httptest.NewRecorder()
	srv.router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusOK {
		t.Fatalf("create session status = %d, body = %s", createRec.Code, createRec.Body.String())
	}

	var createResp CreateSessionResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &createResp); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	chatReq, err := http.NewRequest(http.MethodPost, "/session/"+createResp.SessionID+"/chat", bytes.NewBufferString(`{"message":"hello"}`))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	chatRec := httptest.NewRecorder()
	srv.router.ServeHTTP(chatRec, chatReq)
	if chatRec.Code != http.StatusOK {
		t.Fatalf("chat status = %d, body = %s", chatRec.Code, chatRec.Body.String())
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(chatRec.Body.Bytes(), &chatResp); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if chatResp.Response != mock.output {
		t.Fatalf("chat response = %q, want %q", chatResp.Response, mock.output)
	}

	sess, err := store.Get(context.Background(), createResp.SessionID)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got := len(sess.Messages); got != 2 {
		t.Fatalf("len(messages) = %d, want 2", got)
	}
	if got := mock.msgs; len(got) != 1 || got[0].Content != "hello" {
		t.Fatalf("provider saw msgs = %#v, want user hello", got)
	}
}

func TestSessionConfigUpdateSwitchesProvider(t *testing.T) {
	t.Parallel()

	storePath := filepath.Join(t.TempDir(), "sessions.db")
	store, err := session.NewStore(storePath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	defer store.Close()

	srv := New(&config.Config{}, store)
	mock := &mockProvider{output: "switched provider reply", model: "mock-model"}
	srv.gateway = nil
	srv.providerResolver = func(ctx context.Context, sess *session.Session) (provider.Provider, error) {
		if sess.State.Provider == "mock" {
			return mock, nil
		}
		return nil, nil
	}

	createReq, err := http.NewRequest(http.MethodPost, "/session", bytes.NewBufferString(`{}`))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	createRec := httptest.NewRecorder()
	srv.router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusOK {
		t.Fatalf("create session status = %d, body = %s", createRec.Code, createRec.Body.String())
	}

	var createResp CreateSessionResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &createResp); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	configReq, err := http.NewRequest(http.MethodPatch, "/session/"+createResp.SessionID+"/config", bytes.NewBufferString(`{"provider":"mock","model":"mock-model"}`))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	configRec := httptest.NewRecorder()
	srv.router.ServeHTTP(configRec, configReq)
	if configRec.Code != http.StatusOK {
		t.Fatalf("config status = %d, body = %s", configRec.Code, configRec.Body.String())
	}

	chatReq, err := http.NewRequest(http.MethodPost, "/session/"+createResp.SessionID+"/chat", bytes.NewBufferString(`{"message":"hello"}`))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	chatRec := httptest.NewRecorder()
	srv.router.ServeHTTP(chatRec, chatReq)
	if chatRec.Code != http.StatusOK {
		t.Fatalf("chat status = %d, body = %s", chatRec.Code, chatRec.Body.String())
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(chatRec.Body.Bytes(), &chatResp); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if chatResp.Response != mock.output {
		t.Fatalf("chat response = %q, want %q", chatResp.Response, mock.output)
	}

	sess, err := store.Get(context.Background(), createResp.SessionID)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if sess.State.Provider != "mock" || sess.State.Model != "mock-model" {
		t.Fatalf("stored state = %#v, want provider/model switched", sess.State)
	}
}

func TestModelsHandlerReturnsAvailableModels(t *testing.T) {
	t.Parallel()

	storePath := filepath.Join(t.TempDir(), "sessions.db")
	store, err := session.NewStore(storePath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	defer store.Close()

	srv := New(&config.Config{}, store)
	mock := &mockProvider{models: []string{"llama3.2", "mistral", "qwen2.5"}}
	gateway := provider.NewGateway()
	gateway.Register("ollama", mock)
	srv.gateway = gateway

	req, err := http.NewRequest(http.MethodGet, "/models", nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	rec := httptest.NewRecorder()
	srv.router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("models status = %d, body = %s", rec.Code, rec.Body.String())
	}

	var resp ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if resp.Provider != "ollama" {
		t.Fatalf("provider = %q, want %q", resp.Provider, "ollama")
	}
	if resp.CurrentModel != "mock-model" {
		t.Fatalf("current model = %q, want %q", resp.CurrentModel, "mock-model")
	}
	if len(resp.Models) != 3 {
		t.Fatalf("models = %#v, want 3 items", resp.Models)
	}
}
