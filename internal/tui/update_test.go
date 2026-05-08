package tui

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCreateSessionCmd(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/session" {
			http.Error(w, "unexpected request", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"session_id": "sess_123"})
	}))
	defer server.Close()

	msg := createSessionCmd(server.URL)()
	created, ok := msg.(sessionCreatedMsg)
	if !ok {
		t.Fatalf("message type = %T, want sessionCreatedMsg", msg)
	}
	if created.err != nil {
		t.Fatalf("createSessionCmd() error = %v", created.err)
	}
	if created.sessionID != "sess_123" {
		t.Fatalf("sessionID = %q, want %q", created.sessionID, "sess_123")
	}
}

func TestSendUserMessage(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/session":
			_ = json.NewEncoder(w).Encode(map[string]string{"session_id": "sess_123"})
		case r.Method == http.MethodPost && r.URL.Path == "/session/sess_123/chat":
			_ = json.NewEncoder(w).Encode(map[string]string{"response": "hello back"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	msg := sendUserMessage(server.URL, "sess_123", "hello")()
	response, ok := msg.(llmResponseMsg)
	if !ok {
		t.Fatalf("message type = %T, want llmResponseMsg", msg)
	}
	if response.err != nil {
		t.Fatalf("sendUserMessage() error = %v", response.err)
	}
	if response.content != "hello back" {
		t.Fatalf("content = %q, want %q", response.content, "hello back")
	}
}

func TestExecuteCommandLoad(t *testing.T) {
	t.Parallel()

	file := t.TempDir() + string(filepath.Separator) + "note.txt"
	if err := os.WriteFile(file, []byte("hello file"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	m := NewModel("http://example.com", "sess_1")
	if cmd := m.executeCommand(".load " + file); cmd != nil {
		t.Fatalf("/load should not return a tea.Cmd")
	}
	if got := m.input.Value(); got != "hello file" {
		t.Fatalf("input = %q, want %q", got, "hello file")
	}
}

func TestExecuteCommandAliasSlashHelp(t *testing.T) {
	t.Parallel()

	m := NewModel("http://example.com", "sess_1")
	if cmd := m.executeCommand("/help"); cmd != nil {
		t.Fatalf("/help should not return a tea.Cmd")
	}
	if got := len(m.messages); got == 0 {
		t.Fatal("expected help command to append a message")
	}
	if got := m.messages[len(m.messages)-1].Content; !strings.Contains(got, ".help") {
		t.Fatalf("help output = %q, want dot commands", got)
	}
}

func TestAutocompleteCommand(t *testing.T) {
	t.Parallel()

	next, hint, ok := autocompleteCommand(".lo")
	if !ok {
		t.Fatal("expected autocomplete to match .lo")
	}
	if next != ".load " {
		t.Fatalf("next = %q, want %q", next, ".load ")
	}
	if !strings.Contains(hint, ".load") {
		t.Fatalf("hint = %q, want load hint", hint)
	}
}

func TestCommandHelpText(t *testing.T) {
	t.Parallel()

	help := commandHelpText()
	if !strings.Contains(help, ".help") || !strings.Contains(help, ".add") || !strings.Contains(help, ".models") {
		t.Fatalf("help text missing expected commands: %s", help)
	}
}

func TestDisplayProviderName(t *testing.T) {
	t.Parallel()

	if got := displayProviderName("ollama"); got != "Ollama" {
		t.Fatalf("displayProviderName(ollama) = %q, want %q", got, "Ollama")
	}
	if got := displayProviderName(""); got != "configured" {
		t.Fatalf("displayProviderName(empty) = %q, want configured", got)
	}
}

func TestFormatModelsMessage(t *testing.T) {
	t.Parallel()

	msg := formatModelsMessage("ollama", "llama3.2", []string{"llama3.2", "mistral", "qwen2.5"})
	if !strings.Contains(msg, "Current Ollama model: llama3.2") {
		t.Fatalf("message missing current model line: %s", msg)
	}
	firstModelLine := strings.Index(msg, "- mistral")
	if firstModelLine == -1 {
		t.Fatalf("message missing remaining models: %s", msg)
	}
	if strings.Index(msg, "Current Ollama model: llama3.2") > firstModelLine {
		t.Fatalf("current model should appear before model list: %s", msg)
	}
}
