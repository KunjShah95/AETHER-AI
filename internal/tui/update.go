package tui

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// LLM Response event
type llmResponseMsg struct {
	content string
	err     error
}

type sessionCreatedMsg struct {
	sessionID string
	err       error
}

type modelsListMsg struct {
	provider     string
	currentModel string
	models       []string
	err          error
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyCtrlD:
			return m, tea.Quit
		case tea.KeyTab:
			if next, hint, ok := autocompleteCommand(m.input.Value()); ok {
				m.input.SetValue(next)
				m.status = hint
				return m, nil
			}
			return m, nil
		}
		if m.sessionID == "" {
			m.input, cmd = m.input.Update(msg)
			return m, cmd
		}
		switch msg.Type {
		case tea.KeyEnter:
			userInput := strings.TrimSpace(m.input.Value())
			if userInput != "" && !m.processing {
				if isCommandInput(userInput) {
					m.input.SetValue("")
					return m, m.executeCommand(userInput)
				}
				// Add user message
				m.messages = append(m.messages, Message{
					Role:    "user",
					Content: userInput,
				})
				m.input.SetValue("")
				m.processing = true
				m.err = nil

				return m, sendUserMessage(m.baseURL, m.sessionID, userInput)
			}

		case tea.KeyCtrlU:
			m.input.SetValue("")
		}

		// Pass other keys to input
		m.input, cmd = m.input.Update(msg)
		return m, cmd

	case tea.WindowSizeMsg:
		m.SetSize(msg.Width, msg.Height)

	case sessionCreatedMsg:
		if msg.err != nil {
			m.err = msg.err
			return m, nil
		}
		m.sessionID = msg.sessionID
		m.err = nil
		m.status = "new session ready"

	case llmResponseMsg:
		m.processing = false
		if msg.err != nil {
			m.err = msg.err
		} else {
			m.messages = append(m.messages, Message{
				Role:    "assistant",
				Content: msg.content,
			})
			m.status = ""
		}

	case modelsListMsg:
		if msg.err != nil {
			m.err = msg.err
			return m, nil
		}
		providerName := displayProviderName(msg.provider)
		m.messages = append(m.messages, Message{Role: "assistant", Content: formatModelsMessage(providerName, msg.currentModel, msg.models)})
		m.status = ""

	default:
		// Let the input handle other messages
		m.input, cmd = m.input.Update(msg)
	}

	return m, nil
}

func (m *Model) executeCommand(raw string) tea.Cmd {
	fields := strings.Fields(strings.TrimSpace(raw))
	if len(fields) == 0 {
		return nil
	}

	cmd := normalizeCommandName(fields[0])
	args := fields[1:]

	switch cmd {
	case "help":
		m.messages = append(m.messages, Message{Role: "assistant", Content: commandHelpText()})
		m.status = ""
		return nil
	case "commands":
		m.messages = append(m.messages, Message{Role: "assistant", Content: commandHelpText()})
		m.status = ""
		return nil
	case "models":
		m.status = "loading available Ollama models..."
		return listModelsCmd(m.baseURL, m.sessionID)
	case "clear":
		m.messages = nil
		m.status = "chat cleared"
		return nil
	case "new":
		m.status = "creating new session..."
		m.sessionID = ""
		m.err = nil
		m.messages = nil
		return createSessionCmd(m.baseURL)
	case "load":
		if len(args) == 0 {
			m.err = fmt.Errorf("usage: .load <path>")
			return nil
		}
		content, err := os.ReadFile(filepath.Clean(args[0]))
		if err != nil {
			m.err = err
			return nil
		}
		m.input.SetValue(string(content))
		m.status = fmt.Sprintf("loaded %s into the editor", args[0])
		return nil
	case "add":
		if len(args) == 0 {
			m.err = fmt.Errorf("usage: .add <path>")
			return nil
		}
		content, err := os.ReadFile(filepath.Clean(args[0]))
		if err != nil {
			m.err = err
			return nil
		}
		payload := fmt.Sprintf("Added file %s:\n\n%s", args[0], string(content))
		m.messages = append(m.messages, Message{Role: "user", Content: payload})
		m.processing = true
		m.status = "sending file contents to the model"
		return sendUserMessage(m.baseURL, m.sessionID, payload)
	case "session":
		if len(args) > 0 && args[0] == "new" {
			m.status = "creating new session..."
			m.sessionID = ""
			m.err = nil
			m.messages = nil
			return createSessionCmd(m.baseURL)
		}
		m.err = fmt.Errorf("usage: /session new")
		return nil
	case "provider", "model":
		if len(args) == 0 {
			m.messages = append(m.messages, Message{Role: "assistant", Content: providerHelpText(cmd)})
			m.status = ""
			return nil
		}
		if strings.TrimSpace(m.sessionID) == "" {
			m.err = fmt.Errorf("create a session before switching %s", cmd)
			return nil
		}
		m.status = fmt.Sprintf("updating %s...", cmd)
		return updateSessionConfigCmd(m.baseURL, m.sessionID, cmd, args[0])
	default:
		m.err = fmt.Errorf("unknown command: %s", cmd)
		return nil
	}
}

func isCommandInput(input string) bool {
	return strings.HasPrefix(input, ".") || strings.HasPrefix(input, "/")
}

func normalizeCommandName(token string) string {
	return strings.TrimLeft(strings.TrimSpace(token), "/.")
}

func providerHelpText(cmd string) string {
	switch cmd {
	case "models":
		return "Usage: .models"
	case "provider":
		return "Usage: .provider <ollama|anthropic>"
	case "model":
		return "Usage: .model <model-name>"
	default:
		return ""
	}
}

func displayProviderName(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "ollama":
		return "Ollama"
	case "anthropic":
		return "Anthropic"
	case "":
		return "configured"
	default:
		return provider
	}
}

func updateSessionConfigCmd(baseURL, sessionID, key, value string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		payload := map[string]string{}
		switch key {
		case "provider":
			payload["provider"] = strings.ToLower(strings.TrimSpace(value))
		case "model":
			payload["model"] = strings.TrimSpace(value)
		}

		body, err := json.Marshal(payload)
		if err != nil {
			return llmResponseMsg{err: err}
		}

		endpoint := strings.TrimRight(baseURL, "/") + "/session/" + sessionID + "/config"
		req, err := http.NewRequestWithContext(ctx, http.MethodPatch, endpoint, bytes.NewReader(body))
		if err != nil {
			return llmResponseMsg{err: err}
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return llmResponseMsg{err: err}
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			if resp.StatusCode == http.StatusMethodNotAllowed {
				resp.Body.Close()
				req, err = http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
				if err != nil {
					return llmResponseMsg{err: err}
				}
				req.Header.Set("Content-Type", "application/json")
				resp, err = http.DefaultClient.Do(req)
				if err != nil {
					return llmResponseMsg{err: err}
				}
				defer resp.Body.Close()
				if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					// continue to decode the successful response below
				} else {
					return llmResponseMsg{err: fmt.Errorf("config update failed: %s", resp.Status)}
				}
			} else {
				return llmResponseMsg{err: fmt.Errorf("config update failed: %s", resp.Status)}
			}
		}

		var decoded struct {
			Provider string `json:"provider"`
			Model    string `json:"model"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
			return llmResponseMsg{err: err}
		}

		provider := strings.TrimSpace(decoded.Provider)
		model := strings.TrimSpace(decoded.Model)
		if provider == "" {
			provider = key
		}
		if model == "" {
			model = value
		}
		return llmResponseMsg{content: fmt.Sprintf("switched to %s %s", provider, model)}
	}
}

func listModelsCmd(baseURL, sessionID string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		tryEndpoints := []string{}
		if strings.TrimSpace(sessionID) != "" {
			tryEndpoints = append(tryEndpoints, strings.TrimRight(baseURL, "/")+"/session/"+sessionID+"/models")
		}
		tryEndpoints = append(tryEndpoints, strings.TrimRight(baseURL, "/")+"/models")

		var lastErr error
		for _, endpoint := range tryEndpoints {
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
			if err != nil {
				lastErr = err
				continue
			}

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				lastErr = err
				continue
			}

			var decoded struct {
				Provider     string   `json:"provider"`
				CurrentModel string   `json:"current_model"`
				Models       []string `json:"models"`
			}

			func() {
				defer resp.Body.Close()
				if resp.StatusCode == http.StatusNotFound {
					lastErr = fmt.Errorf("models endpoint not found")
					return
				}
				if resp.StatusCode < 200 || resp.StatusCode >= 300 {
					lastErr = fmt.Errorf("list models failed: %s", resp.Status)
					return
				}
				if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
					lastErr = err
					return
				}
				lastErr = nil
			}()

			if lastErr == nil {
				return modelsListMsg{provider: decoded.Provider, currentModel: decoded.CurrentModel, models: decoded.Models}
			}
		}

		return modelsListMsg{err: lastErr}
	}
}

func formatModelsMessage(providerName, currentModel string, models []string) string {
	providerName = displayProviderName(providerName)
	currentModel = strings.TrimSpace(currentModel)

	var lines []string
	if currentModel != "" {
		lines = append(lines, fmt.Sprintf("Current %s model: %s", providerName, currentModel))
	}

	seen := make(map[string]struct{}, len(models))
	if currentModel != "" {
		seen[strings.ToLower(currentModel)] = struct{}{}
	}

	filtered := make([]string, 0, len(models))
	for _, model := range models {
		trimmed := strings.TrimSpace(model)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		filtered = append(filtered, trimmed)
	}

	if len(filtered) == 0 {
		if len(lines) == 0 {
			return fmt.Sprintf("No %s models found.", providerName)
		}
		return strings.Join(lines, "\n")
	}

	if len(lines) == 0 {
		lines = append(lines, fmt.Sprintf("Available %s models:", providerName))
	} else {
		lines = append(lines, fmt.Sprintf("Available %s models:", providerName))
	}
	for _, model := range filtered {
		lines = append(lines, "- "+model)
	}
	return strings.Join(lines, "\n")
}

func autocompleteCommand(input string) (string, string, bool) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" || !(strings.HasPrefix(trimmed, ".") || strings.HasPrefix(trimmed, "/")) {
		return "", "", false
	}

	prefix := normalizeCommandName(trimmed)
	matches := commandMatches(prefix)
	if len(matches) == 0 {
		return "", "", false
	}

	if len(matches) == 1 {
		next := "." + matches[0].Name
		if len(strings.Fields(trimmed)) == 1 {
			next += " "
		}
		return next, fmt.Sprintf("command: %s", matches[0].Usage), true
	}

	parts := make([]string, 0, len(matches))
	for _, match := range matches {
		parts = append(parts, match.Usage)
	}
	return "", "matches: " + strings.Join(parts, ", "), true
}

func createSessionCmd(baseURL string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()

		client := &http.Client{Timeout: 1 * time.Second}
		endpoint := strings.TrimRight(baseURL, "/") + "/session"
		var lastErr error
		for {
			select {
			case <-ctx.Done():
				if lastErr != nil {
					return sessionCreatedMsg{err: lastErr}
				}
				return sessionCreatedMsg{err: fmt.Errorf("create session timed out")}
			default:
			}

			req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBufferString(`{}`))
			if err != nil {
				return sessionCreatedMsg{err: err}
			}

			req.Header.Set("Content-Type", "application/json")
			resp, err := client.Do(req)
			if err != nil {
				lastErr = err
				time.Sleep(100 * time.Millisecond)
				continue
			}

			var decoded struct {
				SessionID string `json:"session_id"`
			}
			func() {
				defer resp.Body.Close()

				if resp.StatusCode < 200 || resp.StatusCode >= 300 {
					lastErr = fmt.Errorf("create session failed: %s", resp.Status)
					return
				}
				if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
					lastErr = err
					return
				}

				if decoded.SessionID == "" {
					lastErr = fmt.Errorf("create session failed: empty session id")
					return
				}

				lastErr = nil
				return
			}()

			if lastErr == nil {
				return sessionCreatedMsg{sessionID: decoded.SessionID}
			}
			time.Sleep(100 * time.Millisecond)
		}
	}
}

// Command that sends message to LLM
func sendUserMessage(baseURL, sessionID, content string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()

		payload := map[string]string{"message": content}
		body, err := json.Marshal(payload)
		if err != nil {
			return llmResponseMsg{err: err}
		}

		endpoint := strings.TrimRight(baseURL, "/") + "/session/" + sessionID + "/chat"
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return llmResponseMsg{err: err}
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return llmResponseMsg{err: err}
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return llmResponseMsg{err: fmt.Errorf("chat failed: %s", resp.Status)}
		}

		var decoded struct {
			Response string `json:"response"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
			return llmResponseMsg{err: err}
		}

		return llmResponseMsg{content: decoded.Response}
	}
}
