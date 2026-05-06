package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

type CommandSpec struct {
	Name        string
	Usage       string
	Description string
}

var commandSpecs = []CommandSpec{
	{Name: "help", Usage: ".help", Description: "Show available commands"},
	{Name: "commands", Usage: ".commands", Description: "Show available commands"},
	{Name: "clear", Usage: ".clear", Description: "Clear the current conversation"},
	{Name: "new", Usage: ".new", Description: "Start a new session"},
	{Name: "session", Usage: ".session new", Description: "Session management commands"},
	{Name: "load", Usage: ".load <path>", Description: "Load a file into the input editor"},
	{Name: "add", Usage: ".add <path>", Description: "Add a file into the conversation"},
	{Name: "models", Usage: ".models", Description: "List available Ollama models"},
	{Name: "provider", Usage: ".provider <name>", Description: "Switch provider for this session"},
	{Name: "model", Usage: ".model <name>", Description: "Switch model for this session"},
}

type Model struct {
	baseURL    string
	sessionID  string
	input      textinput.Model
	messages   []Message
	processing bool
	err        error
	width      int
	height     int
	status     string
}

type Message struct {
	Role    string // "user" or "assistant"
	Content string
}

func NewModel(baseURL, sessionID string) *Model {
	ti := textinput.New()
	ti.Placeholder = "Ask Sentinel AI..."
	ti.Focus()
	ti.CharLimit = 500
	ti.Width = 80

	return &Model{
		baseURL:   baseURL,
		sessionID: sessionID,
		input:     ti,
		messages:  []Message{},
	}
}

func (m *Model) Init() tea.Cmd {
	if m.sessionID != "" {
		return textinput.Blink
	}
	return createSessionCmd(m.baseURL)
}

func commandMatches(prefix string) []CommandSpec {
	prefix = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(prefix, "/"), "."))
	if prefix == "" {
		return commandSpecs
	}
	results := make([]CommandSpec, 0)
	for _, spec := range commandSpecs {
		if strings.HasPrefix(spec.Name, prefix) {
			results = append(results, spec)
		}
	}
	return results
}

func commandHelpText() string {
	var b strings.Builder
	b.WriteString("Commands:\n")
	for _, spec := range commandSpecs {
		b.WriteString(fmt.Sprintf("%s  - %s\n", spec.Usage, spec.Description))
	}
	b.WriteString("Tip: press Tab after typing . to autocomplete a command.")
	return strings.TrimSpace(b.String())
}

func (m *Model) SetSize(width, height int) {
	m.width = width
	m.height = height
	m.input.Width = width - 4
}
