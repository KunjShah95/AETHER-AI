package tui

import (
	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/textinput"
)

type Model struct {
	sessionID    string
	input        textinput.Model
	messages     []Message
	messageIndex int
	processing   bool
	err          error
	width        int
	height       int
	scrollOffset int
}

type Message struct {
	Role    string // "user" or "assistant"
	Content string
}

func NewModel(sessionID string) *Model {
	ti := textinput.New()
	ti.Placeholder = "Ask Sentinel AI..."
	ti.Focus()
	ti.CharLimit = 500
	ti.Width = 80

	return &Model{
		sessionID: sessionID,
		input:     ti,
		messages:  []Message{},
	}
}

func (m *Model) Init() tea.Cmd {
	return textinput.Blink
}

func (m *Model) SetSize(width, height int) {
	m.width = width
	m.height = height
	m.input.Width = width - 4
}
