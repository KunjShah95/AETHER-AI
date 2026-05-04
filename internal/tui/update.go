package tui

import (
	tea "github.com/charmbracelet/bubbletea"
)

// LLM Response event
type llmResponseMsg struct {
	content string
	err     error
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyCtrlD:
			return m, tea.Quit

		case tea.KeyEnter:
			userInput := m.input.Value()
			if userInput != "" && !m.processing {
				// Add user message
				m.messages = append(m.messages, Message{
					Role:    "user",
					Content: userInput,
				})
				m.input.SetValue("")
				m.processing = true
				m.err = nil

				// Send to LLM (command would be injected by parent)
				return m, sendUserMessage(userInput)
			}

		case tea.KeyCtrlU:
			m.input.SetValue("")
		}

		// Pass other keys to input
		m.input, cmd = m.input.Update(msg)
		return m, cmd

	case tea.WindowSizeMsg:
		m.SetSize(msg.Width, msg.Height)

	case llmResponseMsg:
		m.processing = false
		if msg.err != nil {
			m.err = msg.err
		} else {
			m.messages = append(m.messages, Message{
				Role:    "assistant",
				Content: msg.content,
			})
		}

	default:
		// Let the input handle other messages
		m.input, cmd = m.input.Update(msg)
	}

	return m, nil
}

// Command that sends message to LLM
func sendUserMessage(content string) tea.Cmd {
	return func() tea.Msg {
		// This will be replaced with actual LLM integration
		return llmResponseMsg{
			content: "This is a placeholder response from the LLM.",
			err:     nil,
		}
	}
}
