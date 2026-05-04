package tui

import (
	"fmt"
	"strings"
)

func (m *Model) View() string {
	if m.width == 0 || m.height == 0 {
		return "Loading..."
	}

	var s strings.Builder

	// Header
	s.WriteString(headerStyle.Render("╭─ Sentinel AI"))
	s.WriteString("\n")
	s.WriteString(fmt.Sprintf("│ Session: %s\n", m.sessionID))
	s.WriteString(separatorStyle.Render("├" + strings.Repeat("─", m.width-2)))
	s.WriteString("\n")

	// Message display area (80% of height)
	messageHeight := (m.height * 80) / 100
	if messageHeight < 5 {
		messageHeight = 5
	}

	messageLines := m.renderMessages()
	
	// Calculate scroll and display messages
	startIdx := 0
	if len(messageLines) > messageHeight {
		startIdx = len(messageLines) - messageHeight
		if startIdx < 0 {
			startIdx = 0
		}
	}

	for i := startIdx; i < len(messageLines) && i-startIdx < messageHeight; i++ {
		s.WriteString(messageLines[i])
		s.WriteString("\n")
	}

	// Padding to fill message area
	displayedLines := len(messageLines) - startIdx
	for i := displayedLines; i < messageHeight; i++ {
		s.WriteString("│\n")
	}

	// Separator before input
	s.WriteString(separatorStyle.Render("├" + strings.Repeat("─", m.width-2)))
	s.WriteString("\n")

	// Status line
	if m.processing {
		s.WriteString(processingStyle.Render("⟳ Thinking..."))
	} else if m.err != nil {
		s.WriteString(errorStyle.Render(fmt.Sprintf("✗ Error: %v", m.err)))
	} else {
		s.WriteString("│")
	}
	s.WriteString("\n")

	// Input area
	s.WriteString(inputStyle.Render("> " + m.input.View()))
	s.WriteString("\n")
	s.WriteString("╰");s.WriteString(strings.Repeat("─", m.width-1))

	return s.String()
}

func (m *Model) renderMessages() []string {
	var lines []string

	for _, msg := range m.messages {
		if msg.Role == "user" {
			lines = append(lines, userStyle.Render("│ You: "+msg.Content))
		} else {
			lines = append(lines, assistantStyle.Render("│ Sentinel: "+msg.Content))
		}
	}

	return lines
}
