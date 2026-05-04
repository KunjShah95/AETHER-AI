package tui

import (
	"github.com/charmbracelet/lipgloss"
)

var (
	// Colors
	primaryColor   = lipgloss.Color("39")  // Cyan
	accentColor    = lipgloss.Color("212") // Magenta
	successColor   = lipgloss.Color("42")  // Green
	warningColor   = lipgloss.Color("214") // Orange
	errorColor     = lipgloss.Color("196") // Red

	// Header Style
	headerStyle = lipgloss.NewStyle().
		Foreground(primaryColor).
		Bold(true).
		PaddingBottom(1)

	// Message Styles
	userStyle = lipgloss.NewStyle().
		Foreground(successColor).
		PaddingLeft(2)

	assistantStyle = lipgloss.NewStyle().
		Foreground(accentColor).
		PaddingLeft(2)

	// Input Style
	inputStyle = lipgloss.NewStyle().
		Foreground(primaryColor).
		PaddingLeft(1)

	// Status Styles
	processingStyle = lipgloss.NewStyle().
		Foreground(warningColor).
		Italic(true).
		PaddingLeft(2)

	errorStyle = lipgloss.NewStyle().
		Foreground(errorColor).
		Bold(true).
		PaddingLeft(2)

	// Separator
	separatorStyle = lipgloss.NewStyle().
		Foreground(lipgloss.Color("240"))
)
