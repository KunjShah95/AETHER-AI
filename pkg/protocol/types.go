package protocol

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

const Version = "0.1.0"

type MessageKind string

const (
	KindHandshake MessageKind = "handshake"
	KindMessage   MessageKind = "message"
	KindHandoff   MessageKind = "handoff"
	KindAck       MessageKind = "ack"
	KindError     MessageKind = "error"
)

type Agent struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	Role         string   `json:"role,omitempty"`
	Capabilities []string `json:"capabilities,omitempty"`
}

type Envelope struct {
	Version   string      `json:"version"`
	Kind      MessageKind `json:"kind"`
	ID        string      `json:"id,omitempty"`
	From      Agent       `json:"from"`
	To        Agent       `json:"to,omitempty"`
	SessionID string      `json:"session_id,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
	Payload   any         `json:"payload,omitempty"`
	Error     string      `json:"error,omitempty"`
}

type HandshakeRequest struct {
	Agent     Agent    `json:"agent"`
	Protocols []string `json:"protocols,omitempty"`
	Workspace string   `json:"workspace,omitempty"`
	SessionID string   `json:"session_id,omitempty"`
}

type HandshakeResponse struct {
	Accepted  bool     `json:"accepted"`
	Agent     Agent    `json:"agent"`
	Protocols []string `json:"protocols,omitempty"`
	Message   string   `json:"message,omitempty"`
}

type HandoffRequest struct {
	SessionID string   `json:"session_id"`
	From      Agent    `json:"from"`
	To        Agent    `json:"to"`
	Summary   string   `json:"summary,omitempty"`
	Recent    []string `json:"recent,omitempty"`
}

type HandoffResponse struct {
	Accepted bool   `json:"accepted"`
	Message  string `json:"message,omitempty"`
}

func NewEnvelope(kind MessageKind, from Agent, to Agent, sessionID string, payload any) Envelope {
	return Envelope{
		Version:   Version,
		Kind:      kind,
		ID:        fmt.Sprintf("env_%d", time.Now().UnixNano()),
		From:      from,
		To:        to,
		SessionID: sessionID,
		Timestamp: time.Now().UTC(),
		Payload:   payload,
	}
}

func (e Envelope) Validate() error {
	if strings.TrimSpace(e.Version) == "" {
		return errors.New("protocol version is required")
	}
	if strings.TrimSpace(e.From.ID) == "" {
		return errors.New("from agent id is required")
	}
	if strings.TrimSpace(e.From.Name) == "" {
		return errors.New("from agent name is required")
	}
	if e.Kind == "" {
		return errors.New("message kind is required")
	}
	return nil
}
