package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"sentinel-ai/internal/session"
	"sentinel-ai/pkg/protocol"

	"github.com/julienschmidt/httprouter"
)

type ProtocolStatusResponse struct {
	Version string           `json:"version"`
	Local   protocol.Agent   `json:"local"`
	Peers   []protocol.Agent `json:"peers"`
}

type ProtocolAckResponse struct {
	Accepted bool   `json:"accepted"`
	Message  string `json:"message,omitempty"`
}

type protocolMessageRequest struct {
	Envelope protocol.Envelope `json:"envelope"`
}

func (s *Server) protocolStatusHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ProtocolStatusResponse{
		Version: protocol.Version,
		Local:   s.protocolHub.LocalAgent(),
		Peers:   s.protocolHub.Peers(),
	})
}

func (s *Server) protocolHandshakeHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	var req protocol.HandshakeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Agent.ID) == "" || strings.TrimSpace(req.Agent.Name) == "" {
		http.Error(w, "agent id and name are required", http.StatusBadRequest)
		return
	}

	s.protocolHub.RegisterPeer(req.Agent)
	response := protocol.HandshakeResponse{
		Accepted:  true,
		Agent:     s.protocolHub.LocalAgent(),
		Protocols: []string{"acp", "a2a", "session-summary"},
		Message:   "handshake accepted",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) protocolMessageHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	var req protocolMessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	if err := req.Envelope.Validate(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.protocolHub.Record(req.Envelope)

	if req.Envelope.SessionID != "" {
		content := envelopeContent(req.Envelope)
		if content != "" {
			_ = s.sessionManager.AddMessage(r.Context(), req.Envelope.SessionID, session.Message{
				Role: req.Envelope.From.Name,
				Parts: []session.Part{{
					Type:    "text",
					Content: content,
				}},
			})
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ProtocolAckResponse{
		Accepted: true,
		Message:  "message recorded",
	})
}

func (s *Server) protocolHandoffHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	var req protocol.HandoffRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.SessionID) == "" {
		http.Error(w, "session_id is required", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.To.ID) == "" || strings.TrimSpace(req.To.Name) == "" {
		http.Error(w, "target agent id and name are required", http.StatusBadRequest)
		return
	}

	s.protocolHub.RegisterPeer(req.To)
	if req.Summary != "" {
		if _, err := s.sessionManager.SetSummary(r.Context(), req.SessionID, req.Summary); err != nil {
			http.Error(w, fmt.Sprintf("failed to save summary: %v", err), http.StatusInternalServerError)
			return
		}
	}
	if _, err := s.sessionManager.CompactSession(r.Context(), req.SessionID, 4); err != nil {
		http.Error(w, fmt.Sprintf("failed to compact session: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(protocol.HandoffResponse{
		Accepted: true,
		Message:  "handoff accepted",
	})
}

func envelopeContent(env protocol.Envelope) string {
	switch payload := env.Payload.(type) {
	case string:
		return payload
	case map[string]any:
		if text, ok := payload["text"].(string); ok {
			return text
		}
		if text, ok := payload["content"].(string); ok {
			return text
		}
	}
	return ""
}
