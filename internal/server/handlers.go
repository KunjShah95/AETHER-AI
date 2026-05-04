package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"sentinel-ai/internal/provider"
	"sentinel-ai/internal/session"
	"sentinel-ai/internal/tool"

	"github.com/julienschmidt/httprouter"
)

type HealthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
}

type CreateSessionRequest struct {
	ModelID string `json:"model_id,omitempty"`
}

type CreateSessionResponse struct {
	SessionID string `json:"session_id"`
	Status    string `json:"status"`
}

type GetSessionResponse struct {
	SessionID string            `json:"session_id"`
	Messages  []session.Message `json:"messages"`
	State     string            `json:"state"`
	Summary   string            `json:"summary,omitempty"`
}

type SkillResponse struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Trigger     []string `json:"trigger,omitempty"`
	ApplyTo     []string `json:"apply_to,omitempty"`
	Path        string   `json:"path"`
}

type ChatRequest struct {
	Message string `json:"message"`
}

type ChatResponse struct {
	Response string `json:"response"`
	Role     string `json:"role"`
}

type ToolRequest struct {
	ToolName string                 `json:"tool_name"`
	Input    map[string]interface{} `json:"input"`
	Approved bool                   `json:"approved,omitempty"`
}

type ToolResponse struct {
	Result   string `json:"result,omitempty"`
	Error    string `json:"error,omitempty"`
	Approved bool   `json:"approved,omitempty"`
	Status   string `json:"status,omitempty"`
}

// Health check endpoint
func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(HealthResponse{
		Status:  "ok",
		Version: "0.1.0",
	})
}

// Create a new session
func (s *Server) createSessionHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CreateSessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	projectID := "default"
	if req.ModelID != "" {
		projectID = req.ModelID
	}

	sess, err := s.sessionManager.CreateSession(ctx, projectID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create session: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(CreateSessionResponse{
		SessionID: sess.ID,
		Status:    "created",
	})
}

// Get session details
func (s *Server) getSessionHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	sessionID := ps.ByName("id")

	ctx := r.Context()
	sess, err := s.sessionManager.GetSession(ctx, sessionID)
	if err != nil {
		http.Error(w, "Session not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(GetSessionResponse{
		SessionID: sessionID,
		Messages:  sess.Messages,
		State:     sess.State.Model,
		Summary:   sess.State.Summary,
	})
}

func (s *Server) listSkillsHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	w.Header().Set("Content-Type", "application/json")
	items := make([]SkillResponse, 0)
	if s.skills != nil {
		for _, skill := range s.skills.List() {
			items = append(items, SkillResponse{
				Name:        skill.Name,
				Description: skill.Description,
				Trigger:     skill.Trigger,
				ApplyTo:     skill.ApplyTo,
				Path:        skill.Path,
			})
		}
	}
	json.NewEncoder(w).Encode(items)
}

// Send chat message
func (s *Server) chatHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sessionID := ps.ByName("id")

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	ctx := r.Context()

	// Add user message to session
	userMsg := session.Message{
		Role: "user",
		Parts: []session.Part{
			{
				Type:    "text",
				Content: req.Message,
			},
		},
		Created: time.Now(),
	}

	err := s.sessionManager.AddMessage(ctx, sessionID, userMsg)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to add message: %v", err), http.StatusInternalServerError)
		return
	}

	// TODO: Send to LLM provider and get response
	response := "Placeholder response from LLM"

	// Add assistant message to session
	assistantMsg := session.Message{
		Role: "assistant",
		Parts: []session.Part{
			{
				Type:    "text",
				Content: response,
			},
		},
		Created: time.Now(),
	}

	s.sessionManager.AddMessage(ctx, sessionID, assistantMsg)

	if current, err := s.sessionManager.GetSession(ctx, sessionID); err == nil && len(current.Messages) > 10 {
		if compacted, compactErr := s.sessionManager.CompactSession(ctx, sessionID, 6); compactErr == nil {
			_ = compacted
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ChatResponse{
		Response: response,
		Role:     "assistant",
	})
}

// Execute a tool
func (s *Server) toolHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sessionID := ps.ByName("id")
	_ = sessionID // For future use

	var req ToolRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// Get tool from registry
	t := s.tools.Get(req.ToolName)
	if t == nil {
		http.Error(w, fmt.Sprintf("Tool not found: %s", req.ToolName), http.StatusNotFound)
		return
	}

	if err := tool.Authorize(t.Name(), t.PermissionLevel(), req.Approved); err != nil {
		permErr, _ := err.(*tool.PermissionError)
		status := http.StatusForbidden
		if permErr != nil && permErr.Level == tool.PermAsk {
			status = http.StatusPreconditionRequired
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		json.NewEncoder(w).Encode(ToolResponse{
			Error:    err.Error(),
			Approved: false,
			Status:   "approval_required",
		})
		return
	}

	// Execute tool
	result, err := t.Execute(r.Context(), req.Input)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ToolResponse{
			Error: err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ToolResponse{
		Result:   result,
		Approved: true,
		Status:   "executed",
	})
}

// Server-Sent Events stream
func (s *Server) streamHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	sessionID := ps.ByName("id")
	_ = sessionID // For future use

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// TODO: Implement streaming responses
	fmt.Fprintf(w, "data: {\"message\": \"streaming not yet implemented\"}\n\n")
}

var (
	_ provider.Provider
)
