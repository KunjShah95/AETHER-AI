package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
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
	Provider  string            `json:"provider,omitempty"`
	Model     string            `json:"model,omitempty"`
	Summary   string            `json:"summary,omitempty"`
}

type UpdateSessionConfigRequest struct {
	Provider string `json:"provider,omitempty"`
	Model    string `json:"model,omitempty"`
}

type ModelsResponse struct {
	Provider     string   `json:"provider"`
	CurrentModel string   `json:"current_model,omitempty"`
	Models       []string `json:"models"`
}

type modelLister interface {
	ListModels(ctx context.Context) ([]string, error)
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

type CreateMilestoneRequest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Version     string `json:"version,omitempty"`
}

type CreateMilestoneResponse struct {
	MilestoneID string `json:"milestone_id"`
	Status      string `json:"status"`
}

type CreatePhaseRequest struct {
	MilestoneID string `json:"milestone_id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Order       int    `json:"order,omitempty"`
}

type CreatePhaseResponse struct {
	PhaseID string `json:"phase_id"`
	Status  string `json:"status"`
}

type CreateTodoRequest struct {
	PhaseID     string `json:"phase_id,omitempty"`
	MilestoneID string `json:"milestone_id"`
	Content     string `json:"content"`
	Description string `json:"description,omitempty"`
	Priority    string `json:"priority,omitempty"`
}

type CreateTodoResponse struct {
	TodoID string `json:"todo_id"`
	Status string `json:"status"`
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
		Provider:  sess.State.Provider,
		Model:     sess.State.Model,
		Summary:   sess.State.Summary,
	})
}

func (s *Server) updateSessionConfigHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodPatch && r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sessionID := ps.ByName("id")
	var req UpdateSessionConfigRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	sess, err := s.sessionManager.GetSession(r.Context(), sessionID)
	if err != nil {
		http.Error(w, "Session not found", http.StatusNotFound)
		return
	}

	if strings.TrimSpace(req.Provider) != "" {
		sess.State.Provider = strings.ToLower(strings.TrimSpace(req.Provider))
	}
	if strings.TrimSpace(req.Model) != "" {
		sess.State.Model = strings.TrimSpace(req.Model)
	}

	if _, err := s.sessionManager.UpdateState(r.Context(), sessionID, sess.State); err != nil {
		http.Error(w, fmt.Sprintf("Failed to update session config: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(GetSessionResponse{
		SessionID: sessionID,
		State:     sess.State.Model,
		Provider:  sess.State.Provider,
		Model:     sess.State.Model,
		Summary:   sess.State.Summary,
	})
}

func (s *Server) modelsHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.writeModelsResponse(w, r, nil)
}

func (s *Server) sessionModelsHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	sess, err := s.sessionManager.GetSession(r.Context(), ps.ByName("id"))
	if err != nil {
		http.Error(w, "Session not found", http.StatusNotFound)
		return
	}

	s.writeModelsResponse(w, r, sess)
}

func (s *Server) writeModelsResponse(w http.ResponseWriter, r *http.Request, sess *session.Session) {
	w.Header().Set("Content-Type", "application/json")
	if s.gateway == nil || s.gateway.Get() == nil {
		json.NewEncoder(w).Encode(ModelsResponse{Provider: "", CurrentModel: "", Models: []string{}})
		return
	}

	providerName := s.gateway.Active()
	providerInstance := s.gateway.Get()
	currentModel := providerInstance.Model()
	if sess != nil {
		if resolved, err := s.resolveProvider(r.Context(), sess); err == nil && resolved != nil {
			providerInstance = resolved
			currentModel = resolved.Model()
			if strings.TrimSpace(sess.State.Provider) != "" {
				providerName = sess.State.Provider
			}
		} else if strings.TrimSpace(sess.State.Model) != "" {
			currentModel = sess.State.Model
		}
	}

	if lister, ok := providerInstance.(modelLister); ok {
		models, err := lister.ListModels(r.Context())
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to list models: %v", err), http.StatusBadGateway)
			return
		}
		json.NewEncoder(w).Encode(ModelsResponse{Provider: providerName, CurrentModel: currentModel, Models: models})
		return
	}

	json.NewEncoder(w).Encode(ModelsResponse{Provider: providerName, CurrentModel: currentModel, Models: []string{providerInstance.Model()}})
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

	response := s.respondToSession(ctx, sessionID)

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

func (s *Server) respondToSession(ctx context.Context, sessionID string) string {
	sess, err := s.sessionManager.GetSession(ctx, sessionID)
	if err != nil {
		return "Placeholder response from LLM"
	}

	providerInstance, err := s.resolveProvider(ctx, sess)
	if err != nil || providerInstance == nil {
		return "Placeholder response from LLM"
	}

	msgs := make([]provider.Message, 0, len(sess.Messages))
	for _, msg := range sess.Messages {
		content := sessionMessageText(msg)
		if content == "" {
			continue
		}
		msgs = append(msgs, provider.Message{
			Role:    providerRole(msg.Role),
			Content: content,
		})
	}

	resp, err := providerInstance.Send(ctx, msgs, nil)
	if err != nil || resp == nil || strings.TrimSpace(resp.Content) == "" {
		return "Placeholder response from LLM"
	}

	return resp.Content
}

func (s *Server) resolveProvider(ctx context.Context, sess *session.Session) (provider.Provider, error) {
	if s.providerResolver != nil {
		return s.providerResolver(ctx, sess)
	}
	if s.gateway != nil && s.gateway.Get() != nil && strings.TrimSpace(sess.State.Provider) == "" && strings.TrimSpace(sess.State.Model) == "" {
		return s.gateway.Get(), nil
	}
	merged := s.cfg.LLMs
	if strings.TrimSpace(sess.State.Provider) != "" {
		merged.Provider = sess.State.Provider
	}
	if strings.TrimSpace(sess.State.Model) != "" {
		merged.Model = sess.State.Model
	}
	gw := provider.NewGateway()
	if err := gw.InitializeFromLLMConfig(merged); err != nil && gw.Get() == nil {
		return nil, err
	}
	return gw.Get(), nil
}

func providerRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "assistant":
		return "assistant"
	case "system":
		return "system"
	default:
		return "user"
	}
}

func sessionMessageText(msg session.Message) string {
	var parts []string
	for _, part := range msg.Parts {
		if strings.TrimSpace(part.Content) != "" {
			parts = append(parts, part.Content)
		}
	}
	return strings.Join(parts, "\n")
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

func (s *Server) createMilestoneHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CreateMilestoneRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Name == "" {
		http.Error(w, "Name is required", http.StatusBadRequest)
		return
	}

	milestone := s.workflowManager.CreateMilestone(req.Name, req.Description)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(CreateMilestoneResponse{
		MilestoneID: milestone.ID,
		Status:      "created",
	})
}

func (s *Server) listMilestonesHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	milestones := s.workflowManager.ListMilestones()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(milestones)
}

func (s *Server) getMilestoneHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	milestoneID := ps.ByName("milestone_id")
	milestone := s.workflowManager.GetMilestone(milestoneID)
	if milestone == nil {
		http.Error(w, "Milestone not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(milestone)
}

func (s *Server) completeMilestoneHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	milestoneID := ps.ByName("milestone_id")
	if err := s.workflowManager.CompleteMilestone(milestoneID); err != nil {
		http.Error(w, fmt.Sprintf("Failed to complete milestone: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "completed"})
}

func (s *Server) createPhaseHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CreatePhaseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.MilestoneID == "" || req.Name == "" {
		http.Error(w, "MilestoneID and Name are required", http.StatusBadRequest)
		return
	}

	phase := s.workflowManager.CreatePhase(req.MilestoneID, req.Name, req.Description, req.Order)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(CreatePhaseResponse{
		PhaseID: phase.ID,
		Status:  "created",
	})
}

func (s *Server) listPhasesHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	milestoneID := ps.ByName("milestone_id")
	phases := s.workflowManager.ListPhases(milestoneID)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(phases)
}

func (s *Server) completePhaseHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	phaseID := ps.ByName("id")
	if err := s.workflowManager.CompletePhase(phaseID); err != nil {
		http.Error(w, fmt.Sprintf("Failed to complete phase: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "completed"})
}

func (s *Server) createTodoHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CreateTodoRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.PhaseID == "" || req.Content == "" {
		http.Error(w, "PhaseID and Content are required", http.StatusBadRequest)
		return
	}

	todo := s.workflowManager.CreateTodo(req.PhaseID, req.Content, req.Description)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(CreateTodoResponse{
		TodoID: todo.ID,
		Status: "created",
	})
}

func (s *Server) listTodosHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	milestoneID := ps.ByName("milestone_id")
	todos := s.workflowManager.ListTodos(milestoneID)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(todos)
}

func (s *Server) completeTodoHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	todoID := ps.ByName("id")
	if err := s.workflowManager.CompleteTodo(todoID); err != nil {
		http.Error(w, fmt.Sprintf("Failed to complete todo: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "completed"})
}

func (s *Server) getRoadmapHandler(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	milestoneID := ps.ByName("milestone_id")
	_ = milestoneID // TODO: filter by milestone
	roadmap := s.workflowManager.GetRoadmap()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(roadmap)
}
