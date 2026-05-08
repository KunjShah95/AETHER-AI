package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"sentinel-ai/internal/skills"

	"github.com/julienschmidt/httprouter"
)

type CommandRequest struct {
	Command string            `json:"command"`
	Args    map[string]string `json:"args,omitempty"`
}

type CommandResponse struct {
	Result string `json:"result"`
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

type CommandInfo struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Trigger     []string `json:"trigger,omitempty"`
}

func (s *Server) commandHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CommandRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(CommandResponse{
			Status: "error",
			Error:  "Invalid request body",
		})
		return
	}

	if req.Command == "" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(CommandResponse{
			Status: "error",
			Error:  "Command is required",
		})
		return
	}

	ctx := r.Context()
	result, err := s.executeCommand(ctx, req.Command, req.Args)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(CommandResponse{
			Status: "error",
			Error:  err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(CommandResponse{
		Result: result,
		Status: "success",
	})
}

func (s *Server) executeCommand(ctx context.Context, cmd string, args map[string]string) (string, error) {
	if s.skills == nil {
		return "", fmt.Errorf("skills registry not available")
	}

	skill := s.skills.Get(cmd)
	if skill == nil {
		return "", fmt.Errorf("command not found: %s", cmd)
	}

	parsedSkill, err := skills.ParseSkillContent(skill.Path, skill.Content)
	if err != nil {
		return "", fmt.Errorf("failed to parse skill: %w", err)
	}

	parsedSkillForExec := &skills.ParsedSkill{
		Name:             parsedSkill.Name,
		Objective:        parsedSkill.Objective,
		ExecutionContext: parsedSkill.ExecutionContext,
		Process:          parsedSkill.Process,
		AllowedTools:     parsedSkill.AllowedTools,
	}

	executor := skills.NewExecutor(nil)
	if !executor.CanExecute(parsedSkillForExec) {
		return "", fmt.Errorf("insufficient tools to execute command")
	}

	result, err := executor.Execute(ctx, parsedSkillForExec, args)

	if err != nil {
		return "", fmt.Errorf("execution failed: %w", err)
	}

	return result, nil
}

func (s *Server) listCommandsHandler(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	w.Header().Set("Content-Type", "application/json")

	commands := make([]CommandInfo, 0)
	if s.skills != nil {
		for _, skill := range s.skills.List() {
			commands = append(commands, CommandInfo{
				Name:        skill.Name,
				Description: skill.Description,
				Trigger:     skill.Trigger,
			})
		}
	}

	json.NewEncoder(w).Encode(commands)
}
