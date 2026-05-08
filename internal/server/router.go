package server

import (
	"context"
	"fmt"
	"net/http"
	"os"

	"sentinel-ai/internal/config"
	"sentinel-ai/internal/mcp"
	"sentinel-ai/internal/provider"
	"sentinel-ai/internal/session"
	"sentinel-ai/internal/skills"
	"sentinel-ai/internal/tool"
	"sentinel-ai/internal/workflow"
	"sentinel-ai/pkg/protocol"

	"github.com/julienschmidt/httprouter"
)

type Server struct {
	router           *httprouter.Router
	cfg              *config.Config
	sessionManager   *session.Manager
	gateway          *provider.Gateway
	providerResolver func(context.Context, *session.Session) (provider.Provider, error)
	skills           *skills.Registry
	tools            *tool.Registry
	mcpBridge        *mcp.Bridge
	protocolHub      *protocol.Hub
	workflowManager  *workflow.Manager
}

func New(cfg *config.Config, store *session.Store, wfManager *workflow.Manager) *Server {
	if cfg == nil {
		cfg = &config.Config{}
	}

	sm := session.NewManager(store)
	registry := tool.NewRegistryWithBuiltins()
	bridge := mcp.NewBridge()
	gateway := provider.NewGateway()
	_ = gateway.InitializeFromLLMConfig(cfg.LLMs)
	skillRegistry := skills.NewRegistry()
	for _, root := range skills.DefaultRoots() {
		if info, err := os.Stat(root); err == nil && info.IsDir() {
			if loaded, err := skills.LoadDefault(context.Background()); err == nil && loaded != nil {
				skillRegistry = loaded
			}
			break
		}
	}
	if len(cfg.MCPs) > 0 {
		if err := bridge.LoadFromConfig(context.Background(), cfg.MCPs); err == nil {
			for _, remoteTool := range bridge.Tools() {
				registry.Register(remoteTool)
			}
		}
	}
	localAgent := protocol.Agent{
		ID:           "sentinel-ai",
		Name:         "Sentinel AI",
		Role:         "coding-assistant",
		Capabilities: []string{"chat", "tool-use", "skill-loading", "session-compaction", "acp", "a2a"},
	}
	protocolHub := protocol.NewHub(localAgent)

	r := httprouter.New()

	s := &Server{
		router:          r,
		cfg:             cfg,
		sessionManager:  sm,
		gateway:         gateway,
		skills:          skillRegistry,
		tools:           registry,
		mcpBridge:       bridge,
		protocolHub:     protocolHub,
		workflowManager: wfManager,
	}

	s.registerRoutes()
	return s
}

func (s *Server) registerRoutes() {
	// Health check
	s.router.GET("/health", s.healthHandler)

	// Session management
	s.router.POST("/session", s.createSessionHandler)
	s.router.GET("/session/:id", s.getSessionHandler)
	s.router.GET("/session/:id/models", s.sessionModelsHandler)
	s.router.GET("/models", s.modelsHandler)
	s.router.GET("/skills", s.listSkillsHandler)
	s.router.GET("/protocol", s.protocolStatusHandler)
	s.router.POST("/protocol/handshake", s.protocolHandshakeHandler)
	s.router.POST("/protocol/message", s.protocolMessageHandler)
	s.router.POST("/protocol/handoff", s.protocolHandoffHandler)

	// Chat and tools
	s.router.POST("/session/:id/chat", s.chatHandler)
	s.router.PATCH("/session/:id/config", s.updateSessionConfigHandler)
	s.router.POST("/session/:id/config", s.updateSessionConfigHandler)
	s.router.POST("/session/:id/tool", s.toolHandler)

	// Server-Sent Events for streaming
	s.router.GET("/session/:id/stream", s.streamHandler)

	// Workflow management - Milestones
	s.router.POST("/milestones", s.createMilestoneHandler)
	s.router.GET("/milestones", s.listMilestonesHandler)

	// Workflow management - Milestone sub-resources (use milestone_id to avoid conflict)
	s.router.GET("/milestones/:milestone_id/phases", s.listPhasesHandler)
	s.router.GET("/milestones/:milestone_id/todos", s.listTodosHandler)
	s.router.GET("/milestones/:milestone_id/roadmap", s.getRoadmapHandler)

	// Single milestone operations
	s.router.GET("/milestones/:milestone_id", s.getMilestoneHandler)
	s.router.PUT("/milestones/:milestone_id/complete", s.completeMilestoneHandler)

	// Workflow management - Phases
	s.router.POST("/phases", s.createPhaseHandler)
	s.router.PUT("/phases/:id/complete", s.completePhaseHandler)

	// Workflow management - Todos
	s.router.POST("/todos", s.createTodoHandler)
	s.router.PUT("/todos/:id/complete", s.completeTodoHandler)
}

func (s *Server) Start(addr string) error {
	fmt.Printf("Starting Sentinel AI Server on %s\n", addr)
	return http.ListenAndServe(addr, s.router)
}

func (s *Server) Handler() http.Handler {
	return s.router
}
