package server

import (
	"context"
	"fmt"
	"net/http"

	"sentinel-ai/internal/config"
	"sentinel-ai/internal/mcp"
	"sentinel-ai/internal/session"
	"sentinel-ai/internal/tool"

	"github.com/julienschmidt/httprouter"
)

type Server struct {
	router         *httprouter.Router
	cfg            *config.Config
	sessionManager *session.Manager
	tools          *tool.Registry
	mcpBridge      *mcp.Bridge
}

func New(cfg *config.Config, store *session.Store) *Server {
	sm := session.NewManager(store)
	registry := tool.NewRegistryWithBuiltins()
	bridge := mcp.NewBridge()
	if err := bridge.LoadFromConfig(context.Background(), cfg.MCPs); err == nil {
		for _, remoteTool := range bridge.Tools() {
			registry.Register(remoteTool)
		}
	}

	r := httprouter.New()

	s := &Server{
		router:         r,
		cfg:            cfg,
		sessionManager: sm,
		tools:          registry,
		mcpBridge:      bridge,
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

	// Chat and tools
	s.router.POST("/session/:id/chat", s.chatHandler)
	s.router.POST("/session/:id/tool", s.toolHandler)

	// Server-Sent Events for streaming
	s.router.GET("/session/:id/stream", s.streamHandler)
}

func (s *Server) Start(addr string) error {
	fmt.Printf("Starting Sentinel AI Server on %s\n", addr)
	return http.ListenAndServe(addr, s.router)
}
