package server

import (
	"fmt"
	"net/http"

	"github.com/julienschmidt/httprouter"
	"sentinel-ai/internal/config"
	"sentinel-ai/internal/session"
)

type Server struct {
	router         *httprouter.Router
	cfg            *config.Config
	sessionManager *session.Manager
}

func New(cfg *config.Config, store *session.Store) *Server {
	sm := session.NewManager(store)

	r := httprouter.New()

	s := &Server{
		router:         r,
		cfg:            cfg,
		sessionManager: sm,
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
