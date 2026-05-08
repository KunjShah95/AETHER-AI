package main

import (
	"fmt"
	"os"

	"sentinel-ai/internal/config"
	"sentinel-ai/internal/server"
	"sentinel-ai/internal/session"
	"sentinel-ai/internal/workflow"
)

func main() {
	fmt.Println("Sentinel AI Server starting...")

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Error validating config: %v\n", err)
		os.Exit(1)
	}

	// Create session store
	store, err := session.NewStore("sentinel_sessions.db")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating session store: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	// Create workflow manager
	wfManager := workflow.NewManager()

	// Create and start HTTP server
	httpServer := server.New(cfg, store, wfManager)
	if err := httpServer.Start(":8080"); err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}
}
