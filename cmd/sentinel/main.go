package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"sentinel-ai/internal/config"
	"sentinel-ai/internal/server"
	"sentinel-ai/internal/session"
	"sentinel-ai/internal/tui"
	"sentinel-ai/internal/workflow"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Error validating config: %v\n", err)
		os.Exit(1)
	}

	shutdown, sessionID, err := startServer(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if shutdown != nil {
			_ = shutdown()
		}
	}()

	program := tea.NewProgram(tui.NewModel("http://127.0.0.1:8080", sessionID), tea.WithAltScreen())
	if _, err := program.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
		os.Exit(1)
	}
}

func startServer(cfg *config.Config) (func() error, string, error) {
	store, err := session.NewStore("sentinel_sessions.db")
	if err != nil {
		return nil, "", err
	}

	wfStore, err := workflow.NewStore("workflow.db")
	if err != nil {
		_ = store.Close()
		return nil, "", err
	}

	wfManager := workflow.NewManager(wfStore)

	manager := session.NewManager(store)
	initialSession, err := manager.CreateSession(context.Background(), "default")
	if err != nil {
		_ = store.Close()
		_ = wfStore.Close()
		return nil, "", err
	}
	app := server.New(cfg, store, wfManager)
	httpServer := &http.Server{
		Addr:    ":8080",
		Handler: app.Handler(),
	}

	go func() {
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		}
	}()

	if err := waitForServer("http://127.0.0.1:8080/health", 5*time.Second); err != nil {
		_ = httpServer.Shutdown(context.Background())
		_ = store.Close()
		_ = wfStore.Close()
		return nil, "", err
	}

	return func() error {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil {
			_ = store.Close()
			_ = wfStore.Close()
			return err
		}
		_ = wfStore.Close()
		return store.Close()
	}, initialSession.ID, nil
}

func waitForServer(url string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	client := &http.Client{Timeout: 750 * time.Millisecond}
	var lastErr error
	for {
		select {
		case <-ctx.Done():
			if lastErr != nil {
				return fmt.Errorf("server did not become ready: %w", lastErr)
			}
			return fmt.Errorf("server did not become ready")
		default:
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(req)
		if err == nil && resp != nil && resp.StatusCode >= 200 && resp.StatusCode < 500 {
			_ = resp.Body.Close()
			return nil
		}
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}
		lastErr = err
		time.Sleep(75 * time.Millisecond)
	}
}
