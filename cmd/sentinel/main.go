package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"sentinel-ai/internal/config"
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

	// Start the server
	if err := startServer(); err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}
}

func startServer() error {
	// Get the path to the server binary
	serverPath, err := getServerPath()
	if err != nil {
		return fmt.Errorf("failed to get server path: %w", err)
	}

	// Spawn server as child process
	srvCmd := exec.Command(serverPath)
	srvCmd.Stdin = os.Stdin
	srvCmd.Stdout = os.Stdout
	srvCmd.Stderr = os.Stderr

	return srvCmd.Run()
}

func getServerPath() (string, error) {
	// Try to find the server binary in the same directory as the sentinel binary
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	exeDir := filepath.Dir(exe)
	serverPath := filepath.Join(exeDir, "sentinel-server.exe")
	return serverPath, nil
}
