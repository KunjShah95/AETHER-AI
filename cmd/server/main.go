package main

import (
	"fmt"
	"os"

	"sentinel-ai/internal/config"
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

	fmt.Printf("Config loaded: provider=%s, model=%s\n", cfg.LLMs.Provider, cfg.LLMs.Model)
	fmt.Println("Server is ready to accept connections")
}
