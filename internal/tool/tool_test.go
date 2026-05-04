package tool

import (
	"context"
	"os"
	"testing"
)

func TestRegistry(t *testing.T) {
	registry := NewRegistryWithBuiltins()

	tools := registry.List()
	if len(tools) == 0 {
		t.Error("expected at least one tool to be registered")
	}

	// Test Get
	readTool := registry.Get("read")
	if readTool == nil {
		t.Error("expected read tool to be registered")
	}

	if readTool.Name() != "read" {
		t.Error("expected tool name to be 'read'")
	}
}

func TestReadTool(t *testing.T) {
	registry := NewRegistryWithBuiltins()

	readTool := registry.Get("read")
	if readTool == nil {
		t.Fatal("read tool not registered")
	}

	// Create a test file
	testFile := "test_read.txt"
	testContent := "Hello, Sentinel!"
	os.WriteFile(testFile, []byte(testContent), 0644)
	defer os.Remove(testFile)

	// Execute the read tool
	result, err := readTool.Execute(context.Background(), map[string]interface{}{
		"path": testFile,
	})
	if err != nil {
		t.Fatal(err)
	}

	if result != testContent {
		t.Error("expected content to match")
	}
}

func TestWriteTool(t *testing.T) {
	registry := NewRegistryWithBuiltins()

	writeTool := registry.Get("write")
	if writeTool == nil {
		t.Fatal("write tool not registered")
	}

	testFile := "test_write.txt"
	testContent := "Hello, World!"
	defer os.Remove(testFile)

	// Execute the write tool
	_, err := writeTool.Execute(context.Background(), map[string]interface{}{
		"path":    testFile,
		"content": testContent,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify the file was written
	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatal(err)
	}

	if string(data) != testContent {
		t.Error("expected written content to match")
	}
}

func TestGlobTool(t *testing.T) {
	registry := NewRegistryWithBuiltins()

	globTool := registry.Get("glob")
	if globTool == nil {
		t.Fatal("glob tool not registered")
	}

	// Find Go files
	result, err := globTool.Execute(context.Background(), map[string]interface{}{
		"pattern": "*.go",
	})
	if err != nil {
		t.Fatal(err)
	}

	if result == "" {
		t.Error("expected glob to find files")
	}
}
