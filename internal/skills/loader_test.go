package skills

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestLoader(t *testing.T) {
	root := t.TempDir()
	skillDir := filepath.Join(root, "graphify")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	data := []byte(`---
name: graphify
description: graph knowledge
trigger:
  - /graphify
---
content
`)
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), data, 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	reg, err := NewLoader(root).Load(context.Background())
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got := reg.Get("graphify"); got == nil {
		t.Fatalf("expected skill graphify to be loaded")
	}
}
