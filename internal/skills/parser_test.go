package skills

import "testing"

func TestParseFile(t *testing.T) {
	data := []byte(`---
name: graphify
description: any input to knowledge graph
trigger:
  - /graphify
applyTo:
  - markdown
---
# Graphify
Transform input into a knowledge graph.
`)

	skill, err := ParseFile("C:/skills/graphify/SKILL.md", data)
	if err != nil {
		t.Fatalf("ParseFile() error = %v", err)
	}
	if skill.Name != "graphify" {
		t.Fatalf("Name = %q, want %q", skill.Name, "graphify")
	}
	if len(skill.Trigger) != 1 || skill.Trigger[0] != "/graphify" {
		t.Fatalf("Trigger = %#v", skill.Trigger)
	}
	if skill.Content == "" {
		t.Fatalf("expected content to be populated")
	}
}

func TestParseFile_InferName(t *testing.T) {
	data := []byte("No front matter\nJust content\n")

	skill, err := ParseFile("C:/skills/notes/SKILL.md", data)
	if err != nil {
		t.Fatalf("ParseFile() error = %v", err)
	}
	if skill.Name != "notes" {
		t.Fatalf("Name = %q, want %q", skill.Name, "notes")
	}
}
