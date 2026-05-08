package workflow

import (
	"context"
	"testing"
)

func setup(t *testing.T) (*Manager, func()) {
	dbPath := t.TempDir() + "/test.db"
	store, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	mgr := NewManager(store)
	return mgr, func() { store.Close() }
}

func TestCreateMilestone(t *testing.T) {
	mgr, cleanup := setup(t)
	defer cleanup()
	ctx := context.Background()

	tests := []struct {
		name        string
		description string
		version     string
		wantName    string
		wantStatus  string
	}{
		{
			name:        "v1.0",
			description: "First release",
			version:     "1.0.0",
			wantName:    "v1.0",
			wantStatus:  "active",
		},
		{
			name:        "v2.0",
			description: "Second release",
			version:     "2.0.0",
			wantName:    "v2.0",
			wantStatus:  "active",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := mgr.CreateMilestone(ctx, tt.name, tt.description, tt.version)
			if err != nil {
				t.Fatalf("CreateMilestone failed: %v", err)
			}
			if m.Name != tt.wantName {
				t.Errorf("Name = %v, want %v", m.Name, tt.wantName)
			}
			if m.Status != tt.wantStatus {
				t.Errorf("Status = %v, want %v", m.Status, tt.wantStatus)
			}
			if m.ID == "" {
				t.Error("ID should not be empty")
			}
		})
	}
}

func TestCreatePhase(t *testing.T) {
	mgr, cleanup := setup(t)
	defer cleanup()
	ctx := context.Background()

	m, err := mgr.CreateMilestone(ctx, "test-milestone", "desc", "1.0")
	if err != nil {
		t.Fatalf("CreateMilestone failed: %v", err)
	}

	tests := []struct {
		name        string
		description string
		order       int
		wantName    string
		wantStatus  string
	}{
		{
			name:        "Phase 1",
			description: "First phase",
			order:       1,
			wantName:    "Phase 1",
			wantStatus:  "pending",
		},
		{
			name:        "Phase 2",
			description: "Second phase",
			order:       2,
			wantName:    "Phase 2",
			wantStatus:  "pending",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := mgr.CreatePhase(ctx, m.ID, tt.name, tt.description, tt.order)
			if err != nil {
				t.Fatalf("CreatePhase failed: %v", err)
			}
			if p.Name != tt.wantName {
				t.Errorf("Name = %v, want %v", p.Name, tt.wantName)
			}
			if p.Status != tt.wantStatus {
				t.Errorf("Status = %v, want %v", p.Status, tt.wantStatus)
			}
			if p.MilestoneID != m.ID {
				t.Errorf("MilestoneID = %v, want %v", p.MilestoneID, m.ID)
			}
			if p.Order != tt.order {
				t.Errorf("Order = %v, want %v", p.Order, tt.order)
			}
		})
	}
}

func TestCreateTodo(t *testing.T) {
	mgr, cleanup := setup(t)
	defer cleanup()
	ctx := context.Background()

	m, err := mgr.CreateMilestone(ctx, "test-milestone", "desc", "1.0")
	if err != nil {
		t.Fatalf("CreateMilestone failed: %v", err)
	}

	p, err := mgr.CreatePhase(ctx, m.ID, "phase1", "desc", 1)
	if err != nil {
		t.Fatalf("CreatePhase failed: %v", err)
	}

	tests := []struct {
		content      string
		priority     string
		wantContent  string
		wantPriority string
	}{
		{
			content:      "Implement feature X",
			priority:     "high",
			wantContent:  "Implement feature X",
			wantPriority: "high",
		},
		{
			content:      "Fix bug Y",
			priority:     "medium",
			wantContent:  "Fix bug Y",
			wantPriority: "medium",
		},
	}

	for _, tt := range tests {
		t.Run(tt.content, func(t *testing.T) {
			todo, err := mgr.CreateTodo(ctx, p.ID, m.ID, tt.content, tt.priority)
			if err != nil {
				t.Fatalf("CreateTodo failed: %v", err)
			}
			if todo.Content != tt.wantContent {
				t.Errorf("Content = %v, want %v", todo.Content, tt.wantContent)
			}
			if todo.Priority != tt.wantPriority {
				t.Errorf("Priority = %v, want %v", todo.Priority, tt.wantPriority)
			}
			if todo.Status != "pending" {
				t.Errorf("Status = %v, want pending", todo.Status)
			}
		})
	}
}

func TestRoadmap(t *testing.T) {
	mgr, cleanup := setup(t)
	defer cleanup()
	ctx := context.Background()

	m, err := mgr.CreateMilestone(ctx, "test-milestone", "desc", "1.0")
	if err != nil {
		t.Fatalf("CreateMilestone failed: %v", err)
	}

	phase1, err := mgr.CreatePhase(ctx, m.ID, "Phase 1", "desc", 1)
	if err != nil {
		t.Fatalf("CreatePhase failed: %v", err)
	}

	_, err = mgr.CreatePhase(ctx, m.ID, "Phase 2", "desc", 2)
	if err != nil {
		t.Fatalf("CreatePhase failed: %v", err)
	}

	_, err = mgr.CreateTodo(ctx, phase1.ID, m.ID, "Todo 1", "high")
	if err != nil {
		t.Fatalf("CreateTodo failed: %v", err)
	}

	mgr.CompletePhase(ctx, phase1.ID)

	roadmap, err := mgr.GetRoadmap(ctx, m.ID)
	if err != nil {
		t.Fatalf("GetRoadmap failed: %v", err)
	}

	if len(roadmap.Phases) != 2 {
		t.Errorf("Phases length = %v, want 2", len(roadmap.Phases))
	}

	if roadmap.Total != 2 {
		t.Errorf("Total = %v, want 2", roadmap.Total)
	}

	if roadmap.Completed != 1 {
		t.Errorf("Completed = %v, want 1", roadmap.Completed)
	}

	if roadmap.Percent != 50 {
		t.Errorf("Percent = %v, want 50", roadmap.Percent)
	}
}

func TestCompleteMilestone(t *testing.T) {
	mgr, cleanup := setup(t)
	defer cleanup()
	ctx := context.Background()

	m, err := mgr.CreateMilestone(ctx, "test-milestone", "desc", "1.0")
	if err != nil {
		t.Fatalf("CreateMilestone failed: %v", err)
	}

	if err := mgr.CompleteMilestone(ctx, m.ID); err != nil {
		t.Fatalf("CompleteMilestone failed: %v", err)
	}

	updated, err := mgr.GetMilestone(ctx, m.ID)
	if err != nil {
		t.Fatalf("GetMilestone failed: %v", err)
	}

	if updated.Status != "completed" {
		t.Errorf("Status = %v, want completed", updated.Status)
	}
	if updated.CompletedAt == nil {
		t.Error("CompletedAt should not be nil")
	}
}
