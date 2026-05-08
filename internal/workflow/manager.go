package workflow

import (
	"context"
	"time"

	"github.com/google/uuid"
)

type Manager struct {
	store *Store
}

func NewManager(store *Store) *Manager {
	return &Manager{store: store}
}

func (m *Manager) CreateMilestone(ctx context.Context, name, description, version string) (*Milestone, error) {
	milestone := &Milestone{
		ID:          "ms_" + uuid.New().String(),
		Name:        name,
		Description: description,
		Version:     version,
		Status:      "active",
	}
	if err := m.store.CreateMilestone(ctx, milestone); err != nil {
		return nil, err
	}
	return milestone, nil
}

func (m *Manager) GetMilestone(ctx context.Context, id string) (*Milestone, error) {
	return m.store.GetMilestone(ctx, id)
}

func (m *Manager) ListMilestones(ctx context.Context) ([]Milestone, error) {
	return m.store.ListMilestones(ctx)
}

func (m *Manager) CompleteMilestone(ctx context.Context, id string) error {
	milestone, err := m.store.GetMilestone(ctx, id)
	if err != nil {
		return err
	}
	now := time.Now()
	milestone.Status = "completed"
	milestone.CompletedAt = &now
	return m.store.UpdateMilestone(ctx, milestone)
}

func (m *Manager) CreatePhase(ctx context.Context, milestoneID, name, description string, order int) (*Phase, error) {
	phase := &Phase{
		ID:          "phase_" + uuid.New().String(),
		MilestoneID: milestoneID,
		Name:        name,
		Description: description,
		Status:      "pending",
		Order:       order,
	}
	if err := m.store.CreatePhase(ctx, phase); err != nil {
		return nil, err
	}
	return phase, nil
}

func (m *Manager) GetPhase(ctx context.Context, id string) (*Phase, error) {
	return m.store.GetPhase(ctx, id)
}

func (m *Manager) ListPhases(ctx context.Context, milestoneID string) ([]Phase, error) {
	return m.store.ListPhases(ctx, milestoneID)
}

func (m *Manager) UpdatePhaseStatus(ctx context.Context, id, status string) error {
	phase, err := m.store.GetPhase(ctx, id)
	if err != nil {
		return err
	}
	phase.Status = status
	return m.store.UpdatePhase(ctx, phase)
}

func (m *Manager) CompletePhase(ctx context.Context, id string) error {
	phase, err := m.store.GetPhase(ctx, id)
	if err != nil {
		return err
	}
	now := time.Now()
	phase.Status = "completed"
	phase.CompletedAt = &now
	return m.store.UpdatePhase(ctx, phase)
}

func (m *Manager) CreateTodo(ctx context.Context, phaseID, milestoneID, content, priority string) (*Todo, error) {
	todo := &Todo{
		ID:          "todo_" + uuid.New().String(),
		PhaseID:     phaseID,
		MilestoneID: milestoneID,
		Content:     content,
		Status:      "pending",
		Priority:    priority,
	}
	if err := m.store.CreateTodo(ctx, todo); err != nil {
		return nil, err
	}
	return todo, nil
}

func (m *Manager) ListTodos(ctx context.Context, milestoneID, phaseID string) ([]Todo, error) {
	return m.store.ListTodos(ctx, milestoneID, phaseID)
}

func (m *Manager) CompleteTodo(ctx context.Context, id string) error {
	todos, err := m.store.ListTodos(ctx, "", "")
	if err != nil {
		return err
	}
	for _, todo := range todos {
		if todo.ID == id {
			now := time.Now()
			todo.Status = "completed"
			todo.CompletedAt = &now
			return m.store.UpdateTodo(ctx, &todo)
		}
	}
	return nil
}

func (m *Manager) GetRoadmap(ctx context.Context, milestoneID string) (*Roadmap, error) {
	milestone, err := m.store.GetMilestone(ctx, milestoneID)
	if err != nil {
		return nil, err
	}

	phases, err := m.store.ListPhases(ctx, milestoneID)
	if err != nil {
		return nil, err
	}

	todos, err := m.store.ListTodos(ctx, milestoneID, "")
	if err != nil {
		return nil, err
	}

	completed := 0
	total := len(phases)
	for _, p := range phases {
		if p.Status == "completed" {
			completed++
		}
	}

	percent := 0
	if total > 0 {
		percent = (completed * 100) / total
	}

	return &Roadmap{
		Milestone: *milestone,
		Phases:    phases,
		Todos:     todos,
		Completed: completed,
		Total:     total,
		Percent:   percent,
	}, nil
}
