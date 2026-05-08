package workflow

import (
	"time"

	"github.com/google/uuid"
)

type Milestone struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Status      string     `json:"status"`
	CreatedAt   time.Time  `json:"created_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

type Phase struct {
	ID          string     `json:"id"`
	MilestoneID string     `json:"milestone_id"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Status      string     `json:"status"`
	Order       int        `json:"order"`
	CreatedAt   time.Time  `json:"created_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

type Todo struct {
	ID          string     `json:"id"`
	PhaseID     string     `json:"phase_id"`
	Title       string     `json:"title"`
	Description string     `json:"description"`
	Status      string     `json:"status"`
	CreatedAt   time.Time  `json:"created_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

type Roadmap struct {
	Milestones []Milestone        `json:"milestones"`
	Phases     map[string][]Phase `json:"phases"`
	Todos      map[string][]Todo  `json:"todos"`
}

type Manager struct {
	milestones map[string]*Milestone
	phases     map[string]*Phase
	todos      map[string]*Todo
}

func NewManager() *Manager {
	return &Manager{
		milestones: make(map[string]*Milestone),
		phases:     make(map[string]*Phase),
		todos:      make(map[string]*Todo),
	}
}

func (m *Manager) CreateMilestone(name, description string) *Milestone {
	milestone := &Milestone{
		ID:          uuid.New().String(),
		Name:        name,
		Description: description,
		Status:      "active",
		CreatedAt:   time.Now(),
	}
	m.milestones[milestone.ID] = milestone
	return milestone
}

func (m *Manager) GetMilestone(id string) *Milestone {
	return m.milestones[id]
}

func (m *Manager) ListMilestones() []*Milestone {
	result := make([]*Milestone, 0, len(m.milestones))
	for _, m := range m.milestones {
		result = append(result, m)
	}
	return result
}

func (m *Manager) CompleteMilestone(id string) error {
	milestone, ok := m.milestones[id]
	if !ok {
		return nil
	}
	now := time.Now()
	milestone.Status = "completed"
	milestone.CompletedAt = &now
	return nil
}

func (m *Manager) CreatePhase(milestoneID, name, description string, order int) *Phase {
	phase := &Phase{
		ID:          uuid.New().String(),
		MilestoneID: milestoneID,
		Name:        name,
		Description: description,
		Status:      "pending",
		Order:       order,
		CreatedAt:   time.Now(),
	}
	m.phases[phase.ID] = phase
	return phase
}

func (m *Manager) GetPhase(id string) *Phase {
	return m.phases[id]
}

func (m *Manager) ListPhases(milestoneID string) []*Phase {
	var result []*Phase
	for _, p := range m.phases {
		if p.MilestoneID == milestoneID {
			result = append(result, p)
		}
	}
	if result == nil {
		return []*Phase{}
	}
	return result
}

func (m *Manager) UpdatePhaseStatus(id, status string) error {
	phase, ok := m.phases[id]
	if !ok {
		return nil
	}
	phase.Status = status
	return nil
}

func (m *Manager) CompletePhase(id string) error {
	phase, ok := m.phases[id]
	if !ok {
		return nil
	}
	now := time.Now()
	phase.Status = "completed"
	phase.CompletedAt = &now
	return nil
}

func (m *Manager) CreateTodo(phaseID, title, description string) *Todo {
	todo := &Todo{
		ID:          uuid.New().String(),
		PhaseID:     phaseID,
		Title:       title,
		Description: description,
		Status:      "pending",
		CreatedAt:   time.Now(),
	}
	m.todos[todo.ID] = todo
	return todo
}

func (m *Manager) ListTodos(phaseID string) []*Todo {
	var result []*Todo
	for _, t := range m.todos {
		if t.PhaseID == phaseID {
			result = append(result, t)
		}
	}
	if result == nil {
		return []*Todo{}
	}
	return result
}

func (m *Manager) CompleteTodo(id string) error {
	todo, ok := m.todos[id]
	if !ok {
		return nil
	}
	now := time.Now()
	todo.Status = "completed"
	todo.CompletedAt = &now
	return nil
}

func (m *Manager) GetRoadmap() *Roadmap {
	roadmap := &Roadmap{
		Milestones: make([]Milestone, 0),
		Phases:     make(map[string][]Phase),
		Todos:      make(map[string][]Todo),
	}

	for _, ms := range m.milestones {
		roadmap.Milestones = append(roadmap.Milestones, *ms)
	}

	for _, p := range m.phases {
		roadmap.Phases[p.MilestoneID] = append(roadmap.Phases[p.MilestoneID], *p)
	}

	for _, t := range m.todos {
		roadmap.Todos[t.PhaseID] = append(roadmap.Todos[t.PhaseID], *t)
	}

	return roadmap
}
