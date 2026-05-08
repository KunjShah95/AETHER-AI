package workflow

import (
	"time"
)

type Milestone struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Version     string     `json:"version"`
	Status      string     `json:"status"` // active, completed, archived
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

type Phase struct {
	ID          string     `json:"id"`
	MilestoneID string     `json:"milestone_id"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Status      string     `json:"status"` // pending, in_progress, completed, blocked
	Order       int        `json:"order"`
	DependsOn   []string   `json:"depends_on"` // Phase IDs
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

type Todo struct {
	ID          string     `json:"id"`
	PhaseID     string     `json:"phase_id,omitempty"`
	MilestoneID string     `json:"milestone_id"`
	Content     string     `json:"content"`
	Status      string     `json:"status"`   // pending, in_progress, completed
	Priority    string     `json:"priority"` // high, medium, low
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

type Roadmap struct {
	Milestone Milestone `json:"milestone"`
	Phases    []Phase   `json:"phases"`
	Todos     []Todo    `json:"todos"`
	Completed int       `json:"completed"`
	Total     int       `json:"total"`
	Percent   int       `json:"percent"`
}
