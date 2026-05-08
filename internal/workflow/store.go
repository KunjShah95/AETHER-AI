package workflow

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	_ "modernc.org/sqlite"
)

type Store struct {
	db *sql.DB
}

func NewStore(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}

	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *Store) migrate() error {
	schema := `
		CREATE TABLE IF NOT EXISTS milestones (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL,
			description TEXT NOT NULL DEFAULT '',
			version TEXT NOT NULL DEFAULT '',
			status TEXT NOT NULL DEFAULT 'active',
			created_at INTEGER NOT NULL,
			updated_at INTEGER NOT NULL,
			completed_at INTEGER
		);

		CREATE TABLE IF NOT EXISTS phases (
			id TEXT PRIMARY KEY,
			milestone_id TEXT NOT NULL,
			name TEXT NOT NULL,
			description TEXT NOT NULL DEFAULT '',
			status TEXT NOT NULL DEFAULT 'pending',
			order_num INTEGER NOT NULL DEFAULT 0,
			depends_on TEXT NOT NULL DEFAULT '[]',
			created_at INTEGER NOT NULL,
			updated_at INTEGER NOT NULL,
			completed_at INTEGER,
			FOREIGN KEY(milestone_id) REFERENCES milestones(id) ON DELETE CASCADE
		);

		CREATE TABLE IF NOT EXISTS todos (
			id TEXT PRIMARY KEY,
			phase_id TEXT,
			milestone_id TEXT NOT NULL,
			content TEXT NOT NULL,
			status TEXT NOT NULL DEFAULT 'pending',
			priority TEXT NOT NULL DEFAULT 'medium',
			created_at INTEGER NOT NULL,
			updated_at INTEGER NOT NULL,
			completed_at INTEGER,
			FOREIGN KEY(phase_id) REFERENCES phases(id) ON DELETE SET NULL,
			FOREIGN KEY(milestone_id) REFERENCES milestones(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS idx_phases_milestone ON phases(milestone_id);
		CREATE INDEX IF NOT EXISTS idx_todos_milestone ON todos(milestone_id);
		CREATE INDEX IF NOT EXISTS idx_todos_phase ON todos(phase_id);
	`
	_, err := s.db.Exec(schema)
	return err
}

func (s *Store) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

func generateID() string {
	return fmt.Sprintf("wf_%d", time.Now().UnixNano())
}

func intToTime(ts *int64) *time.Time {
	if ts == nil {
		return nil
	}
	t := time.Unix(*ts, 0)
	return &t
}

func timeToInt(t *time.Time) *int64 {
	if t == nil {
		return nil
	}
	ts := t.Unix()
	return &ts
}

func (s *Store) CreateMilestone(ctx context.Context, m *Milestone) error {
	if m.ID == "" {
		m.ID = generateID()
	}
	now := time.Now()
	m.CreatedAt = now
	m.UpdatedAt = now

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO milestones (id, name, description, version, status, created_at, updated_at, completed_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		m.ID, m.Name, m.Description, m.Version, m.Status,
		m.CreatedAt.Unix(), m.UpdatedAt.Unix(), timeToInt(m.CompletedAt))
	return err
}

func (s *Store) GetMilestone(ctx context.Context, id string) (*Milestone, error) {
	var m Milestone
	var createdAt, updatedAt int64
	var completedAt *int64

	err := s.db.QueryRowContext(ctx,
		`SELECT id, name, description, version, status, created_at, updated_at, completed_at
		 FROM milestones WHERE id = ?`, id).
		Scan(&m.ID, &m.Name, &m.Description, &m.Version, &m.Status,
			&createdAt, &updatedAt, &completedAt)
	if err != nil {
		return nil, err
	}

	m.CreatedAt = time.Unix(createdAt, 0)
	m.UpdatedAt = time.Unix(updatedAt, 0)
	m.CompletedAt = intToTime(completedAt)

	return &m, nil
}

func (s *Store) ListMilestones(ctx context.Context) ([]Milestone, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, name, description, version, status, created_at, updated_at, completed_at
		 FROM milestones ORDER BY created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var milestones []Milestone
	for rows.Next() {
		var m Milestone
		var createdAt, updatedAt int64
		var completedAt *int64
		if err := rows.Scan(&m.ID, &m.Name, &m.Description, &m.Version, &m.Status,
			&createdAt, &updatedAt, &completedAt); err != nil {
			return nil, err
		}
		m.CreatedAt = time.Unix(createdAt, 0)
		m.UpdatedAt = time.Unix(updatedAt, 0)
		m.CompletedAt = intToTime(completedAt)
		milestones = append(milestones, m)
	}

	return milestones, rows.Err()
}

func (s *Store) UpdateMilestone(ctx context.Context, m *Milestone) error {
	m.UpdatedAt = time.Now()

	_, err := s.db.ExecContext(ctx,
		`UPDATE milestones SET name = ?, description = ?, version = ?, status = ?, updated_at = ?, completed_at = ?
		 WHERE id = ?`,
		m.Name, m.Description, m.Version, m.Status, m.UpdatedAt.Unix(), timeToInt(m.CompletedAt), m.ID)
	return err
}

func (s *Store) CreatePhase(ctx context.Context, p *Phase) error {
	if p.ID == "" {
		p.ID = generateID()
	}
	now := time.Now()
	p.CreatedAt = now
	p.UpdatedAt = now

	dependsOnJSON, err := json.Marshal(p.DependsOn)
	if err != nil {
		dependsOnJSON = []byte("[]")
	}

	_, err = s.db.ExecContext(ctx,
		`INSERT INTO phases (id, milestone_id, name, description, status, order_num, depends_on, created_at, updated_at, completed_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		p.ID, p.MilestoneID, p.Name, p.Description, p.Status, p.Order, string(dependsOnJSON),
		p.CreatedAt.Unix(), p.UpdatedAt.Unix(), timeToInt(p.CompletedAt))
	return err
}

func (s *Store) GetPhase(ctx context.Context, id string) (*Phase, error) {
	var p Phase
	var createdAt, updatedAt int64
	var completedAt *int64
	var dependsOnJSON string

	err := s.db.QueryRowContext(ctx,
		`SELECT id, milestone_id, name, description, status, order_num, depends_on, created_at, updated_at, completed_at
		 FROM phases WHERE id = ?`, id).
		Scan(&p.ID, &p.MilestoneID, &p.Name, &p.Description, &p.Status, &p.Order, &dependsOnJSON,
			&createdAt, &updatedAt, &completedAt)
	if err != nil {
		return nil, err
	}

	p.CreatedAt = time.Unix(createdAt, 0)
	p.UpdatedAt = time.Unix(updatedAt, 0)
	p.CompletedAt = intToTime(completedAt)

	if err := json.Unmarshal([]byte(dependsOnJSON), &p.DependsOn); err != nil {
		p.DependsOn = []string{}
	}

	return &p, nil
}

func (s *Store) ListPhases(ctx context.Context, milestoneID string) ([]Phase, error) {
	query := `SELECT id, milestone_id, name, description, status, order_num, depends_on, created_at, updated_at, completed_at
		 FROM phases`
	var rows *sql.Rows
	var err error

	if milestoneID != "" {
		query += " WHERE milestone_id = ?"
		rows, err = s.db.QueryContext(ctx, query, milestoneID)
	} else {
		rows, err = s.db.QueryContext(ctx, query)
	}

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var phases []Phase
	for rows.Next() {
		var p Phase
		var createdAt, updatedAt int64
		var completedAt *int64
		var dependsOnJSON string
		if err := rows.Scan(&p.ID, &p.MilestoneID, &p.Name, &p.Description, &p.Status, &p.Order, &dependsOnJSON,
			&createdAt, &updatedAt, &completedAt); err != nil {
			return nil, err
		}
		p.CreatedAt = time.Unix(createdAt, 0)
		p.UpdatedAt = time.Unix(updatedAt, 0)
		p.CompletedAt = intToTime(completedAt)
		if err := json.Unmarshal([]byte(dependsOnJSON), &p.DependsOn); err != nil {
			p.DependsOn = []string{}
		}
		phases = append(phases, p)
	}

	return phases, rows.Err()
}

func (s *Store) UpdatePhase(ctx context.Context, p *Phase) error {
	p.UpdatedAt = time.Now()

	dependsOnJSON, err := json.Marshal(p.DependsOn)
	if err != nil {
		dependsOnJSON = []byte("[]")
	}

	_, err = s.db.ExecContext(ctx,
		`UPDATE phases SET name = ?, description = ?, status = ?, order_num = ?, depends_on = ?, updated_at = ?, completed_at = ?
		 WHERE id = ?`,
		p.Name, p.Description, p.Status, p.Order, string(dependsOnJSON), p.UpdatedAt.Unix(), timeToInt(p.CompletedAt), p.ID)
	return err
}

func (s *Store) CreateTodo(ctx context.Context, t *Todo) error {
	if t.ID == "" {
		t.ID = generateID()
	}
	now := time.Now()
	t.CreatedAt = now
	t.UpdatedAt = now

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO todos (id, phase_id, milestone_id, content, status, priority, created_at, updated_at, completed_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		t.ID, t.PhaseID, t.MilestoneID, t.Content, t.Status, t.Priority,
		t.CreatedAt.Unix(), t.UpdatedAt.Unix(), timeToInt(t.CompletedAt))
	return err
}

func (s *Store) ListTodos(ctx context.Context, milestoneID string, phaseID string) ([]Todo, error) {
	query := `SELECT id, phase_id, milestone_id, content, status, priority, created_at, updated_at, completed_at
		 FROM todos WHERE 1=1`
	var args []interface{}

	if milestoneID != "" {
		query += " AND milestone_id = ?"
		args = append(args, milestoneID)
	}
	if phaseID != "" {
		query += " AND phase_id = ?"
		args = append(args, phaseID)
	}

	query += " ORDER BY created_at DESC"

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var todos []Todo
	for rows.Next() {
		var t Todo
		var createdAt, updatedAt int64
		var completedAt *int64
		if err := rows.Scan(&t.ID, &t.PhaseID, &t.MilestoneID, &t.Content, &t.Status, &t.Priority,
			&createdAt, &updatedAt, &completedAt); err != nil {
			return nil, err
		}
		t.CreatedAt = time.Unix(createdAt, 0)
		t.UpdatedAt = time.Unix(updatedAt, 0)
		t.CompletedAt = intToTime(completedAt)
		todos = append(todos, t)
	}

	return todos, rows.Err()
}

func (s *Store) UpdateTodo(ctx context.Context, t *Todo) error {
	t.UpdatedAt = time.Now()

	_, err := s.db.ExecContext(ctx,
		`UPDATE todos SET phase_id = ?, content = ?, status = ?, priority = ?, updated_at = ?, completed_at = ?
		 WHERE id = ?`,
		t.PhaseID, t.Content, t.Status, t.Priority, t.UpdatedAt.Unix(), timeToInt(t.CompletedAt), t.ID)
	return err
}
