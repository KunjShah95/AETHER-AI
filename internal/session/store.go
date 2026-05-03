package session

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"modernc.org/sqlite"
)

type Store struct {
	db *sqlite.Conn
}

func NewStore(path string) (*Store, error) {
	conn, err := sqlite.Open(path)
	if err != nil {
		return nil, err
	}

	s := &Store{db: conn}
	if err := s.migrate(); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *Store) migrate() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS sessions (
			id TEXT PRIMARY KEY,
			project_id TEXT NOT NULL,
			created_at INTEGER NOT NULL,
			state TEXT NOT NULL DEFAULT '{}'
		);
		CREATE TABLE IF NOT EXISTS messages (
			id INTEGER PRIMARY KEY,
			session_id TEXT NOT NULL,
			role TEXT NOT NULL,
			parts TEXT NOT NULL,
			created_at INTEGER NOT NULL,
			FOREIGN KEY(session_id) REFERENCES sessions(id)
		);
		CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
	`)
	return err
}

func (s *Store) Create(ctx context.Context, projectID string) (*Session, error) {
	id := generateID()
	now := time.Now().Unix()

	_, err := s.db.Exec(
		"INSERT INTO sessions (id, project_id, created_at, state) VALUES (?, ?, ?, ?)",
		id, projectID, now, "{}",
	)
	if err != nil {
		return nil, err
	}

	return &Session{
		ID:        id,
		ProjectID: projectID,
		CreatedAt: time.Now(),
		Messages:  []Message{},
		State:     SessionState{},
	}, nil
}

func (s *Store) Get(ctx context.Context, id string) (*Session, error) {
	var row struct {
		ID        string
		ProjectID string
		CreatedAt int64
		State     string
	}

	err := s.db.QueryRow("SELECT id, project_id, created_at, state FROM sessions WHERE id = ?", id).
		Scan(&row.ID, &row.ProjectID, &row.CreatedAt, &row.State)
	if err != nil {
		return nil, err
	}

	var state SessionState
	if err := json.Unmarshal([]byte(row.State), &state); err != nil {
		state = SessionState{}
	}

	return &Session{
		ID:        row.ID,
		ProjectID: row.ProjectID,
		CreatedAt: time.Unix(row.CreatedAt, 0),
		Messages:  []Message{},
		State:     state,
	}, nil
}

func (s *Store) AddMessage(ctx context.Context, sessionID string, msg Message) error {
	now := time.Now().Unix()

	partsJSON, err := json.Marshal(msg.Parts)
	if err != nil {
		return err
	}

	_, err = s.db.Exec(
		"INSERT INTO messages (session_id, role, parts, created_at) VALUES (?, ?, ?, ?)",
		sessionID, msg.Role, string(partsJSON), now,
	)
	return err
}

func (s *Store) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

func generateID() string {
	return fmt.Sprintf("sess_%d", time.Now().UnixNano())
}
