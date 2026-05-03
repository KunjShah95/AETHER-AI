package session

import (
	"context"
)

type Manager struct {
	store  *Store
	active map[string]*Session
}

func NewManager(store *Store) *Manager {
	return &Manager{
		store:  store,
		active: make(map[string]*Session),
	}
}

func (m *Manager) CreateSession(ctx context.Context, projectID string) (*Session, error) {
	sess, err := m.store.Create(ctx, projectID)
	if err != nil {
		return nil, err
	}
	m.active[sess.ID] = sess
	return sess, nil
}

func (m *Manager) GetSession(ctx context.Context, id string) (*Session, error) {
	if sess, ok := m.active[id]; ok {
		return sess, nil
	}
	return m.store.Get(ctx, id)
}

func (m *Manager) AddMessage(ctx context.Context, sessionID string, msg Message) error {
	sess, err := m.GetSession(ctx, sessionID)
	if err != nil {
		return err
	}
	sess.Messages = append(sess.Messages, msg)
	return m.store.AddMessage(ctx, sessionID, msg)
}

func (m *Manager) CloseSession(ctx context.Context, id string) error {
	if _, ok := m.active[id]; !ok {
		return nil
	}
	delete(m.active, id)
	return nil
}
