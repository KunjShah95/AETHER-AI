package session

import (
	"context"
	"fmt"
	"strings"
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

func (m *Manager) CompactSession(ctx context.Context, sessionID string, keepLast int) (*Session, error) {
	if keepLast < 2 {
		keepLast = 2
	}

	sess, err := m.GetSession(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	if len(sess.Messages) <= keepLast {
		return sess, nil
	}

	cutoff := len(sess.Messages) - keepLast
	old := sess.Messages[:cutoff]
	kept := append([]Message{}, sess.Messages[cutoff:]...)

	var summaryLines []string
	for _, msg := range old {
		text := messageText(msg)
		if text == "" {
			continue
		}
		summaryLines = append(summaryLines, fmt.Sprintf("%s: %s", msg.Role, text))
	}

	if len(summaryLines) > 0 {
		prefix := sess.State.Summary
		if prefix != "" {
			prefix += "\n"
		}
		sess.State.Summary = prefix + "Summary of earlier conversation:\n- " + strings.Join(summaryLines, "\n- ")
	}
	sess.Messages = kept

	if err := m.store.UpdateState(ctx, sessionID, sess.State); err != nil {
		return nil, err
	}
	if err := m.store.DeleteMessagesBefore(ctx, sessionID, int64(keepLast)); err != nil {
		return nil, err
	}

	m.active[sessionID] = sess
	return sess, nil
}

func (m *Manager) CloseSession(ctx context.Context, id string) error {
	if _, ok := m.active[id]; !ok {
		return nil
	}
	delete(m.active, id)
	return nil
}

func messageText(msg Message) string {
	for _, part := range msg.Parts {
		if part.Content != "" {
			return part.Content
		}
	}
	return ""
}
