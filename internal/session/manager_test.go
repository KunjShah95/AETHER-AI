package session

import (
	"context"
	"os"
	"testing"
)

func TestSessionManager(t *testing.T) {
	tmpFile := os.TempDir() + "/sentinel_test.db"
	defer os.Remove(tmpFile)

	store, err := NewStore(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	manager := NewManager(store)

	sess, err := manager.CreateSession(context.Background(), "test-project")
	if err != nil {
		t.Fatal(err)
	}

	if sess.ID == "" {
		t.Error("session ID should not be empty")
	}

	// Test GetSession
	retrieved, err := manager.GetSession(context.Background(), sess.ID)
	if err != nil {
		t.Fatal(err)
	}

	if retrieved.ID != sess.ID {
		t.Error("retrieved session ID should match created session ID")
	}

	// Test AddMessage
	msg := Message{
		Role: "user",
		Parts: []Part{
			{
				Type:    "text",
				Content: "Hello, Sentinel!",
			},
		},
	}
	if err := manager.AddMessage(context.Background(), sess.ID, msg); err != nil {
		t.Fatal(err)
	}

	// Test CloseSession
	if err := manager.CloseSession(context.Background(), sess.ID); err != nil {
		t.Fatal(err)
	}

	// Cleanup
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionManagerCompactSession(t *testing.T) {
	tmpFile := os.TempDir() + "/sentinel_compact_test.db"
	defer os.Remove(tmpFile)

	store, err := NewStore(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	manager := NewManager(store)
	sess, err := manager.CreateSession(context.Background(), "test-project")
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 8; i++ {
		msg := Message{Role: "user", Parts: []Part{{Type: "text", Content: "message"}}}
		if i%2 == 1 {
			msg.Role = "assistant"
			msg.Parts[0].Content = "response"
		}
		if err := manager.AddMessage(context.Background(), sess.ID, msg); err != nil {
			t.Fatal(err)
		}
	}

	compacted, err := manager.CompactSession(context.Background(), sess.ID, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(compacted.Messages) != 4 {
		t.Fatalf("len(compacted.Messages) = %d, want 4", len(compacted.Messages))
	}
	if compacted.State.Summary == "" {
		t.Fatal("expected summary to be populated")
	}

	reloaded, err := manager.GetSession(context.Background(), sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(reloaded.Messages) != 4 {
		t.Fatalf("len(reloaded.Messages) = %d, want 4", len(reloaded.Messages))
	}
	if reloaded.State.Summary == "" {
		t.Fatal("expected persisted summary to be populated")
	}
}
