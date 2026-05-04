package protocol

import "testing"

func TestHub(t *testing.T) {
	hub := NewHub(Agent{ID: "sentinel-ai", Name: "Sentinel AI"})
	hub.RegisterPeer(Agent{ID: "agent-1", Name: "Other Agent"})

	if hub.LocalAgent().ID != "sentinel-ai" {
		t.Fatalf("LocalAgent() = %q, want sentinel-ai", hub.LocalAgent().ID)
	}
	if len(hub.Peers()) != 1 {
		t.Fatalf("Peers() len = %d, want 1", len(hub.Peers()))
	}

	env := NewEnvelope(KindMessage, Agent{ID: "agent-1", Name: "Other Agent"}, Agent{ID: "sentinel-ai", Name: "Sentinel AI"}, "sess_1", "hello")
	hub.Record(env)
	if !hub.Has(env.ID) {
		t.Fatal("expected envelope to be recorded")
	}
}
