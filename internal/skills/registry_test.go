package skills

import "testing"

func TestRegistry(t *testing.T) {
	r := NewRegistry()
	r.Register(&Skill{Metadata: Metadata{Name: "alpha", Description: "first", Trigger: []string{"/alpha"}}})
	r.Register(&Skill{Metadata: Metadata{Name: "beta", Description: "second", ApplyTo: []string{"go"}}})

	if got := r.Get("alpha"); got == nil || got.Name != "alpha" {
		t.Fatalf("Get(alpha) = %#v", got)
	}
	if len(r.List()) != 2 {
		t.Fatalf("List() len = %d, want 2", len(r.List()))
	}
	if len(r.Find("/alpha")) != 1 {
		t.Fatalf("Find(/alpha) len = %d, want 1", len(r.Find("/alpha")))
	}
	if len(r.Find("go")) != 1 {
		t.Fatalf("Find(go) len = %d, want 1", len(r.Find("go")))
	}
}
