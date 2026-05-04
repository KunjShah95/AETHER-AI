package protocol

import "testing"

func TestEnvelopeValidate(t *testing.T) {
	env := NewEnvelope(KindMessage, Agent{ID: "a1", Name: "Agent One"}, Agent{ID: "a2", Name: "Agent Two"}, "sess_1", "hello")
	if err := env.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestEnvelopeValidateRejectsMissingFields(t *testing.T) {
	if err := (Envelope{}).Validate(); err == nil {
		t.Fatal("expected validation error")
	}
}
