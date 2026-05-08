package protocol

import (
	"sync"
)

type Hub struct {
	mu      sync.RWMutex
	agent   Agent
	peer    map[string]Agent
	handled map[string]Envelope
}

func NewHub(agent Agent) *Hub {
	return &Hub{
		agent:   agent,
		peer:    make(map[string]Agent),
		handled: make(map[string]Envelope),
	}
}

func (h *Hub) LocalAgent() Agent {
	return h.agent
}

func (h *Hub) RegisterPeer(agent Agent) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if agent.ID == "" {
		return
	}
	h.peer[agent.ID] = agent
}

func (h *Hub) Peers() []Agent {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make([]Agent, 0, len(h.peer))
	for _, agent := range h.peer {
		out = append(out, agent)
	}
	return out
}

func (h *Hub) Record(envelope Envelope) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.handled[envelope.ID] = envelope
}

func (h *Hub) Has(id string) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	_, ok := h.handled[id]
	return ok
}
