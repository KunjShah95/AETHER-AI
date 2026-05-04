package skills

import "sort"

type Registry struct {
	skills map[string]*Skill
}

func NewRegistry() *Registry {
	return &Registry{skills: make(map[string]*Skill)}
}

func (r *Registry) Register(skill *Skill) {
	if skill == nil {
		return
	}
	if skill.Name == "" {
		return
	}
	r.skills[skill.Name] = skill
}

func (r *Registry) Get(name string) *Skill {
	return r.skills[name]
}

func (r *Registry) List() []*Skill {
	out := make([]*Skill, 0, len(r.skills))
	for _, skill := range r.skills {
		out = append(out, skill)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (r *Registry) Find(query string) []*Skill {
	if query == "" {
		return nil
	}
	matches := make([]*Skill, 0)
	for _, skill := range r.skills {
		if skill.Matches(query) {
			matches = append(matches, skill)
		}
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].Name < matches[j].Name })
	return matches
}
