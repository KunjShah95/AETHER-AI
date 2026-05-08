package skills

import "fmt"

type ValidationError struct {
	Skill   string
	Field   string
	Message string
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("skill %q: %s: %s", e.Skill, e.Field, e.Message)
}

func ValidateSkill(skill *Skill) []ValidationError {
	var errors []ValidationError

	if skill.Name == "" {
		errors = append(errors, ValidationError{
			Skill:   skill.Name,
			Field:   "name",
			Message: "name is required",
		})
	}

	if skill.Description == "" {
		errors = append(errors, ValidationError{
			Skill:   skill.Name,
			Field:   "description",
			Message: "description is recommended",
		})
	}

	return errors
}

func ValidateToolAccess(skill *Skill, toolName string) error {
	if len(skill.AllowedTools) == 0 {
		return nil
	}

	for _, allowed := range skill.AllowedTools {
		if allowed == "*" {
			return nil
		}
		if allowed == toolName {
			return nil
		}
	}

	return fmt.Errorf("tool %q is not allowed for skill %q", toolName, skill.Name)
}
