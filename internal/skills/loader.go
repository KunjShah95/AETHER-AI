package skills

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

type Loader struct {
	roots []string
}

func NewLoader(roots ...string) *Loader {
	filtered := make([]string, 0, len(roots))
	for _, root := range roots {
		if strings.TrimSpace(root) != "" {
			filtered = append(filtered, root)
		}
	}
	return &Loader{roots: filtered}
}

func DefaultRoots() []string {
	roots := []string{}
	if home, err := os.UserHomeDir(); err == nil {
		// Sentinel's own skills
		roots = append(roots, filepath.Join(home, ".sentinel", "skills"))
		// Claude Code skills (gsd-*, caveman-*, etc.)
		roots = append(roots, filepath.Join(home, ".claude", "skills"))
		// OpenCode/Agents skills (azure-*, shadcn, etc.)
		roots = append(roots, filepath.Join(home, ".agents", "skills"))
	}
	// Project-local skills
	roots = append(roots, filepath.Join(".", "skills"))
	roots = append(roots, filepath.Join(".", ".agents", "skills"))
	return roots
}

func LoadDefault(ctx context.Context) (*Registry, error) {
	return NewLoader(DefaultRoots()...).Load(ctx)
}

func (l *Loader) Load(ctx context.Context) (*Registry, error) {
	reg := NewRegistry()
	for _, root := range l.roots {
		if err := l.loadRoot(ctx, root, reg); err != nil {
			return nil, err
		}
	}
	return reg, nil
}

func (l *Loader) loadRoot(ctx context.Context, root string, reg *Registry) error {
	info, err := os.Stat(root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("skill root is not a directory: %s", root)
	}

	return filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if d.IsDir() {
			return nil
		}
		if strings.EqualFold(filepath.Base(path), "SKILL.md") || strings.EqualFold(filepath.Ext(path), ".md") {
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				return readErr
			}
			skill, parseErr := ParseFile(path, data)
			if parseErr != nil {
				return parseErr
			}
			reg.Register(skill)
		}
		return nil
	})
}
