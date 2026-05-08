package skills

import (
	"bytes"
	"fmt"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

func ParseFile(path string, data []byte) (*Skill, error) {
	meta, body, err := splitFrontMatter(data)
	if err != nil {
		return nil, err
	}

	var md Metadata
	if len(meta) > 0 {
		if err := yaml.Unmarshal(meta, &md); err != nil {
			return nil, fmt.Errorf("parse skill metadata: %w", err)
		}
	}

	if md.Name == "" {
		md.Name = inferNameFromPath(path)
	}
	if md.Description == "" {
		md.Description = strings.TrimSpace(firstLine(string(body)))
	}

	return &Skill{
		Metadata: md,
		Path:     path,
		Content:  strings.TrimSpace(string(body)),
	}, nil
}

func splitFrontMatter(data []byte) ([]byte, []byte, error) {
	trimmed := bytes.TrimSpace(data)
	if !bytes.HasPrefix(trimmed, []byte("---")) {
		return nil, trimmed, nil
	}

	parts := bytes.SplitN(trimmed, []byte("\n---"), 2)
	if len(parts) != 2 {
		return nil, nil, fmt.Errorf("invalid front matter")
	}

	meta := bytes.TrimPrefix(parts[0], []byte("---"))
	body := bytes.TrimSpace(bytes.TrimPrefix(parts[1], []byte("\n")))
	return bytes.TrimSpace(meta), body, nil
}

func inferNameFromPath(path string) string {
	base := filepath.Base(filepath.Dir(path))
	if base == "." || base == string(filepath.Separator) || base == "" {
		base = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}
	return base
}

func firstLine(s string) string {
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		return s[:idx]
	}
	return s
}
