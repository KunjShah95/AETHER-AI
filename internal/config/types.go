package config

type Config struct {
	LLMs    LLMConfig     `mapstructure:"llm" yaml:"llm"`
	MCPs    []MCPServer   `mapstructure:"mcp" yaml:"mcp"`
	LSP     LSPConfig     `mapstructure:"lsp" yaml:"lsp"`
	Project ProjectConfig `mapstructure:"project" yaml:"project"`
}

type LLMConfig struct {
	Provider string `mapstructure:"provider" yaml:"provider"`
	Model    string `mapstructure:"model" yaml:"model"`
	APIKey   string `mapstructure:"api_key" yaml:"api_key"`
	BaseURL  string `mapstructure:"base_url" yaml:"base_url"`
}

type MCPServer struct {
	Name    string            `mapstructure:"name" yaml:"name"`
	Type    string            `mapstructure:"type" yaml:"type"`
	Command string            `mapstructure:"command" yaml:"command"`
	Args    []string          `mapstructure:"args" yaml:"args"`
	Env     map[string]string `mapstructure:"env" yaml:"env"`
}

type LSPConfig struct {
	Servers map[string]string `mapstructure:"servers" yaml:"servers"`
}

type ProjectConfig struct {
	WorkDir string `mapstructure:"work_dir" yaml:"work_dir"`
}
