package config

import (
	"os"
	"path/filepath"

	"github.com/spf13/viper"
)

func Load() (*Config, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	defaultPath := filepath.Join(home, ".sentinel", "config.yaml")

	viper.SetConfigType("yaml")
	viper.SetConfigFile(defaultPath)
	viper.SetDefault("project.work_dir", ".")

	if err := viper.ReadInConfig(); err != nil {
		if os.IsNotExist(err) {
			return &Config{}, nil
		}
		return nil, err
	}

	var cfg Config
	if err := viper.Unmarshal(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}

func (c *Config) Validate() error {
	if c.LLMs.Provider == "" {
		c.LLMs.Provider = "ollama"
	}
	if c.LLMs.Model == "" {
		c.LLMs.Model = "llama3.2"
	}
	if c.LLMs.Provider == "ollama" && c.LLMs.BaseURL == "" {
		c.LLMs.BaseURL = "http://localhost:11434"
	}
	return nil
}
