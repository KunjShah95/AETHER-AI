# Security

AETHER AI is designed with security as a core priority.

## Security Model

- **Input Sanitization**: Blocks dangerous patterns and suspicious Unicode characters to prevent injection attacks.
- **Safe Command Allowlist**: The `/run` command only executes allowlisted commands.
  - No shells, pipes, redirection, or wildcards are allowed.
  - File arguments are enforced to be within the current directory boundary.
- **API Key Validation**: API keys for major providers are validated for correct format.
- **Local Data Storage**: Configuration and user databases are stored per-user in `~/.aether` with restricted permissions where supported.

## Security Commands

| Command | Description |
|---------|-------------|
| `/encrypt [message]` | Encrypt messages |
| `/decrypt [message]` | Decrypt messages |
| `/rotate-key [service] [key]` | Rotate API keys |
| `/biometric-auth [data]` | Biometric authentication simulation |
| `/secure-password [len]` | Generate cryptographically secure passwords |
| `/security-report` | View security report |
| `/threat-scan [text]` | Scan text for security threats |

## Reporting Issues

If you discover a security vulnerability, please refer to `SECURITY.md` in the repository for reporting instructions.
