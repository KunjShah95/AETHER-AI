"""
MCP (Model Context Protocol) Manager for AetherAI.

This module provides universal MCP integration for connecting to various
AI tools, data sources, and services using the Model Context Protocol.

Supported MCP Servers:
- File system access
- Web browsing
- Database connections
- API integrations
- Custom tool servers
"""

import os
import json
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class MCPServer:
    """Represents an MCP server configuration."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    status: str = "disconnected"
    process: Optional[subprocess.Popen] = None


@dataclass 
class MCPTool:
    """Represents a tool exposed by an MCP server."""
    name: str
    description: str
    server: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    

class MCPManager:
    """Manages MCP server connections and tool invocations."""
    
    # Default MCP servers that are commonly available
    DEFAULT_SERVERS = {
        "filesystem": {
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
            "description": "Access local files and directories",
            "capabilities": ["read_file", "write_file", "list_directory", "search_files"]
        },
        "fetch": {
            "name": "fetch",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-fetch"],
            "description": "Fetch web pages and APIs",
            "capabilities": ["fetch_url", "get_json"]
        },
        "git": {
            "name": "git",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-git"],
            "description": "Git repository operations",
            "capabilities": ["git_status", "git_log", "git_diff", "git_commit"]
        },
        "sqlite": {
            "name": "sqlite",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sqlite"],
            "description": "SQLite database operations",
            "capabilities": ["query", "execute", "list_tables"]
        },
        "memory": {
            "name": "memory",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "description": "Persistent memory/knowledge graph",
            "capabilities": ["store", "retrieve", "search", "relate"]
        },
        "brave-search": {
            "name": "brave-search",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "description": "Web search via Brave",
            "capabilities": ["web_search", "local_search"]
        },
        "puppeteer": {
            "name": "puppeteer",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
            "description": "Browser automation",
            "capabilities": ["navigate", "screenshot", "click", "type", "evaluate"]
        },
        "slack": {
            "name": "slack",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-slack"],
            "description": "Slack integration",
            "capabilities": ["list_channels", "post_message", "search"]
        },
        "github": {
            "name": "github",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "description": "GitHub operations",
            "capabilities": ["list_repos", "create_issue", "create_pr", "search_code"]
        },
        "postgres": {
            "name": "postgres",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres"],
            "description": "PostgreSQL database operations",
            "capabilities": ["query", "execute", "list_tables", "describe_table"]
        },
        "sequential-thinking": {
            "name": "sequential-thinking",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "description": "Step-by-step reasoning",
            "capabilities": ["think_step", "plan", "reflect"]
        },
        "everything": {
            "name": "everything",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-everything"],
            "description": "Demo server with all capabilities",
            "capabilities": ["demo"]
        }
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize MCP manager.
        
        Args:
            config_dir: Directory for MCP configuration files.
        """
        self.config_dir = config_dir or os.path.join(
            os.getenv('HOME') or os.getenv('USERPROFILE') or os.path.expanduser('~'),
            '.aether', 'mcp'
        )
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.config_file = os.path.join(self.config_dir, 'mcp-config.json')
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load MCP configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                for name, server_config in config.get('servers', {}).items():
                    self.servers[name] = MCPServer(
                        name=name,
                        command=server_config.get('command', ''),
                        args=server_config.get('args', []),
                        env=server_config.get('env', {}),
                        enabled=server_config.get('enabled', True),
                        description=server_config.get('description', ''),
                        capabilities=server_config.get('capabilities', [])
                    )
        except Exception as e:
            print(f"Warning: Could not load MCP config: {e}")
    
    def _save_config(self):
        """Save MCP configuration to file."""
        try:
            config = {
                "servers": {
                    name: {
                        "command": server.command,
                        "args": server.args,
                        "env": server.env,
                        "enabled": server.enabled,
                        "description": server.description,
                        "capabilities": server.capabilities
                    }
                    for name, server in self.servers.items()
                },
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save MCP config: {e}")
    
    def add_server(self, name: str, command: str, args: List[str] = None,
                   env: Dict[str, str] = None, description: str = "",
                   capabilities: List[str] = None) -> MCPServer:
        """Add a new MCP server configuration.
        
        Args:
            name: Unique server name.
            command: Command to run the server.
            args: Command arguments.
            env: Environment variables.
            description: Server description.
            capabilities: List of capability names.
            
        Returns:
            Created MCPServer object.
        """
        server = MCPServer(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            description=description,
            capabilities=capabilities or []
        )
        
        self.servers[name] = server
        self._save_config()
        return server
    
    def add_default_server(self, name: str) -> Optional[MCPServer]:
        """Add a default MCP server by name.
        
        Args:
            name: Name from DEFAULT_SERVERS.
            
        Returns:
            Created MCPServer or None if not found.
        """
        if name not in self.DEFAULT_SERVERS:
            return None
        
        config = self.DEFAULT_SERVERS[name]
        return self.add_server(
            name=config['name'],
            command=config['command'],
            args=config['args'],
            description=config['description'],
            capabilities=config['capabilities']
        )
    
    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration.
        
        Args:
            name: Server name to remove.
            
        Returns:
            True if removed.
        """
        if name in self.servers:
            # Stop if running
            self.stop_server(name)
            del self.servers[name]
            self._save_config()
            return True
        return False
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all configured MCP servers.
        
        Returns:
            List of server info dictionaries.
        """
        return [
            {
                "name": server.name,
                "description": server.description,
                "enabled": server.enabled,
                "status": server.status,
                "capabilities": server.capabilities
            }
            for server in self.servers.values()
        ]
    
    def list_available_servers(self) -> List[Dict[str, Any]]:
        """List all available default servers.
        
        Returns:
            List of available server configurations.
        """
        return [
            {
                "name": name,
                "description": config['description'],
                "capabilities": config['capabilities'],
                "installed": name in self.servers
            }
            for name, config in self.DEFAULT_SERVERS.items()
        ]
    
    def start_server(self, name: str) -> Tuple[bool, str]:
        """Start an MCP server.
        
        Args:
            name: Server name to start.
            
        Returns:
            Tuple of (success, message).
        """
        if name not in self.servers:
            return False, f"Server '{name}' not configured"
        
        server = self.servers[name]
        
        if server.process and server.process.poll() is None:
            return False, f"Server '{name}' is already running"
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(server.env)
            
            # Start process
            cmd = [server.command] + server.args
            server.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            server.status = "running"
            return True, f"Server '{name}' started (PID: {server.process.pid})"
            
        except FileNotFoundError:
            server.status = "error"
            return False, f"Command not found: {server.command}. Install Node.js and run 'npm install -g {server.args[1]}'"
        except Exception as e:
            server.status = "error"
            return False, f"Failed to start server: {str(e)}"
    
    def stop_server(self, name: str) -> Tuple[bool, str]:
        """Stop an MCP server.
        
        Args:
            name: Server name to stop.
            
        Returns:
            Tuple of (success, message).
        """
        if name not in self.servers:
            return False, f"Server '{name}' not configured"
        
        server = self.servers[name]
        
        if not server.process or server.process.poll() is not None:
            server.status = "disconnected"
            return False, f"Server '{name}' is not running"
        
        try:
            server.process.terminate()
            server.process.wait(timeout=5)
            server.status = "disconnected"
            server.process = None
            return True, f"Server '{name}' stopped"
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.status = "disconnected"
            server.process = None
            return True, f"Server '{name}' force killed"
        except Exception as e:
            return False, f"Error stopping server: {str(e)}"
    
    def get_server_status(self, name: str) -> Dict[str, Any]:
        """Get detailed status of an MCP server.
        
        Args:
            name: Server name.
            
        Returns:
            Status dictionary.
        """
        if name not in self.servers:
            return {"error": f"Server '{name}' not configured"}
        
        server = self.servers[name]
        
        status = {
            "name": server.name,
            "status": server.status,
            "enabled": server.enabled,
            "description": server.description,
            "capabilities": server.capabilities,
            "command": f"{server.command} {' '.join(server.args)}",
            "pid": server.process.pid if server.process and server.process.poll() is None else None
        }
        
        return status
    
    def format_server_list(self) -> str:
        """Format server list for display.
        
        Returns:
            Formatted string.
        """
        lines = ["📡 **MCP Servers**\n"]
        
        configured = self.list_servers()
        if configured:
            lines.append("**Configured:**")
            for server in configured:
                status_icon = "🟢" if server['status'] == 'running' else "🔴"
                lines.append(f"  {status_icon} **{server['name']}** - {server['description']}")
                if server['capabilities']:
                    lines.append(f"      Capabilities: {', '.join(server['capabilities'][:5])}")
        else:
            lines.append("No servers configured yet.")
        
        lines.append("\n**Available (not installed):**")
        available = [s for s in self.list_available_servers() if not s['installed']]
        for server in available[:8]:
            lines.append(f"  ⚪ **{server['name']}** - {server['description']}")
        
        if len(available) > 8:
            lines.append(f"  ... and {len(available) - 8} more")
        
        lines.append("\n💡 Use `/mcp add <name>` to add a server")
        lines.append("💡 Use `/mcp start <name>` to start a server")
        
        return "\n".join(lines)



