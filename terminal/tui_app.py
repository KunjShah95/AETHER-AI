from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, TabbedContent, TabPane, Static, Button
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from rich.text import Text
from datetime import datetime
import asyncio

# Import AetherAI from main (assuming main.py is in the same package or path)
# We need to handle imports carefully to avoid circular dependency if main imports this
try:
    from terminal.main import AetherAI
    from terminal.dashboard_tui import SystemMonitor
except ImportError:
    # Fallback for direct execution or different path structure
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from terminal.main import AetherAI
    from terminal.dashboard_tui import SystemMonitor

class ChatView(Container):
    """Chat interface with history and input."""
    
    def compose(self) -> ComposeResult:
        yield RichLog(id="chat_log", markup=True, wrap=True)
        yield Input(placeholder="Type a command or chat with AI...", id="chat_input")

class ToolsView(Container):
    """Quick access tools."""
    
    def compose(self) -> ComposeResult:
        yield Static("Quick Tools", classes="header")
        with Horizontal():
            yield Button("System Status", id="btn_status", variant="primary")
            yield Button("List Models", id="btn_models", variant="default")
            yield Button("Clear Chat", id="btn_clear", variant="error")

class AetherTUI(App):
    """The Unified Aether AI Terminal Interface."""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    ChatView {
        height: 100%;
        layout: vertical;
    }
    
    #chat_log {
        height: 1fr;
        border: solid green;
        background: $surface;
        padding: 1;
    }
    
    #chat_input {
        dock: bottom;
        border: solid cyan;
    }
    
    ToolsView {
        padding: 1;
        height: auto;
    }
    
    Button {
        margin: 1;
    }
    
    .header {
        text-align: center;
        text-style: bold;
        padding: 1;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "clear_chat", "Clear Chat"),
        ("d", "toggle_dark", "Toggle Dark Mode"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="chat"):
            with TabPane("Chat", id="chat"):
                yield ChatView()
            with TabPane("Dashboard", id="dashboard"):
                yield SystemMonitor()
            with TabPane("Tools", id="tools"):
                yield ToolsView()
        yield Footer()

    def on_mount(self) -> None:
        """Initialize AetherAI when app starts."""
        self.query_one(RichLog).write(Text("Initializing AETHER AI...", style="bold green"))
        
        # Initialize AI in background to not block UI
        self.run_worker(self.init_ai)

    async def init_ai(self):
        """Async initialization of AI."""
        try:
            self.ai = AetherAI(quiet=True)
            log = self.query_one(RichLog)
            log.write(Text(f"Welcome to AETHER AI v3.1", style="bold cyan"))
            log.write(Text(f"Current Model: {self.ai.current_model.upper()}", style="yellow"))
            log.write(Text("Type /help for commands.", style="dim"))
        except Exception as e:
            self.query_one(RichLog).write(Text(f"Error initializing AI: {e}", style="bold red"))

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle input submission."""
        user_input = message.value.strip()
        if not user_input:
            return
            
        input_widget = self.query_one(Input)
        input_widget.value = ""
        
        log = self.query_one(RichLog)
        log.write(Text(f"> {user_input}", style="bold white"))
        
        if user_input.lower() in ["/quit", "exit", "/exit"]:
            self.exit()
            return
            
        if user_input.lower() in ["/clear", "clear"]:
            log.clear()
            return

        # Process input via AetherAI
        if hasattr(self, 'ai'):
            # Show thinking indicator
            log.write(Text("Thinking...", style="dim blue"))
            
            # Run processing in a worker thread to keep UI responsive
            self.run_worker(self.process_ai_input(user_input))
        else:
            log.write(Text("AI not initialized yet.", style="red"))

    async def process_ai_input(self, user_input: str):
        """Process input in background."""
        try:
            response = self.ai.process_input(user_input)
            
            # Update UI with response
            log = self.query_one(RichLog)
            # Remove "Thinking..." (simple way is just to print response, Textual log appends)
            # In a real app we might want to replace the last line, but appending is fine.
            
            if response:
                log.write(Text(str(response), style="green"))
            else:
                log.write(Text("No response.", style="dim red"))
                
        except Exception as e:
             self.query_one(RichLog).write(Text(f"Error: {e}", style="bold red"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn_clear":
            self.query_one(RichLog).clear()
        elif event.button.id == "btn_status":
            self.query_one(RichLog).write(Text("System Status Checked.", style="italic"))
            # Could trigger a command here
        elif event.button.id == "btn_models":
             if hasattr(self, 'ai'):
                 self.run_worker(self.process_ai_input("/models"))

    def action_clear_chat(self) -> None:
        self.query_one(RichLog).clear()

if __name__ == "__main__":
    app = AetherTUI()
    app.run()
