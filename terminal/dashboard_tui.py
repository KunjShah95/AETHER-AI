from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Log, Digits, RichLog
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.reactive import reactive
from textual.binding import Binding
from datetime import datetime
import psutil
import platform
import time
import os

class SystemMonitor(Static):
    """Displays system information."""
    
    cpu_usage = reactive(0.0)
    memory_usage = reactive(0.0)

    def on_mount(self) -> None:
        self.set_interval(1, self.update_stats)

    def update_stats(self) -> None:
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent

    def watch_cpu_usage(self, usage: float) -> None:
        self.update_display()

    def update_display(self) -> None:
        self.update(f"""
[bold green]SYSTEM STATUS[/bold green]
----------------
OS:  {platform.system()} {platform.release()}
CPU: {self.cpu_usage}%
RAM: {self.memory_usage}%
Disk: {psutil.disk_usage('/').percent}%
        """)

class ActiveTask(Static):
    """Displays the current active task."""
    
    task_text = reactive("Idle")
    
    def on_mount(self) -> None:
        self.update_display()
        
    def update_display(self) -> None:
        self.update(f"""
[bold blue]ACTIVE MISSION[/bold blue]
----------------
{self.task_text}
        """)

class LiveLog(RichLog):
    """Reads from ai_assistant.log and displays it."""
    
    def on_mount(self) -> None:
        self.markup = True
        self.set_interval(0.5, self.tail_log)
        self.log_file = "ai_assistant.log"
        self.last_pos = 0

    def tail_log(self) -> None:
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                f.seek(self.last_pos)
                new_lines = f.readlines()
                self.last_pos = f.tell()
                
                for line in new_lines:
                    self.write(line.strip())

class AetherDashboard(App):
    """Mission Control for Aether AI."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 1;
        background: $surface;
    }
    
    .box {
        height: 100%;
        border: heavy $accent;
        padding: 1;
        background: $surface-lighten-1;
    }
    
    #sys_mon {
        row-span: 1;
        col-span: 1;
    }

    #task_panel {
        row-span: 1;
        col-span: 1;
    }

    #log_panel {
        row-span: 1;
        col-span: 2;
        height: 100%;
        overflow-y: scroll;
    }
    """

    BINDINGS = [
        Binding("d", "toggle_dark", "Toggle Dark Mode"),
        Binding("q", "quit", "Quit Dashboard"),
    ]
    
    TITLE = "AETHER AI MISSION CONTROL"
    SUB_TITLE = "Agentic Coding Assistant"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(SystemMonitor(id="sys_mon"), classes="box")
        yield Container(ActiveTask(id="task_panel"), classes="box")
        yield LiveLog(id="log_panel", classes="box", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(LiveLog).write("[bold green]Aether AI Dashboard Initialized...[/bold green]")
        self.query_one(LiveLog).write("Monitoring 'ai_assistant.log' for activity...")

if __name__ == "__main__":
    app = AetherDashboard()
    app.run()
