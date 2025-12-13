"""
Streaming Response Handler for AetherAI.

This module provides real-time streaming of AI responses for a more
interactive user experience.
"""

import sys
import time
from typing import Generator, Optional, Callable, Any
from functools import wraps

# Try importing rich for better terminal output
try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.panel import Panel
    from rich.spinner import Spinner
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


class StreamingHandler:
    """Handles streaming responses from AI models."""
    
    def __init__(self, console: Optional[Any] = None):
        """Initialize streaming handler.
        
        Args:
            console: Rich Console instance for output.
        """
        if RICH_AVAILABLE and console is None:
            self.console = Console()
        else:
            self.console = console
        
        self.is_streaming = False
        self.current_content = ""
        self.stream_callback: Optional[Callable[[str], None]] = None
    
    def stream_print(self, text: str, end: str = "", style: str = None):
        """Print text in streaming fashion.
        
        Args:
            text: Text to print.
            end: End character (default no newline).
            style: Rich style to apply.
        """
        if RICH_AVAILABLE and self.console:
            if style:
                self.console.print(text, end=end, style=style)
            else:
                self.console.print(text, end=end)
        else:
            print(text, end=end, flush=True)
    
    def stream_gemini(self, model, prompt: str) -> str:
        """Stream response from Gemini model.
        
        Args:
            model: Gemini GenerativeModel instance.
            prompt: Prompt to send.
            
        Returns:
            Complete response text.
        """
        try:
            import google.generativeai as genai
            
            self.is_streaming = True
            self.current_content = ""
            
            # Generate with streaming
            response = model.generate_content(
                prompt,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4000
                )
            )
            
            if RICH_AVAILABLE and self.console:
                with Live(Text("▌", style="bold cyan"), refresh_per_second=10, console=self.console) as live:
                    for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            self.current_content += chunk.text
                            # Update display with markdown rendering
                            try:
                                display = Markdown(self.current_content + "▌")
                            except Exception:
                                display = Text(self.current_content + "▌")
                            live.update(display)
                            
                            if self.stream_callback:
                                self.stream_callback(chunk.text)
            else:
                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        self.current_content += chunk.text
                        print(chunk.text, end="", flush=True)
                        
                        if self.stream_callback:
                            self.stream_callback(chunk.text)
                print()  # Final newline
            
            self.is_streaming = False
            return self.current_content
            
        except Exception as e:
            self.is_streaming = False
            raise e
    
    def stream_groq(self, client, prompt: str, model: str = "mixtral-8x7b-32768") -> str:
        """Stream response from Groq.
        
        Args:
            client: Groq client instance.
            prompt: Prompt to send.
            model: Model name.
            
        Returns:
            Complete response text.
        """
        try:
            self.is_streaming = True
            self.current_content = ""
            
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=4000
            )
            
            if RICH_AVAILABLE and self.console:
                with Live(Text("▌", style="bold cyan"), refresh_per_second=10, console=self.console) as live:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            self.current_content += content
                            try:
                                display = Markdown(self.current_content + "▌")
                            except Exception:
                                display = Text(self.current_content + "▌")
                            live.update(display)
                            
                            if self.stream_callback:
                                self.stream_callback(content)
            else:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        self.current_content += content
                        print(content, end="", flush=True)
                        
                        if self.stream_callback:
                            self.stream_callback(content)
                print()
            
            self.is_streaming = False
            return self.current_content
            
        except Exception as e:
            self.is_streaming = False
            raise e
    
    def stream_openai(self, client, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Stream response from OpenAI.
        
        Args:
            client: OpenAI client instance.
            prompt: Prompt to send.
            model: Model name.
            
        Returns:
            Complete response text.
        """
        try:
            self.is_streaming = True
            self.current_content = ""
            
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=4000
            )
            
            if RICH_AVAILABLE and self.console:
                with Live(Text("▌", style="bold cyan"), refresh_per_second=10, console=self.console) as live:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            self.current_content += content
                            try:
                                display = Markdown(self.current_content + "▌")
                            except Exception:
                                display = Text(self.current_content + "▌")
                            live.update(display)
                            
                            if self.stream_callback:
                                self.stream_callback(content)
            else:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        self.current_content += content
                        print(content, end="", flush=True)
                        
                        if self.stream_callback:
                            self.stream_callback(content)
                print()
            
            self.is_streaming = False
            return self.current_content
            
        except Exception as e:
            self.is_streaming = False
            raise e
    
    def stream_ollama(self, model: str, prompt: str) -> str:
        """Stream response from Ollama.
        
        Args:
            model: Ollama model name.
            prompt: Prompt to send.
            
        Returns:
            Complete response text.
        """
        try:
            import ollama
            
            self.is_streaming = True
            self.current_content = ""
            
            stream = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            if RICH_AVAILABLE and self.console:
                with Live(Text("▌", style="bold cyan"), refresh_per_second=10, console=self.console) as live:
                    for chunk in stream:
                        content = ""
                        if hasattr(chunk, 'message'):
                            if hasattr(chunk.message, 'content'):
                                content = chunk.message.content
                            elif isinstance(chunk.message, dict):
                                content = chunk.message.get('content', '')
                        elif isinstance(chunk, dict) and 'message' in chunk:
                            content = chunk['message'].get('content', '')
                        
                        if content:
                            self.current_content += content
                            try:
                                display = Markdown(self.current_content + "▌")
                            except Exception:
                                display = Text(self.current_content + "▌")
                            live.update(display)
                            
                            if self.stream_callback:
                                self.stream_callback(content)
            else:
                for chunk in stream:
                    content = ""
                    if hasattr(chunk, 'message'):
                        if hasattr(chunk.message, 'content'):
                            content = chunk.message.content
                        elif isinstance(chunk.message, dict):
                            content = chunk.message.get('content', '')
                    elif isinstance(chunk, dict) and 'message' in chunk:
                        content = chunk['message'].get('content', '')
                    
                    if content:
                        self.current_content += content
                        print(content, end="", flush=True)
                        
                        if self.stream_callback:
                            self.stream_callback(content)
                print()
            
            self.is_streaming = False
            return self.current_content
            
        except Exception as e:
            self.is_streaming = False
            raise e
    
    def simulate_stream(self, text: str, delay: float = 0.02) -> str:
        """Simulate streaming for non-streaming responses.
        
        Args:
            text: Complete text to simulate streaming.
            delay: Delay between characters in seconds.
            
        Returns:
            Complete text.
        """
        self.is_streaming = True
        self.current_content = ""
        
        words = text.split(' ')
        
        if RICH_AVAILABLE and self.console:
            with Live(Text("▌", style="bold cyan"), refresh_per_second=20, console=self.console) as live:
                for word in words:
                    self.current_content += word + " "
                    try:
                        display = Markdown(self.current_content + "▌")
                    except Exception:
                        display = Text(self.current_content + "▌")
                    live.update(display)
                    time.sleep(delay)
                    
                    if self.stream_callback:
                        self.stream_callback(word + " ")
        else:
            for word in words:
                self.current_content += word + " "
                print(word, end=" ", flush=True)
                time.sleep(delay)
                
                if self.stream_callback:
                    self.stream_callback(word + " ")
            print()
        
        self.is_streaming = False
        return self.current_content.strip()
    
    def show_thinking(self, message: str = "Thinking", callback: Optional[Callable] = None) -> str:
        """Show a thinking/loading indicator while processing.
        
        Args:
            message: Message to display.
            callback: Function to execute while showing indicator.
            
        Returns:
            Result from callback or empty string.
        """
        if RICH_AVAILABLE and self.console:
            spinner = Spinner("dots", text=f" {message}...")
            
            if callback:
                result = None
                with Live(spinner, refresh_per_second=10, console=self.console):
                    result = callback()
                return result if result else ""
            else:
                with Live(spinner, refresh_per_second=10, console=self.console):
                    time.sleep(1)
                return ""
        else:
            print(f"{message}...", end="", flush=True)
            if callback:
                result = callback()
                print(" Done!")
                return result if result else ""
            else:
                print(" Done!")
                return ""
    
    def set_callback(self, callback: Callable[[str], None]):
        """Set callback for streaming chunks.
        
        Args:
            callback: Function to call with each chunk.
        """
        self.stream_callback = callback
    
    def stop_stream(self):
        """Stop current streaming."""
        self.is_streaming = False


class PipeInputHandler:
    """Handles piped input from stdin."""
    
    def __init__(self):
        """Initialize pipe handler."""
        self.has_pipe_input = False
        self.pipe_content: Optional[str] = None
        self._check_pipe()
    
    def _check_pipe(self):
        """Check if there's piped input available."""
        try:
            import select
            if sys.platform != 'win32':
                # Unix-like: use select
                if select.select([sys.stdin], [], [], 0)[0]:
                    self.has_pipe_input = True
            else:
                # Windows: check if stdin is not a tty
                if not sys.stdin.isatty():
                    self.has_pipe_input = True
        except Exception:
            self.has_pipe_input = not sys.stdin.isatty()
    
    def read_pipe(self) -> Optional[str]:
        """Read content from pipe if available.
        
        Returns:
            Piped content or None.
        """
        if not self.has_pipe_input:
            return None
        
        if self.pipe_content is not None:
            return self.pipe_content
        
        try:
            self.pipe_content = sys.stdin.read()
            return self.pipe_content
        except Exception:
            return None
    
    def get_pipe_preview(self, max_lines: int = 5) -> str:
        """Get a preview of piped content.
        
        Args:
            max_lines: Maximum lines to preview.
            
        Returns:
            Preview string.
        """
        content = self.read_pipe()
        if not content:
            return ""
        
        lines = content.split('\n')
        if len(lines) <= max_lines:
            return content
        
        preview_lines = lines[:max_lines]
        remaining = len(lines) - max_lines
        return '\n'.join(preview_lines) + f"\n... ({remaining} more lines)"
