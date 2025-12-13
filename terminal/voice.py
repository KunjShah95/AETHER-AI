"""
Enhanced Voice Manager for AetherAI - Text-to-Speech and Speech-to-Text.

This module provides:
- Speech-to-Text (STT) using multiple backends
- Text-to-Speech (TTS) using multiple engines
- Voice activity detection
- Multi-language support
- Continuous listening mode
"""

import os
import sys
import threading
import time
import queue
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class STTBackend(Enum):
    """Available Speech-to-Text backends."""
    GOOGLE = "google"
    WHISPER = "whisper"
    SPHINX = "sphinx"
    AZURE = "azure"


class TTSBackend(Enum):
    """Available Text-to-Speech backends."""
    PYTTSX3 = "pyttsx3"
    GTTS = "gtts"
    AZURE = "azure"
    ELEVEN_LABS = "elevenlabs"


# Try importing speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    sr = None
    SR_AVAILABLE = False

# Try importing pyttsx3 for offline TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False

# Try importing gTTS for Google TTS
try:
    from gtts import gTTS
    import tempfile
    GTTS_AVAILABLE = True
except ImportError:
    gTTS = None
    GTTS_AVAILABLE = False

# Try importing playsound for audio playback
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    playsound = None
    PLAYSOUND_AVAILABLE = False


@dataclass
class VoiceConfig:
    """Voice configuration settings."""
    stt_backend: STTBackend = STTBackend.GOOGLE
    tts_backend: TTSBackend = TTSBackend.PYTTSX3
    language: str = "en-US"
    rate: int = 170  # Words per minute
    volume: float = 0.9
    voice_id: Optional[str] = None
    timeout: int = 5
    phrase_time_limit: int = 15
    auto_speak_responses: bool = False
    continuous_listen: bool = False


class EnhancedVoiceManager:
    """Enhanced voice manager with TTS and STT support."""
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        """Initialize the voice manager.
        
        Args:
            config: Voice configuration settings.
        """
        self.config = config or VoiceConfig()
        self.enabled = False
        self.is_speaking = False
        self.is_listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._callback: Optional[Callable[[str], None]] = None
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None
        if self.recognizer:
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
        
        # Initialize TTS engine
        self.tts_engine = None
        self._init_tts()
        
        # Available voices cache
        self._voices: List[Dict[str, Any]] = []
    
    def _init_tts(self):
        """Initialize the TTS engine based on config."""
        if self.config.tts_backend == TTSBackend.PYTTSX3 and PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.config.rate)
                self.tts_engine.setProperty('volume', self.config.volume)
                
                # Get available voices
                voices = self.tts_engine.getProperty('voices')
                self._voices = [
                    {"id": v.id, "name": v.name, "languages": getattr(v, 'languages', [])}
                    for v in voices
                ]
                
                # Set voice if specified
                if self.config.voice_id:
                    for voice in voices:
                        if self.config.voice_id in voice.id or self.config.voice_id in voice.name:
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                            
            except Exception as e:
                print(f"⚠️ TTS initialization failed: {e}")
                self.tts_engine = None
    
    # =========================================================================
    # Availability Checks
    # =========================================================================
    
    def is_stt_available(self) -> bool:
        """Check if Speech-to-Text is available."""
        return SR_AVAILABLE and self.recognizer is not None
    
    def is_tts_available(self) -> bool:
        """Check if Text-to-Speech is available."""
        if self.config.tts_backend == TTSBackend.PYTTSX3:
            return PYTTSX3_AVAILABLE and self.tts_engine is not None
        elif self.config.tts_backend == TTSBackend.GTTS:
            return GTTS_AVAILABLE and PLAYSOUND_AVAILABLE
        return False
    
    def is_available(self) -> bool:
        """Check if voice features are available."""
        return self.is_stt_available() or self.is_tts_available()
    
    def get_status(self) -> Dict[str, Any]:
        """Get voice system status."""
        return {
            "enabled": self.enabled,
            "stt_available": self.is_stt_available(),
            "tts_available": self.is_tts_available(),
            "stt_backend": self.config.stt_backend.value,
            "tts_backend": self.config.tts_backend.value,
            "language": self.config.language,
            "is_speaking": self.is_speaking,
            "is_listening": self.is_listening,
            "auto_speak": self.config.auto_speak_responses,
            "voice_count": len(self._voices)
        }
    
    # =========================================================================
    # Speech-to-Text (STT)
    # =========================================================================
    
    def listen(self, timeout: Optional[int] = None, 
               phrase_time_limit: Optional[int] = None,
               silent: bool = False) -> Optional[str]:
        """Listen for voice input and convert to text.
        
        Args:
            timeout: Seconds to wait for speech to start.
            phrase_time_limit: Maximum seconds of speech to capture.
            silent: If True, suppress console output.
            
        Returns:
            Recognized text or None if failed.
        """
        if not self.is_stt_available():
            return None
        
        if self.is_listening:
            return None
        
        timeout = timeout or self.config.timeout
        phrase_limit = phrase_time_limit or self.config.phrase_time_limit
        
        self.is_listening = True
        
        try:
            with sr.Microphone() as source:
                if not silent:
                    print("🎤 Listening...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit
                )
                
                if not silent:
                    print("🔄 Processing...")
                
                # Recognize speech based on backend
                if self.config.stt_backend == STTBackend.GOOGLE:
                    text = self.recognizer.recognize_google(
                        audio,
                        language=self.config.language
                    )
                elif self.config.stt_backend == STTBackend.SPHINX:
                    text = self.recognizer.recognize_sphinx(audio)
                elif self.config.stt_backend == STTBackend.WHISPER:
                    # Requires openai-whisper package
                    try:
                        text = self.recognizer.recognize_whisper(
                            audio,
                            language=self.config.language.split('-')[0]
                        )
                    except AttributeError:
                        text = self.recognizer.recognize_google(audio)
                else:
                    text = self.recognizer.recognize_google(audio)
                
                if not silent:
                    print(f"🎤 You said: {text}")
                return text
                
        except sr.WaitTimeoutError:
            if not silent:
                print("🎤 No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            if not silent:
                print("🎤 Could not understand audio")
            return None
        except sr.RequestError as e:
            if not silent:
                print(f"🎤 Speech service error: {e}")
            return None
        except Exception as e:
            if not silent:
                print(f"🎤 Error: {e}")
            return None
        finally:
            self.is_listening = False

    def listen_for_wake_word(self, wake_word: str, callback: Callable[[], None]):
        """Listen for wake word in background.
        
        Args:
            wake_word: The keyword to trigger the callback (e.g., "aether").
            callback: Function to call when wake word is detected.
        """
        if not self.is_stt_available():
            print("❌ Speech recognition not available")
            return
        
        self.config.continuous_listen = True
        
        def _wake_loop():
            print(f"🎤 Waiting for wake word: '{wake_word}'...")
            while self.config.continuous_listen:
                # Use short timeout and silence to avoid spam
                try:
                    text = self.listen(timeout=2, phrase_time_limit=3, silent=True)
                    if text:
                        if wake_word.lower() in text.lower():
                            print(f"🎤 Wake word '{wake_word}' detected!")
                            # Play a chime or visual cue if possible
                            if self.is_tts_available():
                                self.speak("Listening", wait=False)
                            callback()
                except Exception:
                    pass
                time.sleep(0.1)
        
        self._listen_thread = threading.Thread(target=_wake_loop, daemon=True)
        self._listen_thread.start()
    
    def listen_continuous(self, callback: Callable[[str], None],
                          stop_phrase: str = "stop listening"):
        """Start continuous listening mode.
        
        Args:
            callback: Function to call with recognized text.
            stop_phrase: Phrase to stop listening.
        """
        if not self.is_stt_available():
            return
        
        self._callback = callback
        self.config.continuous_listen = True
        
        def _listen_loop():
            print("🎤 Continuous listening started. Say 'stop listening' to end.")
            while self.config.continuous_listen:
                text = self.listen()
                if text:
                    if stop_phrase.lower() in text.lower():
                        print("🎤 Stopping continuous listening")
                        self.config.continuous_listen = False
                        break
                    if self._callback:
                        self._callback(text)
                time.sleep(0.5)
        
        self._listen_thread = threading.Thread(target=_listen_loop, daemon=True)
        self._listen_thread.start()
    
    def stop_continuous_listen(self):
        """Stop continuous listening mode."""
        self.config.continuous_listen = False
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2)
    
    # =========================================================================
    # Text-to-Speech (TTS)
    # =========================================================================
    
    def speak(self, text: str, wait: bool = False):
        """Speak the given text.
        
        Args:
            text: Text to speak.
            wait: If True, wait for speech to complete.
        """
        if not self.is_tts_available() or not self.enabled:
            return
        
        if not text or not text.strip():
            return
        
        # Clean up text for speech
        text = self._prepare_text_for_speech(text)
        
        if not text:
            return
        
        def _speak_thread():
            self.is_speaking = True
            try:
                if self.config.tts_backend == TTSBackend.PYTTSX3 and self.tts_engine:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    
                elif self.config.tts_backend == TTSBackend.GTTS and GTTS_AVAILABLE:
                    # Use gTTS for online TTS
                    tts = gTTS(text=text, lang=self.config.language.split('-')[0])
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                        tts.save(fp.name)
                        if PLAYSOUND_AVAILABLE:
                            playsound(fp.name)
                        os.unlink(fp.name)
                        
            except Exception as e:
                print(f"🔊 TTS Error: {e}")
            finally:
                self.is_speaking = False
        
        thread = threading.Thread(target=_speak_thread, daemon=True)
        thread.start()
        
        if wait:
            thread.join()
    
    def _prepare_text_for_speech(self, text: str) -> str:
        """Prepare text for speech output.
        
        Args:
            text: Raw text to prepare.
            
        Returns:
            Cleaned text suitable for speech.
        """
        if not text:
            return ""
        
        # Handle code blocks
        if "```" in text:
            # Extract non-code parts only
            parts = text.split("```")
            clean_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Not a code block
                    clean_parts.append(part)
                else:
                    clean_parts.append("I've generated some code. Please check the terminal.")
            text = " ".join(clean_parts)
        
        # Remove markdown formatting
        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code
        text = re.sub(r'#{1,6}\s*', '', text)         # Headers
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Remove emojis (they don't speak well)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Truncate long text
        max_length = 500
        if len(text) > max_length:
            # Find a good breaking point
            text = text[:max_length]
            last_period = text.rfind('.')
            if last_period > max_length // 2:
                text = text[:last_period + 1]
            text += " Text truncated for speech."
        
        return text.strip()
    
    def stop_speaking(self):
        """Stop current speech output."""
        if self.tts_engine and self.is_speaking:
            try:
                self.tts_engine.stop()
            except Exception:
                pass
        self.is_speaking = False
    
    # =========================================================================
    # Voice Configuration
    # =========================================================================
    
    def list_voices(self) -> List[Dict[str, Any]]:
        """List available TTS voices.
        
        Returns:
            List of voice info dictionaries.
        """
        return self._voices
    
    def set_voice(self, voice_id: str) -> bool:
        """Set the TTS voice.
        
        Args:
            voice_id: Voice identifier.
            
        Returns:
            True if successful.
        """
        if not self.tts_engine:
            return False
        
        try:
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if voice_id in voice.id or voice_id.lower() in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    self.config.voice_id = voice.id
                    return True
        except Exception:
            pass
        
        return False
    
    def set_rate(self, rate: int) -> bool:
        """Set speech rate.
        
        Args:
            rate: Words per minute (50-300).
            
        Returns:
            True if successful.
        """
        if not self.tts_engine:
            return False
        
        rate = max(50, min(300, rate))
        try:
            self.tts_engine.setProperty('rate', rate)
            self.config.rate = rate
            return True
        except Exception:
            return False
    
    def set_volume(self, volume: float) -> bool:
        """Set speech volume.
        
        Args:
            volume: Volume level (0.0-1.0).
            
        Returns:
            True if successful.
        """
        if not self.tts_engine:
            return False
        
        volume = max(0.0, min(1.0, volume))
        try:
            self.tts_engine.setProperty('volume', volume)
            self.config.volume = volume
            return True
        except Exception:
            return False
    
    def enable(self):
        """Enable voice features."""
        self.enabled = True
        if self.config.auto_speak_responses:
            self.speak("Voice enabled. I will now speak my responses.")
    
    def disable(self):
        """Disable voice features."""
        if self.is_speaking:
            self.stop_speaking()
        self.stop_continuous_listen()
        self.enabled = False
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def format_status(self) -> str:
        """Format voice status for display.
        
        Returns:
            Formatted status string.
        """
        status = self.get_status()
        
        lines = ["🎤 **Voice System Status**\n"]
        
        # STT Status
        if status['stt_available']:
            lines.append(f"✅ Speech-to-Text: Available ({status['stt_backend']})")
        else:
            lines.append("❌ Speech-to-Text: Not available")
            lines.append("   Install: `pip install SpeechRecognition pyaudio`")
        
        # TTS Status
        if status['tts_available']:
            lines.append(f"✅ Text-to-Speech: Available ({status['tts_backend']})")
            if status['voice_count'] > 0:
                lines.append(f"   Voices available: {status['voice_count']}")
        else:
            lines.append("❌ Text-to-Speech: Not available")
            lines.append("   Install: `pip install pyttsx3`")
        
        # Current state
        lines.append(f"\n**Status:** {'Enabled' if status['enabled'] else 'Disabled'}")
        if status['is_speaking']:
            lines.append("🔊 Currently speaking...")
        if status['is_listening']:
            lines.append("🎤 Currently listening...")
        
        # Commands
        lines.append("\n**Commands:**")
        lines.append("  `/voice on`  - Enable voice")
        lines.append("  `/voice off` - Disable voice")
        lines.append("  `/listen`    - Listen for input")
        lines.append("  `/speak <text>` - Speak text")
        lines.append("  `/voices`    - List available voices")
        
        return "\n".join(lines)


# Factory function for backward compatibility
def VoiceManager():
    """Create a VoiceManager instance (backward compatible)."""
    manager = EnhancedVoiceManager()
    return manager
