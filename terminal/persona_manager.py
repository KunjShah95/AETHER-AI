import json
import os
from typing import Dict, Optional

class PersonaManager:
    def __init__(self):
        self.persona_file = os.path.join(os.path.expanduser("~"), ".nexus", "personas.json")
        self.personas = self._load_personas()
        self.current_persona = None

    def _load_personas(self) -> Dict[str, str]:
        defaults = {
            "default": "You are Nexus AI, a helpful and secure terminal assistant.",
            "coder": "You are an expert software engineer. Provide concise, efficient, and secure code solutions.",
            "pirate": "You are a pirate captain. Speak in nautical terms and be adventurous!",
            "teacher": "You are a patient teacher. Explain concepts simply and clearly.",
            "security": "You are an expert Security Engineer. Analyze code for vulnerabilities (OWASP Top 10), suggest hardening measures, and prioritize safety above all else.",
            "qa": "You are a Senior QA Engineer. Focus on edge cases, robust error handling, generating comprehensive test plans, and ensuring code reliability.",
            "frontend": "You are a Frontend Architect expert in React, Tailwind, and UX/UI principles. Focus on accessibility, component composition, and responsive design."
        }
        if os.path.exists(self.persona_file):
            try:
                with open(self.persona_file, 'r') as f:
                    saved = json.load(f)
                    defaults.update(saved)
            except Exception:
                pass
        return defaults

    def _save_personas(self):
        os.makedirs(os.path.dirname(self.persona_file), exist_ok=True)
        with open(self.persona_file, 'w') as f:
            json.dump(self.personas, f, indent=2)

    def create_persona(self, name: str, prompt: str) -> str:
        self.personas[name] = prompt
        self._save_personas()
        return f"✅ Persona '{name}' created."

    def set_persona(self, name: str) -> str:
        if name in self.personas:
            self.current_persona = name
            return f"🎭 Switched to persona: {name}"
        return f"❌ Persona '{name}' not found."

    def get_system_prompt(self) -> str:
        if self.current_persona:
            return self.personas.get(self.current_persona, self.personas["default"])
        return self.personas["default"]

    def list_personas(self) -> str:
        return "\n".join([f"🎭 {k}: {v[:50]}..." for k, v in self.personas.items()])
