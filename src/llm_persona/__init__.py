"""llm-persona-py — named persona registry for LLM agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Persona:
    """A named agent persona with a system prompt and optional metadata."""

    name: str
    system_prompt: str
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def as_message(self) -> dict[str, str]:
        """Return a system message dict ready for the LLM."""
        return {"role": "system", "content": self.system_prompt}

    def with_variables(self, variables: dict[str, Any]) -> "Persona":
        """Return a new Persona with variables interpolated into the system prompt."""
        try:
            rendered = self.system_prompt.format_map(variables)
        except (KeyError, ValueError):
            rendered = self.system_prompt
        return Persona(
            name=self.name,
            system_prompt=rendered,
            description=self.description,
            metadata={**self.metadata},
        )


class PersonaRegistry:
    """
    Registry of named LLM personas.

    Personas are system-prompt templates for different agent roles. Retrieve
    them by name to build per-request system messages.

    Example::

        registry = PersonaRegistry()
        registry.add("assistant", "You are a helpful coding assistant.")
        registry.add("reviewer", "You are a strict code reviewer. Be concise.")
        registry.add("teacher", "You are {subject} teacher. Explain clearly.")

        persona = registry.get("teacher")
        msg = persona.with_variables({"subject": "a Python"}).as_message()
        # {"role": "system", "content": "You are a Python teacher. Explain clearly."}
    """

    def __init__(self) -> None:
        self._personas: dict[str, Persona] = {}

    def add(
        self,
        name: str,
        system_prompt: str,
        description: str = "",
        metadata: dict | None = None,
    ) -> "PersonaRegistry":
        """Register a persona by name. Overwrites any existing persona with the same name."""
        self._personas[name] = Persona(
            name=name,
            system_prompt=system_prompt,
            description=description,
            metadata=metadata or {},
        )
        return self

    def get(self, name: str) -> Persona:
        """Retrieve a persona by name.

        Raises:
            KeyError: If no persona is registered with that name.
        """
        if name not in self._personas:
            raise KeyError(f"No persona registered with name '{name}'.")
        return self._personas[name]

    def get_or_none(self, name: str) -> Persona | None:
        """Return the persona or None if not found."""
        return self._personas.get(name)

    def remove(self, name: str) -> "PersonaRegistry":
        """Remove a persona by name. No-op if not registered."""
        self._personas.pop(name, None)
        return self

    def has(self, name: str) -> bool:
        """Return True if a persona is registered with that name."""
        return name in self._personas

    def names(self) -> list[str]:
        """Return all registered persona names in insertion order."""
        return list(self._personas.keys())

    def all_personas(self) -> list[Persona]:
        """Return all registered Persona objects."""
        return list(self._personas.values())

    def as_message(self, name: str, variables: dict[str, Any] | None = None) -> dict:
        """Get a persona by name and return its system message dict.

        Args:
            name: Persona name.
            variables: Optional dict for format_map interpolation.

        Returns:
            {"role": "system", "content": "..."}
        """
        persona = self.get(name)
        if variables:
            return persona.with_variables(variables).as_message()
        return persona.as_message()

    def clone(self) -> "PersonaRegistry":
        """Return a deep copy of this registry."""
        new = PersonaRegistry()
        for p in self._personas.values():
            new.add(p.name, p.system_prompt, p.description, dict(p.metadata))
        return new

    def __len__(self) -> int:
        return len(self._personas)

    def __contains__(self, name: str) -> bool:
        return self.has(name)


__all__ = ["Persona", "PersonaRegistry"]
