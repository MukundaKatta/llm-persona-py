# llm-persona-py

Named persona registry for LLM agents. Define, store, and retrieve system-prompt templates by name.

## Install

```bash
pip install llm-persona-py
```

## Usage

```python
from llm_persona import PersonaRegistry, Persona

registry = PersonaRegistry()

# Register personas
registry.add("assistant", "You are a helpful coding assistant.", description="General helper")
registry.add("reviewer",  "You are a strict code reviewer. Be concise and direct.")
registry.add("teacher",   "You are a {subject} teacher. Explain step by step.")

# Retrieve and use
persona = registry.get("assistant")
msg = persona.as_message()
# {"role": "system", "content": "You are a helpful coding assistant."}

# Variable interpolation
msg = registry.as_message("teacher", variables={"subject": "Python"})
# {"role": "system", "content": "You are a Python teacher. Explain step by step."}

# Check existence
"assistant" in registry  # True

# List / iterate
print(registry.names())       # ["assistant", "reviewer", "teacher"]
print(len(registry))          # 3

# Clone for per-request modifications
variant = registry.clone()
variant.add("assistant", "You are a strict assistant.")

# Remove
registry.remove("reviewer")
```

## License

MIT
