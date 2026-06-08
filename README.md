# llm-persona-py

Named persona registry for LLM agents. Define, store, and retrieve system-prompt
templates by name, with optional `{variable}` interpolation.

It is a tiny, dependency-free helper: a `Persona` is just a system prompt plus
some metadata, and a `PersonaRegistry` is an ordered, name-keyed collection of
personas that knows how to turn them into chat-style `system` messages.

## Install

```bash
pip install llm-persona-py
```

Requires Python 3.9+. No runtime dependencies.

## Usage

```python
from llm_persona import PersonaRegistry, Persona

registry = PersonaRegistry()

# Register personas. add() returns the registry so calls can be chained.
registry.add("assistant", "You are a helpful coding assistant.", description="General helper")
registry.add("reviewer",  "You are a strict code reviewer. Be concise and direct.")
registry.add("teacher",   "You are a {subject} teacher. Explain step by step.")

# Retrieve a persona and turn it into a system message.
persona = registry.get("assistant")
msg = persona.as_message()
# {"role": "system", "content": "You are a helpful coding assistant."}

# Variable interpolation (uses str.format_map under the hood).
msg = registry.as_message("teacher", variables={"subject": "Python"})
# {"role": "system", "content": "You are a Python teacher. Explain step by step."}

# Membership and listing.
"assistant" in registry        # True
registry.has("assistant")      # True
registry.names()               # ["assistant", "reviewer", "teacher"]  (insertion order)
len(registry)                  # 3

# Clone the registry for per-request modifications without touching the original.
variant = registry.clone()
variant.add("assistant", "You are a strict assistant.")
registry.get("assistant").system_prompt   # unchanged

# Remove a persona (no-op if the name is not registered).
registry.remove("reviewer")
```

### Building a request

A common pattern is to keep one shared registry and pull the right system
message per request:

```python
messages = [
    registry.as_message("assistant"),
    {"role": "user", "content": "Refactor this function."},
]
# pass `messages` to your LLM client of choice
```

## Variable interpolation

`Persona.with_variables(...)` and `PersonaRegistry.as_message(name, variables=...)`
render `{placeholder}` tokens via `str.format_map`. If interpolation fails for any
reason — a missing key, a bad attribute/index access such as `{user.name}` or
`{items[3]}`, or a malformed format spec — the **original, unrendered prompt is
returned instead of raising**. This guarantees callers always get a usable prompt
even when a variable is missing.

```python
p = Persona(name="t", system_prompt="Hello {missing}!")
p.with_variables({"other": "x"}).system_prompt
# "Hello {missing}!"  (left as-is, no KeyError)
```

`with_variables` never mutates the persona it is called on; it returns a new
`Persona`.

## API

### `Persona`

A dataclass describing a single named persona.

| Field           | Type             | Default | Description                          |
| --------------- | ---------------- | ------- | ------------------------------------ |
| `name`          | `str`            | —       | Identifier for the persona.          |
| `system_prompt` | `str`            | —       | The system prompt (may contain `{}`).|
| `description`   | `str`            | `""`    | Optional human-readable description. |
| `metadata`      | `dict`           | `{}`    | Optional free-form metadata.         |

Methods:

- `as_message() -> dict` — return `{"role": "system", "content": system_prompt}`.
- `with_variables(variables: dict) -> Persona` — return a new `Persona` with
  `{...}` placeholders interpolated (see *Variable interpolation* above).

### `PersonaRegistry`

An ordered, name-keyed collection of personas.

- `add(name, system_prompt, description="", metadata=None) -> PersonaRegistry` —
  register (or overwrite) a persona; returns `self` for chaining.
- `get(name) -> Persona` — retrieve by name; raises `KeyError` if absent.
- `get_or_none(name) -> Persona | None` — retrieve by name or `None` if absent.
- `remove(name) -> PersonaRegistry` — remove by name; no-op if absent; returns `self`.
- `has(name) -> bool` — whether a persona with `name` is registered.
- `names() -> list[str]` — all names in insertion order.
- `all_personas() -> list[Persona]` — all `Persona` objects.
- `as_message(name, variables=None) -> dict` — system message for `name`,
  optionally interpolated.
- `clone() -> PersonaRegistry` — independent deep copy (metadata is copied too).
- `len(registry)` — number of registered personas.
- `name in registry` — membership test (alias for `has`).

## Development

Run the test suite with the standard library only — no third-party tools required:

```bash
python -m unittest discover -s tests
```

## License

MIT
