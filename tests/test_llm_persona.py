"""Tests for llm-persona-py."""

import pytest
from llm_persona import Persona, PersonaRegistry


def test_persona_as_message():
    p = Persona(name="assistant", system_prompt="You are helpful.")
    msg = p.as_message()
    assert msg == {"role": "system", "content": "You are helpful."}


def test_persona_with_variables():
    p = Persona(name="teacher", system_prompt="You are a {subject} teacher.")
    p2 = p.with_variables({"subject": "Python"})
    assert p2.system_prompt == "You are a Python teacher."
    assert p.system_prompt == "You are a {subject} teacher."  # original unchanged


def test_persona_with_variables_missing_key_passthrough():
    p = Persona(name="t", system_prompt="Hello {missing}!")
    p2 = p.with_variables({"other": "x"})
    assert "{missing}" in p2.system_prompt


def test_persona_with_variables_bad_attribute_passthrough():
    # {user.nope} triggers AttributeError inside format_map; should fall back
    # to the original prompt rather than raising.
    p = Persona(name="t", system_prompt="Hi {user.nope}")
    p2 = p.with_variables({"user": "astring"})
    assert p2.system_prompt == "Hi {user.nope}"


def test_persona_with_variables_bad_index_passthrough():
    # {nums[5]} triggers IndexError inside format_map; should fall back.
    p = Persona(name="t", system_prompt="Hi {nums[5]}")
    p2 = p.with_variables({"nums": [1, 2]})
    assert p2.system_prompt == "Hi {nums[5]}"


def test_registry_add_and_get():
    reg = PersonaRegistry()
    reg.add("assistant", "You are helpful.")
    p = reg.get("assistant")
    assert p.name == "assistant"
    assert p.system_prompt == "You are helpful."


def test_registry_get_missing_raises():
    reg = PersonaRegistry()
    with pytest.raises(KeyError) as exc_info:
        reg.get("nonexistent")
    assert "nonexistent" in str(exc_info.value)


def test_registry_get_or_none_missing():
    reg = PersonaRegistry()
    assert reg.get_or_none("x") is None


def test_registry_get_or_none_found():
    reg = PersonaRegistry()
    reg.add("x", "system")
    assert reg.get_or_none("x") is not None


def test_registry_overwrite():
    reg = PersonaRegistry()
    reg.add("x", "original")
    reg.add("x", "updated")
    assert reg.get("x").system_prompt == "updated"


def test_registry_remove():
    reg = PersonaRegistry()
    reg.add("a", "A")
    reg.remove("a")
    assert not reg.has("a")


def test_registry_remove_missing_is_noop():
    reg = PersonaRegistry()
    reg.remove("nonexistent")  # should not raise


def test_registry_has():
    reg = PersonaRegistry()
    reg.add("a", "A")
    assert reg.has("a") is True
    assert reg.has("b") is False


def test_registry_contains_operator():
    reg = PersonaRegistry()
    reg.add("a", "A")
    assert "a" in reg
    assert "b" not in reg


def test_registry_names():
    reg = PersonaRegistry()
    reg.add("b", "B")
    reg.add("a", "A")
    names = reg.names()
    assert "a" in names
    assert "b" in names


def test_registry_all_personas():
    reg = PersonaRegistry()
    reg.add("a", "A", description="desc A")
    reg.add("b", "B")
    personas = reg.all_personas()
    assert len(personas) == 2


def test_registry_len():
    reg = PersonaRegistry()
    assert len(reg) == 0
    reg.add("a", "A")
    assert len(reg) == 1
    reg.add("b", "B")
    assert len(reg) == 2


def test_registry_as_message():
    reg = PersonaRegistry()
    reg.add("assistant", "You are helpful.")
    msg = reg.as_message("assistant")
    assert msg == {"role": "system", "content": "You are helpful."}


def test_registry_as_message_with_variables():
    reg = PersonaRegistry()
    reg.add("teacher", "You are a {subject} teacher.")
    msg = reg.as_message("teacher", variables={"subject": "Python"})
    assert msg["content"] == "You are a Python teacher."


def test_registry_clone_independence():
    reg = PersonaRegistry()
    reg.add("x", "original")
    clone = reg.clone()
    clone.add("x", "modified")
    assert reg.get("x").system_prompt == "original"
    assert clone.get("x").system_prompt == "modified"


def test_registry_chaining():
    reg = PersonaRegistry()
    result = reg.add("a", "A").add("b", "B").remove("a")
    assert result is reg
    assert "b" in reg
    assert "a" not in reg


def test_persona_metadata():
    reg = PersonaRegistry()
    reg.add("x", "X", metadata={"version": "1.0"})
    p = reg.get("x")
    assert p.metadata["version"] == "1.0"


def test_persona_description():
    reg = PersonaRegistry()
    reg.add("x", "X", description="A test persona")
    p = reg.get("x")
    assert p.description == "A test persona"
