"""Tests for llm-persona-py.

Uses only the Python standard library (``unittest``) so the suite runs with::

    python3 -m unittest discover -s tests

without installing any third-party test dependencies.
"""

import os
import sys
import unittest

# Make the package importable without an editable install by putting the
# ``src`` layout directory on the import path.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_persona import Persona, PersonaRegistry  # noqa: E402


class PersonaTests(unittest.TestCase):
    def test_persona_as_message(self):
        p = Persona(name="assistant", system_prompt="You are helpful.")
        self.assertEqual(
            p.as_message(),
            {"role": "system", "content": "You are helpful."},
        )

    def test_persona_with_variables(self):
        p = Persona(name="teacher", system_prompt="You are a {subject} teacher.")
        p2 = p.with_variables({"subject": "Python"})
        self.assertEqual(p2.system_prompt, "You are a Python teacher.")
        # Original is unchanged (with_variables returns a new Persona).
        self.assertEqual(p.system_prompt, "You are a {subject} teacher.")

    def test_persona_with_variables_returns_new_instance(self):
        p = Persona(name="t", system_prompt="No placeholders here.")
        p2 = p.with_variables({"unused": "x"})
        self.assertIsNot(p, p2)

    def test_persona_with_variables_missing_key_passthrough(self):
        p = Persona(name="t", system_prompt="Hello {missing}!")
        p2 = p.with_variables({"other": "x"})
        self.assertIn("{missing}", p2.system_prompt)

    def test_persona_with_variables_bad_attribute_passthrough(self):
        # {user.nope} triggers AttributeError inside format_map; should fall
        # back to the original prompt rather than raising.
        p = Persona(name="t", system_prompt="Hi {user.nope}")
        p2 = p.with_variables({"user": "astring"})
        self.assertEqual(p2.system_prompt, "Hi {user.nope}")

    def test_persona_with_variables_bad_index_passthrough(self):
        # {nums[5]} triggers IndexError inside format_map; should fall back.
        p = Persona(name="t", system_prompt="Hi {nums[5]}")
        p2 = p.with_variables({"nums": [1, 2]})
        self.assertEqual(p2.system_prompt, "Hi {nums[5]}")

    def test_persona_with_variables_copies_metadata(self):
        p = Persona(name="t", system_prompt="hi", metadata={"v": "1"})
        p2 = p.with_variables({})
        self.assertEqual(p2.metadata, {"v": "1"})
        # Mutating the copy must not affect the original.
        p2.metadata["v"] = "2"
        self.assertEqual(p.metadata["v"], "1")

    def test_persona_defaults(self):
        p = Persona(name="x", system_prompt="X")
        self.assertEqual(p.description, "")
        self.assertEqual(p.metadata, {})


class RegistryTests(unittest.TestCase):
    def test_add_and_get(self):
        reg = PersonaRegistry()
        reg.add("assistant", "You are helpful.")
        p = reg.get("assistant")
        self.assertEqual(p.name, "assistant")
        self.assertEqual(p.system_prompt, "You are helpful.")

    def test_get_missing_raises(self):
        reg = PersonaRegistry()
        with self.assertRaises(KeyError) as ctx:
            reg.get("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_get_or_none_missing(self):
        reg = PersonaRegistry()
        self.assertIsNone(reg.get_or_none("x"))

    def test_get_or_none_found(self):
        reg = PersonaRegistry()
        reg.add("x", "system")
        self.assertIsNotNone(reg.get_or_none("x"))

    def test_overwrite(self):
        reg = PersonaRegistry()
        reg.add("x", "original")
        reg.add("x", "updated")
        self.assertEqual(reg.get("x").system_prompt, "updated")
        # Overwriting does not grow the registry.
        self.assertEqual(len(reg), 1)

    def test_remove(self):
        reg = PersonaRegistry()
        reg.add("a", "A")
        reg.remove("a")
        self.assertFalse(reg.has("a"))

    def test_remove_missing_is_noop(self):
        reg = PersonaRegistry()
        # Should not raise.
        reg.remove("nonexistent")

    def test_has(self):
        reg = PersonaRegistry()
        reg.add("a", "A")
        self.assertTrue(reg.has("a"))
        self.assertFalse(reg.has("b"))

    def test_contains_operator(self):
        reg = PersonaRegistry()
        reg.add("a", "A")
        self.assertIn("a", reg)
        self.assertNotIn("b", reg)

    def test_names_insertion_order(self):
        reg = PersonaRegistry()
        reg.add("b", "B")
        reg.add("a", "A")
        reg.add("c", "C")
        # names() documents insertion order.
        self.assertEqual(reg.names(), ["b", "a", "c"])

    def test_all_personas(self):
        reg = PersonaRegistry()
        reg.add("a", "A", description="desc A")
        reg.add("b", "B")
        personas = reg.all_personas()
        self.assertEqual(len(personas), 2)
        self.assertTrue(all(isinstance(p, Persona) for p in personas))

    def test_len(self):
        reg = PersonaRegistry()
        self.assertEqual(len(reg), 0)
        reg.add("a", "A")
        self.assertEqual(len(reg), 1)
        reg.add("b", "B")
        self.assertEqual(len(reg), 2)

    def test_as_message(self):
        reg = PersonaRegistry()
        reg.add("assistant", "You are helpful.")
        self.assertEqual(
            reg.as_message("assistant"),
            {"role": "system", "content": "You are helpful."},
        )

    def test_as_message_with_variables(self):
        reg = PersonaRegistry()
        reg.add("teacher", "You are a {subject} teacher.")
        msg = reg.as_message("teacher", variables={"subject": "Python"})
        self.assertEqual(msg["content"], "You are a Python teacher.")

    def test_as_message_missing_raises(self):
        reg = PersonaRegistry()
        with self.assertRaises(KeyError):
            reg.as_message("nope")

    def test_clone_independence(self):
        reg = PersonaRegistry()
        reg.add("x", "original", metadata={"k": "v"})
        clone = reg.clone()
        clone.add("x", "modified")
        self.assertEqual(reg.get("x").system_prompt, "original")
        self.assertEqual(clone.get("x").system_prompt, "modified")

    def test_clone_metadata_is_deep(self):
        reg = PersonaRegistry()
        reg.add("x", "original", metadata={"k": "v"})
        clone = reg.clone()
        # Mutating the clone's metadata must not leak back to the source.
        clone.get("x").metadata["k"] = "changed"
        self.assertEqual(reg.get("x").metadata["k"], "v")

    def test_chaining(self):
        reg = PersonaRegistry()
        result = reg.add("a", "A").add("b", "B").remove("a")
        self.assertIs(result, reg)
        self.assertIn("b", reg)
        self.assertNotIn("a", reg)

    def test_metadata(self):
        reg = PersonaRegistry()
        reg.add("x", "X", metadata={"version": "1.0"})
        self.assertEqual(reg.get("x").metadata["version"], "1.0")

    def test_description(self):
        reg = PersonaRegistry()
        reg.add("x", "X", description="A test persona")
        self.assertEqual(reg.get("x").description, "A test persona")


if __name__ == "__main__":
    unittest.main()
