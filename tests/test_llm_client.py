"""Tests for sift_kg.extract.llm_client — specifically parse_llm_json."""

import pytest

from sift_kg.extract.llm_client import parse_llm_json


class TestParseLlmJson:
    """Test JSON parsing from LLM responses."""

    def test_clean_json(self):
        """Parse clean JSON object."""
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_markdown_fences(self):
        """Parse JSON wrapped in markdown code fences."""
        text = '```json\n{"key": "value"}\n```'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_json_with_plain_fences(self):
        """Parse JSON wrapped in plain code fences."""
        text = '```\n{"key": "value"}\n```'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_json_with_trailing_text(self):
        """Parse JSON followed by explanation text."""
        text = '{"key": "value"}\n\nThis is the extracted data.'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_json_with_leading_text(self):
        """Parse JSON preceded by explanation text."""
        text = 'Here is the result:\n{"key": "value"}'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_nested_json(self):
        """Parse nested JSON objects."""
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_llm_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_json_with_braces_in_trailing_text(self):
        """Handles trailing text with stray braces (greedy regex fix)."""
        text = '{"key": "value"} Note: use {brackets} for grouping.'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_json_with_entities_and_relations(self):
        """Parse a realistic extraction response."""
        text = '''```json
{
    "entities": [
        {"name": "Alice", "entity_type": "PERSON", "confidence": 0.9}
    ],
    "relations": []
}
```'''
        result = parse_llm_json(text)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Alice"

    def test_empty_object(self):
        """Parse empty JSON object."""
        result = parse_llm_json("{}")
        assert result == {}

    def test_invalid_json_raises(self):
        """Non-JSON text raises ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_json("This is just plain text with no JSON.")

    def test_empty_string_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            parse_llm_json("")

    def test_json_with_unicode(self):
        """Parse JSON with unicode characters."""
        text = '{"name": "José García", "city": "São Paulo"}'
        result = parse_llm_json(text)
        assert result["name"] == "José García"
        assert result["city"] == "São Paulo"

    def test_json_with_escaped_quotes(self):
        """Parse JSON with escaped quotes in values."""
        text = '{"quote": "She said \\"hello\\"."}'
        result = parse_llm_json(text)
        assert 'hello' in result["quote"]
