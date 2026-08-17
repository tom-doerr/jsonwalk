"""Prompt construction and JSON string-boundary rules.

The object we ask the model to produce is always the same shape::

    {"<field>": "<value>", "<bool_field>": true}

The walker generates ``<value>`` and stops at the closing quote; the scorer
then measures how much the model prefers ``true`` over ``false`` in the slot
after ``<bool_field>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

FIELD_PLACEHOLDER = "{field}"
BOOL_PLACEHOLDER = "{bool_field}"

# Two worked examples, deliberately from unrelated domains.
#
# Measured on Qwen3.5-0.8B-Base. With no preamble only ~4-20% of the mass at
# the boolean slot lands on true/false -- the model wants to write a number
# there. Examples fix that (~99%), but examples that reuse the *queried* field
# name hijack it: asking for city_name with startup examples returns Stripe
# and Google. Foreign-domain examples teach the shape without the subject, and
# they judge better too -- is_in_europe then scores Paris +1.75 / Berlin +1.75
# against Los Angeles -1.13 / New York -1.00. A third example made both the
# separation and the bool mass worse.
DEFAULT_PREAMBLE = (
    "Expert judgements, one JSON object per line.\n"
    '{"movie_title": "Casablanca", "is_a_masterpiece": true}\n'
    '{"chemical_element": "Helium", "is_a_metal": false}\n'
)

# Lower bool-mass but no worked values to echo back. Worth switching to when
# the enumerated values matter more than the sharpness of the verdict.
SCHEMA_ONLY_PREAMBLE = (
    "// Dataset of expert judgements, one JSON object per line.\n"
    '// Schema: {"{field}": <string>, "{bool_field}": <true|false>}\n'
)

# A real JSON Schema. Measured on Qwen3.5-0.8B-Base it is WORSE than examples,
# and it fails in an interesting way: the schema text names the fields, so the
# model starts emitting the field names as values -- asking for startup_name
# returned "My Startup", "Good Sounding", "good_sounding_name". A base model
# continues the pattern it was shown, and a block of type declarations is a
# pattern of declarations, not of records. Kept because it is the obvious
# thing to try and being able to reproduce the result is worth more than an
# assertion that it does not work.
JSON_SCHEMA_PREAMBLE = (
    "// Each line is a JSON object matching this schema:\n"
    '// {"type": "object", "properties": {"{field}": {"type": "string"}, '
    '"{bool_field}": {"type": "boolean"}}, '
    '"required": ["{field}", "{bool_field}"]}\n'
)

PREAMBLE_STYLES = {
    "examples": DEFAULT_PREAMBLE,
    "comment": SCHEMA_ONLY_PREAMBLE,
    "json-schema": JSON_SCHEMA_PREAMBLE,
}


def first_unescaped_quote(text: str) -> int | None:
    """Index of the first ``"`` not preceded by an odd run of backslashes.

    This is the "second quote" of the spec: the opening quote lives in the
    prompt, so the first unescaped quote the model emits closes the value.
    """
    backslashes = 0
    for i, ch in enumerate(text):
        if ch == "\\":
            backslashes += 1
            continue
        if ch == '"' and backslashes % 2 == 0:
            return i
        backslashes = 0
    return None


class InvalidJSONString(ValueError):
    """Raised when generated text is not a decodable JSON string body."""


def unescape(raw: str) -> str:
    """Turn the model's raw in-string bytes into the real Python string.

    ``raw`` is what sits between the two quotes, still JSON-escaped. A base
    model can emit nonsense like a trailing lone backslash; that is a real
    dead branch and is reported as such instead of being papered over.
    """
    try:
        decoded = json.loads('"' + raw + '"')
    except json.JSONDecodeError as exc:
        raise InvalidJSONString(f"not a valid JSON string body: {raw!r}") from exc
    if not isinstance(decoded, str):  # pragma: no cover - json cannot do this
        raise InvalidJSONString(f"decoded to {type(decoded).__name__}")
    return decoded


@dataclass(frozen=True)
class Schema:
    """The two-field JSON object being explored."""

    field: str = "startup_name"
    bool_field: str = "good_sounding_name"
    preamble: str = DEFAULT_PREAMBLE

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("field name must not be empty")
        if not self.bool_field:
            raise ValueError("bool_field name must not be empty")

    def rendered_preamble(self) -> str:
        """Preamble with the field-name placeholders substituted.

        Plain ``str.format`` is unusable here: a preamble full of JSON braces
        would blow up on every literal ``{``. Explicit replacement it is.
        """
        return self.preamble.replace(FIELD_PLACEHOLDER, self.field).replace(
            BOOL_PLACEHOLDER, self.bool_field
        )

    def value_prefix(self) -> str:
        """Everything up to and including the value's opening quote."""
        return f"{self.rendered_preamble()}{{{json.dumps(self.field)}: \""

    def bool_prefix(self, raw_value: str) -> str:
        """Prompt ending at the bool field's colon.

        ``raw_value`` is the still-escaped text the model generated, so it is
        spliced back verbatim -- re-escaping it would change the tokens the
        model conditioned on.

        The prompt deliberately stops at ``:`` with no trailing space; the
        space is part of the scored word. See :mod:`jsonwalk.score`.
        """
        return f'{self.value_prefix()}{raw_value}", {json.dumps(self.bool_field)}:'

    def as_object(self, value: str, truth: bool) -> dict[str, object]:
        return {self.field: value, self.bool_field: truth}
