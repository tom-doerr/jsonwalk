import pytest

from jsonwalk.prompt import (
    InvalidJSONString,
    Schema,
    first_unescaped_quote,
    unescape,
)


def test_finds_plain_quote():
    assert first_unescaped_quote('Acme"') == 4


def test_skips_escaped_quote():
    assert first_unescaped_quote('say \\"hi\\" now"') == 14


def test_escaped_backslash_does_not_escape_the_quote():
    # \\ is a literal backslash, so the quote right after it still closes.
    assert first_unescaped_quote('path\\\\"') == 6


def test_no_quote_returns_none():
    assert first_unescaped_quote("Acme Corp") is None


def test_unescape_decodes_json_escapes():
    assert unescape('a\\"b\\u0041') == 'a"bA'


def test_unescape_rejects_dangling_backslash():
    with pytest.raises(InvalidJSONString):
        unescape("trailing\\")


def test_value_prefix_ends_at_the_opening_quote():
    schema = Schema(field="startup_name", bool_field="good", preamble="")
    assert schema.value_prefix() == '{"startup_name": "'


def test_bool_prefix_stops_at_the_colon():
    # No trailing space: the space belongs to the scored word so the tokenizer
    # cannot merge it into " true".
    schema = Schema(field="name", bool_field="is_good", preamble="")
    assert schema.bool_prefix("Acme") == '{"name": "Acme", "is_good":'


def test_bool_prefix_keeps_the_raw_escaped_value():
    schema = Schema(field="n", bool_field="b", preamble="")
    assert schema.bool_prefix('say \\"hi\\"') == '{"n": "say \\"hi\\"", "b":'


def test_preamble_placeholders_survive_literal_json_braces():
    # str.format would choke on the braces in the schema line; this is the
    # regression guard for using plain replacement instead.
    schema = Schema(
        field="pet", bool_field="cute", preamble='{"{field}": s, "{bool_field}": b}\n'
    )
    assert schema.rendered_preamble() == '{"pet": s, "cute": b}\n'


def test_empty_field_names_are_rejected():
    with pytest.raises(ValueError):
        Schema(field="", bool_field="b")
    with pytest.raises(ValueError):
        Schema(field="a", bool_field="")
