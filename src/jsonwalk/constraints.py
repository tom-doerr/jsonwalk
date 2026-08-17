"""Constraints that prune the walk to values which are valid by construction.

The useful property here is that **substring-ness is prefix-closed**: if the
finished value must appear verbatim in some text, then so must every prefix of
it. That makes "is this still a substring" a sound *incremental* test, so the
tree walk can drop a branch the moment it leaves the text instead of
generating a whole string and checking afterwards.

The practical consequence is that a search anchor cannot be hallucinated. The
usual failure of model-generated search/replace -- an anchor that does not
occur in the document -- is not filtered out here, it is unreachable.
"""

from __future__ import annotations

from typing import Protocol


class ValueConstraint(Protocol):
    """Decides which partial and complete values the walk may keep."""

    def allows(self, partial: str) -> bool:
        """Could this partial value still grow into an acceptable one?"""
        ...

    def accepts(self, value: str) -> bool:
        """Is this finished value acceptable?"""
        ...


class SubstringOf:
    """Value must occur verbatim in ``text``.

    ``min_length`` matters more than it looks. Shorter strings are always more
    probable, so without a floor the ranking fills with two-character anchors.
    ``require_unique`` does the same job from the other side: a short anchor
    usually occurs many times, so demanding exactly one occurrence pushes the
    search towards anchors that are long enough to be distinctive *and* safe
    to replace.
    """

    def __init__(
        self, text: str, *, min_length: int = 6, require_unique: bool = True
    ) -> None:
        if not text:
            raise ValueError("cannot constrain against empty text")
        self.text = text
        self.min_length = min_length
        self.require_unique = require_unique

    def allows(self, partial: str) -> bool:
        # Empty is a substring of everything, so the walk starts unpruned.
        return partial in self.text

    def accepts(self, value: str) -> bool:
        if len(value) < self.min_length:
            return False
        count = self.occurrences(value)
        return count == 1 if self.require_unique else count >= 1

    def occurrences(self, value: str) -> int:
        if not value:
            return 0
        n = start = 0
        while (i := self.text.find(value, start)) != -1:
            n += 1
            start = i + 1  # overlapping counts, e.g. "aa" in "aaa"
        return n
