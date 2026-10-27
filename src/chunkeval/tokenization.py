"""Full-input token accounting and lossless source-coordinate splitting."""

from __future__ import annotations

from typing import Any

from .config import PipelineError, data_path


class SourceTokenizer:
    """A fast tokenizer whose offsets always refer to the unprefixed source."""

    def __init__(self, hf_tokenizer: Any):
        if not getattr(hf_tokenizer, "is_fast", False):
            raise PipelineError("E5 requires a fast tokenizer with offset mapping.")
        self.hf_tokenizer = hf_tokenizer

    def count(self, text: str, prefix: str) -> int:
        if not isinstance(text, str) or not isinstance(prefix, str):
            raise PipelineError("Token input and prefix must be strings.")
        encoded = self.hf_tokenizer(
            prefix + text, add_special_tokens=True, truncation=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self.hf_tokenizer(
            text, add_special_tokens=False, truncation=False,
            return_offsets_mapping=True, return_attention_mask=False,
        )
        result = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        if len(result) != len(encoded["input_ids"]):
            raise PipelineError("Tokenizer offsets and source tokens are not aligned.")
        previous_start = previous_end = -1
        for start, end in result:
            if not 0 <= start < end <= len(text):
                raise PipelineError("Tokenizer returned an invalid Unicode source offset.")
            if start < previous_start or end < previous_end:
                raise PipelineError("Tokenizer source offsets must be ordered.")
            previous_start, previous_end = start, end
        if not result and text.strip():
            raise PipelineError("Tokenizer produced no source tokens for substantive text.")
        return result

    def split_span(
        self, text: str, start: int, end: int, limit: int, prefix: str,
    ) -> list[tuple[int, int]]:
        """Split one source span into contiguous slices, without decoded text.

        Candidate cuts use source-token starts. Whitespace before the next token
        belongs to the preceding fragment. Every candidate is retokenized with
        the complete prefix and special tokens before it can be emitted.
        """
        if not 0 <= start <= end <= len(text):
            raise PipelineError("Invalid source span for token splitting.")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise PipelineError("Token input limit must be a positive integer.")
        if start == end:
            return []
        if self.count(text[start:end], prefix) <= limit:
            return [(start, end)]
        # Tokenizing this source slice preserves source coordinates after adding
        # its origin. It also permits a partial sentence/paragraph parent span.
        offsets = self.offsets(text[start:end])
        cuts = sorted({start + token_start for token_start, _ in offsets
                       if start < start + token_start < end}) + [end]
        output: list[tuple[int, int]] = []
        cursor = start
        first_candidate = 0
        while cursor < end:
            # A full input cannot usually hold more than `limit` source tokens.
            # Start near that budget, then correct using actual full input counts.
            candidate = min(first_candidate + limit - 1, len(cuts) - 1)
            while candidate >= first_candidate and self.count(text[cursor:cuts[candidate]], prefix) > limit:
                candidate -= 1
            if candidate < first_candidate:
                raise PipelineError(
                    f"A source token at character {cursor} cannot fit the {limit}-token input budget."
                )
            # Recover capacity when prefix/source boundary tokenization reduced
            # the count; each expanded endpoint receives the same exact check.
            while candidate + 1 < len(cuts) and self.count(text[cursor:cuts[candidate + 1]], prefix) <= limit:
                candidate += 1
            next_cursor = cuts[candidate]
            if next_cursor <= cursor:
                raise PipelineError("Token splitter could not make forward progress.")
            output.append((cursor, next_cursor))
            cursor = next_cursor
            first_candidate = candidate + 1
        return output


def load_tokenizer(config: dict) -> SourceTokenizer:
    """Load exactly the model revision used by the embedding encoder."""
    embedding = config["embedding"]
    revision = embedding.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in revision
    ):
        raise PipelineError("Pin embedding.revision to a concrete 40-character model commit before loading E5.")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            embedding["model"], revision=revision, use_fast=True,
            cache_dir=str(data_path(config, "cache", "model")),
        )
    except Exception as exc:
        raise PipelineError(f"Could not load the pinned E5 tokenizer: {exc}") from exc
    return SourceTokenizer(tokenizer)
