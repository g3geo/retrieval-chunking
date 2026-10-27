"""Synthetic tokenizer fixtures: these tests do not download an E5 model."""

import pytest

from chunkeval.config import PipelineError
from chunkeval.tokenization import SourceTokenizer, load_tokenizer


class SyntheticCharacterFastTokenizer:
    """One synthetic source token per non-whitespace Unicode character."""

    is_fast = True

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        offsets = [(index, index + 1) for index, character in enumerate(text) if not character.isspace()]
        ids = list(range(len(offsets)))
        if kwargs["add_special_tokens"]:
            ids = [-1, *ids, -2]
        result = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


def test_count_tokenizes_full_prefixed_input_with_special_tokens_without_truncation():
    fixture = SyntheticCharacterFastTokenizer()
    tokenizer = SourceTokenizer(fixture)
    assert tokenizer.count("é 😀", "passage: ") == 12
    text, options = fixture.calls[-1]
    assert text == "passage: é 😀"
    assert options["add_special_tokens"] is True
    assert options["truncation"] is False


def test_prefix_and_source_cannot_be_counted_separately():
    class SyntheticContextTokenizer(SyntheticCharacterFastTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            # Synthetic boundary fusion demonstrates why prefix counts cannot
            # simply be added to a separately tokenized source count.
            if text.startswith("passage: é") and kwargs["add_special_tokens"]:
                result["input_ids"].pop()
            return result

    tokenizer = SourceTokenizer(SyntheticContextTokenizer())
    actual = tokenizer.count("é", "passage: ")
    separate = tokenizer.count("", "passage: ") + len(tokenizer.offsets("é"))
    assert actual == separate - 1


def test_unicode_source_split_preserves_all_characters_and_token_limits():
    text = "\n" + "é, 😀法律 " * 120 + "\n"
    tokenizer = SourceTokenizer(SyntheticCharacterFastTokenizer())
    spans = tokenizer.split_span(text, 0, len(text), 32, "passage: ")
    assert len(spans) > 1
    assert "".join(text[start:end] for start, end in spans) == text
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert all(tokenizer.count(text[start:end], "passage: ") <= 32 for start, end in spans)
    token_starts = {start for start, _ in tokenizer.offsets(text)}
    assert all(end in token_starts for _, end in spans[:-1])


def test_split_only_selected_parent_span_and_fail_unrepresentable_budget():
    tokenizer = SourceTokenizer(SyntheticCharacterFastTokenizer())
    text = "LEFT " + "isi 😀 " * 50 + " RIGHT"
    start, end = 5, len(text) - 6
    spans = tokenizer.split_span(text, start, end, 20, "passage: ")
    assert "".join(text[left:right] for left, right in spans) == text[start:end]
    assert all(start <= left < right <= end for left, right in spans)
    with pytest.raises(PipelineError, match="cannot fit"):
        tokenizer.split_span("x", 0, 1, 10, "passage: ")


def test_slow_tokenizer_and_bad_offsets_fail_clearly():
    with pytest.raises(PipelineError, match="fast tokenizer"):
        SourceTokenizer(object())

    class InvalidSyntheticTokenizer(SyntheticCharacterFastTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if kwargs.get("return_offsets_mapping"):
                result["offset_mapping"] = [(0, len(text) + 1)]
            return result

    with pytest.raises(PipelineError, match="Unicode source offset"):
        SourceTokenizer(InvalidSyntheticTokenizer()).offsets("é")


@pytest.mark.parametrize("revision", [None, "main", "a" * 39, "z" * 40])
def test_unpinned_tokenizer_revision_rejected_before_any_model_download(revision):
    with pytest.raises(PipelineError, match="concrete 40-character"):
        load_tokenizer({"embedding": {"model": "intfloat/multilingual-e5-small", "revision": revision}})
