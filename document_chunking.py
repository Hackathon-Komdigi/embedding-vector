import html
from typing import Any

from pydantic import ConfigDict

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.serializer.base import BaseSerializerProvider
from docling_core.transforms.serializer.markdown import (
    ImageRefMode,
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc import DoclingDocument


class MarkdownChunkingSerializerProvider(BaseSerializerProvider):
    """Use Markdown pipe tables instead of the chunker's triplet table format."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(
                image_mode=ImageRefMode.PLACEHOLDER,
                image_placeholder="",
                escape_underscores=False,
                escape_html=True,
                include_hyperlinks=False,
            ),
        )


class LimitedTokenizer(BaseTokenizer):
    """Apply a request-local token limit without mutating a shared tokenizer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: BaseTokenizer
    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.tokenizer.get_tokenizer()


def _label_value(item: Any) -> str:
    label = getattr(item, "label", "")
    return getattr(label, "value", str(label))


def _chunk_contains_table(chunk: Any) -> bool:
    return any(_label_value(item) == "table" for item in chunk.meta.doc_items)


def _headings(chunk: Any) -> tuple[str, ...]:
    return tuple(chunk.meta.headings or ())


def _render_headings(headings: tuple[str, ...]) -> str:
    return "\n".join(
        f"{'#' * min(level, 6)} {html.escape(heading)}"
        for level, heading in enumerate(headings, 1)
    )


def _tail_tokens(text: str, tokenizer: BaseTokenizer, token_count: int) -> str:
    if token_count <= 0:
        return ""
    raw_tokenizer = tokenizer.get_tokenizer()
    tokens = raw_tokenizer.encode(text, add_special_tokens=False)
    return raw_tokenizer.decode(tokens[-token_count:], skip_special_tokens=True).strip()


def _compose_text(headings: tuple[str, ...], overlap: str, body: str) -> str:
    parts = [_render_headings(headings)] if headings else []
    if overlap:
        parts.append(overlap)
    if body:
        parts.append(body)
    return "\n\n".join(part for part in parts if part)


def _fit_overlap(
    headings: tuple[str, ...],
    body: str,
    previous_body: str,
    tokenizer: BaseTokenizer,
    overlap_tokens: int,
    max_tokens: int,
) -> str:
    for count in range(overlap_tokens, -1, -1):
        overlap = _tail_tokens(previous_body, tokenizer, count)
        candidate = _compose_text(headings, overlap, body)
        if tokenizer.count_tokens(candidate) <= max_tokens:
            return candidate
    return _compose_text(headings, "", body)


def _page_number(chunk: Any) -> int:
    for item in chunk.meta.doc_items:
        provenance = getattr(item, "prov", None) or []
        if provenance:
            return provenance[0].page_no
    return 0


def _sequence(chunk: Any, item_order: dict[str, int], fallback: int) -> int:
    positions = [
        item_order[item.self_ref]
        for item in chunk.meta.doc_items
        if getattr(item, "self_ref", None) in item_order
    ]
    return min(positions, default=fallback)


def build_text_chunks(
    document: DoclingDocument,
    tokenizer: BaseTokenizer,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Create hierarchy-aware, semantic, overlapping Markdown text chunks."""
    if max_tokens <= 0:
        raise ValueError("max_tokens harus lebih besar dari nol")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens harus nol atau lebih dan lebih kecil dari max_tokens")

    heading_reserve = min(16, max(1, (max_tokens - overlap_tokens) // 4))
    body_limit = max_tokens - overlap_tokens - heading_reserve
    if body_limit <= 0:
        raise ValueError("max_tokens terlalu kecil untuk heading dan overlap")
    limited_tokenizer = LimitedTokenizer(tokenizer=tokenizer, max_tokens=body_limit)
    chunker = HybridChunker(
        tokenizer=limited_tokenizer,
        serializer_provider=MarkdownChunkingSerializerProvider(),
        repeat_table_header=True,
        merge_peers=False,
        always_emit_headings=True,
    )
    item_order = {
        item.self_ref: index
        for index, (item, _) in enumerate(document.iterate_items())
        if getattr(item, "self_ref", None)
    }
    picture_positions = {
        index
        for index, (item, _) in enumerate(document.iterate_items())
        if _label_value(item) == "picture"
    }

    result: list[dict[str, Any]] = []
    previous_chunk = None
    previous_sequence = -1
    for index, chunk in enumerate(chunker.chunk(document)):
        body = chunk.text.strip()
        headings = _headings(chunk)
        if not body and not headings:
            continue
        sequence = _sequence(chunk, item_order, index)
        crosses_picture = any(
            previous_sequence < position < sequence for position in picture_positions
        )

        can_overlap = (
            bool(body)
            and previous_chunk is not None
            and not crosses_picture
            and headings == _headings(previous_chunk)
            and not _chunk_contains_table(previous_chunk)
            and not _chunk_contains_table(chunk)
        )
        if can_overlap:
            assert previous_chunk is not None
            text = _fit_overlap(
                headings,
                body,
                previous_chunk.text,
                tokenizer,
                overlap_tokens,
                max_tokens,
            )
        else:
            text = _compose_text(headings, "", body)
        result.append(
            {
                "type": "text",
                "page": _page_number(chunk),
                "text": text,
                "_sequence": sequence,
            }
        )
        previous_chunk = chunk
        previous_sequence = sequence

    return result
