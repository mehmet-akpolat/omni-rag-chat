import re
from collections.abc import Iterable

from .domain import Chunk, ChunkingConfig, ChunkingStrategy


def _group_units(units: list[str], size: int, overlap: int, separator: str = " ") -> list[str]:
    """Group complete text units without cutting a unit to satisfy size or overlap targets."""
    if not units:
        return []

    chunks: list[str] = []
    start = 0
    previous_end = 0
    while start < len(units):
        end = start
        current_length = 0
        while end < len(units):
            added_length = len(units[end]) + (len(separator) if end > start else 0)
            if end > start and current_length + added_length > size:
                break
            current_length += added_length
            end += 1

        # An overlap can leave the next window containing only already emitted units.
        # Include one new complete unit even when that makes chunk_size a soft limit.
        if end <= previous_end and previous_end < len(units):
            end = previous_end + 1

        chunks.append(separator.join(units[start:end]).strip())
        if end == len(units):
            break

        previous_end = end
        if not overlap:
            start = end
            continue

        overlap_start = end
        overlap_length = 0
        while overlap_start > start and overlap_length < overlap:
            overlap_start -= 1
            overlap_length = len(separator.join(units[overlap_start:end]))

        # Never repeat an entire chunk: advance one complete unit when the overlap
        # target reaches the beginning of the current window.
        start = max(overlap_start, start + 1)

    return chunks


def _sentence_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        units.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip())
            if sentence.strip()
        )
    return units


def _windows(text: str, size: int, overlap: int) -> list[str]:
    return _group_units(_sentence_units(text), size, overlap)


def _recursive(text: str, size: int, overlap: int) -> list[str]:
    return _group_units(_sentence_units(text), size, overlap)


def _semantic(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 2:
        return _group_units(_sentence_units(text), size, overlap)
    return _group_units(paragraphs, size, overlap, separator="\n")


def _section(text: str) -> str | None:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if (
        first
        and len(first) <= 100
        and (first.isupper() or first.endswith(":") or first.startswith("#"))
    ):
        return first.lstrip("# ").rstrip(":")
    return None


def _heading(line: str) -> tuple[int, str] | None:
    clean = line.strip()
    if not clean or len(clean) > 100:
        return None
    markdown = re.match(r"^(#{1,6})\s+(.+)$", clean)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip().rstrip(":")
    numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+(.+)$", clean)
    if numbered and any(character.isalpha() for character in numbered.group(2)):
        return numbered.group(1).count(".") + 1, numbered.group(2).strip().rstrip(":")
    if clean.isupper() or (clean.endswith(":") and not re.search(r"[.!?]", clean[:-1])):
        return 1, clean.rstrip(":")
    return None


def _section_blocks(text: str) -> list[tuple[list[str], str]]:
    blocks: list[tuple[list[str], str]] = []
    path: list[str] = []
    block_path: list[str] = []
    lines: list[str] = []

    def flush() -> None:
        content = "\n".join(lines).strip()
        if content:
            blocks.append((block_path.copy(), content))

    for line in text.splitlines():
        heading = _heading(line)
        if heading:
            flush()
            level, title = heading
            path = [*path[: level - 1], title]
            block_path = path.copy()
            lines = [line]
        else:
            lines.append(line)
    flush()
    return blocks or [([], text)]


def chunk_pages(
    pages: Iterable[tuple[int | None, str]],
    knowledge_base_id: str,
    company_id: str,
    config: ChunkingConfig,
) -> list[Chunk]:
    output: list[Chunk] = []
    for page_number, text in pages:
        if config.strategy == ChunkingStrategy.HIERARCHICAL:
            for breadcrumb, section_text in _section_blocks(text):
                section = breadcrumb[-1] if breadcrumb else None
                parent_texts = _recursive(section_text, config.parent_size, config.overlap)
                for parent_text in parent_texts:
                    parent_metadata = {
                        "section": section,
                        "breadcrumb": breadcrumb,
                        "level": "parent",
                    }
                    if page_number is not None:
                        parent_metadata["page_number"] = page_number
                    parent = Chunk(
                        knowledge_base_id=knowledge_base_id,
                        company_id=company_id,
                        text=parent_text,
                        metadata=parent_metadata,
                    )
                    output.append(parent)
                    children = _recursive(parent_text, config.chunk_size, config.overlap)
                    for child_text in children:
                        child_metadata = {
                            "section": section,
                            "breadcrumb": breadcrumb,
                            "parent_id": parent.id,
                            "level": "child",
                        }
                        if page_number is not None:
                            child_metadata["page_number"] = page_number
                        output.append(
                            Chunk(
                                knowledge_base_id=knowledge_base_id,
                                company_id=company_id,
                                text=child_text,
                                metadata=child_metadata,
                            )
                        )
            continue

        blocks = _section_blocks(text) if page_number is None else [([], text)]
        for breadcrumb, block_text in blocks:
            section = breadcrumb[-1] if breadcrumb else _section(block_text)
            if config.strategy == ChunkingStrategy.FIXED:
                pieces = _windows(block_text, config.chunk_size, config.overlap)
            elif config.strategy == ChunkingStrategy.SEMANTIC:
                pieces = _semantic(block_text, config.chunk_size, config.overlap)
            else:
                pieces = _recursive(block_text, config.chunk_size, config.overlap)
            for piece in pieces:
                metadata = {"section": section}
                if page_number is not None:
                    metadata["page_number"] = page_number
                output.append(
                    Chunk(
                        knowledge_base_id=knowledge_base_id,
                        company_id=company_id,
                        text=piece,
                        metadata=metadata,
                    )
                )
    return output
