import re

# end of sentence: . ! ? followed by space/end, not inside "e.g." or a version like 3.2
_END = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\d)[.!?](?=\s|$)")


def pop_sentence(buf: str, min_chars: int = 20) -> tuple[str | None, str]:
    """(sentence, rest) if buf contains a complete sentence of reasonable length."""
    for m in _END.finditer(buf):
        if m.end() >= min_chars:
            return buf[: m.end()].strip(), buf[m.end():]
    return None, buf
