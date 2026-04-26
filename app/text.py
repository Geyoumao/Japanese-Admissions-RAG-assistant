from __future__ import annotations

import re
from dataclasses import dataclass, field


WHITESPACE_RE = re.compile(r"[ \t\u3000]+")
NAME_SPACE_RE = re.compile(r"[\s\u3000]+")
YEAR_RE = re.compile(r"(20\d{2}|\u4ee4\u548c\d+|\u5e73\u6210\d+)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f\.\?!])")
UNIVERSITY_NAME_RE = re.compile(
    r"(.+?(?:\u77ed\u671f\u5927\u5b66|\u5927\u5b66\u6821|\u5c02\u9580\u5b66\u6821|\u5927\u5b66))"
)
NOISE_SUFFIX_RE = re.compile(
    r"("
    r"\u52df\u96c6\u8981\u9805|"
    r"\u5b66\u751f\u52df\u96c6\u8981\u9805|"
    r"\u5165\u8a66\u8981\u9805|"
    r"\u51fa\u9858\u8981\u9805|"
    r"\u4e00\u822c\u9078\u629c|"
    r"\u7279\u5225\u9078\u629c|"
    r"\u79c1\u8cbb\u5916\u56fd\u4eba\u7559\u5b66\u751f|"
    r"\u5927\u5b66\u9662|"
    r"\u7814\u7a76\u79d1|"
    r"\u5c02\u653b|"
    r"\u53d7\u9a13\u6848\u5185"
    r")+$"
)
MATCH_NORMALIZE_RE = re.compile(r"[\s_\-()\[\]{}<>/\\|,:;\u3000\u30fb\u3010\u3011\u300c\u300d]+")
GENERIC_UNIVERSITY_NAMES = {
    "\u56fd\u7acb\u5927\u5b66",
    "\u516c\u7acb\u5927\u5b66",
    "\u79c1\u7acb\u5927\u5b66",
}


def normalize_text(text: str) -> str:
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.replace("\r", "\n").split("\n")]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()


def normalize_for_match(text: str) -> str:
    normalized = normalize_text(text).lower()
    return MATCH_NORMALIZE_RE.sub("", normalized)


def _compact_line(text: str) -> str:
    return NAME_SPACE_RE.sub("", normalize_text(text))


def _extract_university_from_line(line: str) -> str:
    compact = _compact_line(line)
    if not compact:
        return ""

    exact_match = re.search(
        r"(.+?(?:\u77ed\u671f\u5927\u5b66|\u5927\u5b66\u6821|\u5c02\u9580\u5b66\u6821|\u5927\u5b66))(?!\u9662)",
        compact,
    )
    if exact_match:
        return exact_match.group(1).strip()

    graduate_prefix = re.search(r"(.+?\u5927\u5b66)(?=\u5927\u5b66\u9662)", compact)
    if graduate_prefix:
        return graduate_prefix.group(1).strip()
    return ""


def extract_university_name(pdf_name: str, first_page_text: str, full_text: str) -> str:
    page_candidates: list[str] = []
    for raw_line in normalize_text(first_page_text).splitlines():
        candidate = _extract_university_from_line(raw_line)
        if candidate:
            page_candidates.append(candidate)

    if page_candidates:
        page_candidates = [candidate for candidate in page_candidates if candidate not in GENERIC_UNIVERSITY_NAMES]
        page_candidates.sort(key=len, reverse=True)
        if page_candidates:
            return page_candidates[0]

    basename = pdf_name.rsplit(".", 1)[0]
    cleaned = re.sub(r"[_\-()\[\]{}]", " ", basename)
    cleaned = normalize_text(cleaned)
    year_match = YEAR_RE.search(cleaned)
    if year_match:
        cleaned = cleaned.replace(year_match.group(1), "").strip()
    cleaned = NOISE_SUFFIX_RE.sub("", cleaned).strip()

    university_match = UNIVERSITY_NAME_RE.search(cleaned)
    if university_match:
        return university_match.group(1).strip()

    head_match = UNIVERSITY_NAME_RE.search(normalize_text(full_text[:1000]))
    if head_match:
        return head_match.group(1).strip()
    return cleaned.strip()


def infer_university_and_year(pdf_name: str, full_text: str, first_page_text: str = "") -> tuple[str, str]:
    basename = pdf_name.rsplit(".", 1)[0]
    year_match = YEAR_RE.search(basename) or YEAR_RE.search(full_text[:400])
    year = year_match.group(1) if year_match else ""
    university = extract_university_name(pdf_name, first_page_text or full_text[:1000], full_text)
    return university, year


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 40:
        return False
    if stripped.endswith((":", "\uff1a")):
        return True
    return bool(
        re.match(
            r"^[0-9\uff10-\uff19\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341A-Za-z\uff21-\uff3a\uff41-\uff5a].{0,20}$",
            stripped,
        )
    )


def guess_section_title(block_text: str) -> str:
    for line in block_text.splitlines()[:3]:
        if looks_like_heading(line):
            return line.strip()
    return ""


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if paragraphs:
        return paragraphs
    pieces = [p.strip() for p in text.split("\n") if p.strip()]
    return pieces if pieces else [text]


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    if not sentences:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current}{sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_japanese_text(text: str, chunk_size: int = 420, overlap: int = 90) -> list[tuple[str, str]]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    segments: list[str] = []
    for paragraph in _split_paragraphs(normalized):
        segments.extend(_split_long_paragraph(paragraph, chunk_size))

    chunks: list[tuple[str, str]] = []
    current = ""
    for segment in segments:
        candidate = f"{current}\n{segment}".strip() if current else segment
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append((guess_section_title(current), current))
        if overlap and chunks:
            previous_text = chunks[-1][1]
            tail = previous_text[-overlap:]
            current = f"{tail}\n{segment}".strip()
        else:
            current = segment
        while len(current) > chunk_size:
            partial = current[:chunk_size]
            chunks.append((guess_section_title(partial), partial))
            current = current[chunk_size - overlap :]
    if current:
        chunks.append((guess_section_title(current), current))
    return chunks


@dataclass(slots=True)
class JapaneseTokenizer:
    mode: str = "A"
    _tokenizer: object | None = field(init=False, default=None, repr=False)
    _split_mode: object | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from sudachipy import SplitMode, dictionary

            self._tokenizer = dictionary.Dictionary().create()
            self._split_mode = getattr(SplitMode, self.mode)
        except Exception:
            self._tokenizer = None
            self._split_mode = None

    def tokenize(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        if not normalized:
            return []
        if self._tokenizer is not None and self._split_mode is not None:
            return [m.surface() for m in self._tokenizer.tokenize(normalized, self._split_mode) if m.surface().strip()]
        fallback = re.findall(r"[\u3040-\u30ff\u3400-\u9fffA-Za-z0-9]+", normalized)
        if fallback:
            return fallback
        return [normalized[i : i + 2] for i in range(0, len(normalized), 2)]
