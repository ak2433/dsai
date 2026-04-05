# syllabus_rag.py
# Chunk and search course markdown for RAG-style retrieval
# Tim Fraser

# Splits a syllabus (.md) into sections by headings, then scores chunks with
# simple keyword overlap so we can pass the best excerpts to the LLM.

import re
from typing import Any

# 1. Parse markdown into sections #############################################

# Match ATX headings (# through ######)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown_sections(text: str) -> list[dict[str, str]]:
    """
    Split markdown into sections using heading lines as boundaries.

    Returns:
        List of dicts with keys: id, title, body (markdown text under that heading).
    """
    lines = text.splitlines()
    sections: list[dict[str, str]] = []
    current_title = "Course document"
    current_lines: list[str] = []

    for line in lines:
        m = _HEADING.match(line)
        if m:
            # Flush previous section (keep intro content if file started without a heading)
            if current_lines or sections:
                sections.append(
                    {
                        "id": f"sec-{len(sections)}",
                        "title": current_title,
                        "body": "\n".join(current_lines).strip(),
                    }
                )
            current_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    sections.append(
        {
            "id": f"sec-{len(sections)}",
            "title": current_title,
            "body": "\n".join(current_lines).strip(),
        }
    )

    # If the whole file had no headings, treat as one section
    if len(sections) == 1 and not sections[0]["body"] and text.strip():
        return [{"id": "sec-0", "title": current_title, "body": text.strip()}]

    return [s for s in sections if s["body"] or s["title"]]


# 2. Keyword search over sections ##############################################


def _tokenize(q: str) -> set[str]:
    """Lowercase alphanumeric tokens; ignores very short noise tokens."""
    raw = re.findall(r"[a-z0-9]+", q.lower())
    return {t for t in raw if len(t) > 1}


def search_sections(
    query: str,
    sections: list[dict[str, str]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Score sections by overlap between query tokens and title+body text.

    Returns:
        List of up to top_k dicts: id, title, score, excerpt (first ~800 chars of body).
    """
    if not query.strip():
        # No query: return start of course for context
        out = []
        for s in sections[:top_k]:
            excerpt = (s["body"] or "")[:800]
            out.append({"id": s["id"], "title": s["title"], "score": 0.0, "excerpt": excerpt})
        return out

    q_tokens = _tokenize(query)
    if not q_tokens:
        q_tokens = {query.lower().strip()}

    scored: list[tuple[float, dict[str, str]]] = []
    for s in sections:
        hay = f"{s['title']}\n{s['body']}".lower()
        score = sum(1 for t in q_tokens if t in hay)
        scored.append((float(score), s))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, s in scored[:top_k]:
        if score == 0 and results:
            continue
        body = s["body"] or ""
        excerpt = body[:1200] + ("…" if len(body) > 1200 else "")
        results.append({"id": s["id"], "title": s["title"], "score": score, "excerpt": excerpt})
    return results[:top_k]


def sections_context_for_llm(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as a single string for the LLM."""
    parts = []
    for c in chunks:
        parts.append(f"### Section: {c['title']} (id={c['id']})\n{c['excerpt']}")
    return "\n\n".join(parts)


def format_course_outline(sections: list[dict[str, str]]) -> str:
    """
    Compact list of every syllabus section so the model sees full course scope.

    Used to orient the tutor (how the course is directed) alongside keyword-matched excerpts.
    """
    if not sections:
        return "(No sections parsed.)"
    lines = [f"- {s['id']}: {s['title']}" for s in sections]
    return "\n".join(lines)
