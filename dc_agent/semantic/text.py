from __future__ import annotations

import re

_STOP = {"the", "and", "for", "with", "what", "which", "show", "give", "list", "from", "that", "this", "are", "per", "all", "any"}
_FOLLOW_UP = re.compile(r"^\s*(now|and|also|then|same|for|what about|how about|ok(ay)?)\b", re.I)


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def tokens(text: str | None) -> set:
    out = set()
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        w = _stem(w)
        if len(w) > 2 and w not in _STOP:
            out.add(w)
    return out


def is_follow_up(question: str) -> bool:
    """True for questions that refine the previous turn ("now for tier IV", "what about the West?").

    Only follow-ups see the previous plan. Self-contained questions are planned without it,
    so they are cacheable and cannot inherit stale filters from earlier in the conversation.
    """
    return bool(_FOLLOW_UP.match(question))


def phrase_windows(question: str, sizes=(2, 3), cap: int = 24) -> list:
    """Short word windows of the question. Synonyms are short phrases, so comparing them with windows
    rather than the whole question keeps the similarity from being diluted by unrelated words."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    out = []
    for n in sizes:
        out += [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return list(dict.fromkeys(out))[:cap]
