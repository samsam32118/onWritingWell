"""Cheap, judge-free signals of robotic prose.

These are not the reward. They are tripwires: if RL pushes the model toward winning the
judge by some trick, these numbers tend to move first (length balloons, bullets creep in,
stock phrases return). Every eval reports them next to win rates.
"""
from __future__ import annotations

import re
import statistics

STOCK_PHRASES = [
    "delve", "tapestry", "testament to", "navigate the complexities", "navigating the",
    "it's important to note", "it is important to note", "it's worth noting", "it is worth noting",
    "in today's fast-paced", "in today's world", "ever-evolving", "ever-changing", "landscape of",
    "unlock", "game-changer", "game changer", "at the end of the day", "plays a crucial role",
    "plays a pivotal role", "crucial role", "pivotal", "multifaceted", "nuanced", "realm",
    "embark", "foster", "leverage", "seamless", "robust", "holistic", "synergy", "paradigm",
    "a rich", "vibrant", "intricate", "in conclusion", "in summary", "to sum up", "overall,",
    "great question", "i hope this helps", "feel free to", "let me know if", "happy to help",
    "certainly!", "absolutely!", "here's the thing", "the key takeaway", "key takeaways",
    "underscores", "highlights the importance", "stands as", "serves as a", "not only",
    "whether you're", "dive into", "deep dive", "buckle up", "the bottom line",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_NOT_X_BUT_Y = re.compile(r"\b(it'?s|is|was|this isn'?t|isn'?t)\s+not\s+(just|only|merely|about)\b[^.]{0,80}?\b(but|it'?s)\b", re.I)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para or para.startswith(("#", "-", "*", "|")) or re.match(r"^\d+\.", para):
            continue
        out += [s for s in _SENT_SPLIT.split(para) if len(s.split()) >= 2]
    return out


def metrics(text: str) -> dict:
    words = text.split()
    n = max(len(words), 1)
    lower = text.lower()
    lines = [l for l in text.split("\n") if l.strip()]
    sents = _sentences(text)
    lens = [len(s.split()) for s in sents] or [0]
    stock = sum(lower.count(p) for p in STOCK_PHRASES)
    return {
        "words": len(words),
        "stock_per_1k": 1000 * stock / n,
        "em_dash_per_1k": 1000 * text.count("—") / n,
        "bullet_line_frac": sum(bool(re.match(r"^\s*([-*•]|\d+\.)\s", l)) for l in lines) / max(len(lines), 1),
        "header_count": sum(l.lstrip().startswith("#") for l in lines),
        "bold_per_1k": 1000 * text.count("**") / 2 / n,
        "not_x_but_y": len(_NOT_X_BUT_Y.findall(text)),
        "sent_len_mean": statistics.fmean(lens),
        # Burstiness: human prose varies sentence length a lot; flat prose doesn't.
        "sent_len_cv": (statistics.pstdev(lens) / statistics.fmean(lens)) if len(lens) > 1 and statistics.fmean(lens) else 0.0,
        "question_open": bool(re.match(r"^[^.?!]{0,120}\?", text.strip())),
    }


def aggregate(texts: list[str]) -> dict:
    ms = [metrics(t) for t in texts if t]
    if not ms:
        return {}
    return {k: round(statistics.fmean(float(m[k]) for m in ms), 4) for k in ms[0]}
