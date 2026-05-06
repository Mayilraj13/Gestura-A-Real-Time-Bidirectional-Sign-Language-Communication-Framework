"""
Improved NLP text processor (FINAL FIXED VERSION)

Fixes:
- Phrase-level matching (good morning → correct label)
- Prevents wrong matches (good → good-afternoon ❌)
- Removes duplicates
- Better noise filtering
"""

from __future__ import annotations

import re
from pathlib import Path
from loguru import logger

_nlp = None
_dataset_labels: set[str] = set()
_label_list: list[str] = []

NOISE_PATTERNS = re.compile(r"[^a-z\s']")
STOP_WORDS_EXTRA = {"um", "uh", "er", "ah", "like", "okay", "ok", "yeah"}
KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

IGNORE_WORDS = {"you", "the", "a", "an", "is", "are"}


def load_spacy() -> None:
    global _nlp
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy loaded")
    except:
        logger.warning("spaCy model not found. Downloading...")
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )
        _nlp = spacy.load("en_core_web_sm")


def load_labels(labels_path: str | Path) -> None:
    global _dataset_labels, _label_list
    labels_path = Path(labels_path)

    if labels_path.exists():
        _label_list = labels_path.read_text(encoding="utf-8").strip().splitlines()
        _dataset_labels = set(l.lower().strip() for l in _label_list)
        logger.info(f"Loaded {len(_dataset_labels)} labels")
    else:
        logger.warning(f"Labels not found: {labels_path}")


def process_text(raw_text: str) -> list[str]:
    if _nlp is None:
        raise RuntimeError("spaCy not loaded")

    # 🔹 Clean text
    text = raw_text.lower().strip()
    text = NOISE_PATTERNS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text or len(text) < 3:
        return []

    if len(set(text.split())) == 1 and len(text.split()) > 1:
        return []

    doc = _nlp(text)

    keywords = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        if token.pos_ not in KEEP_POS:
            continue

        if token.is_stop:
            continue

        lemma = token.lemma_.lower().strip()

        if len(lemma) < 2:
            continue

        if lemma in STOP_WORDS_EXTRA:
            continue

        if lemma in IGNORE_WORDS:
            continue

        keywords.append(lemma)

    logger.debug(f"Keywords: {keywords}")

    if not keywords:
        return []

    matched = _match_labels(keywords, text)

    # 🔥 Remove duplicates (preserve order)
    return list(dict.fromkeys(matched))


# 🔥 FINAL MATCHING FUNCTION
def _match_labels(keywords: list[str], text: str) -> list[str]:
    matched = []
    used = set()

    # ✅ STEP 1 — PHRASE MATCH (MOST IMPORTANT)
    for label in _label_list:
        label_clean = label.lower().replace("-", " ")
        if label_clean in text:
            matched.append(label.lower())
            used.add(label.lower())

    # ✅ STEP 2 — EXACT MATCH
    for kw in keywords:
        if kw in _dataset_labels and kw not in used:
            matched.append(kw)
            used.add(kw)

    # ✅ STEP 3 — STRICT PARTIAL MATCH
    for kw in keywords:
        if kw in used:
            continue

        for label in _label_list:
            label_lower = label.lower()

            # strict word boundary match
            if f" {kw} " in f" {label_lower} " and label_lower not in used:
                matched.append(label_lower)
                used.add(label_lower)
                break

    return matched