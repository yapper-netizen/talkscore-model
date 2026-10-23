"""
Stage A2 - segmentation, structure parsing, language detection.

Takes the page-level record from A1 and turns it into a flat list of
paragraph-level segments, each tagged with:
  - which TCFD/ISSB pillar it came from (Governance / Strategy / Risk
    Management / Metrics and Targets / Other)
  - language (en / zh / other)
  - position in the report (firm, page, paragraph index)

Why do this before the B-stage filter/classifier: ISSB reports are literally
structured around the four TCFD pillars - they have headings called
"Governance", "Strategy", etc. If a paragraph sits under one of those
headings we know its pillar for free, and the classifier only has to guess
on the un-headed text (tables, exec summary, loose narrative).

Install: langdetect (small, deterministic when seeded). Everything else is
stdlib.
"""

import re
import json
import os
import sys
import glob
from langdetect import detect, DetectorFactory, LangDetectException

# reports are bilingual (English + Chinese); the eyeball-print loop below
# prints raw segment text, and Windows' legacy console codepage crashes on
# non-Latin characters otherwise
sys.stdout.reconfigure(encoding="utf-8")

# without a seed, langdetect can classify the same text as 'en' one run and
# 'no' the next, which would quietly break reproducibility
DetectorFactory.seed = 0


# the four ISSB/TCFD pillar headings we look for in the report text itself.
# order matters a little - more specific patterns first, so "Risk
# Management" matches before a bare "Risk"
PILLAR_PATTERNS = [
    ("Governance",
     r"\b(governance|board\s+oversight|board\s+of\s+directors)\b",
     "TCFD pillar 1 - who is accountable"),

    ("Strategy",
     r"\b(strategy|strategic|business\s+model|transition\s+plan)\b",
     "TCFD pillar 2 - how climate is integrated into business plans"),

    ("Risk Management",
     r"\b(risk\s+management|risk\s+identification|risk\s+assessment)\b",
     "TCFD pillar 3 - how climate risks are identified and managed"),

    ("Metrics and Targets",
     r"\b(metrics\s+and\s+targets|metrics\s+&\s+targets|emissions?\s+target|"
     r"performance\s+indicator|kpi|scope\s+[123])\b",
     "TCFD pillar 4 - what is measured and what targets are set"),
]


def split_into_paragraphs(page_text):
    """
    Splits one page's text into paragraphs on blank-line boundaries.

    Paragraphs, not sentences: a pledge often runs 2-3 sentences and
    splitting to sentence level would fragment it before the specificity
    checklist in Stage D ever sees the whole claim. Sentence-level splitting
    happens later inside ClimateBERT anyway, which works per sentence.

    Returns list of paragraph strings, in document order.
    """
    if not page_text:
        return []

    # PDF extraction leaves multiple spaces, non-breaking spaces, and stray
    # line breaks inside what's logically one paragraph - normalize blank
    # lines first, then collapse the single newlines inside a paragraph
    text = page_text.replace(" ", " ")
    text = re.sub(r"\n\s*\n", "\n\n", text)
    paragraphs = text.split("\n\n")

    cleaned = []
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p).strip()
        # drop short fragments (page numbers, headers/footers, stray
        # artifacts) - 20 chars is a heuristic, lower it if real short
        # headings are getting dropped
        if len(p) >= 20:
            cleaned.append(p)
    return cleaned


def is_chapter_divider(paragraph):
    """
    True if this paragraph looks like a chapter-divider title rather than a
    TCFD/ISSB pillar heading.

    The pillar patterns above are deliberately loose (bare "governance",
    "risk management") so they catch real section headings, but reports also
    reuse those words as chapter titles for unrelated content - a "Corporate
    Governance" chapter about board pay, a general "Risk Management" chapter
    with no climate content. Without this check the carry-over logic below
    would tag the whole chapter with the wrong pillar.

    Heuristic: short (<80 chars) and starts with a chapter-number marker,
    e.g. "03 / Low-Carbon Development". Only catches that one convention -
    check what your own reports' divider pages look like if a publisher uses
    "PART 3:", roman numerals, or no numbering at all.
    """
    stripped = paragraph.strip()
    if len(stripped) > 80:
        return False
    return bool(re.match(r"^\d+\s*[/|]\s*.+", stripped))


def classify_heading(paragraph):
    """
    Returns the pillar name if this paragraph looks like a heading for one of
    the four TCFD/ISSB pillars, "RESET" if it's a non-climate chapter divider
    that should clear the current pillar, or None otherwise.

    Heading heuristics: short (<80 chars), matches a pillar pattern, and is
    mostly title-case or all-caps. We don't use font size since pdfplumber's
    font metadata isn't reliable across publishers - a text-only regex is
    weaker but works on every PDF.
    """
    # check dividers first - "03 / Risk Management" would otherwise match the
    # Risk Management pattern below and mislabel the whole following chapter
    if is_chapter_divider(paragraph):
        return "RESET"

    if len(paragraph) > 80:
        return None

    is_title = paragraph.istitle()
    is_upper = paragraph.isupper()
    words = paragraph.split()
    if words:
        cap_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
    else:
        cap_ratio = 0
    is_heading_case = is_title or is_upper or cap_ratio >= 0.6

    if not is_heading_case:
        return None

    for pillar, pattern, _ in PILLAR_PATTERNS:
        if re.search(pattern, paragraph, flags=re.IGNORECASE):
            return pillar

    return None


def tag_paragraphs_with_section(paragraphs):
    """
    Walks paragraphs in order. Once a paragraph is identified as a pillar
    heading, every following paragraph inherits that pillar until the next
    heading (or a chapter divider resets it back to "Other").

    Paragraphs before any heading default to "Other".

    Returns list of dicts {paragraph, pillar_from_structure, is_heading}.
    """
    tagged = []
    current_pillar = "Other"

    for p in paragraphs:
        pillar = classify_heading(p)
        if pillar == "RESET":
            current_pillar = "Other"
            tagged.append({
                "paragraph": p,
                "pillar_from_structure": "Other",
                "is_heading": False,
            })
        elif pillar is not None:
            current_pillar = pillar
            tagged.append({
                "paragraph": p,
                "pillar_from_structure": pillar,
                "is_heading": True,
            })
        else:
            tagged.append({
                "paragraph": p,
                "pillar_from_structure": current_pillar,
                "is_heading": False,
            })
    return tagged


def detect_language_safe(text):
    """
    langdetect raises on text that's too short/noisy to classify (common for
    table fragments), so treat those as 'unknown' instead of crashing.

    Returns ISO 639-1 code ('en', 'zh-cn', 'zh-tw', ...) or 'unknown'.
    """
    if len(text.strip()) < 20:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def single_char_token_ratio(text):
    """
    Fraction of whitespace-split tokens in `text` that are a single
    character long.

    pdfplumber's extract_text() can interleave columns character-by-character
    on pages with vector-drawn multi-column diagrams, so most "words" end up
    being stray single characters. Normal prose has a low ratio; a garbled
    diagram page has a much higher one.

    Returns 0.0 for empty/whitespace-only text. The 0.20 threshold used by
    callers came from an observed median of ~2% vs 39-84% on known-affected
    pages in our corpus - re-check it on your own reports, especially ones
    with lots of legitimate single-char tokens (dense numbered lists etc).
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    single_char = sum(1 for tok in tokens if len(tok) == 1)
    return single_char / len(tokens)


def segment_report(ingest_record):
    """
    Takes the dict from stage_a1_ingest.ingest_report() and returns a flat
    list of tagged segments ready for Stage B.

    Each segment:
      {
        "firm": str,
        "page_num": int,
        "para_idx": int,
        "text": str,
        "pillar_from_structure": str,
        "is_heading": bool,
        "language": str,
        "quality_flag": str,             # carried from A1
        "source": str,                   # 'text' / 'table_cell' / 'image_ocr'
                                          # / 'image_vision_llm'
      }

    Things worth checking on a new run:
      - if 'Other' dominates the pillar counts, the structure parser isn't
        catching headings - open the report and add patterns
      - if 'unknown' dominates language, paragraphs are too short/noisy
      - spot check a few non-heading segments per pillar against the source
      - table_cell segments duplicate the 'text' segments on the same
        garbled page by design - Stage B filters by quality
    """
    firm = ingest_record["firm"]
    all_segments = []

    for page in ingest_record["pages"]:
        page_num = page["page_num"]
        quality_flag = page.get("quality_flag", "ok")
        page_text = page.get("text", "")

        paragraphs = split_into_paragraphs(page_text)
        tagged = tag_paragraphs_with_section(paragraphs)

        idx = 0
        for t in tagged:
            all_segments.append({
                "firm": firm,
                "page_num": page_num,
                "para_idx": idx,
                "text": t["paragraph"],
                "pillar_from_structure": t["pillar_from_structure"],
                "is_heading": t["is_heading"],
                "language": detect_language_safe(t["paragraph"]),
                "quality_flag": quality_flag,
                "source": "text",
            })
            idx += 1

        # table-cell fallback for garbled diagram pages: a high proportion of
        # single-char tokens means extract_text() likely interleaved
        # multi-column diagram text. Minimum word count avoids flagging a
        # near-empty cover page off a tiny sample.
        word_count = len(page_text.split())
        ratio = single_char_token_ratio(page_text)
        page_is_garbled = ratio > 0.20 and word_count > 50

        if page_is_garbled:
            for table in page.get("tables", []) or []:
                for row in table:
                    for cell in row:
                        if not cell:
                            continue
                        cell_text = cell.strip()
                        if len(cell_text) < 20:
                            continue
                        all_segments.append({
                            "firm": firm,
                            "page_num": page_num,
                            "para_idx": idx,
                            "text": cell_text,
                            # not walked by the heading/carry-over logic, so
                            # no structural pillar signal - default to Other
                            "pillar_from_structure": "Other",
                            "is_heading": False,
                            "language": detect_language_safe(cell_text),
                            "quality_flag": quality_flag,
                            "source": "table_cell",
                        })
                        idx += 1

        # image-extraction fallback: A1 runs OCR (or a vision LLM) on every
        # page with an embedded image, regardless of quality_flag - a page
        # can read fine as prose and still carry a chart in the margin
        for extraction in page.get("image_extractions", []) or []:
            method = extraction.get("method")

            if method == "tesseract":
                text = (extraction.get("text") or "").strip()
                if len(text) < 20:
                    continue
                source = "image_ocr"
            elif method == "vision_llm":
                data = extraction.get("data") or {}
                if data.get("result") in ("no_data", "parse_error"):
                    continue
                # serialize the structured JSON to text so it flows through
                # the same paragraph-shaped segment schema as everything else
                text = json.dumps(data, ensure_ascii=False)
                source = "image_vision_llm"
            else:
                continue

            all_segments.append({
                "firm": firm,
                "page_num": page_num,
                "para_idx": idx,
                "text": text,
                "pillar_from_structure": "Other",
                "is_heading": False,
                "language": detect_language_safe(text),
                "quality_flag": quality_flag,
                "source": source,
            })
            idx += 1

    return all_segments


def summarise_segmentation(segments):
    """Counts to eyeball before passing results to Stage B."""
    from collections import Counter
    by_pillar = Counter(s["pillar_from_structure"] for s in segments)
    by_language = Counter(s["language"] for s in segments)
    by_source = Counter(s["source"] for s in segments)
    headings_found = sum(1 for s in segments if s["is_heading"])
    return {
        "total_segments": len(segments),
        "headings_detected": headings_found,
        "by_pillar": dict(by_pillar),
        "by_language": dict(by_language),
        "by_source": dict(by_source),
    }


if __name__ == "__main__":
    EXTRACTED_DIR = "data/extracted"

    # A1 already wrote one *_segments.json per firm (its ingest record, not
    # yet segmented) - read each back in, segment it, overwrite with the
    # final segmented output
    ingest_paths = sorted(glob.glob(os.path.join(EXTRACTED_DIR, "*_segments.json")))

    for ingest_path in ingest_paths:
        with open(ingest_path, "r", encoding="utf-8") as f:
            ingested = json.load(f)

        segments = segment_report(ingested)
        summary = summarise_segmentation(segments)

        print(f"\n=== Firm: {ingested['firm']} ===")
        print(f"Total segments:    {summary['total_segments']}")
        print(f"Headings detected: {summary['headings_detected']}")
        print(f"By pillar:         {summary['by_pillar']}")
        print(f"By language:       {summary['by_language']}")
        print(f"By source:         {summary['by_source']}")

        # eyeball one segment per pillar to confirm the tagging is right
        print("--- One segment per pillar ---")
        seen = set()
        for s in segments:
            p = s["pillar_from_structure"]
            if p in seen or s["is_heading"]:
                continue
            seen.add(p)
            print(f"[{p}] page {s['page_num']} para {s['para_idx']} "
                  f"({s['language']})")
            print(s["text"][:200])

        out_path = os.path.join(EXTRACTED_DIR, f"{ingested['firm']}_segments.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        print(f"Segments saved to {out_path}")

    # if 'Other' dominates the pillar distribution, PILLAR_PATTERNS isn't
    # catching the headings in these reports - go add the real heading text.
    # if 'unknown' dominates language, paragraphs are too short - B3 gets a
    # second chance at filtering anyway
