"""
=============================================================================
STAGE A2  -  SEGMENTATION, STRUCTURE PARSING, LANGUAGE DETECTION
TalkScore pipeline | A1 output -> tagged segments ready for filtering (B3)
=============================================================================
 
WHAT THIS FILE DOES
-------------------
Takes the page-level record produced by Stage A1 and turns it into a flat list
of paragraph-level segments, each tagged with:
  - which TCFD/ISSB section it came from  (Governance / Strategy / Risk
    Management / Metrics and Targets / Other)
  - what language it is in                 (en / zh / other)
  - where in the report it sits            (firm, page, paragraph index)
 
WHY DO THIS BEFORE FILTERING (B3) OR CLASSIFYING (B5)
-----------------------------------------------------
The TCFD classifier in B3 makes a probabilistic guess about pillar. But ISSB
reports are *structured* around those exact four pillars - they literally have
headings called "Governance", "Strategy", "Risk Management", "Metrics and
Targets". If a paragraph sits under one of those headings in the report, we
know its pillar with certainty. The model only has to work on un-headed text
(tables, executive summary, narrative without clear headings).
 
This is the same logic as a teacher checking your worked answer against the
chapter title before grading the method. The chapter title is the strongest
prior we have, and it is free.
 
DECISIONS THIS STAGE EMBEDS (see Decision Log DL-005 through DL-008, DL-016, DL-019)
--------------------------------------------------------------------
  DL-005  Unit of analysis = paragraph (not sentence, not whole report)
  DL-006  Heading detection by regex + font size, with a "section carry-over"
          rule so paragraphs under a heading inherit its label
  DL-007  Language detection at the segment level, not document level
  DL-008  Reject sentence-splitting with NLTK punkt in favour of a lightweight
          regex splitter, to avoid an extra ~30MB dependency for negligible gain
  DL-016  Chapter-divider reset: non-climate chapter headings (e.g. "Corporate
          Governance", "Risk Management" as chapter titles rather than TCFD
          section headings) must not carry a pillar label across an entire
          chapter - detect them and reset current_pillar back to "Other"
  DL-019  Table-cell fallback: on pages where extract_text() is garbled by
          vector-drawn multi-column diagrams, also emit segments built from
          A1's already-correctly-ordered page["tables"] cells
 
DEPENDENCIES
------------
  pip install langdetect
  # Everything else is standard library (re, json).
  # langdetect is small (~1MB) and deterministic when seeded.
 
=============================================================================
"""
 
import re
import json
from langdetect import detect, DetectorFactory, LangDetectException
 
# Make langdetect deterministic. Without this seed, the same text can be
# classified as 'en' one run and 'no' the next, which would silently corrupt
# the reproducibility of the whole pipeline. This is a non-obvious gotcha.
DetectorFactory.seed = 0
 
 
# ---------------------------------------------------------------------------
# The four ISSB / TCFD pillar headings we look for in the report itself.
# Order matters slightly: more specific patterns first, so "Risk Management"
# is matched before a bare "Risk".
# ---------------------------------------------------------------------------
PILLAR_PATTERNS = [
    # (pillar_label, regex pattern, notes)
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
 
 
# ---------------------------------------------------------------------------
# STEP A2  -  Split page text into paragraphs
# ---------------------------------------------------------------------------
def split_into_paragraphs(page_text):
    """
    Splits one page's text into paragraphs using blank-line boundaries.
 
    Why paragraphs and not sentences (decision DL-005):
      - A pledge often spans 2-3 sentences ("We will reduce X. Our baseline
        is Y. This is validated by SBTi."). Splitting to sentences fragments
        the pledge and the specificity checklist (Stage D9) loses context.
      - Paragraphs are the natural unit in ISSB reports - they are usually
        one coherent claim each.
      - Sentence-level becomes useful later inside ClimateBERT (B3), which
        operates per sentence anyway. We can re-split there if needed.
 
    Returns: list of paragraph strings, in document order.
    """
    if not page_text:
        return []
 
    # Normalise weird whitespace - PDF extraction often leaves multiple spaces,
    # non-breaking spaces, and stray line breaks inside what is logically one
    # paragraph. Collapse single newlines inside a paragraph to spaces, but
    # KEEP blank lines as paragraph boundaries.
    #
    # The two-step regex: first protect blank-line boundaries, then collapse
    # single newlines inside paragraphs.
    text = page_text.replace("\u00a0", " ")          # non-breaking space -> space
    text = re.sub(r"\n\s*\n", "\n\n", text)          # normalise blank-line markers
    paragraphs = text.split("\n\n")
 
    cleaned = []
    for p in paragraphs:
        # Collapse internal line breaks to single spaces. Real paragraph
        # breaks were already preserved above.
        p = re.sub(r"\s+", " ", p).strip()
        # Drop very short fragments (page numbers, headers/footers, stray
        # artifacts from PDF extraction). 20 characters is a heuristic
        # >>> VERIFY THIS, DO NOT TRUST IT <<<
        # If real short headings get dropped, lower the threshold.
        if len(p) >= 20:
            cleaned.append(p)
    return cleaned
 
 
# ---------------------------------------------------------------------------
# STEP A2b  -  Identify pillar headings and carry the label forward
# ---------------------------------------------------------------------------
def is_chapter_divider(paragraph):
    """
    Returns True if this paragraph looks like a major chapter-divider title
    rather than a TCFD/ISSB pillar heading (decision DL-016).

    Why this exists: the pillar patterns in PILLAR_PATTERNS are deliberately
    loose (e.g. bare "governance", "risk management") so they catch real TCFD
    section headings. But many reports also use those same words as titles for
    unrelated, non-climate chapters - e.g. a "Corporate Governance" chapter
    about board composition and executive pay, or a general "Risk Management"
    chapter about enterprise risk with no climate content. Without this check,
    tag_paragraphs_with_section() would carry that pillar label across every
    paragraph in the chapter (potentially 20+ pages) until the next heading.

    Heuristic (decision DL-016): a chapter divider is short (under 80 chars,
    same bound as a heading) and starts with a chapter-number marker, e.g.
    "03 / Low-Carbon Development" or "12 | Governance and Risk".

    >>> VERIFY THIS, DO NOT TRUST IT <<<
    The regex only catches "<number> <slash-or-pipe> <title>" dividers. Open a
    few real reports and check what their chapter-divider pages actually look
    like - publishers that use a different convention (e.g. "PART 3:", roman
    numerals, or no numbering at all) will slip past this and still trigger
    the mislabeling bug this function exists to prevent.
    """
    stripped = paragraph.strip()
    if len(stripped) > 80:
        return False
    return bool(re.match(r"^\d+\s*[/|]\s*.+", stripped))


def classify_heading(paragraph):
    """
    Returns the pillar name if this paragraph looks like a heading for one of
    the four TCFD/ISSB pillars, "RESET" if it looks like a non-climate chapter
    divider that should clear the current pillar (decision DL-016), or None
    if it is neither.

    Heuristics for "this is a heading" (decision DL-006):
      - short (under 80 characters)
      - matches one of the pillar regex patterns
      - mostly title-case or all-caps

    We deliberately do NOT rely on font size, because pdfplumber's font
    metadata is unreliable across publishers. A regex on the text itself is
    less powerful but works on every PDF, which matters for reproducibility.
    """
    # Chapter dividers are checked first, before the pillar patterns, because
    # a divider like "03 / Risk Management" would otherwise match the "Risk
    # Management" pillar pattern below and silently mislabel the whole
    # following chapter (DL-016).
    if is_chapter_divider(paragraph):
        return "RESET"

    if len(paragraph) > 80:
        return None  # too long to be a heading
 
    # Heading-ish casing test. Many ISSB reports use either Title Case or
    # ALL CAPS for section headers.
    is_title = paragraph.istitle()
    is_upper = paragraph.isupper()
    # A more forgiving check: most words start with a capital letter.
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
    Walks paragraphs in order. When a paragraph is identified as a pillar
    heading, every subsequent paragraph inherits that pillar label until a
    new heading is encountered. This is the "carry-over" rule.
 
    Worked example:
        [P1] "Governance"                       -> heading, pillar=Governance
        [P2] "Our board oversees climate..."     -> inherits Governance
        [P3] "The Sustainability Committee..."   -> inherits Governance
        [P4] "Metrics and Targets"               -> new heading
        [P5] "We will cut Scope 1 by 42%..."     -> inherits Metrics and Targets
 
    Paragraphs that appear before any pillar heading get pillar = "Other".

    Chapter dividers (decision DL-016) reset the carry-over instead of setting
    a pillar:
        [P6] "12 / Corporate Governance"        -> RESET, pillar=Other
        [P7] "Our executive pay framework..."    -> inherits Other, not
                                                     Governance

    Returns: list of dicts {paragraph, pillar_from_structure, is_heading}.
    """
    tagged = []
    current_pillar = "Other"  # default until the first heading is seen

    for p in paragraphs:
        pillar = classify_heading(p)
        if pillar == "RESET":
            # Non-climate chapter divider (DL-016) - clear the carry-over so
            # the next chapter doesn't inherit a stale pillar label. The
            # divider itself is not a pillar heading, so is_heading is False.
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
 
 
# ---------------------------------------------------------------------------
# STEP A2c  -  Language detection per segment
# ---------------------------------------------------------------------------
def detect_language_safe(text):
    """
    Wraps langdetect with a fail-safe. langdetect raises an exception on text
    that is too short or too noisy to classify (very common for table fragments
    extracted from a PDF). We treat those as 'unknown' rather than crashing.
 
    Returns: ISO 639-1 code ('en', 'zh-cn', 'zh-tw', ...) or 'unknown'.
    """
    if len(text.strip()) < 20:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
 
 
# ---------------------------------------------------------------------------
# STEP A2d  -  Garbled-page detection for the table-cell fallback (DL-019)
# ---------------------------------------------------------------------------
def single_char_token_ratio(text):
    """
    Returns the fraction of whitespace-split tokens in `text` that are a
    single character long.

    Why this matters (decision DL-019): pdfplumber's extract_text() can
    interleave columns character-by-character on pages with vector-drawn
    multi-column diagrams, producing text where most "words" are stray single
    characters. A normal prose page has a low ratio (numbers, "a", "I",
    bullet glyphs); a garbled diagram page has a very high one because entire
    lines have been shredded into single-character fragments.

    Returns 0.0 for empty/whitespace-only text.

    >>> VERIFY THIS, DO NOT TRUST IT <<<
    The 0.20 ratio threshold used by callers is based on an observed document
    median of ~2% versus 39-84% on known-affected pages. That gap was
    measured on a specific set of reports - re-check it against your own
    corpus before trusting it at scale, especially for reports with a lot of
    legitimate single-character tokens (e.g. dense numbered/lettered lists).
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    single_char = sum(1 for tok in tokens if len(tok) == 1)
    return single_char / len(tokens)


# ---------------------------------------------------------------------------
# MASTER FUNCTION  -  full Stage A2 for one ingested report
# ---------------------------------------------------------------------------
def segment_report(ingest_record):
    """
    Takes the dict produced by stage_a1_ingest.ingest_report() and returns
    a flat list of tagged segments ready for Stage B (filter & classify).
 
    Output schema (each segment):
      {
        "firm": str,
        "page_num": int,
        "para_idx": int,                 # paragraph index within the page
        "text": str,
        "pillar_from_structure": str,    # Governance / Strategy / Risk Management
                                         # / Metrics and Targets / Other
        "is_heading": bool,              # True if this paragraph IS the heading
        "language": str,                 # 'en', 'zh-cn', etc, or 'unknown'
        "quality_flag": str,             # carried from A1
        "source": str,                   # 'text' (from extract_text) or
                                         # 'table_cell' (DL-019 fallback)
      }

    HOW TO READ THE OUTPUT (verify, do not trust):
      1. Count segments per pillar. If 'Other' dominates, the structure parser
         is not catching headings - inspect the report's actual heading text
         and add patterns to PILLAR_PATTERNS.
      2. Check the language distribution. If 'unknown' is large, paragraphs
         are too short or too noisy - lower the 20-char threshold or trust the
         downstream filter (B3) to drop them.
      3. Spot-check 5 random non-heading segments - is each one really under
         the correct pillar? If not, the carry-over rule is grouping unrelated
         content together because a heading was missed.
      4. Check the 'table_cell' source count. Those segments come from pages
         flagged as garbled (DL-019) - the matching 'text' segments from the
         same page are kept too (not suppressed), so expect some duplication
         between the two sources on those pages. Stage B filters by quality.
    """
    firm = ingest_record["firm"]
    all_segments = []

    for page in ingest_record["pages"]:
        page_num = page["page_num"]
        quality_flag = page.get("quality_flag", "ok")
        page_text = page.get("text", "")

        # A2: split this page into paragraphs
        paragraphs = split_into_paragraphs(page_text)

        # A2b: tag each paragraph with its inferred pillar
        tagged = tag_paragraphs_with_section(paragraphs)

        # A2c: detect language per paragraph and build the final segment record
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

        # A2d: table-cell fallback for garbled diagram pages (DL-019).
        # Detection: a high proportion of single-character tokens means
        # extract_text() likely interleaved multi-column diagram text. We
        # also require a minimum word count so near-empty pages (e.g. a
        # cover page with just a logo) don't get flagged on a tiny sample.
        #
        # >>> VERIFY THIS, DO NOT TRUST IT <<<
        # The 0.20 ratio / 50-word thresholds are carried over from the
        # DL-019 investigation on a specific set of reports. Re-check them
        # against your own corpus.
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
                        # Same 20-char minimum used by split_into_paragraphs,
                        # to drop tiny/empty cells (see note at that call).
                        if len(cell_text) < 20:
                            continue
                        all_segments.append({
                            "firm": firm,
                            "page_num": page_num,
                            "para_idx": idx,
                            "text": cell_text,
                            # Table cells aren't walked by the heading/
                            # carry-over logic above, so we have no
                            # structural pillar signal for them - default to
                            # "Other" rather than guessing.
                            "pillar_from_structure": "Other",
                            "is_heading": False,
                            "language": detect_language_safe(cell_text),
                            "quality_flag": quality_flag,
                            "source": "table_cell",
                        })
                        idx += 1

    return all_segments
 
 
# ---------------------------------------------------------------------------
# Convenience: summarise a segmentation run so you can eyeball before scaling
# ---------------------------------------------------------------------------
def summarise_segmentation(segments):
    """
    Returns counts that tell you whether A2 worked before you pass results to
    Stage B. These are the same checks listed in segment_report's docstring.
    """
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
 
 
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run A1 first, then feed it into A2.
    from stage_a1_ingest import ingest_report
 
    ingested = ingest_report("data/raw_reports/cathay_pacific_2025.pdf",
                             use_vision_llm=False)
    segments = segment_report(ingested)
    summary = summarise_segmentation(segments)
 
    print(f"Firm:              {ingested['firm']}")
    print(f"Total segments:    {summary['total_segments']}")
    print(f"Headings detected: {summary['headings_detected']}")
    print(f"By pillar:         {summary['by_pillar']}")
    print(f"By language:       {summary['by_language']}")
    print(f"By source:         {summary['by_source']}")
 
    # Eyeball one segment per pillar so you can confirm the tagging is right.
    print("\n--- One segment per pillar ---")
    seen = set()
    for s in segments:
        p = s["pillar_from_structure"]
        if p in seen or s["is_heading"]:
            continue
        seen.add(p)
        print(f"\n[{p}] page {s['page_num']} para {s['para_idx']} "
              f"({s['language']})")
        print(s["text"][:200])
 
    # Save for the next stage to consume.
    with open(f"data/extracted/{ingested['firm']}_segments.json", "w") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print("\nSegments saved to data/extracted/")
 
    # NEXT STEP after reading this:
    #   If 'Other' dominates the pillar distribution, the regex in
    #   PILLAR_PATTERNS is not catching the headings in YOUR reports. Open
    #   one report, copy the actual heading text, and add it to the patterns.
    #   If 'unknown' dominates the language distribution, your paragraphs are
    #   too short - the language filter will get a second chance in B3 anyway.