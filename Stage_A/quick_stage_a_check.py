"""
=============================================================================
QUICK STAGE A LITMUS TEST  -  solo, 1-2 hours, not a full validation
=============================================================================
Pulls a small stratified sample from your A2 output (4 segments per firm:
2 clean paragraph, 1 table_cell, 1 flagged quality_flag != ok) and prints
them in a format you can check by eye against the source PDF, then fills
in a scoring sheet as you go.

This is NOT the P10-style two-validator kappa protocol. It's a fast sanity
check. Do the full validation later if the pipeline is going into the paper.

USAGE
-----
  python quick_stage_a_check.py <firm>_segments.json <firm>_segments.json ...

Prints numbered segments to check, then at the end prints a blank scoring
template to fill in (or edit SCORES dict directly and rerun to get the
summary at the bottom).
=============================================================================
"""

import json
import random
import sys
from collections import defaultdict

random.seed(42)  # reproducible sample, not cherry-picked


def stratified_sample(segments, firm_name, n_clean=2, n_table=1, n_flagged=1):
    """
    Pulls a small, deliberately mixed sample from one firm's segments:
      - n_clean paragraph segments with quality_flag == "ok"
      - n_table table_cell segments (if any exist for this firm)
      - n_flagged paragraph segments with quality_flag != "ok" (if any)
    Falls back gracefully if a category is empty (e.g. a firm with no
    table_cell segments, or no flagged pages) rather than crashing.
    """
    clean = [s for s in segments if s.get("source_type", "paragraph") == "paragraph"
             and s.get("quality_flag", "ok") == "ok" and len(s.get("text", "")) > 40]
    table = [s for s in segments if s.get("source_type") == "table_cell"]
    flagged = [s for s in segments if s.get("source_type", "paragraph") == "paragraph"
               and s.get("quality_flag", "ok") != "ok"]

    picked = []
    picked += random.sample(clean, min(n_clean, len(clean)))
    picked += random.sample(table, min(n_table, len(table)))
    picked += random.sample(flagged, min(n_flagged, len(flagged)))

    for seg in picked:
        seg["_firm_tag"] = firm_name
    return picked


def print_check_sheet(all_picked):
    """
    Prints each sampled segment with an ID, so you can open the source PDF
    at that page and eyeball it. Leaves blank lines for you to write your
    yes/no/partial judgment directly under each one.
    """
    print("=" * 90)
    print("STAGE A QUICK LITMUS TEST — fill in the four judgments under each segment")
    print("=" * 90)
    for i, seg in enumerate(all_picked, 1):
        print(f"\n[{i}] firm={seg['_firm_tag']}  page={seg.get('page_num')}  "
              f"para_idx={seg.get('para_idx')}  source_type={seg.get('source_type')}")
        print(f"    pillar_from_structure = {seg.get('pillar_from_structure')}")
        print(f"    is_heading = {seg.get('is_heading')}  language = {seg.get('language')}  "
              f"quality_flag = {seg.get('quality_flag')}")
        print(f"    text: {seg.get('text', '')[:220]}")
        print("    --- open the PDF at this page and judge ---")
        print("    text_correct        [ y / n / partial ]:")
        print("    pillar_correct      [ y / n ] (if n, correct pillar = ____):")
        print("    source_type_correct [ y / n ]:")
        print("    quality_flag_honest [ y / n ]:")


# ---------------------------------------------------------------------------
# SCORING  -  after you have judged all segments, fill this in and rerun to
# get the summary. Format: {segment_id: {"text": "y"/"n"/"partial",
# "pillar": "y"/"n", "source_type": "y"/"n", "quality_flag": "y"/"n"}}
# ---------------------------------------------------------------------------
SCORES = {
    # 1: {"text": "y", "pillar": "y", "source_type": "y", "quality_flag": "y"},
    # 2: {"text": "n", "pillar": "y", "source_type": "y", "quality_flag": "y"},
    # ... fill in as you check each segment, then rerun this file
}


def summarise(scores):
    if not scores:
        print("\n(no scores filled in yet — edit SCORES at the bottom of this file, then rerun)")
        return
    fields = ["text", "pillar", "source_type", "quality_flag"]
    counts = {f: defaultdict(int) for f in fields}
    for s in scores.values():
        for f in fields:
            counts[f][s.get(f, "?")] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n = len(scores)
    for f in fields:
        y = counts[f].get("y", 0)
        print(f"{f:15s}: {y}/{n} correct  ({counts[f]})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_stage_a_check.py <firm1>_segments.json <firm2>_segments.json ...")
        sys.exit(1)

    all_picked = []
    for path in sys.argv[1:]:
        firm_name = path.split("/")[-1].replace("_segments.json", "")
        with open(path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        all_picked += stratified_sample(segments, firm_name)

    print_check_sheet(all_picked)
    summarise(SCORES)

    # NEXT STEP:
    #   1. Run this once to print the sheet, open each firm's PDF at the
    #      printed page numbers, judge each of the 4 fields per segment.
    #   2. Fill in the SCORES dict above with your judgments.
    #   3. Rerun this file. The summary block prints accuracy per field.
    #   That's your litmus test number. Takes about 1-2 hours for ~20 segments.
