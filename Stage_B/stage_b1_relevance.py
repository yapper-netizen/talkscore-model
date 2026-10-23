"""
Stage B1 - climate relevance filter (ClimateBERT).

Takes the segments from A2 and keeps only the ones that are actually about
climate. This is a coarse, fast, free, offline binary classifier - it just
answers "climate-related, yes or no?" and drops the clear no's before they
reach the expensive LLM step in B2.

A2 (segment + pillar tag + language) -> THIS FILE (B1) -> B2 (LLM)
No LLM and no API call happens in this file - see the note at the bottom for
how B2 picks up from here.

Why a ClimateBERT sieve before the LLM: most of an ESG report isn't about
climate (health & safety, governance boilerplate, community programmes).
Sending all of that to the LLM burns tokens for nothing. ClimateBERT is
small, validated, and deterministic, so it's the right tool to thin the
stream first (grounded on Bingler et al. 2022).

Install: transformers, torch. First run downloads the checkpoint (~330MB)
from Hugging Face, then it's cached and runs offline/deterministically. CPU
is fine, GPU just makes it faster.
"""

import json
from datetime import date

# transformers is imported lazily inside _load_classifier() so importing this
# module for its constants/helpers doesn't drag in torch


# CHECKPOINT is the binary relevance classifier. The alternative,
# "climatebert/distilroberta-base-climate-f", is the domain-adapted base LM
# with no classification head - it can't emit a yes/no without extra
# fine-tuning we're not doing.
CHECKPOINT = "climatebert/distilroberta-base-climate-detector"

# keep a segment when P(climate) >= this. 0.3, not the default 0.5, because
# B1 is a first-stage sieve tuned for recall - B2 filters again, and a missed
# pledge is lost for good. Verify this on a hand-labeled pilot before scaling.
KEEP_THRESHOLD = 0.3

# ClimateBERT is English-only. "unknown" is treated as not-English too -
# short/noisy fragments langdetect couldn't resolve usually aren't scorable
# prose anyway.
KEEP_LANGUAGES = {"en"}

# table_cell segments skip the model and pass straight through - ClimateBERT
# is prose-trained and unreliable on fragments.
BYPASS_FILTER_FOR_TABLE_CELLS = True

# drop paragraph segments whose page was flagged bad quality - their real
# content gets recovered separately as table_cell segments.
DROP_LOW_QUALITY_PARAGRAPHS = True

# the classifier is trained on prose sentences and has a 512-token limit
MAX_TOKENS = 512

# label strings meaning "this IS climate-related". Matched case-insensitively
# and not hardcoded to one string since checkpoint versions label the
# positive class differently - the loader prints the raw labels once on
# first run so you can confirm one of these matches.
POSITIVE_LABEL_ALIASES = {"yes", "climate", "climate-related", "1", "label_1"}


_CLASSIFIER = None  # module-level cache so we don't reload the model per call


def _load_classifier():
    """
    Builds the HF text-classification pipeline once and caches it.

    top_k=None so we get every class score (not just the winning label) -
    we need P(climate) specifically to apply our own 0.3 threshold rather
    than the model's built-in 0.5 argmax.
    """
    global _CLASSIFIER
    if _CLASSIFIER is not None:
        return _CLASSIFIER

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1

    _CLASSIFIER = pipeline(
        task="text-classification",
        model=CHECKPOINT,
        tokenizer=CHECKPOINT,
        top_k=None,
        truncation=True,
        max_length=MAX_TOKENS,
        device=device,
    )

    # confirm the positive class really is one of POSITIVE_LABEL_ALIASES on
    # the installed version
    labels = list(_CLASSIFIER.model.config.id2label.values())
    print(f"[B1] Loaded {CHECKPOINT} on {'GPU' if device == 0 else 'CPU'}. "
          f"Class labels: {labels}. "
          f"Positive class matched by: {sorted(POSITIVE_LABEL_ALIASES)}")
    return _CLASSIFIER


def _prob_climate(score_list):
    """
    Given one segment's list of {label, score} dicts, return P(climate).

    Falls back to 1 - P(no-like label) if no positive alias matches (version
    mismatch), and to the top score as a last resort. Either way it doesn't
    crash - the label print above should catch a mismatch before a full run.
    """
    for item in score_list:
        if item["label"].strip().lower() in POSITIVE_LABEL_ALIASES:
            return float(item["score"])
    for item in score_list:
        if item["label"].strip().lower() in {"no", "0", "label_0", "not_climate"}:
            return 1.0 - float(item["score"])
    return float(max(score_list, key=lambda d: d["score"])["score"])


def route_pre_model(seg):
    """
    Rule-based routing that doesn't need the model. Returns a decision dict,
    or the string "ASK_MODEL" if the segment needs to be scored.
    """
    text = seg.get("text", "")
    source_type = seg.get("source_type", "paragraph")
    language = seg.get("language", "unknown")
    quality_flag = seg.get("quality_flag", "ok")

    # shouldn't happen (A2 drops <20 chars) but never hand empty text to the model
    if not text or not text.strip():
        return {"keep": False, "climate_prob": None, "b1_route": "drop_empty"}

    if BYPASS_FILTER_FOR_TABLE_CELLS and source_type == "table_cell":
        return {"keep": True, "climate_prob": None,
                "b1_route": "keep_table_cell_bypass"}

    if language not in KEEP_LANGUAGES:
        return {"keep": False, "climate_prob": None, "b1_route": "drop_non_english"}

    if DROP_LOW_QUALITY_PARAGRAPHS and quality_flag != "ok":
        return {"keep": False, "climate_prob": None,
                "b1_route": "drop_low_quality_text"}

    return "ASK_MODEL"


def run_b1(segments, threshold=KEEP_THRESHOLD, batch_size=32):
    """
    Annotates every segment with its B1 decision and returns the full list -
    nothing gets deleted, dropped segments stay with keep=False so the run
    stays auditable.

    Each segment gets: climate_relevant, climate_prob, b1_route, b1_model,
    b1_threshold, b1_run_date.

    Use passing_segments(result) to get the ones B2 should actually read.

    Two passes: first apply rule-based routing and collect what still needs
    scoring, then run one batched model call over just those texts.
    """
    run_date = date.today().isoformat()

    decisions = [None] * len(segments)
    to_score_idx = []
    to_score_text = []
    for i, seg in enumerate(segments):
        d = route_pre_model(seg)
        if d == "ASK_MODEL":
            to_score_idx.append(i)
            to_score_text.append(seg["text"])
        else:
            decisions[i] = d

    if to_score_text:
        clf = _load_classifier()
        # one entry per input text, each entry itself a list of {label,
        # score} dicts since top_k=None
        raw = clf(to_score_text, batch_size=batch_size)
        for local_i, score_list in enumerate(raw):
            p = _prob_climate(score_list)
            keep = p >= threshold
            decisions[to_score_idx[local_i]] = {
                "keep": keep,
                "climate_prob": round(p, 4),
                "b1_route": "keep_climatebert" if keep else "drop_low_relevance",
            }

    out = []
    for seg, d in zip(segments, decisions):
        enriched = dict(seg)  # don't mutate the caller's list
        enriched["climate_relevant"] = d["keep"]
        enriched["climate_prob"] = d["climate_prob"]
        enriched["b1_route"] = d["b1_route"]
        enriched["b1_model"] = CHECKPOINT
        enriched["b1_threshold"] = threshold
        enriched["b1_run_date"] = run_date
        out.append(enriched)
    return out


def passing_segments(annotated):
    """The subset B2 should actually read: everything B1 kept."""
    return [s for s in annotated if s["climate_relevant"]]


def summarise_b1(annotated):
    """
    Counts to sanity check a B1 run before spending money on B2.

    If it keeps almost everything, the threshold's too low or the model
    isn't discriminating. Eyeball a handful of kept/dropped segments by hand
    - a real pledge in the dropped pile means the threshold is too high.
    """
    from collections import Counter
    routes = Counter(s["b1_route"] for s in annotated)
    kept = sum(1 for s in annotated if s["climate_relevant"])
    return {
        "total_segments": len(annotated),
        "kept": kept,
        "dropped": len(annotated) - kept,
        "by_route": dict(routes),
    }


if __name__ == "__main__":
    # B1 consumes the JSON that A2 saved. Point this at one firm to start.
    FIRM = "cathay_pacific"
    in_path = f"data/extracted/{FIRM}_segments.json"
    out_path = f"data/extracted/{FIRM}_b1.json"

    with open(in_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # bridge for older A2 output that doesn't emit source_type
    missing_st = sum(1 for s in segments if "source_type" not in s)
    if missing_st:
        print(f"[B1] WARNING: {missing_st}/{len(segments)} segments have no "
              f"'source_type'; defaulting them to 'paragraph'. Confirm your A2 "
              f"is the current version.")

    annotated = run_b1(segments)
    summary = summarise_b1(annotated)

    print(f"\nFirm:            {FIRM}")
    print(f"Total segments:  {summary['total_segments']}")
    print(f"Kept (climate):  {summary['kept']}")
    print(f"Dropped:         {summary['dropped']}")
    print(f"By route:        {summary['by_route']}")

    print("\n--- 3 KEPT ---")
    for s in [x for x in annotated if x["climate_relevant"]][:3]:
        print(f"[{s['b1_route']} p={s['climate_prob']}] {s['text'][:160]}")
    print("\n--- 3 DROPPED (low relevance) ---")
    for s in [x for x in annotated if x["b1_route"] == "drop_low_relevance"][:3]:
        print(f"[p={s['climate_prob']}] {s['text'][:160]}")

    # save the full annotated list (kept + dropped) for auditability - B2
    # only reads passing_segments() from it
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)
    print(f"\nAnnotated B1 output saved to {out_path}")

    # 1. confirm the printed class labels include a positive match, or add
    #    the real label to POSITIVE_LABEL_ALIASES
    # 2. hand-check kept vs dropped; lower KEEP_THRESHOLD if a real pledge
    #    got dropped
    # 3. then stage_b2_extract.py takes passing_segments() and does one LLM
    #    call per segment

# B1 is ClimateBERT only - no API key, no network call once the checkpoint is
# cached. stage_b2_extract.py is the next file and the only place an LLM
# enters: it imports passing_segments() from here, calls GPT-4o/Claude per
# segment for {tcfd_pillar, content_type, pledge_text}, adds a table-cell
# note to the prompt when relevant, and retries once on a parse failure. That
# separation is deliberate - B1 stays deterministic, only B2 introduces
# run-to-run LLM variance, which is why only B2's output logs model+run_date.
