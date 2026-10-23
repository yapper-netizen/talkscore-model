"""
Stage B2/B3 - LLM structured extraction.

Takes the segments B1 kept and makes one LLM call per segment, returning:
  - tcfd_pillar   which TCFD pillar the text belongs to
  - content_type  pledge / kpi / boilerplate / etc
  - pledge_text   the pledge itself as a clean string, or null

This replaced an earlier plan (a regex pillar tagger + hand-built router) -
the rules kept multiplying without converging across firms, so the
classification job moved to one LLM call that absorbs the structural
variation. B1 stays in front as a cheap ClimateBERT sieve; this file is the
only place an LLM (and an API key) enters.

A2 -> B1 (ClimateBERT sieve) -> THIS FILE (B2/B3 LLM) -> Stage C/D
Input is passing_segments() from stage_b1_relevance. Output is a flat list
of pledge records for Stage C/D.

Install: openai (if PROVIDER="openai") or anthropic (if "anthropic"). API
key comes from the environment (OPENAI_API_KEY / ANTHROPIC_API_KEY), never
written in code. Unlike B1 this step costs money and isn't fully
deterministic - should stay under ~$1 at n=7, ~$30 at n=200. Every record
logs model + run_date so variance is at least traceable.
"""

import os
import json
from datetime import date


# flip this one line to run the same prompt through each provider for the
# pilot comparison - "openai" or "anthropic"
PROVIDER = "openai"

# update the anthropic string to whichever Sonnet is actually available
# before running the pilot - the exact string gets logged on every record
MODEL = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
}

# temperature 0 for the most stable output an LLM gives, which still isn't
# fully deterministic - hence logging run_date and model
TEMPERATURE = 0.0

# leave modality (will/shall vs ongoing vs past) to Stage D for now. Flip to
# True to have the LLM return a "modality" field in the same call instead.
EXTRACT_MODALITY = False

# fixed label sets - the prompt pins the model to exactly these strings so
# downstream code can trust the values
PILLARS = ["Governance", "Strategy", "Risk Management", "Metrics and Targets", "Other"]
CONTENT_TYPES = ["pledge", "kpi", "time_series_kpi", "materiality_matrix", "boilerplate"]


# keep this string stable - bump PROMPT_VERSION if it changes so a later run
# is distinguishable from an earlier one in the saved output
PROMPT_VERSION = "b2-v1"

SYSTEM_PROMPT = f"""You are a precise information extractor for corporate climate disclosures.
You read one short segment from an ESG or sustainability report and return ONLY a JSON object, nothing else.

Return exactly these fields:
- "tcfd_pillar": one of {PILLARS}. Pick the TCFD pillar the segment is about. Use "Other" only if none of the four fit.
- "content_type": one of {CONTENT_TYPES}.
    "pledge" = a forward-looking commitment, especially with a target, mechanism, or date.
    "kpi" = a reported metric or indicator for a single period, or the name of one.
    "time_series_kpi" = a metric reported across three or more years.
    "materiality_matrix" = a stakeholder-importance grid of ESG topics.
    "boilerplate" = generic aspiration or context with no firm-specific, actionable commitment.
- "pledge_text": if content_type is "pledge", the pledge as a clean single string; otherwise null. If a non-pledge segment still contains a specific target clause, put that clause here; otherwise null.

Rules:
- Base the answer only on the segment text. Do not invent detail that is not present.
- A vague statement of support or alignment with national or global goals, with no firm-specific target or action, is "boilerplate", not "pledge".
- Return valid JSON with those exact keys and nothing outside the JSON."""

TABLE_CELL_NOTE = ("This segment is a single TABLE CELL, not a prose sentence. "
                   "It may be a fragment or just an indicator name. Judge it as a table cell.")


def build_user_prompt(seg):
    """
    Builds the per-segment user message: the A2 structural pillar as a hint
    (the model may agree or override it - resolution happens in code, not
    here), the source type, and the table-cell note when relevant.
    """
    lines = []
    if seg.get("source_type") == "table_cell":
        lines.append(TABLE_CELL_NOTE)
    lines.append(f'Structural pillar hint from report headings: {seg.get("pillar_from_structure", "Other")}')
    lines.append(f'Source type: {seg.get("source_type", "paragraph")}')
    lines.append("Segment text:")
    lines.append(seg["text"])
    return "\n".join(lines)


_CLIENT = None  # cached client so we don't re-create it per segment


def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if PROVIDER == "openai":
        from openai import OpenAI
        _CLIENT = OpenAI()
    elif PROVIDER == "anthropic":
        from anthropic import Anthropic
        _CLIENT = Anthropic()
    else:
        raise ValueError(f"Unknown PROVIDER: {PROVIDER}")
    return _CLIENT


def _call_llm(system_prompt, user_prompt):
    """Sends the prompt to the configured provider and returns the raw text."""
    client = _get_client()
    if PROVIDER == "openai":
        # json_object response format removes most parse failures up front
        resp = client.chat.completions.create(
            model=MODEL["openai"],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content
    else:  # anthropic
        resp = client.messages.create(
            model=MODEL["anthropic"],
            max_tokens=512,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text


def _parse_and_validate(raw):
    """
    Turns the raw model text into a validated dict, or raises ValueError.

    Strips stray code fences, parses JSON, checks the three keys exist and
    that tcfd_pillar/content_type are from the allowed sets, and coerces an
    empty/missing pledge_text to None.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"): text.rfind("}") + 1]

    data = json.loads(text)  # raises JSONDecodeError, caught by caller

    pillar = data.get("tcfd_pillar")
    ctype = data.get("content_type")
    if pillar not in PILLARS:
        raise ValueError(f"tcfd_pillar not in allowed set: {pillar!r}")
    if ctype not in CONTENT_TYPES:
        raise ValueError(f"content_type not in allowed set: {ctype!r}")

    pledge = data.get("pledge_text")
    if isinstance(pledge, str) and not pledge.strip():
        pledge = None

    out = {"tcfd_pillar": pillar, "content_type": ctype, "pledge_text": pledge}
    if EXTRACT_MODALITY:
        out["modality"] = data.get("modality")
    return out


def extract_one(seg, max_retries=1):
    """
    Runs the LLM call for one segment, with one retry on parse/validation
    failure. Falls back to content_type "parse_error" instead of crashing so
    one bad segment doesn't kill a whole firm's run - the failure stays
    visible in the output.
    """
    user_prompt = build_user_prompt(seg)
    last_err = None
    for _ in range(max_retries + 1):
        try:
            raw = _call_llm(SYSTEM_PROMPT, user_prompt)
            return _parse_and_validate(raw)
        except Exception as e:
            last_err = e
    return {"tcfd_pillar": "Other", "content_type": "parse_error",
            "pledge_text": None, "error": str(last_err)}


def resolve_pillar(structural_pillar, llm_pillar):
    """
    A2's structural pillar wins when the text sits under a real report
    heading. When A2 says "Other" (un-headed text, e.g. a table cell), fall
    back to the LLM's guess.
    """
    if structural_pillar and structural_pillar != "Other":
        return structural_pillar
    return llm_pillar


def run_b2(passing, verbose=True):
    """
    Takes passing_segments() from B1 and returns a flat list of pledge
    records:

      {
        "firm": str, "page_num": int, "para_idx": int, "source_type": str,
        "pillar": str,               # resolved
        "pillar_from_structure": str, "tcfd_pillar_llm": str,  # kept for audit
        "content_type": str, "pledge_text": str|None,
        "model": str, "run_date": str, "prompt_version": str
      }

    Sequential, one API call per segment. Could be made concurrent at scale
    but sequential is fine and easier to debug at n=200.
    """
    run_date = date.today().isoformat()
    model_str = f'{PROVIDER}:{MODEL[PROVIDER]}'
    records = []
    for i, seg in enumerate(passing):
        llm = extract_one(seg)
        pillar = resolve_pillar(seg.get("pillar_from_structure", "Other"), llm["tcfd_pillar"])
        rec = {
            "firm": seg.get("firm"),
            "page_num": seg.get("page_num"),
            "para_idx": seg.get("para_idx"),
            "source_type": seg.get("source_type", "paragraph"),
            "pillar": pillar,
            "pillar_from_structure": seg.get("pillar_from_structure", "Other"),
            "tcfd_pillar_llm": llm["tcfd_pillar"],
            "content_type": llm["content_type"],
            "pledge_text": llm["pledge_text"],
            "model": model_str,
            "run_date": run_date,
            "prompt_version": PROMPT_VERSION,
        }
        if EXTRACT_MODALITY:
            rec["modality"] = llm.get("modality")
        records.append(rec)
        if verbose and (i + 1) % 20 == 0:
            print(f"[B2] {i + 1}/{len(passing)} segments done")
    return records


def summarise_b2(records):
    """Quick counts to eyeball before trusting the output or scaling up."""
    from collections import Counter
    return {
        "total": len(records),
        "by_content_type": dict(Counter(r["content_type"] for r in records)),
        "by_pillar": dict(Counter(r["pillar"] for r in records)),
        "pledges_with_text": sum(1 for r in records if r["pledge_text"]),
        "parse_errors": sum(1 for r in records if r["content_type"] == "parse_error"),
    }


if __name__ == "__main__":
    from stage_b1_relevance import passing_segments

    FIRM = "cr_power"
    with open(f"data/extracted/{FIRM}_b1.json", "r", encoding="utf-8") as f:
        annotated = json.load(f)

    passing = passing_segments(annotated)
    print(f"[B2] {len(passing)} segments passed B1, sending to {PROVIDER}:{MODEL[PROVIDER]}")

    records = run_b2(passing)
    summary = summarise_b2(records)
    print("Summary:", summary)

    print("\n--- first 3 pledges ---")
    for r in [x for x in records if x["content_type"] == "pledge"][:3]:
        print(f"[{r['pillar']}] p{r['page_num']}: {r['pledge_text']}")

    out_path = f"data/extracted/{FIRM}_pledges.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nPledge records saved to {out_path}")

    # 1. run the 20-segment pilot: PROVIDER="openai" then "anthropic" on the
    #    same 20 mixed segments, compare parse_errors and hand-check
    #    pillar/content_type, log the winner
    # 2. Stage C/D consume this pledges list; Stage D scores the pledge_text
