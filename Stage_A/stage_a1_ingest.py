"""
=============================================================================
STAGE A1 - REPORT INGESTION
TalkScore pipeline | ESG/sustainability report -> structured page records
=============================================================================
 
WHAT THIS FILE DOES
-------------------
Takes one sustainability/ESG report PDF and turns it into a clean, structured
record that the next stage (A2, segmentation) can consume. It handles the fact
that these reports are NOT uniform text: they mix three content types.
 
THE THREE CONTENT TYPES (this is decision DL-001 in the decision log)
---------------------------------------------------------------------
  Type 1  Native text   - characters stored directly, selectable/copyable.
  Type 2  Native tables - rows/cols drawn as vector lines with text cells.
  Type 3  Image objects - emission charts/infographics baked in as pictures,
                          NO text layer. This is the hard part, because most
                          emission numbers live here.
 
WHY PARTITION BY TYPE: a single text reader silently drops every Type 3 object,
which is exactly where the emissions data is. No paper in the current set
(P10 Xu & Suzuki 2026, M1 Gorovaia & Makrominas 2025) documents this step;
they both jump straight to "we extracted the text." So this stage rests on
engineering judgment, not borrowed method, and that is stated honestly in the
methods section.
 
DEPENDENCIES
------------
  pip install pdfplumber pymupdf pytesseract pillow
  # Tesseract binary also needed for Option A: apt-get install tesseract-ocr
  # Option B (vision LLM) additionally needs the openai package + an API key.
 
=============================================================================
"""
 
import pdfplumber          # Type 1 + Type 2 extraction (native text and tables)
import fitz                # PyMuPDF: pulls embedded image objects (Type 3)
import pytesseract         # Option A: free OCR fallback for image pages
from PIL import Image      # image handling for OCR / vision LLM
import io                  # in-memory byte buffers for images
import json                # parse structured output from the vision LLM
# NOTE: 'import re' was in the earlier draft but unused, so removed.
#       (Small thing, but unused imports are exactly the noise heavy commenting
#        is supposed to catch. Verify, do not trust.)
 
 
# ---------------------------------------------------------------------------
# STEP 1  -  Extract native text + native tables, one page at a time
# ---------------------------------------------------------------------------
def extract_native_text(pdf_path):
    """
    Pulls Type 1 (native text) and Type 2 (native tables) with pdfplumber.
 
    This is the obvious default, NOT a logged decision: pdfplumber is the
    standard library for native PDF text. We only log decisions where real
    alternatives were weighed (see decision_log).
 
    Returns: list of dicts, one per page:
        {page_num, text, tables, has_text}
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # extract_text() returns None on a page with no text layer,
            # so coalesce to "" to keep downstream code simple.
            text = page.extract_text() or ""
 
            # extract_tables() finds Type 2 tables (vector-ruled grids).
            # It will NOT find tables that are actually images (Type 3) -
            # those are handled later by the image path. This is the whole
            # reason native table extraction alone is insufficient.
            tables = page.extract_tables() or []
 
            pages.append({
                "page_num": i + 1,
                "text": text,
                "tables": tables,                 # list of rows, each a list of cells
                "has_text": len(text.strip()) > 50  # cheap "is this a text page" hint
            })
    return pages
 
 
# ---------------------------------------------------------------------------
# STEP 1b  -  Quality check  (decision DL-003)
# ---------------------------------------------------------------------------
def quality_check(page_dict):
    """
    Flags pages that are probably image-only or have garbled text, BEFORE the
    bad text propagates downstream and produces false-low specificity scores
    that are undetectable later.
 
    Returns: (passed: bool, reason: str)
 
    >>> VERIFY THIS, DO NOT TRUST IT <<<
    The thresholds below (50 chars, 10 words, 15% non-ASCII) are heuristic
    starting points I picked, not values from any paper. Run this on YOUR
    reports, eyeball the flagged pages, and tune the numbers. They are almost
    certainly slightly wrong for your specific documents.
    """
    text = page_dict["text"]
    word_count = len(text.split())
 
    # Garbled OCR tends to spray non-ASCII characters (mojibake). A high ratio
    # is a smell that the text layer is junk rather than real prose.
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
 
    if word_count < 10:
        # Almost no text on the page -> probably a full-page image/chart.
        return False, "likely_image_page"
    if non_ascii_ratio > 0.15:
        # Lots of weird characters -> probably garbled extraction.
        return False, "likely_garbled_ocr"
 
    return True, "ok"
 
 
# ---------------------------------------------------------------------------
# STEP 1c  -  Extract embedded image objects (Type 3) with PyMuPDF
# ---------------------------------------------------------------------------
def extract_images_from_pdf(pdf_path):
    """
    Pulls raw embedded images (Type 3) so they can be OCR'd or sent to a
    vision model. These are the emission charts and infographic tables.
 
    Returns: dict {page_num: [PIL.Image, ...]}
 
    Size filter (w>200, h>100) drops logos/icons/decorative rules so we do not
    waste OCR or API calls on a page-footer logo. Another heuristic to verify
    on your own reports - if real data tables are getting filtered out because
    they are small, lower the thresholds.
    """
    doc = fitz.open(pdf_path)
    page_images = {}
 
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)  # every image xref on the page
        page_images[page_num + 1] = []
 
        for img_ref in image_list:
            xref = img_ref[0]                       # the image's reference id
            base_image = doc.extract_image(xref)    # decode it
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))
 
            w, h = pil_image.size
            if w > 200 and h > 100:                 # keep only data-sized images
                page_images[page_num + 1].append(pil_image)
 
    doc.close()
    return page_images
 
 
# ---------------------------------------------------------------------------
# OPTION A  -  Tesseract OCR  (free, reproducible, weak on charts)
# ---------------------------------------------------------------------------
def ocr_image_tesseract(pil_image):
    """
    Free OCR. Good on clean typed tables, BAD on charts, coloured backgrounds,
    and infographics. Use when there is no API budget, and accept (and log in
    the limitations section) that chart-based numbers will be missed.
 
    Cite: Tesseract OCR (Smith, 2007), a standard tool in NLP pipelines.
    """
    return pytesseract.image_to_string(pil_image, lang="eng").strip()
 
 
# ---------------------------------------------------------------------------
# OPTION B  -  Vision LLM extraction  (decision DL-002, recommended at n=7)
# ---------------------------------------------------------------------------
def extract_table_from_image_llm(pil_image, openai_client, model="gpt-4o-mini"):
    """
    Sends the image to a multimodal model and asks for structured extraction.
    This is the only option that reliably handles charts + infographics, which
    is the dominant format for emission disclosures here.
 
    Cost sanity check (this is the DL-002 rationale, verify it for yourself):
      P10 (Xu & Suzuki 2026) spent ~USD 2,400 on GPT-4o for 141 firms.
      At n=7 the proportional cost is roughly USD 10-30. Trivial.
 
    Reproducibility caveat (the trade-off we accepted): model versions drift
    and outputs vary run to run. Mitigation: we log `model` and the run date
    so the exact configuration is recoverable. Adapted from P10's zero-shot
    GPT-4o text extraction, here pointed at image-embedded tables instead.
    """
    import base64
 
    # Encode the image as base64 so it can ride inside the JSON request body.
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
 
    # The prompt is deliberately strict: return JSON only, and an explicit
    # "no_data" sentinel, so the caller never has to parse prose.
    prompt = (
        "This image is from a corporate sustainability or ESG report.\n"
        "If it contains a table or data about emissions (Scope 1, Scope 2, "
        "Scope 3), energy use, or climate targets:\n"
        "  - Extract the data as JSON: {metric, value, unit, year, scope}\n"
        "  - If it is a chart, extract the approximate values you can read\n"
        "  - If no relevant data, return {\"result\": \"no_data\"}\n"
        "Return ONLY valid JSON, no other text."
    )
 
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }],
        max_tokens=500,
    )
 
    raw = response.choices[0].message.content
    # The model is told to return JSON, but never trust that blindly - guard
    # the parse so one malformed reply does not crash the whole ingest.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"result": "parse_error", "raw": raw}
 
 
# ---------------------------------------------------------------------------
# MASTER FUNCTION  -  full Stage A1 for one report
# ---------------------------------------------------------------------------
def ingest_report(pdf_path, use_vision_llm=False, openai_client=None,
                  vision_model="gpt-4o-mini"):
    """
    Runs the whole of Stage A1 on a single report and returns a structured
    record ready for Stage A2 (segmentation).
 
    Parameters
    ----------
    pdf_path        : path to one report PDF
    use_vision_llm  : True  -> Option B (vision LLM) on image pages
                      False -> Option A (Tesseract) on image pages
    openai_client   : an initialised OpenAI client (only needed if use_vision_llm)
    vision_model    : logged so the exact model version is recoverable later
 
    Returns
    -------
    dict:
      {
        "firm": str,
        "pdf_path": str,
        "extraction_method": "vision_llm" | "tesseract",   # for the decision log
        "pages": [ {page_num, text, tables, quality_flag, image_extractions}, ... ],
        "extraction_summary": {
            total_pages, text_pages, image_pages, flagged_pages
        }
      }
 
    HOW TO READ THE OUTPUT (do this before trusting any of it):
      Look at extraction_summary.image_pages first. That single number tells
      you whether the hard problem is real for these reports. If it is high,
      open those flagged_pages by hand and confirm they are emission tables
      and not just decorative photos. This is the "verify, don't trust" check.
    """
    firm_name = pdf_path.split("/")[-1].replace(".pdf", "")
 
    # Step 1: native text + tables
    native_pages = extract_native_text(pdf_path)
 
    # Step 1b: tag each page with a quality flag
    for page in native_pages:
        _, reason = quality_check(page)
        page["quality_flag"] = reason
 
    # Step 1c: pull embedded images once, reuse per page
    all_images = extract_images_from_pdf(pdf_path)
 
    # For flagged image pages, run the chosen extraction route on each image.
    flagged = []
    for page in native_pages:
        pnum = page["page_num"]
        page["image_extractions"] = []
 
        if page["quality_flag"] == "likely_image_page":
            flagged.append(pnum)
            for img in all_images.get(pnum, []):
                if use_vision_llm and openai_client is not None:
                    result = extract_table_from_image_llm(
                        img, openai_client, model=vision_model)
                    page["image_extractions"].append(
                        {"method": "vision_llm", "model": vision_model, "data": result})
                else:
                    ocr_text = ocr_image_tesseract(img)
                    page["image_extractions"].append(
                        {"method": "tesseract", "text": ocr_text})
 
    text_pages = sum(1 for p in native_pages if p["quality_flag"] == "ok")
 
    return {
        "firm": firm_name,
        "pdf_path": pdf_path,
        "extraction_method": "vision_llm" if use_vision_llm else "tesseract",
        "pages": native_pages,
        "extraction_summary": {
            "total_pages": len(native_pages),
            "text_pages": text_pages,
            "image_pages": len(flagged),
            "flagged_pages": flagged,
        },
    }
 
 
# ---------------------------------------------------------------------------
# USAGE  -  run this on ONE report first and read the summary before scaling
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Start with the free path (Tesseract) just to see the page breakdown.
    result = ingest_report("cathay_pacific_2025.pdf", use_vision_llm=False)
 
    print(f"Firm: {result['firm']}")
    print(f"Extraction method: {result['extraction_method']}")
    print(f"Total pages:       {result['extraction_summary']['total_pages']}")
    print(f"Native text pages: {result['extraction_summary']['text_pages']}")
    print(f"Image pages:       {result['extraction_summary']['image_pages']}")
    print(f"Flagged pages:     {result['extraction_summary']['flagged_pages']}")
 
    # Eyeball a sample of real extracted text so you can confirm it is not junk.
    if result["pages"]:
        print("\n--- Page 1 text sample ---")
        print(result["pages"][0]["text"][:500])
 
    # NEXT STEP after reading this:
    #   If image_pages is high AND those pages are emission tables, re-run with
    #   use_vision_llm=True and a real openai_client. Otherwise stay on Tesseract
    #   and note the chart-data gap in your limitations section.