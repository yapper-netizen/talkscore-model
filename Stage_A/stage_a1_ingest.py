"""
Stage A1 - report ingestion.

Turns one ESG/sustainability report PDF into a structured, page-level record
for Stage A2 (segmentation). These reports mix three content types: native
text, native (vector-drawn) tables, and image objects like emission charts
that have no text layer at all. Type 3 is the annoying one since most of the
emission numbers live there, and a plain text extractor just silently drops
it. Neither of the papers we're following (Xu & Suzuki 2026, Gorovaia &
Makrominas 2025) documents this step - they both start from "we extracted the
text" - so this part is our own engineering call, noted as such in methods.

Install: pdfplumber, pymupdf, pytesseract, pillow
Tesseract binary needed for OCR (apt-get install tesseract-ocr).
Vision LLM path additionally needs the openai package + API key.
"""

import pdfplumber          # native text + tables
import fitz                # PyMuPDF, embedded images
import pytesseract         # OCR fallback for image pages
from PIL import Image
import io
import json
import os
import glob


def extract_native_text(pdf_path):
    """
    Pulls native text + native tables per page with pdfplumber.

    Returns list of dicts: {page_num, text, tables, has_text}
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""  # None on pages with no text layer
            tables = page.extract_tables() or []  # misses tables that are actually images

            pages.append({
                "page_num": i + 1,
                "text": text,
                "tables": tables,
                "has_text": len(text.strip()) > 50
            })
    return pages


def quality_check(page_dict):
    """
    Flags pages that are probably image-only or have garbled text, so bad
    text doesn't quietly propagate into false-low specificity scores later.

    Returns (passed, reason). Thresholds below are heuristic starting points,
    not from any paper - tune them against your own reports.
    """
    text = page_dict["text"]
    word_count = len(text.split())

    # garbled OCR sprays non-ascii characters
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)

    if word_count < 10:
        return False, "likely_image_page"
    if non_ascii_ratio > 0.15:
        return False, "likely_garbled_ocr"

    return True, "ok"


def extract_images_from_pdf(pdf_path):
    """
    Pulls embedded image objects (charts/infographics) with PyMuPDF so they
    can be OCR'd or sent to a vision model.

    Returns dict {page_num: [PIL.Image, ...]}. Size filter drops logos/icons -
    lower it if real tables are small enough to get filtered out.
    """
    doc = fitz.open(pdf_path)
    page_images = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        page_images[page_num + 1] = []

        for img_ref in image_list:
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))

            w, h = pil_image.size
            if w > 200 and h > 100:
                page_images[page_num + 1].append(pil_image)

    doc.close()
    return page_images


def ocr_image_tesseract(pil_image):
    """Free OCR. Fine on clean typed tables, weak on charts and infographics."""
    return pytesseract.image_to_string(pil_image, lang="eng").strip()


def extract_table_from_image_llm(pil_image, openai_client, model="gpt-4o-mini"):
    """
    Sends the image to a multimodal model and asks for structured extraction.
    This is the only option that handles charts/infographics reliably, which
    is the dominant format for emission disclosures here.

    Cost is trivial at our scale (Xu & Suzuki spent ~$2,400 on GPT-4o for 141
    firms; at n=7 that's roughly $10-30). Model/run date get logged elsewhere
    since outputs can drift run to run.
    """
    import base64

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

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
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"result": "parse_error", "raw": raw}


def ingest_report(pdf_path, use_vision_llm=False, openai_client=None,
                  vision_model="gpt-4o-mini"):
    """
    Runs Stage A1 end to end for one report and returns a record ready for
    Stage A2.

    use_vision_llm=False uses Tesseract on image pages, True routes them to
    the vision LLM instead (needs openai_client). vision_model gets logged on
    the output so the exact version is recoverable later.

    Look at extraction_summary.image_pages first - it tells you whether the
    image-extraction problem is actually real for these reports. If it's
    high, open a few flagged_pages by hand and confirm they're emission
    tables and not just photos.
    """
    firm_name = os.path.splitext(os.path.basename(pdf_path))[0]

    native_pages = extract_native_text(pdf_path)

    for page in native_pages:
        _, reason = quality_check(page)
        page["quality_flag"] = reason

    all_images = extract_images_from_pdf(pdf_path)

    # flagged = text-quality only, independent of whether the page has images
    flagged = [p["page_num"] for p in native_pages if p["quality_flag"] == "likely_image_page"]

    # any page with an embedded image gets OCR/vision processing regardless of
    # quality_flag - a page can read fine as prose and still carry a chart in
    # the margin that the text-density heuristic would never catch
    pages_with_images = []
    for page in native_pages:
        pnum = page["page_num"]
        page_images = all_images.get(pnum, [])
        page["has_embedded_images"] = len(page_images) > 0
        page["embedded_image_count"] = len(page_images)
        page["image_extractions"] = []

        if page_images:
            pages_with_images.append(pnum)
            for img in page_images:
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
            "pages_with_embedded_images": len(pages_with_images),
            "total_embedded_images_processed": sum(p["embedded_image_count"] for p in native_pages),
        },
    }


if __name__ == "__main__":
    RAW_PDF_DIR = "data"
    OUTPUT_DIR = "data/extracted"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_paths = sorted(glob.glob(os.path.join(RAW_PDF_DIR, "*.pdf")))

    for pdf_path in pdf_paths:
        firm = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Processing {firm}...", end=" ", flush=True)

        # start with the free path just to see the page breakdown
        result = ingest_report(pdf_path, use_vision_llm=False)

        out_path = os.path.join(OUTPUT_DIR, f"{firm}_segments.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("done")
        summary = result["extraction_summary"]
        print(f"  pages={summary['total_pages']} "
              f"pages_with_images={summary['pages_with_embedded_images']} "
              f"images_processed={summary['total_embedded_images_processed']}")

    # if image_pages is high and those pages are actually emission tables,
    # re-run with use_vision_llm=True; otherwise stay on Tesseract and note
    # the chart-data gap in limitations
