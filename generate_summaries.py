#!/usr/bin/env python3
"""
generate_summaries.py

For every reference PDF under References/:
  1. Extract metadata and first-page text.
  2. Cross-check with References.txt for a canonical citation.
  3. Build a context-aware summary tied to main.tex.
  4. Save the summary in Summaries/<relative_path>.txt.
  5. Append a clean BibTeX entry to latex/references.bib.

The script processes one PDF at a time and explicitly frees memory after each file
so references never accumulate in memory/context.
"""

import os
import re
import gc
import fitz  # PyMuPDF
import difflib
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
REFS_DIR = ROOT / "References"
SUMMARIES_DIR = ROOT / "Summaries"
BIB_PATH = ROOT / "latex" / "references.bib"
REFS_TXT = ROOT / "References.txt"
MAIN_TEX = ROOT / "latex" / "main.tex"

BAD_TITLE_PREFIXES = [
    "Abstract", "Introduction", "Keywords", "Copyright", "DOI", "Vol.",
    "No.", "Downloaded from", "http", "www.", "arXiv", "Published",
    "Accepted", "Received", "Revised", "Figure", "Table", "References",
]

AFFILIATION_MARKERS = [
    "Department", "University", "Hospital", "School", "Institute",
    "Centre", "Center", "Corresponding", "Abstract", "Keywords", "DOI",
    "E-mail", "Email", "@", "http", "www", "Division", "College",
    "Laboratory", "Inc.", "Ltd.", "USA", "UK", "China", "India",
]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# Map canonical titles (normalized) to the exact BibTeX keys required by main.tex.
KEY_OVERRIDES = {
    _normalize("Scalable and accurate deep learning with electronic health records"): "rajkomar2018",
    _normalize("ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission"): "huang2019_clinicalbert",
    _normalize("Publicly Available Clinical BERT Embeddings"): "alsentzer2019",
    _normalize("Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis"): "shickel2017",
    _normalize("BioBERT: a pre-trained biomedical language representation model for biomedical text mining"): "lee2020_biobert",
    _normalize("Deep Patient: An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records"): "miotto2016",
    _normalize("Geometric Deep Learning: Going beyond Euclidean data"): "bronstein2017",
    _normalize("Poincaré Embeddings for Learning Hierarchical Representations"): "nickel2017_poincare",
    _normalize("Symmetric Positive Definite (SPD) Manifolds for Covariance Modeling"): "lopez2021",
    _normalize("A Path Towards Autonomous Machine Intelligence"): "lecun2022",
    _normalize("Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)"): "assran2023_ijepa",
    _normalize("Session-based Recommendations with Recurrent Neural Networks"): "hidasi2015",
    _normalize("Session-based Recommendation with Graph Neural Networks"): "wu2019_session",
    _normalize("High-performance medicine: the convergence of human and artificial intelligence"): "topol2019",
    _normalize("Riemannian Intent Manifolds for Session-Based Search: A Joint Embedding Predictive Architecture for Intent Forecasting"): "cma2026",
    _normalize("Continuum Memory Architectures for Long-Horizon LLM Agents"): "logan2026",
}

CONTEXT_KEYWORDS = {
    "clinical": ["electronic health record", "EHR", "clinical", "patient", "hospital", "medicine", "diagnosis", "clinician"],
    "bert_nlp": ["BERT", "clinical NLP", "natural language processing", "clinical notes", "biomedical"],
    "ehr_ml": ["deep learning", "machine learning", "predictive model", "representation", "healthcare"],
    "llm": ["large language model", "LLM", "GPT", "generative AI", "foundation model"],
    "geom": ["manifold", "Riemannian", "SPD", "symmetric positive definite", "hyperbolic", "Poincaré", "geometric deep learning", "curvature"],
    "ssl_jepa": ["JEPA", "self-supervised", "contrastive learning", "predictive architecture", "forecasting", "latent"],
    "search": ["session-based", "recommendation", "search", "intent", "conversational search", "context drift"],
    "cognitive": ["cognitive load", "NASA-TLX", "usability", "mental load", "clinician"],
}


def parse_reference_list(path: Path):
    """Parse References.txt into canonical records."""
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("part"):
            continue
        # skip descriptive prose paragraphs
        if len(line) > 100 and ("this section" in line.lower() or
                                "this segment" in line.lower() or
                                "covers" in line.lower()):
            continue
        m = re.match(r'^(.*?)\s+\((\d{4})\)\.\s*"(.*?)"\s*(.*?)\.?\s*$', line)
        if not m:
            continue
        authors, year, title, venue = m.groups()
        records.append({
            "authors": authors.strip(),
            "year": year.strip(),
            "title": title.strip(),
            "venue": venue.strip(),
            "title_norm": _normalize(title),
        })
    return records


def _title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def clean_text(text: str) -> str:
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_pdf_info(pdf_path: Path):
    """Extract metadata + cleaned text from first pages."""
    doc = fitz.open(str(pdf_path))
    meta = doc.metadata
    blocks = []
    for i, page in enumerate(doc):
        if i >= 5:
            break
        txt = page.get_text("text")
        if txt:
            blocks.append(txt)
    full_text = "\n".join(blocks)
    doc.close()
    return {
        "meta_title": (meta.get("title") or "").strip(),
        "meta_author": (meta.get("author") or "").strip(),
        "meta_subject": (meta.get("subject") or "").strip(),
        "text": clean_text(full_text + "\n" + meta.get("subject", "")),
    }


def _is_header_footer_line(text: str) -> bool:
    return bool(re.search(r"^arXiv:|^Published as|^Downloaded from|^DOI:|^http|^www|^Vol\.|^No\.|^Page |^©|^Preprint|^Submitted to|^Accepted|^Received|^Revised", text, re.I))


def extract_font_title(page):
    """
    Read the title from the first page by finding the largest non-header text.
    Returns (title, title_end_y).
    """
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return "", 0
    lines = []
    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            size = max(span.get("size", 0) for span in line.get("spans", []))
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text or size <= 0:
                continue
            if _is_header_footer_line(text):
                continue
            if re.fullmatch(r"\d+", text):
                continue
            lines.append((line["bbox"][1], line["bbox"][3], size, text))
    if not lines:
        return "", 0

    max_size = max(l[2] for l in lines)
    tol = 1.5
    title_lines = []
    # scan top-down
    for y0, y1, size, text in sorted(lines, key=lambda x: x[0]):
        if size >= max_size - tol:
            if not _looks_like_title_line(text):
                if title_lines:
                    break
                continue
            title_lines.append((y0, y1, size, text))
        elif title_lines and size >= max_size - 4:
            if _looks_like_title_line(text):
                title_lines.append((y0, y1, size, text))
            else:
                break
        elif title_lines:
            break

    if not title_lines:
        # fallback: largest remaining candidate line
        for y0, y1, size, text in sorted(lines, key=lambda x: -x[2]):
            if _looks_like_title_line(text):
                return text, y1
        return "", 0

    parts = [t[3] for t in sorted(title_lines, key=lambda x: x[0])]
    title = " ".join(parts)
    end_y = max(t[1] for t in title_lines)
    return title, end_y


def _looks_like_title_line(text: str) -> bool:
    if len(text) < 10 or len(text) > 300:
        return False
    if any(text.startswith(p) or text.lower().startswith(p.lower()) for p in BAD_TITLE_PREFIXES):
        return False
    if re.search(r"@|http|www|vol\.?\s*\d|pp\.?\s*\d|doi:\s*|arXiv:", text, re.I):
        return False
    if re.match(r"^[0-9\-.,;()]+$", text):
        return False
    return True


def _metadata_title_looks_valid(title: str) -> bool:
    if not title or len(title) < 10 or "untitled" in title.lower():
        return False
    # reject all-caps / no-lowercase strings that are often article IDs/headers
    if not any(c.islower() for c in title):
        return False
    # reject patterns like OP-CBIO190693 1234..1240
    if re.search(r"\d+\.\.\d+", title):
        return False
    return True


def guess_title(pdf_path: Path, info: dict) -> str:
    meta = info["meta_title"]
    if _metadata_title_looks_valid(meta):
        return meta

    doc = fitz.open(str(pdf_path))
    try:
        title, _ = extract_font_title(doc[0])
    finally:
        doc.close()
    if title:
        return title

    # final fallback: scan first text lines
    for line in info["text"].splitlines()[:20]:
        if _looks_like_title_line(line):
            return line
    return "Unknown title"


def _looks_like_author_fragment(text: str) -> bool:
    if len(text) < 3 or len(text) > 50:
        return False
    txt = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\s\-'.]+", "", text)
    if not re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ\-\s\.']+$", txt):
        return False
    parts = txt.split()
    if len(parts) < 2 or len(parts) > 6:
        return False
    stop_words = ("the and for not have has from with are was were been that this these those will would there their they such can had but about into through during before after")
    if any(w in stop_words for w in [w.lower() for w in parts]):
        return False
    capitalized = sum(1 for w in parts if w and w[0].isupper())
    if capitalized < len(parts) / 2:
        return False
    if re.search(r"\b(University|College|Institute|Department|Laboratory|Hospital|School|Center|Centre|Inc\.|Ltd\.|Group|Team|Research|Foundation|Corporation|DeepMind|Google|OpenAI|Microsoft|Amazon|Facebook|Meta)\b", text, re.I):
        return False
    return True


def _clean_author_fragment(text: str) -> str:
    # strip symbols/digits while keeping letters, spaces, hyphen, apostrophe, dot
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\s\-'.]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(",; ")
    if not text:
        return ""
    # drop single capital/abbreviation tokens
    words = text.split()
    if len(words) == 1 and len(words[0]) <= 2:
        return ""
    return text


def _looks_like_author_line(text: str) -> bool:
    if len(text) < 3 or len(text) > 200:
        return False
    txt = re.sub(r"[*†‡§#]", "", text)
    # email/url/abstract marker
    if re.search(r"@|http|www|Tel:|Fax:|\+\d|Page \d|^Abstract\b|^Introduction\b|^Keywords\b", txt, re.I):
        return False
    if any(marker in txt for marker in AFFILIATION_MARKERS):
        return False
    return True


def extract_authors_from_page(pdf_path: Path, title: str):
    """Attempt to extract authors from the first page using layout."""
    title_norm = _normalize(title)
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        blocks = page.get_text("dict").get("blocks", [])
        lines = []
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not text:
                    continue
                y0, y1 = line["bbox"][1], line["bbox"][3]
                lines.append((y0, y1, text))

        # determine where the first title block ends (ignore repeated citation blocks)
        title_end_y = 0
        first_title_y = None
        for y0, y1, text in sorted(lines, key=lambda x: x[0]):
            tnorm = _normalize(text)
            if tnorm and (tnorm in title_norm or title_norm in tnorm):
                if first_title_y is None:
                    first_title_y = y0
                if first_title_y is not None and y0 <= first_title_y + 120:
                    title_end_y = max(title_end_y, y1)
                # Ignore later repeats of the title (e.g., reference-format footers)
        if not title_end_y and lines:
            # heuristic: skip the largest line as title
            title_end_y = max(l[1] for l in lines[:1])

        names = []
        for y0, y1, text in sorted(lines, key=lambda x: x[0]):
            if y1 <= title_end_y + 1:
                continue
            if not _looks_like_author_line(text):
                # stop once we clearly hit the abstract/introduction body
                if re.match(r"^(Abstract|Introduction|1\.?\s*Introduction|Keywords)\b", text, re.I):
                    break
                continue
            # split a line that contains multiple authors
            parts = re.split(r",\s*|\s+and\s+|\s*;\s*", text)
            for p in parts:
                p = _clean_author_fragment(p)
                if _looks_like_author_fragment(p):
                    names.append(p)
            # stop once we hit an abstract section
            if re.match(r"^Abstract\b", text, re.I):
                break
            # safety: stop after collecting a reasonable number of authors
            if len(names) >= 15:
                break
        if names:
            return " and ".join(names)
    finally:
        doc.close()
    return ""


def clean_authors(raw: str, canonical_authors: str = "") -> str:
    if not raw:
        return canonical_authors or "Unknown author"
    raw = raw.replace(";", " and ").replace(",", ", ")
    raw = re.sub(r"[\x00-\x1f\x7f]", " ", raw)
    # strip document/artifact prefixes often embedded in metadata
    raw = re.sub(r"Version\s+[\d.]+[-\d.]*\s*", "", raw, flags=re.I)
    raw = re.sub(r"Published as a conference paper at.*?\d{4}", "", raw, flags=re.I)
    # remove emails, urls, footnote markers, degrees
    raw = re.sub(r"\S+@\S+\.\S+", "", raw)
    raw = re.sub(r"https?://\S+|www\.\S+", "", raw)
    raw = re.sub(r"[*†‡§#]+", "", raw)
    raw = re.sub(r",?\s*(?:MS|PhD|MD|MBA|MPH|MSc|BS|BSc|Dr\.?|Prof\.?)\b", "", raw, flags=re.I)
    raw = re.sub(r",\s*\d+", "", raw)
    raw = re.sub(r"(?<=[A-Za-z])\d+\b", "", raw)
    # split et al in a fixed-width manner
    if re.search(r"\bet\s+al\.?\b", raw, re.I):
        parts = re.split(r"\bet\s+al\.?\b", raw, flags=re.I)
        raw = parts[0].strip() + " et al."
    raw = re.sub(r"\s+", " ", raw).strip().strip(",; ")
    if not raw:
        return canonical_authors or "Unknown author"

    # Heuristic: if raw is dominated by institution/artifact terms or a single very long token, prefer canonical
    artifact_score = 0
    bad_tokens = ["SESSION-BASED", "RECOMMENDATIONS", "RECURRENT", "CONTRASTIVE",
                  "PREDICTIVE", "REPRESENTATION", "LEARNING", "BOOTSTRAP", "YOUR",
                  "OWN", "LATENT", "BIG", "BIRD", "LONGFORMER"]
    for t in bad_tokens:
        if t in raw.upper().split():
            artifact_score += 1
    if artifact_score >= 2 and canonical_authors:
        return canonical_authors

    # Otherwise, split on commas/and, keep fragments that look like names
    parts = re.split(r",\s*|\s+and\s+", raw)
    names = []
    for p in parts:
        p = p.strip().strip(",; ")
        if len(p) < 3:
            continue
        if re.search(r"\b(Department|University|Hospital|School|Institute|Center|Centre|United States|College|Laboratory|Inc\.|Ltd\.|Gravity)\b", p, re.I):
            continue
        if re.match(r"^[A-Za-z\-]+(\s+[A-Za-z\-]+\.?)+$", p):
            names.append(p)
    if names:
        return " and ".join(names)
    # if canonical available and raw has only one token or looks weird, use canonical
    if canonical_authors and (len(raw.split()) <= 1 or len(raw) > 200):
        return canonical_authors
    return raw


def guess_authors(pdf_path: Path, info: dict, title: str, canonical_authors: str = "") -> str:
    # Prefer PDF metadata author if it looks like author names
    meta = info["meta_author"]
    if meta and len(meta) > 4:
        cleaned = clean_authors(meta, canonical_authors)
        if cleaned and cleaned != "Unknown author":
            return cleaned
    # Try layout-based extraction using the already-determined title
    page_authors = extract_authors_from_page(pdf_path, title)
    if page_authors:
        cleaned = clean_authors(page_authors, canonical_authors)
        if cleaned and cleaned != "Unknown author":
            return cleaned
    return canonical_authors or "Unknown author"


def guess_year(text: str, meta_subject: str = "") -> str:
    search = meta_subject + "\n" + text[:6000]
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", search)]
    current_year = datetime.now().year
    plausible = [y for y in years if 1970 <= y <= current_year]
    if not plausible:
        return ""
    for w in ["published", "copyright", "publication date", "date"]:
        m = re.search(rf"{re.escape(w)}\D{{0,30}}\b((?:19|20)\d{{2}})\b", search, re.I)
        if m:
            return m.group(1)
    counter = {}
    for y in plausible:
        counter[y] = counter.get(y, 0) + 1
    return str(max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0])


def guess_venue(text: str, meta_subject: str, title: str, filename: str) -> str:
    patterns = [
        r"\b(Nature(?:\s+(?:Medicine|Digital Medicine|Computational Science|Reviews?\s+\w+|\.\.\.))?)\b",
        r"\b(npj\s+Digital\s+Medicine)\b",
        r"\b(Scientific\s+Reports)\b",
        r"\b(Bioinformatics)\b",
        r"\b(Journal\s+of\s+Biomedical\s+Informatics)\b",
        r"\b(Journal\s+of\s+the\s+American\s+Medical\s+Informatics\s+Association)\b",
        r"\b(JAMA)\b",
        r"\b(The\s+Lancet(?:\s+Digital\s+Health)?)\b",
        r"\b(BMJ)\b",
        r"\b(BMC\s+Medical\s+Informatics\s+and\s+Decision\s+Making)\b",
        r"\b(Medical\s+Image\s+Computing\s+and\s+Computer\s+Assisted\s+Intervention|MICCAI)\b",
        r"\b(Proceedings\s+of\s+[A-Za-z\s]+?)\b",
        r"\b(ACM\s+Transactions\s+on\s+[A-Za-z\s]+?)\b",
        r"\b(IEEE\s+Transactions\s+on\s+[A-Za-z\s]+?)\b",
        r"\b(NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|SIGIR|KDD|AAAI|IJCAI|WWW|WSDM|CHIL)\b",
        r"\b(arXiv:\d{4}\.\d+(?:v\d+)?)\b",
    ]
    # Canonical metadata subject often has the complete citation
    if meta_subject:
        for pat in patterns:
            m = re.search(pat, meta_subject, re.I)
            if m:
                return m.group(1).strip()
    lines = text.splitlines()
    candidate_text = "\n".join(lines[:12] + lines[-10:])
    for pat in patterns[:-2]:
        m = re.search(pat, candidate_text, re.I)
        if m:
            return m.group(1).strip()
    # arXiv id from filename/title for arXiv papers
    m = re.search(r"(arXiv:\d{4}\.\d+(?:v\d+)?)", text + "\n" + title, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(arXiv[\s:]\d{4}\.\d+(?:v\d+)?)", filename, re.I)
    if m:
        return m.group(1)
    return ""


def extract_abstract(text: str) -> str:
    # Common abstract boundaries
    m = re.search(
        r"\bAbstract\b[\s:]*\n?(.*?)(?:\n\s*(?:Keywords?|Key\s+words?|Introduction|1\.?\s*Introduction|Background|Methods|Objective|Summary)\b)",
        text, re.IGNORECASE | re.DOTALL)
    if m:
        abstract = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(abstract) > 80:
            return abstract[:1800]
    # fallback: first long block before Introduction/1. heading
    m = re.search(r"^(.{250,3000}?)\n\s*(?=Introduction|1\.?\s*Introduction|Background|2\.?)",
                  text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if m:
        candidate = re.sub(r"\s+", " ", m.group(1)).strip()
        return candidate[:1800]
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 200:
            return line[:1800]
    return ""


def identify_context(abstract: str, title: str) -> list:
    text = (abstract + " " + title).lower()
    found = []
    for bucket, kws in CONTEXT_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text:
                found.append(bucket)
                break
    return found


def match_canonical(title: str, year: str, records: list):
    """Return best canonical record if title is sufficiently similar."""
    best = None
    best_score = 0.0
    for rec in records:
        score = _title_similarity(title, rec["title"])
        # boost if years match
        if year and rec["year"] == year:
            score += 0.10
        if score > best_score:
            best_score = score
            best = rec
    # Require high similarity; be stricter when years disagree
    threshold = 0.85
    if best and year and best["year"] != year:
        # allow a one-year publication/preprint drift
        if abs(int(year) - int(best["year"])) > 1:
            threshold = 0.93
    if best_score >= threshold:
        return best
    return None


def make_summary(rel_path: str, filename: str, title: str, authors: str, year: str,
                 venue: str, abstract: str, contexts: list) -> str:
    context_map = {
        "clinical": "clinical data / EHR retrieval",
        "bert_nlp": "clinical NLP and biomedical language models",
        "ehr_ml": "deep learning for structured EHR data",
        "llm": "large language models in medicine",
        "geom": "geometric / Riemannian / SPD representations",
        "ssl_jepa": "self-supervised and JEPA-style predictive learning",
        "search": "session-based search, intent, and context drift",
        "cognitive": "cognitive load, usability, and clinician workflow",
    }
    ctx_lines = [context_map.get(c, c) for c in contexts] or ["general clinical informatics / machine learning background"]

    lines = [
        f"# Summary: {title}",
        "",
        f"**Source PDF:** `{filename}`",
        f"**Relative path:** `{rel_path}`",
        f"**Authors:** {authors or 'Not extracted'}",
        f"**Year:** {year or 'Not extracted'}",
        f"**Venue:** {venue or 'Not extracted'}",
        "",
        "## Manuscript Context",
        "This reference contributes to the following themes in *Continuum Memory Architecture (CMA) for Clinical Search*:",
    ]
    for ctx in ctx_lines:
        lines.append(f"- {ctx}")
    lines += [
        "",
        "## Extracted Abstract / Overview",
        abstract or "No abstract or substantial opening paragraph could be automatically extracted from the PDF.",
        "",
        "## Relevance to CMA / main.tex",
    ]
    relevance_bits = []
    joined = " ".join(contexts).lower()
    if "geom" in contexts or "SPD" in title.upper() or "manifold" in title.lower():
        relevance_bits.append("Provides geometric or Riemannian machinery related to the SPD intent representations and curvature-aware gating used by CMA.")
    if "ssl_jepa" in contexts or "JEPA" in title.upper():
        relevance_bits.append("Aligns with the JEPA predictor component that forecasts next-intent states for anticipatory prefetch in CMA.")
    if "search" in contexts or "session" in title.lower() or "intent" in title.lower():
        relevance_bits.append("Supports the session-based search motivation and the problem of latent-context pollution / context drift addressed by the CAI gate.")
    if "clinical" in contexts or "EHR" in title.upper() or "health record" in title.lower():
        relevance_bits.append("Directly informs the clinical EHR retrieval task and benchmark design in main.tex.")
    if "bert_nlp" in contexts or "BERT" in title.upper():
        relevance_bits.append("Supports the clinical NLP foundation used to encode notes/queries in the CMA clinical pipeline.")
    if "llm" in contexts:
        relevance_bits.append("Provides context on contemporary generative clinical AI and decision-support safety considerations relevant to CMA deployment.")
    if "cognitive" in contexts:
        relevance_bits.append("Relates to the NASA-TLX and usability outcomes measured in the CMA evaluation.")
    if not relevance_bits:
        relevance_bits.append("Provides background or methodology that may contextualize the CMA clinical retrieval evaluation.")

    for bit in relevance_bits:
        lines.append(f"- {bit}")
    lines.append("")
    return "\n".join(lines)


def assign_key(title: str, authors: str, year: str, used_keys: set) -> str:
    """Use a predefined key when the title matches a main.tex citation."""
    norm = _normalize(title)
    base = KEY_OVERRIDES.get(norm)
    # Also allow partial matches for multi-line titles that were truncated
    if not base:
        for norm_key, mapped_key in KEY_OVERRIDES.items():
            if norm in norm_key or norm_key in norm:
                if len(norm) >= 15 and len(norm_key) >= 15:
                    base = mapped_key
                    break
    if base:
        key = base
        suffix = ""
        while key in used_keys:
            suffix = chr(ord(suffix or "a") + 1) if suffix else "a"
            key = f"{base}{suffix}"
        used_keys.add(key)
        return key
    return make_bib_key(authors, year, title, used_keys)


def make_bib_key(authors: str, year: str, title: str, used_keys: set) -> str:
    first = authors.split(" and ")[0] if " and " in authors else authors
    # prefer last name
    last = first.split()[-1] if first.split() else ""
    last = re.sub(r"[^a-zA-Z]", "", last).lower()
    if not last:
        last = re.sub(r"[^a-zA-Z]", "", title.split()[0]).lower() if title else "ref"
    yr = year[-2:] if year else "00"
    base = f"{last}{yr}"
    key = base
    suffix = ""
    while key in used_keys:
        # a, b, c, ...
        suffix = chr(ord(suffix or "a") + 1) if suffix else "a"
        key = f"{base}{suffix}"
    used_keys.add(key)
    return key


def normalize_author_string(authors: str) -> str:
    # Remove leading/trailing "and" introduced by parsing separators
    authors = re.sub(r"^\s*and\s+|\s+and\s*$", "", authors, flags=re.I)
    # Collapse repeated "and and"
    authors = re.sub(r"\s+and\s+and\s+", " and ", authors, flags=re.I)
    # Normalize whitespace
    authors = re.sub(r"\s+", " ", authors).strip()
    return authors


def bib_escape(s: str) -> str:
    s = s.replace("{", "{{").replace("}", "}}").replace("&", "\\&")
    # normalize spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_bib_entry(key: str, title: str, authors: str, year: str, venue: str) -> str:
    entry_type = "article"
    v = venue or ""
    if re.search(r"arXiv|preprint|OpenReview|techrxiv|ssrn|bioRxiv", v, re.I):
        entry_type = "misc"
    elif re.search(r"Proceedings|NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|SIGIR|KDD|AAAI|IJCAI|MICCAI|CHIL|WWW|WSDM|ICDM", v, re.I):
        entry_type = "inproceedings"
    elif re.search(r"Journal|Transactions|Medicine|npj|Nature|Science|Bioinformatics|Reports|BMJ|Lancet|BMC", v, re.I):
        entry_type = "article"
    else:
        # if title looks like survey/review and no venue, default misc
        entry_type = "misc"

    lines = [f"@{entry_type}{{{key},"]
    lines.append(f"  title={{{bib_escape(title)}}},")
    lines.append(f"  author={{{bib_escape(normalize_author_string(authors))}}},")
    if entry_type == "article":
        lines.append(f"  journal={{{bib_escape(v)}}},")
    elif entry_type == "inproceedings":
        lines.append(f"  booktitle={{{bib_escape(v)}}},")
    elif entry_type == "misc":
        if v:
            lines.append(f"  howpublished={{{bib_escape(v)}}},")
    if year:
        lines.append(f"  year={{{year}}},")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def pdf_summary_name(pdf_path: Path) -> str:
    rel = pdf_path.relative_to(REFS_DIR)
    stem = str(rel.with_suffix(""))
    # sanitize for filesystem
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem + ".txt"


def process_all():
    SUMMARIES_DIR.mkdir(exist_ok=True)
    BIB_PATH.parent.mkdir(exist_ok=True)

    # fresh outputs
    if SUMMARIES_DIR.exists():
        for f in SUMMARIES_DIR.glob("*.txt"):
            f.unlink()
    with open(BIB_PATH, "w", encoding="utf-8") as bf:
        bf.write("% References generated from References/ PDFs (recursive).\n")
        bf.write("% Manual review recommended for author/venue details.\n\n")

    records = parse_reference_list(REFS_TXT)
    print(f"Loaded {len(records)} canonical records from References.txt")

    pdfs = sorted(p for p in REFS_DIR.rglob("*.pdf") if p.is_file())
    # Optionally also include the MECO/CMA paper that lives at repo root.
    meco_pdf = ROOT / "MECO_2026_Corrected_paper_14.pdf"
    if meco_pdf.exists() and meco_pdf not in pdfs:
        pdfs.append(meco_pdf)

    print(f"Found {len(pdfs)} PDFs to process")

    used_keys = set()
    seen_titles = set()
    errors = []

    def handle_pdf(pdf: Path, idx: int, total: int, rel_for_summary: str = ""):
        if rel_for_summary:
            rel = rel_for_summary
        elif pdf.is_relative_to(REFS_DIR):
            rel = pdf.relative_to(REFS_DIR)
        else:
            rel = pdf.name
        print(f"[{idx}/{total}] {rel}")
        try:
            info = extract_pdf_info(pdf)
            title = guess_title(pdf, info)
            year = guess_year(info["text"], info["meta_subject"])

            canonical = match_canonical(title, year, records)
            canonical_authors = canonical["authors"] if canonical else ""
            canonical_title = canonical["title"] if canonical else ""
            canonical_year = canonical["year"] if canonical else ""
            canonical_venue = canonical["venue"] if canonical else ""

            # Only let canonical metadata overrule extracted values when it is very confident
            if canonical_title:
                title = canonical_title
            if canonical_year:
                year = canonical_year

            authors = guess_authors(pdf, info, title, canonical_authors)

            # venue: prefer canonical or metadata subject-based
            venue = canonical_venue or guess_venue(info["text"], info["meta_subject"], title, pdf.name)

            abstract = extract_abstract(info["text"])
            contexts = identify_context(abstract, title)

            # avoid duplicate entries from duplicated PDFs (e.g., print + arXiv copies)
            title_norm = _normalize(title)
            if title_norm in seen_titles:
                print(f"    (duplicate title; skipping BibTeX entry)")
            else:
                seen_titles.add(title_norm)
                key = assign_key(title, authors, year, used_keys)
                entry = make_bib_entry(key, title, authors, year, venue)
                with open(BIB_PATH, "a", encoding="utf-8") as bf:
                    bf.write(entry)

            summary_path = SUMMARIES_DIR / (pdf_summary_name(pdf) if pdf.is_relative_to(REFS_DIR) else re.sub(r"[^A-Za-z0-9_\-]+", "_", pdf.stem) + ".txt")
            summary = make_summary(str(rel), pdf.name, title, authors, year, venue,
                                   abstract, contexts)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)

            # explicit per-reference cleanup
            del info, title, authors, year, venue, abstract, contexts, canonical
            del summary, summary_path
            if 'entry' in locals():
                del entry, key
            gc.collect()

        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append((str(rel), str(e)))
            err_path = SUMMARIES_DIR / pdf_summary_name(pdf) if pdf.is_relative_to(REFS_DIR) else SUMMARIES_DIR / (re.sub(r"[^A-Za-z0-9_\-]+", "_", pdf.name) + ".txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"# Summary: {pdf.name}\n\n**Error:** Could not process PDF: {e}\n")

    total = len(pdfs)
    for idx, pdf in enumerate(pdfs, 1):
        handle_pdf(pdf, idx, total)

    # If Lopez et al. 2021 SPD paper is missing from the PDF set, add it from References.txt
    # because main.tex cites lopez2021.
    if "lopez2021" not in used_keys:
        lopez = next((r for r in records if _title_similarity(r["title"], "Symmetric Positive Definite (SPD) Manifolds for Covariance Modeling") > 0.95), None)
        if lopez:
            with open(BIB_PATH, "a", encoding="utf-8") as bf:
                bf.write(make_bib_entry(
                    "lopez2021",
                    lopez["title"],
                    lopez["authors"].replace(" and ", " "),
                    lopez["year"],
                    lopez["venue"]))
            used_keys.add("lopez2021")
            print("Added Lopez 2021 SPD entry from References.txt")

    print("\nDone.")
    print(f"  Summaries:  {len(list(SUMMARIES_DIR.glob('*.txt')))}")
    print(f"  BibTeX entries: {len(used_keys)}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for rel, err in errors[:10]:
            print(f"    - {rel}: {err}")


if __name__ == "__main__":
    process_all()
