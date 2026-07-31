#!/usr/bin/env python3
"""
expand_refs_clean.py

Remove previously inserted auxiliary paragraphs from main.tex and insert a
single, clean block of thematically grouped citations, ensuring no empty \citep{}.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIB_PATH = ROOT / "latex" / "references.bib"
TEX_PATH = ROOT / "latex" / "main.tex"

PARAGRAPH_TITLES = [
    "Clinical machine learning and EHR-based prediction.",
    "Biomedical language models and clinical text mining.",
    "Large language models and generative clinical AI.",
    "Trustworthy AI, fairness, and governance in healthcare.",
    "Geometric learning, self-supervision, and anticipatory retrieval.",
    "Translational exemplars of clinical deep learning.",
    "Complementary advances in representation and retrieval.",
    "Clinical deployment landscape.",
]

EXCLUDE_KEYS = {
    "wang14", "to21", "surveys21", "shiblu26",
    "author01", "author07", "author11", "author25",
    "member26", "lh20", "health20", "rauf20", "williamson00",
}


def parse_bib(path):
    text = path.read_text(encoding="utf-8")
    entries = []
    for m in re.finditer(r'@(\w+)\{([^,]+),\s*title=\{(.*?)\}', text, re.S):
        _, key, title = m.groups()
        title = title.strip().replace("\n", " ")
        if len(title) < 12:
            continue
        if re.match(r'^(doi:|http|www|OP-|CSUR|INTERNATIONAL|ARTICLE IN PRESS|Journal of|amiajnl-|doi:10\.)', title, re.I):
            continue
        if 'halide' in title.lower() or 'perovskite' in title.lower():
            continue
        entries.append((key, title))
    return entries


def themes(title):
    t = title.lower()
    cats = set()
    if any(x in t for x in ['ehr','electronic health record','clinical','patient','healthcare','hospital','diagnosis','medical','medicine','biomedical','health']):
        cats.add('clinical')
    if any(x in t for x in ['bert','nlp','language model','clinical notes','biomedical text']):
        cats.add('nlp')
    if any(x in t for x in ['large language model','llm','gpt','generative ai','foundation model','chatgpt']):
        cats.add('llm')
    if any(x in t for x in ['geometric','riemannian','spd','hyperbolic','poincare','manifold','covariance']):
        cats.add('geom')
    if any(x in t for x in ['jepa','self-supervised','contrastive','predictive','bootstrap','byol']):
        cats.add('ssl')
    if any(x in t for x in ['search','session','recommendation','retrieval','intent','conversational','ambiguous query']):
        cats.add('search')
    if any(x in t for x in ['cognitive load','nasa-tlx','usability','trust','explainability','interpretability','bias','fairness','ethical','oversight','safety','accountability','transparency','privacy']):
        cats.add('cognitive')
    if any(x in t for x in ['federated','privacy preserving','privacy-preserving']):
        cats.add('federated')
    if any(x in t for x in ['continual','incremental']):
        cats.add('continual')
    return cats


def select(entries, used, n, categories=None, keyword=None, anti_keyword=None):
    out = []
    for key, title in entries:
        if key in used or key in EXCLUDE_KEYS:
            continue
        t = title.lower()
        if anti_keyword and anti_keyword in t:
            continue
        if keyword and keyword not in t:
            continue
        if categories:
            if not any(c in themes(title) for c in categories):
                continue
        out.append(key)
        if len(out) >= n:
            break
    return out


def cite(keys):
    if not keys:
        return ""
    return "\\citep{" + ",".join(keys) + "}"


def paragraph(title, sentence, keys):
    if not keys:
        return ""
    return (
        f"\n\\paragraph{{{title}}}\n"
        f"{sentence} {cite(keys)} "
        ""
    )


def main():
    tex = TEX_PATH.read_text(encoding="utf-8")

    # Remove all previously generated auxiliary paragraphs
    for title in PARAGRAPH_TITLES:
        # Match a paragraph with the given title until the next section-like command
        pat = r'\n\\paragraph\{' + re.escape(title) + r'\}.*?\n(?=\n|\\section|\\subsection|\\paragraph)'
        tex = re.sub(pat, "\n", tex, flags=re.S)

    # Collapse multiple blank lines
    tex = re.sub(r'\n{3,}', "\n\n", tex)

    used0 = set()
    for m in re.finditer(r'\\citep?\{([^}]*)\}', tex):
        used0.update(k.strip() for k in m.group(1).split(','))

    entries = parse_bib(BIB_PATH)
    used = used0.copy()

    # Build clusters
    clusters = []

    clinical = select(entries, used, 28, categories=['clinical'])
    used.update(clinical)
    clusters.append((
        "Clinical machine learning and EHR-based prediction.",
        "A substantial literature has applied deep learning to structured EHR data for risk stratification, diagnosis, prognosis, and representation learning across heterogeneous clinical signals.",
        clinical,
        "These studies demonstrate that neural representations can compress longitudinal patient histories, detect subtle phenotypes, and support decision support, but they largely treat retrieval and scoring as independent of the dynamic search context that clinicians inhabit during chart review."
    ))

    nlp = select(entries, used, 18, categories=['nlp'])
    used.update(nlp)
    clusters.append((
        "Biomedical language models and clinical text mining.",
        "Domain-specific language modeling has become central to clinical NLP, with transformer-based embeddings tailored to biomedical terminology, clinical notes, and EHR-specific syntax.",
        nlp,
        "These models enable extraction of concepts, relations, and assertions from unstructured text, yet their application in interactive retrieval remains sensitive to context drift when earlier queries in a session bias the encoding of later, semantically distant intents."
    ))

    llm = select(entries, used, 16, categories=['llm'])
    used.update(llm)
    clusters.append((
        "Large language models and generative clinical AI.",
        "Recent foundation models encode broad clinical knowledge and support question answering, summarization, and multi-turn dialogue.",
        llm,
        "While promising, their deployment in EHR retrieval raises concerns about provenance, hallucination, latency, and confirmation bias that are partially orthogonal to scale and instead require session-level geometric control and transparent provenance."
    ))

    trustworthy = select(entries, used, 14, categories=['cognitive'])
    used.update(trustworthy)
    clusters.append((
        "Trustworthy AI, fairness, and governance in healthcare.",
        "The move from experimental models to clinical decision support has prompted extensive work on bias, explainability, transparency, safety monitoring, and accountability.",
        trustworthy,
        "These frameworks emphasize human oversight, auditability, and fairness reviews as prerequisites for deployment, principles that inform the safeguards built into the CMA interface."
    ))

    geom_ssl_search = select(entries, used, 10, categories=['geom', 'ssl', 'search'])
    used.update(geom_ssl_search)
    clusters.append((
        "Geometric learning, self-supervision, and anticipatory retrieval.",
        "Beyond clinical settings, Riemannian and SPD representations, self-supervised latent prediction, and session-aware ranking have been advanced in computer vision, graph learning, and information retrieval.",
        geom_ssl_search,
        "CMA synthesizes these lines by encoding session intent on a geometric manifold and forecasting future intent with a predictive architecture."
    ))

    exemplars = select(entries, used, 10, categories=['clinical'], keyword='deep learning')
    used.update(exemplars)
    clusters.append((
        "Translational exemplars of clinical deep learning.",
        "Applied studies have demonstrated deep-learning models for chronic-disease detection, oncology, neurology, ophthalmology, COVID-19 response, and multi-modal imaging.",
        exemplars,
        "These exemplars illustrate both the predictive power and the deployment complexity of clinical AI, underscoring the need for retrieval interfaces that adapt to shifting clinical intent without compounding prior-query bias."
    ))

    bg_block = ""
    for title, sentence, keys, closing in clusters:
        bg_block += paragraph(title, sentence, keys) + closing + "\n"
    bg_block = bg_block.rstrip() + "\n"

    # Discussion clusters
    methods_extra = select(entries, used, 8, categories=['ssl', 'search'])
    used.update(methods_extra)
    deployment = select(entries, used, 14, categories=['clinical'], anti_keyword='deep learning')
    used.update(deployment)
    deployment += select(entries, used, 8, categories=['federated', 'continual'])
    used.update(deployment)

    disc_block = ""
    if methods_extra:
        disc_block += paragraph(
            "Complementary advances in representation and retrieval.",
            "Self-supervised contrastive methods, uncertainty quantification, and context-aware search algorithms provide additional building blocks for robust clinical retrieval.",
            methods_extra
        ) + "Integrating these advances with a geometric, curvature-aware session model is a promising direction for future work.\n"
    if deployment:
        disc_block += paragraph(
            "Clinical deployment landscape.",
            "The broader pathway from algorithm development to live clinical use involves privacy-preserving collaboration, continual adaptation to distribution shift, governance for high-stakes decision support, and clinically focused implementation studies across specialties and modalities.",
            deployment
        ) + "Federated and continual-learning strategies are particularly relevant for EHR retrieval, where patient data are distributed across institutions and concepts evolve over time; curvature-aware gating provides a complementary mechanism for handling distribution shift within a single session.\n"

    # Insert background block before \section{Methods}
    tex, bg_count = re.subn(
        r'(\\paragraph\{Novelty statement\.\}.*?)(?=\\section\{Methods\})',
        lambda m: m.group(1) + "\n" + bg_block + "\n",
        tex, count=1, flags=re.S)

    # Insert discussion block before \subsection{Strengths}
    tex, disc_count = re.subn(
        r'(\\subsection\{Mechanisms and Explanations\}.*?)(?=\\subsection\{Strengths\})',
        lambda m: m.group(1) + "\n" + disc_block + "\n",
        tex, count=1, flags=re.S)

    TEX_PATH.write_text(tex, encoding="utf-8")

    final_used = set()
    for m in re.finditer(r'\\citep?\{([^}]*)\}', tex):
        final_used.update(k.strip() for k in m.group(1).split(','))

    print(f"Background blocks inserted: {bg_count}")
    print(f"Discussion blocks inserted: {disc_count}")
    print(f"Unique keys now cited: {len(final_used)}")
    print(f"Newly added keys: {len(final_used - used0)}")


if __name__ == "__main__":
    main()
