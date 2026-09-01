# IEEE Format Manuscript

This directory contains the **IEEE Transactions** format version of:

> **Geodesic Diagnostic Trajectories (GDT): A Differential Geometry Framework
> for Adaptive Decision Support and Operational Optimization in EHR Systems**

## Document class

```latex
\documentclass[journal,10pt,twoside]{IEEEtran}
```

Targets **IEEE Transactions on Biomedical Engineering**, **IEEE Journal of
Biomedical and Health Informatics (JBHI)**, or similar double-column IEEE
journal venues.

## Files

| File | Description |
|---|---|
| `main.tex` | IEEE-format manuscript (self-contained, two-column) |
| `main.pdf` | Compiled output (generated after build) |

The bibliography is shared with the parent directory:

```latex
\bibliography{../references}   % → latex/references.bib
```

## Build

```bash
cd latex/ieee
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Requires a full **TeX Live** installation with the `IEEEtran` class:

```bash
# macOS (MacTeX)
sudo tlmgr install IEEEtran

# Ubuntu/Debian
sudo apt install texlive-publishers
```

## Key differences from `../main.tex` (journal format)

| Feature | `../main.tex` | `ieee/main.tex` |
|---|---|---|
| Document class | `article` (11pt) | `IEEEtran` (journal, 10pt) |
| Layout | Single column | **Two column** |
| Bibliography | `natbib` + `unsrtnat` | `cite` + **`IEEEtran`** |
| Author block | Plain `\author` | `\IEEEauthorblockN/A` |
| Keywords | Abstract only | `\begin{IEEEkeywords}` |
| Proof env | `amsthm` proof | **`IEEEproof`** |
| Page margins | `geometry` 1 in | IEEEtran default |

## Adapting for IEEE Conference (e.g., EMBC, BIBM)

Change the document class to conference mode:

```latex
\documentclass[conference]{IEEEtran}
```

Then remove `\IEEEpubid` if present, shorten the abstract to ≤150 words, and
target 6--8 pages.
