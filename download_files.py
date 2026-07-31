#!/usr/bin/env python3
"""Download files from DOI numbers, URLs, or local paths.

Examples:
    python download_files.py 10.1038/srep26094 ./downloads
    python download_files.py https://example.com/files ./downloads
    python download_files.py /path/to/folder ./downloads
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, List
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)
                break


def is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def is_doi(value: str) -> bool:
    return bool(DOI_RE.fullmatch(value.strip()))


def normalize_doi(doi: str) -> str:
    return doi.strip().rstrip("/")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "download"


def fetch_html(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def clean_bib_text(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace("\\'", "'").replace('\\"', '"')
    cleaned = re.sub(r"\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_bib_entries(path: str) -> List[dict[str, str]]:
    bib_path = Path(path)
    if not bib_path.exists():
        raise FileNotFoundError(f"BibTeX file not found: {path}")

    text = bib_path.read_text(encoding="utf-8", errors="ignore")
    entries: List[dict[str, str]] = []

    for match in re.finditer(r"@(?P<type>[^\s{]+)\s*\{(?P<key>[^,]+),(?P<body>.*?)\n\}", text, re.S):
        body = match.group("body")
        entry: dict[str, str] = {"key": clean_bib_text(match.group("key"))}
        for field in re.finditer(r"(?P<name>[A-Za-z0-9_:-]+)\s*=\s*\{(?P<value>.*?)\}", body, re.S):
            entry[field.group("name").lower()] = clean_bib_text(field.group("value"))
        if entry:
            entries.append(entry)

    return entries


def collect_sources(source: str) -> List[Any]:
    if is_doi(source):
        return [normalize_doi(source)]

    path = Path(source)
    if path.exists() and path.suffix.lower() == ".bib":
        return parse_bib_entries(str(path))

    parsed = urlparse(source)

    if parsed.scheme in {"http", "https"}:
        html = fetch_html(source)
        parser = LinkParser()
        parser.feed(html)
        parser.close()

        sources: List[str] = []
        seen = set()
        for href in parser.links:
            if href.startswith("#") or href.startswith("mailto:"):
                continue
            if href.startswith("/"):
                candidate = f"{parsed.scheme}://{parsed.netloc}{href}"
            elif is_http_url(href):
                candidate = href
            else:
                candidate = urljoin(source, href)

            if candidate.endswith(("/", "#")):
                continue

            if candidate in seen:
                continue
            seen.add(candidate)
            sources.append(candidate)
        return sources

    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [str(file_path) for file_path in sorted(path.iterdir()) if file_path.is_file()]

    raise FileNotFoundError(f"Source does not exist: {source}")


def guess_extension(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix:
        return suffix
    if "pdf" in content_type.lower():
        return ".pdf"
    if "html" in content_type.lower():
        return ".html"
    if "json" in content_type.lower():
        return ".json"
    return ".bin"


def build_reference_query(entry: dict[str, str]) -> str:
    title = entry.get("title", "").strip()
    authors = entry.get("author", "").strip()
    parts = [part for part in [title, authors] if part]
    return " ".join(parts)


def extract_primary_author(authors: str) -> str:
    if not authors:
        return ""
    first = authors.split(" and ")[0]
    first = first.split(",")[0].strip()
    return first.split()[0] if first.split() else ""


def search_crossref_for_doi(entry: dict[str, str]) -> str | None:
    title = entry.get("title", "").strip()
    authors = entry.get("author", "").strip()
    if not title and not authors:
        return None

    params: dict[str, str] = {"rows": "1", "select": "DOI,title,author"}
    if title:
        params["query.title"] = title
    primary_author = extract_primary_author(authors)
    if primary_author:
        params["query.author"] = primary_author

    url = "https://api.crossref.org/works?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        payload = json.load(response)

    items = payload.get("message", {}).get("items", [])
    if not items:
        return None
    doi = items[0].get("DOI")
    return doi if doi else None


def download_doi(doi: str, destination_dir: Path) -> Path:
    doi_value = normalize_doi(doi)
    doi_url = f"https://doi.org/{doi_value}"
    extension = ".bin"
    destination = destination_dir / f"{sanitize_name(doi_value)}{extension}"

    if destination.exists():
        return destination

    req = Request(doi_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")

    extension = guess_extension(final_url, content_type)
    destination = destination_dir / f"{sanitize_name(doi_value)}{extension}"
    if destination.exists():
        return destination

    destination.write_bytes(data)
    return destination


def download_reference(entry: dict[str, str], destination_dir: Path) -> Path:
    doi = entry.get("doi")
    if doi:
        return download_doi(doi, destination_dir)

    query = build_reference_query(entry)
    if not query:
        raise ValueError("Reference entry is missing title and authors")

    found_doi = search_crossref_for_doi(entry)
    if found_doi:
        return download_doi(found_doi, destination_dir)

    search_query = sanitize_name(query)
    destination = destination_dir / f"{search_query}.html"
    if destination.exists():
        return destination

    search_url = "https://scholar.google.com/scholar?q=" + urlencode({"q": query})
    req = Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")

    destination.write_text(html, encoding="utf-8")
    return destination


def download_file(source: Any, destination_dir: Path) -> Path:
    if isinstance(source, dict):
        return download_reference(source, destination_dir)

    if is_doi(source):
        return download_doi(source, destination_dir)

    parsed = urlparse(source)
    filename = Path(parsed.path).name or "downloaded_file"
    destination = destination_dir / filename

    if destination.exists():
        return destination

    if is_http_url(source):
        req = Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as response:
            data = response.read()
        destination.write_bytes(data)
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        destination.write_bytes(source_path.read_bytes())

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download files from DOI numbers, URLs, or local paths")
    parser.add_argument("sources", nargs="+", help="One or more DOI numbers, URLs, or local file/directory paths")
    parser.add_argument("output_dir", nargs="?", default="./downloads", help="Directory to save downloads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for source in args.sources:
        try:
            sources = collect_sources(source)
        except Exception as exc:  # pragma: no cover - simple CLI error handling
            print(f"Error for {source}: {exc}", file=sys.stderr)
            continue

        for item in sources:
            try:
                destination = download_file(item, output_dir)
                downloaded.append(destination)
                print(f"Downloaded: {item} -> {destination}")
            except Exception as exc:
                print(f"Failed to download {item}: {exc}", file=sys.stderr)

    if not downloaded:
        print("No files were downloaded.")
        return

    print(f"\nFinished. Downloaded {len(downloaded)} file(s) into {output_dir}")


if __name__ == "__main__":
    main()
