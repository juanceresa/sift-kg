"""Parse the AstraZeneca Ontoverse Zotero RDF into a sift-ready manifest.

Source: https://github.com/AstraZeneca/ontoverse-kg-choreographer/blob/main/zotero_library/OntoverseSandbox.rdf
"""
from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "z": "http://www.zotero.org/namespaces/export#",
    "dcterms": "http://purl.org/dc/terms/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "bib": "http://purl.org/net/biblio#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "prism": "http://prismstandard.org/namespaces/1.2/basic/",
}
RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"


def _txt(node, path: str) -> str:
    el = node.find(path, NS)
    return (el.text or "").strip() if el is not None and el.text else ""


def _doi(art) -> str:
    for ident in art.findall("dc:identifier", NS):
        if ident.text and "DOI" in ident.text:
            return ident.text.replace("DOI", "").strip()
    ji = art.find("dcterms:isPartOf/bib:Journal/dc:identifier", NS)
    if ji is not None and ji.text and "DOI" in ji.text:
        return ji.text.replace("DOI", "").strip()
    return ""


def _authors(art) -> list[str]:
    out = []
    for p in art.findall("bib:authors/rdf:Seq/rdf:li/foaf:Person", NS):
        s = _txt(p, "foaf:surname")
        g = _txt(p, "foaf:givenName")
        if s or g:
            out.append(f"{g} {s}".strip())
    return out


def _pdf_url(url: str, doi: str) -> str:
    """Best-effort guess for an open PDF URL."""
    m = re.search(r"arxiv\.org/abs/([\d\.v]+)", url, re.I)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    m = re.search(r"arxiv\.org/abs/([\w\.\-/]+)", url, re.I)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    if "biorxiv.org" in url or "medrxiv.org" in url:
        u = url.replace("http://", "https://")
        u = re.sub(r"v\d+$", "", u)
        if "/lookup/doi/" in u and doi:
            return f"https://www.biorxiv.org/content/{doi}v1.full.pdf"
        if u.endswith(".full.pdf"):
            return u
        return u + ".full.pdf" if "/content/" in u else u
    return ""


def parse(rdf_path: Path):
    tree = ET.parse(rdf_path)
    root = tree.getroot()

    articles = []
    for art in root.findall("bib:Article", NS):
        aid = art.attrib.get(RDF_ABOUT, "")
        articles.append(
            {
                "id": aid,
                "title": _txt(art, "dc:title"),
                "journal": _txt(art, "dcterms:isPartOf/bib:Journal/dc:title"),
                "date": _txt(art, "dc:date"),
                "doi": _doi(art),
                "authors": _authors(art),
                "url": aid,
                "pdf_url": _pdf_url(aid, _doi(art)),
            }
        )

    by_id = {a["id"]: a for a in articles}

    # Collections: name -> [article ids]
    collections: dict[str, list[str]] = {}
    article_collections: dict[str, list[str]] = defaultdict(list)
    for c in root.findall("z:Collection", NS):
        name = _txt(c, "dc:title")
        members = [hp.attrib.get(RDF_RESOURCE, "") for hp in c.findall("dcterms:hasPart", NS)]
        # Filter to only article members (others are sub-collections)
        article_members = [m for m in members if m in by_id]
        if article_members:
            collections[name] = article_members
            for m in article_members:
                article_collections[m].append(name)

    for a in articles:
        a["collections"] = article_collections.get(a["id"], [])

    return articles, collections


def main():
    here = Path(__file__).parent
    rdf = here / "OntoverseSandbox.rdf"
    if not rdf.exists():
        # Fallback to /tmp where it was downloaded
        rdf = Path("/tmp/ontoverse_sandbox.rdf")

    articles, collections = parse(rdf)

    (here / "manifest.json").write_text(json.dumps(articles, indent=2))

    with (here / "manifest.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["title", "journal", "date", "doi", "first_author", "pdf_url", "collections"])
        for a in articles:
            w.writerow(
                [
                    a["title"],
                    a["journal"],
                    a["date"],
                    a["doi"],
                    a["authors"][0] if a["authors"] else "",
                    a["pdf_url"],
                    " | ".join(a["collections"]),
                ]
            )

    # Reverse index for "what's in each theme"
    (here / "collections.json").write_text(
        json.dumps(
            {name: [a for a in members] for name, members in sorted(collections.items())},
            indent=2,
        )
    )

    arxiv = sum(1 for a in articles if "arxiv.org" in a["url"].lower())
    biorxiv = sum(1 for a in articles if "biorxiv.org" in a["url"].lower() or "medrxiv.org" in a["url"].lower())
    print(f"Articles:        {len(articles)}")
    print(f"  arXiv:         {arxiv}")
    print(f"  bioRxiv/med:   {biorxiv}")
    print(f"  Other:         {len(articles) - arxiv - biorxiv}")
    print(f"Collections w/ articles: {len(collections)}")
    print(f"Wrote manifest.json, manifest.csv, collections.json")


if __name__ == "__main__":
    main()
