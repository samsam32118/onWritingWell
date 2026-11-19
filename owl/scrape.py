"""Collect essays from the writers listed in configs/sources.yaml.

Output: one JSON object per document
    {doc_id, source, url, title, text, words, sha256}

`doc_id` is a stable hash of the URL, `sha256` is a hash of the cleaned text. The repo
commits only the manifest (doc_id, url, sha256); Modal re-fetches the text and checks the
hash, so copyrighted text never enters git.

    python -m owl.scrape --out data/corpus.jsonl [--only paul_graham,dan_luu] [--limit 20]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
import trafilatura
import yaml
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; onWritingWell-research/0.1; +https://github.com/samsam32118/onwritingwell)"

# Lines that are site furniture, not prose.
_JUNK_LINE = re.compile(
    r"^(\w+ translation|translations?:.*|thanks( to)?\b.*reading drafts.*|\[return\]|"
    r"share this:?|subscribe.*|related posts?:?|comments?|previous post|next post|"
    r"posted (on|by) .*|tags?:.*|filed under.*|leave a comment.*|\d+ comments?)$",
    re.I,
)


def doc_id_for(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def clean_text(raw: str) -> str:
    """Normalise extracted text into paragraphs separated by blank lines."""
    raw = raw.replace("\r", "")
    raw = re.sub(r"</?(sup|sub|span|small)[^>]*>", "", raw)
    raw = re.sub(r"\\([\[\]\*_#`~|>-])", r"\1", raw)  # markdown escapes
    raw = re.sub(r"\s*\[return\]", "", raw)
    raw = raw.replace("↩", "")
    paras: list[str] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _JUNK_LINE.match(line):
            continue
        paras.append(line)
    # Drop a trailing "Thanks to X, Y and Z for reading drafts" block that trafilatura splits
    # across lines ("Thanks" / "to Trevor ... for reading drafts of this.").
    while paras and re.search(r"(reading drafts|translation)\b", paras[-1], re.I):
        paras.pop()
    if paras and paras[-1].lower() == "thanks":
        paras.pop()
    return "\n\n".join(paras).strip()


class Fetcher:
    def __init__(self, delay: float = 0.5):
        self.client = httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": UA})
        self.delay = delay

    def get(self, url: str, tries: int = 3) -> httpx.Response | None:
        for i in range(tries):
            try:
                r = self.client.get(url)
                if 200 <= r.status_code < 300:
                    return r
                if r.status_code in (404, 410, 403):
                    return None
            except httpx.HTTPError:
                pass
            time.sleep(self.delay * (2**i) + 0.5)
        return None


def _links(fetcher: Fetcher, page: str) -> list[str]:
    r = fetcher.get(page)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    base = str(r.url)
    out = []
    for a in soup.find_all("a", href=True):
        u = urljoin(base, a["href"]).split("#")[0].split("?")[0]
        out.append(u)
    return out


def _substack_posts(fetcher: Fetcher, host: str, max_posts: int = 2000) -> list[str]:
    urls: list[str] = []
    offset = 0
    while offset < max_posts:
        q = f"&offset={offset}" if offset else ""
        r = fetcher.get(f"https://{host}/api/v1/archive?sort=new&limit=50{q}")
        if r is None:
            break
        batch = r.json()
        if isinstance(batch, dict):  # newer API wraps the list
            batch = batch.get("posts", [])
        if not batch:
            break
        for p in batch:
            if p.get("audience") not in (None, "everyone"):
                continue  # skip paywalled posts
            if p.get("type") not in (None, "newsletter"):
                continue  # skip podcasts/threads
            if p.get("canonical_url"):
                urls.append(p["canonical_url"].split("?")[0])
        offset += len(batch)
        time.sleep(fetcher.delay)
    return urls


def discover(fetcher: Fetcher, src: dict) -> list[str]:
    """Return post URLs for one source, filtered by its include/exclude regexes."""
    candidates: list[str] = []
    if src.get("substack"):
        candidates += _substack_posts(fetcher, src["substack"])
    pages = list(src.get("index", []))
    if src.get("pages"):  # e.g. {"template": "https://x/?page={n}", "max": 20}
        pages += [src["pages"]["template"].format(n=n) for n in range(1, src["pages"]["max"] + 1)]
    for page in pages:
        links = _links(fetcher, page)
        if src.get("index_follow"):
            follow = re.compile(src["index_follow"])
            for sub in sorted({u for u in links if follow.match(u)}):
                links += _links(fetcher, sub)
                time.sleep(fetcher.delay)
        candidates += links
    inc = re.compile(src["include"])
    exc = re.compile(src.get("exclude") or r"(?!x)x")
    seen, out = set(), []
    for u in candidates:
        u = u.replace("http://", "https://")
        if u in seen or not inc.search(u) or exc.search(u):
            continue
        seen.add(u)
        out.append(u)
    return out


def extract(html: str) -> tuple[str, str]:
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta else "") or ""
    body = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        include_links=False,
        include_images=False,
        favor_recall=True,
        output_format="markdown",
    )
    return title.strip(), clean_text(body or "")


def fetch_doc(fetcher: Fetcher, source: str, url: str) -> dict | None:
    r = fetcher.get(url)
    if r is None:
        return None
    title, text = extract(r.text)
    if not text:
        return None
    return {
        "doc_id": doc_id_for(url),
        "source": source,
        "url": url,
        "title": title,
        "text": text,
        "words": len(text.split()),
        "sha256": text_hash(text),
    }


def scrape(cfg: dict, only: set[str] | None, limit: int | None, workers: int = 6):
    fetcher = Fetcher()
    for src in cfg["sources"]:
        if only and src["name"] not in only:
            continue
        lo = src.get("min_words", cfg.get("min_words", 500))
        hi = src.get("max_words", cfg.get("max_words", 9000))
        urls = discover(fetcher, src)
        if limit:
            urls = urls[:limit]
        print(f"[scrape] {src['name']}: {len(urls)} candidate posts", file=sys.stderr)
        kept = 0
        with cf.ThreadPoolExecutor(workers) as ex:
            for doc in ex.map(lambda u: fetch_doc(fetcher, src["name"], u), urls):
                if doc and lo <= doc["words"] <= hi:
                    kept += 1
                    yield doc
        print(f"[scrape] {src['name']}: kept {kept}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    only = set(filter(None, a.only.split(","))) or None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w") as f:
        for doc in scrape(cfg, only, a.limit):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    print(f"[scrape] wrote {n} docs to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
