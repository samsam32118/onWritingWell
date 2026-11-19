"""Collect essays from the writers listed in configs/sources.yaml.

Output: one JSON object per document {doc_id, source, url, title, text, words, sha256}.
doc_id hashes the URL, sha256 hashes the cleaned text. The repo commits only the manifest;
Modal re-fetches the text and checks the hash, so copyrighted text never enters git.

    python -m owl.scrape --out corpus.jsonl [--only paul_graham,dan_luu] [--limit 20]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import time
from urllib.parse import urljoin

import httpx
import trafilatura
import yaml
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (compatible; onWritingWell-research/0.1; "
    "+https://github.com/samsam32118/onwritingwell)")
DELAY = 0.5
client = httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": UA})

# Lines that are site furniture, not prose.
JUNK_LINE = re.compile(
    r"^(\w+ translation|translations?:.*|thanks( to)?\b.*reading drafts.*|\[return\]|"
    r"share this:?|subscribe.*|related posts?:?|comments?|previous post|next post|"
    r"posted (on|by) .*|tags?:.*|filed under.*|leave a comment.*|\d+ comments?)$", re.I)


def get(url, tries=3):
    """The response on 2xx, None when the page is gone or the site keeps failing."""
    for i in range(tries):
        try:
            r = client.get(url)
        except httpx.HTTPError:
            time.sleep(DELAY * 2**i + 0.5)
            continue
        if 200 <= r.status_code < 300:
            return r
        if r.status_code in (403, 404, 410):
            return None
        time.sleep(DELAY * 2**i + 0.5)
    return None


def links(page):
    r = get(page)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    return [urljoin(str(r.url), a["href"]).split("#")[0].split("?")[0]
        for a in soup.find_all("a", href=True)]


def clean_text(raw):
    """Normalise extracted text into paragraphs separated by blank lines."""
    raw = re.sub(r"</?(sup|sub|span|small)[^>]*>", "", raw.replace("\r", ""))
    raw = re.sub(r"\\([\[\]\*_#`~|>-])", r"\1", raw)
    raw = re.sub(r"\s*\[return\]", "", raw).replace("↩", "")
    paras = [l.strip() for l in raw.split("\n") if l.strip() and not JUNK_LINE.match(l.strip())]
    # trafilatura splits "Thanks / to X for reading drafts of this." across lines
    while paras and re.search(r"(reading drafts|translation)\b", paras[-1], re.I):
        paras.pop()
    if paras and paras[-1].lower() == "thanks":
        paras.pop()
    return "\n\n".join(paras).strip()


def fetch_doc(source, url):
    r = get(url)
    if r is None:
        return None
    # Let BeautifulSoup sniff the charset (paulgraham.com sends none, so httpx garbles em
    # dashes) and unwrap inline tags (trafilatura splits a paragraph at a link around <u>).
    soup = BeautifulSoup(r.content, "lxml")
    for tag in soup.find_all(["a", "u", "font", "span"]):
        tag.unwrap()
    html = str(soup)
    meta = trafilatura.extract_metadata(html)
    text = clean_text(trafilatura.extract(html, include_comments=False, include_tables=False,
        include_links=False, include_images=False, favor_recall=True,
        output_format="markdown") or "")
    if not text:
        return None
    return {"doc_id": hashlib.sha1(url.encode()).hexdigest()[:12], "source": source,
        "url": url, "title": ((meta.title if meta else "") or "").strip(), "text": text,
        "words": len(text.split()), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def substack_posts(host, max_posts=3000):
    """Free public posts. The API pages by opaque nextCursor and caps limit at 25; older
    publications still answer the list-shaped, offset-paged form."""
    urls, cursor, offset = [], None, 0
    while len(urls) < max_posts:
        q = f"&cursor={cursor}" if cursor else (f"&offset={offset}" if offset else "")
        r = get(f"https://{host}/api/v1/archive?sort=new&limit=25{q}")
        if r is None:
            break
        data = r.json()
        batch = data.get("posts", []) if isinstance(data, dict) else data
        cursor = data.get("nextCursor") if isinstance(data, dict) else None
        offset += len(batch)
        urls += [p["canonical_url"].split("?")[0] for p in batch if p.get("canonical_url")
            and p.get("audience") in (None, "everyone") and p.get("type") in (None, "newsletter")]
        if not batch or isinstance(data, dict) and not cursor:
            break
        time.sleep(DELAY)
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    only = set(filter(None, a.only.split(",")))
    n = 0
    out = open(a.out, "w")
    for src in cfg["sources"]:
        if only and src["name"] not in only:
            continue
        candidates = substack_posts(src["substack"]) if src.get("substack") else []
        pages = list(src.get("index", []))
        if src.get("pages"):
            pages += [src["pages"]["template"].format(n=i)
                for i in range(1, src["pages"]["max"] + 1)]
        follow = re.compile(src.get("index_follow") or r"(?!x)x")
        for page in pages:
            found = links(page)
            for sub in sorted({u for u in found if follow.match(u)}):
                found += links(sub)
                time.sleep(DELAY)
            candidates += found
        inc = re.compile(src["include"])
        exc = re.compile(src.get("exclude") or r"(?!x)x")
        urls = [u.replace("http://", "https://") for u in candidates]
        urls = list(dict.fromkeys(u for u in urls if inc.search(u) and not exc.search(u)))
        urls = urls[:a.limit] if a.limit else urls
        print(f"[scrape] {src['name']}: {len(urls)} candidate posts", file=sys.stderr)
        lo = src.get("min_words", cfg.get("min_words", 500))
        hi = src.get("max_words", cfg.get("max_words", 9000))
        kept = 0
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for doc in ex.map(lambda u: fetch_doc(src["name"], u), urls):
                if doc and lo <= doc["words"] <= hi:
                    out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    kept += 1
        n += kept
        print(f"[scrape] {src['name']}: kept {kept}", file=sys.stderr)
    out.close()
    print(f"[scrape] wrote {n} docs to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
