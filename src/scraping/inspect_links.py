"""Repérer l'image d'une œuvre WGA et tester un petit échantillon."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, unquote
from urllib.request import Request, urlopen
import io
import time
import pandas as pd
from PIL import Image


class ImageLinksParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.base_href = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "base":
            self.base_href = a.get("href")
        for attribute in {"a": ["href"], "img": ["src", "data-src"]}.get(tag, []):
            if a.get(attribute):
                self.links.append((tag, a[attribute]))
        if tag == "meta" and a.get("property", a.get("name")) in {"og:image", "twitter:image"}:
            if a.get("content"):
                self.links.append(("meta", a["content"]))


def allowed_url(url):
    try:
        u = urlsplit(url)
        return (u.scheme in {"http", "https"} and u.hostname in {"wga.hu", "www.wga.hu"}
                and not u.username and not u.password)
    except ValueError:
        return False


def fetch_bytes(url, max_bytes, referer=None):
    if not allowed_url(url):
        raise ValueError("URL hors du domaine WGA")
    headers = {"User-Agent": "GANSchoolProject/1.0 (educational project)"}
    if referer:
        headers["Referer"] = referer
    with urlopen(Request(url, headers=headers), timeout=20) as response:
        if not allowed_url(response.geturl()):
            raise ValueError("Redirection hors du domaine WGA")
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("Réponse trop volumineuse")
        return data, response.headers, response.geturl(), response.status


def decode_html(data, headers):
    encoding = headers.get_content_charset() or "utf-8"
    try:
        return data.decode(encoding), encoding
    except (UnicodeDecodeError, LookupError):
        return data.decode("cp1252", errors="replace"), "cp1252"


def extract_candidates(html, page_url):
    parser = ImageLinksParser()
    parser.feed(html)
    base = urljoin(page_url, parser.base_href or page_url)
    page_stem = Path(unquote(urlsplit(page_url).path)).stem.lower()
    candidates = {}
    for origin, link in parser.links:
        url = urljoin(base, link).split("#")[0]
        path = Path(unquote(urlsplit(url).path))
        if not allowed_url(url) or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        score = 100 * (path.stem.lower() == page_stem) + 40 * ("/art/" in str(path))
        record = {"image_url": url, "origin": origin, "score": score}
        if url not in candidates or score > candidates[url]["score"]:
            candidates[url] = record
    return sorted(candidates.values(), key=lambda row: row["score"], reverse=True)


def choose_candidate(candidates):
    if not candidates:
        return None, "no_image_candidate"
    if candidates[0]["score"] < 40:
        return None, "candidate_requires_review"
    if len(candidates) > 1 and candidates[0]["score"] == candidates[1]["score"]:
        return None, "ambiguous_candidates"
    return candidates[0], "selected"


def image_info(data):
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Format d'image inattendu")
            return image.format, image.width, image.height, image.mode
    except OSError as error:
        raise ValueError(f"Image illisible : {error}") from error


def inspect_sample(project_root, n_pages=8, delay=1.5):
    if not 5 <= n_pages <= 10 or delay < 0:
        raise ValueError("Choisir 5 à 10 pages et un délai positif ou nul")
    root = Path(project_root).resolve()
    output = root / "data/interim/scraping_test"
    images = root / "data/raw/images_test"
    (output / "html").mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "data/interim/paintings.csv", keep_default_na=False)
    sample = df.sample(min(n_pages, len(df)), random_state=42)
    sample.to_csv(output / "sample_pages.csv", index=False)
    records, all_candidates = [], []
    for _, item in sample.iterrows():
        record = {"source_record": int(item["source_record"]), "AUTHOR": item["AUTHOR"],
                  "TITLE": item["TITLE"], "page_url": item["URL"], "error": ""}
        record.update(dict.fromkeys(["final_page_url", "page_http_status", "html_encoding", "image_url",
                                     "image_origin", "image_http_status", "image_content_type", "image_format",
                                     "width_px", "height_px", "image_mode", "image_bytes", "image_path"], ""))
        try:
            data, headers, page_url, status = fetch_bytes(item["URL"], 2_000_000)
            html, encoding = decode_html(data, headers)
            (output / "html" / f'{record["source_record"]}.html').write_text(html, encoding="utf-8")
            candidates = extract_candidates(html, page_url)
            all_candidates.extend(dict(c, source_record=record["source_record"]) for c in candidates)
            chosen, record["status"] = choose_candidate(candidates)
            record.update(page_http_status=status, final_page_url=page_url, html_encoding=encoding,
                          candidate_count=len(candidates))
            if chosen:
                time.sleep(delay)
                data, headers, final_url, status = fetch_bytes(chosen["image_url"], 20_000_000, page_url)
                fmt, width, height, mode = image_info(data)
                extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
                path = images / f'{record["source_record"]}{extension}'
                path.write_bytes(data)
                record.update(image_url=final_url, image_http_status=status, image_format=fmt,
                              image_origin=chosen["origin"], image_content_type=headers.get_content_type(),
                              width_px=width, height_px=height, image_mode=mode, image_bytes=len(data),
                              image_path=str(path.relative_to(root)),
                              status="downloaded_verified" if min(width, height) >= 64 else "downloaded_small_review")
        except (OSError, ValueError) as error:
            record.update(status="error", error=str(error))
        records.append(record)
        pd.DataFrame(records).to_csv(output / "test_manifest.csv", index=False)
        pd.DataFrame(all_candidates).to_csv(output / "image_candidates.csv", index=False)
        print(record["source_record"], record["status"])
        time.sleep(delay)
    return pd.DataFrame(records)