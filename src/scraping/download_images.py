"""Scraper séquentiel avec logs, retries et reprise depuis le manifeste CSV."""
from pathlib import Path
from urllib.error import HTTPError, URLError
import json
import logging
import runpy
import time
import pandas as pd


def download_images(project_root, max_items=None, delay=1.0, retries=2,
                    retry_failed=False, export_every=100, target_success=1000):
    if delay < 0 or retries < 0 or export_every < 1 or target_success < 1:
        raise ValueError("Paramètres de téléchargement invalides")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items doit être positif")
    root = Path(project_root).resolve()
    api = runpy.run_path(str(root / "src/scraping/inspect_links.py"))
    df = pd.read_csv(root / "data/interim/paintings.csv", keep_default_na=False)
    if df["source_record"].duplicated().any():
        raise ValueError("source_record doit être unique")
    images, tracking = root / "data/raw/images", root / "data/interim/scraping"
    images.mkdir(parents=True, exist_ok=True)
    tracking.mkdir(parents=True, exist_ok=True)
    manifest_path = tracking / "download_manifest.csv"
    records = {}
    if manifest_path.exists():
        previous = pd.read_csv(manifest_path, keep_default_na=False)
        if previous["source_record"].duplicated().any():
            raise ValueError("Doublons d'identifiants dans le manifeste")
        records = {int(row["source_record"]): row for row in previous.to_dict("records")}
    current_urls = dict(zip(df["source_record"].astype(int), df["URL"]))
    records = {key: row for key, row in records.items()
               if current_urls.get(key) == row.get("page_url")}
    logger = logging.getLogger("gan_download")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in [logging.FileHandler(tracking / "download.log", encoding="utf-8"), logging.StreamHandler()]:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)

    def existing_success(row):
        path = (root / str(row.get("image_path", ""))).resolve()
        if row.get("status") != "downloaded" or not path.is_relative_to(images.resolve()) or not path.is_file():
            return False
        return path.stat().st_size == float(row.get("image_bytes") or path.stat().st_size)

    successes = {key for key, row in records.items() if existing_success(row)}
    processed, skipped, stop = 0, 0, "completed"

    def save():
        columns = ["source_record", "AUTHOR", "TITLE", "page_url", "image_url", "image_path", "status", "error"]
        frame = pd.DataFrame(records.values()) if records else pd.DataFrame(columns=columns)
        temporary = manifest_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(manifest_path)
        summary = {"catalogue_entries": len(df), "tracked_entries": len(frame),
                   "untracked_entries": len(df) - len(frame), "status_counts": frame["status"].value_counts().to_dict(),
                   "processed_this_run": processed, "skipped_this_run": skipped, "stop_reason": stop,
                   "target_success": target_success, "images_toward_target": len(successes)}
        (tracking / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def fetch(url, limit, referer=None):
        for attempt in range(retries + 1):
            time.sleep(delay)
            try:
                return api["fetch_bytes"](url, limit, referer)
            except HTTPError as error:
                if error.code not in {408, 500, 502, 503, 504} or attempt == retries:
                    raise
            except (URLError, TimeoutError):
                if attempt == retries:
                    raise
            time.sleep(2 ** attempt)

    try:
        for _, item in df.iterrows():
            if len(successes) >= target_success:
                stop = "target_reached"
                break
            if max_items is not None and processed >= max_items:
                stop = "max_items_reached"
                break
            key = int(item["source_record"])
            previous = records.get(key)
            if key in successes or (previous and previous["status"] not in {"downloaded", "temporary_error"} and not retry_failed):
                skipped += 1
                continue
            row = {"source_record": key, "AUTHOR": item["AUTHOR"], "TITLE": item["TITLE"],
                   "page_url": item["URL"], "image_url": "", "image_path": "", "error": ""}
            try:
                data, headers, page_url, _ = fetch(item["URL"], 2_000_000)
                html, _ = api["decode_html"](data, headers)
                chosen, status = api["choose_candidate"](api["extract_candidates"](html, page_url))
                row["status"] = status
                if chosen:
                    data, _, image_url, _ = fetch(chosen["image_url"], 20_000_000, page_url)
                    fmt, width, height, mode = api["image_info"](data)
                    row.update(image_url=image_url, width_px=width, height_px=height,
                               image_format=fmt, image_mode=mode, image_bytes=len(data))
                    if min(width, height) < 64:
                        row["status"] = "small_image"
                    else:
                        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
                        path = images / f"{key}{extension}"
                        temporary = path.with_suffix(path.suffix + ".part")
                        temporary.write_bytes(data)
                        temporary.replace(path)
                        row.update(status="downloaded", image_path=str(path.relative_to(root)))
                        successes.add(key)
            except HTTPError as error:
                row.update(status="temporary_error" if error.code in {408, 429, 500, 502, 503, 504} else "http_error",
                           error=str(error))
                if error.code in {403, 429}:
                    stop = "server_refusal"
            except (URLError, TimeoutError) as error:
                row.update(status="temporary_error", error=str(error))
            except ValueError as error:
                row.update(status="invalid_response", error=str(error))
            except OSError as error:
                row.update(status="io_or_image_error", error=str(error))
                stop = "io_error"
            records[key] = row
            processed += 1
            save()  # Petit corpus : sauvegarder chaque entrée simplifie la reprise.
            logger.info("%s | %s | %s", key, row["status"], row["error"])
            if stop != "completed":
                break
        if len(successes) >= target_success:
            stop = "target_reached"
    except KeyboardInterrupt:
        stop = "interrupted"
    finally:
        summary = save()
        logger.info("Bilan : %s", summary)
    return summary