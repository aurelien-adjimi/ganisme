"""Contrôler les images localement, sans modifier les originaux."""
from pathlib import Path
import hashlib
import json
import warnings
import pandas as pd
from PIL import Image, ImageOps, ImageChops


def check_images(project_root, min_side=64):
    if min_side < 1:
        raise ValueError("min_side doit être positif")
    root = Path(project_root).resolve()
    image_dir = (root / "data/raw/images").resolve()
    tracking = root / "data/interim/scraping"
    output = root / "data/interim/image_quality"
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(tracking / "download_manifest.csv", keep_default_na=False)
    downloaded = manifest.loc[manifest["status"].eq("downloaded")].sort_values("source_record")
    if downloaded["source_record"].duplicated().any():
        raise ValueError("Plusieurs images déclarées pour le même source_record")
    records, referenced, seen = [], set(), {}
    for _, item in downloaded.iterrows():
        row = {key: item[key] for key in ["source_record", "TITLE", "page_url", "image_url", "image_path"]}
        row.update(file_present=False, width_px=None, height_px=None, ratio_width_height=None,
                   format="", mode="", orientation="", has_alpha=False, grayscale_pixels=False,
                   animated=False, extreme_ratio=False, duplicate_of="", qc_status="rejected", reason="")
        try:
            path = (root / str(item["image_path"])).resolve()
            if not path.is_relative_to(image_dir):
                raise ValueError("Chemin hors du dossier des images")
            referenced.add(str(path.relative_to(root)))
            row["file_present"] = path.is_file()
            if not path.is_file():
                raise ValueError("missing_file")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    row.update(format=image.format, mode=image.mode,
                               animated=getattr(image, "is_animated", False),
                               has_alpha="A" in image.getbands() or "transparency" in image.info)
                    image.load()
                    oriented = ImageOps.exif_transpose(image)
                    width, height = oriented.size
                    rgb = oriented.convert("RGB")
                    r, g, b = rgb.split()
                    digest = hashlib.sha256(str(oriented.size).encode() + oriented.convert("RGBA").tobytes()).hexdigest()
                    row.update(width_px=width, height_px=height, ratio_width_height=width / height,
                               orientation="square" if width == height else "landscape" if width > height else "portrait",
                               grayscale_pixels=not ImageChops.difference(r, g).getbbox() and not ImageChops.difference(g, b).getbbox(),
                               extreme_ratio=width / height < 0.1 or width / height > 10,
                               file_bytes=path.stat().st_size, pixel_sha256=digest)
            if min(width, height) < min_side:
                row["reason"] = "too_small"
            elif row["animated"]:
                row["reason"] = "animated_image"
            elif digest in seen:
                row.update(reason="duplicate_pixels", duplicate_of=seen[digest])
            else:
                seen[digest] = item["source_record"]
                row["qc_status"] = "usable"
        except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            row["reason"] = str(error)
        records.append(row)
    columns = ["source_record", "TITLE", "page_url", "image_url", "image_path", "file_present",
               "width_px", "height_px", "ratio_width_height", "format", "mode", "orientation",
               "has_alpha", "grayscale_pixels", "animated", "extreme_ratio", "duplicate_of", "qc_status", "reason"]
    quality = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    usable = quality.loc[quality["qc_status"].eq("usable")].copy()
    rejected = quality.loc[quality["qc_status"].eq("rejected")].copy()
    files = [{"image_path": str(path.relative_to(root)),
              "referenced_in_success_manifest": str(path.relative_to(root)) in referenced,
              "temporary_file": path.suffix in {".part", ".tmp"}}
             for path in image_dir.rglob("*") if path.is_file()]
    inventory = pd.DataFrame(files, columns=["image_path", "referenced_in_success_manifest", "temporary_file"])
    for name, frame in {"image_quality": quality, "usable_images": usable,
                        "rejected_images": rejected, "file_inventory": inventory}.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    distributions = {}
    for column in ["format", "mode", "orientation", "grayscale_pixels", "has_alpha"]:
        counts = quality.loc[quality["width_px"].notna(), column].value_counts()
        counts.rename_axis(column).reset_index(name="count").to_csv(output / f"distribution_{column}.csv", index=False)
        distributions[column] = {str(key): int(value) for key, value in counts.items()}
    reasons = rejected["reason"].value_counts()
    reasons.rename_axis("reason").reset_index(name="count").to_csv(output / "rejection_reasons.csv", index=False)
    summary_file = tracking / "download_summary.json"
    scrape = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    summary = {"expected_target": scrape.get("target_success", 1000),
               "catalogue_paintings": scrape.get("catalogue_entries"),
               "manifest_downloaded": len(downloaded), "referenced_files_present": int(quality["file_present"].sum()),
               "rejected_records": len(rejected), "usable_unique_images": len(usable), "minimum_side_px": min_side,
               "unreferenced_files": int((~inventory["referenced_in_success_manifest"].astype(bool) & ~inventory["temporary_file"].astype(bool)).sum()),
               "temporary_files": int(inventory["temporary_file"].sum()),
               "distributions_decoded": distributions, "rejection_reasons": reasons.to_dict(),
               "extreme_ratio_usable": int(usable["extreme_ratio"].sum())}
    (output / "image_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert len(downloaded) == len(usable) + len(rejected)
    print(f"Téléchargées : {len(downloaded)} | Utilisables : {len(usable)} | Rejetées : {len(rejected)}")
    return {"quality": quality, "usable": usable, "rejected": rejected, "inventory": inventory, "summary": summary}