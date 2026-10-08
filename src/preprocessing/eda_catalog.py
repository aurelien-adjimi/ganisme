"""EDA du catalogue : quelques tableaux et graphiques répondant à des questions."""
from pathlib import Path
import json
import re
import pandas as pd
import matplotlib.pyplot as plt


def extract_dimensions(value):
    """Paire physique simple ; hauteur × largeur est une hypothèse."""
    empty = {"height_assumed_cm": None, "width_assumed_cm": None,
             "ratio_width_height_assumed": None, "area_assumed_cm2": None}
    text = str(value).lower().replace("×", "x")
    if re.search(r"\b(detail|frame|framed|each|overall|diameter|diam)\b", text):
        return empty
    matches = re.findall(r"(\d+(?:[.,]\d+)?)\s*x\s*(\d+(?:[.,]\d+)?)\s*(cm|mm|m)\b", text)
    if len(matches) != 1 or len(re.findall(r"\d\s*x\s*\d", text)) != 1:
        return empty
    h, w, unit = matches[0]
    scale = {"cm": 1, "mm": 0.1, "m": 100}[unit]
    h, w = float(h.replace(",", ".")) * scale, float(w.replace(",", ".")) * scale
    if min(h, w) <= 0:
        return empty
    return dict(height_assumed_cm=h, width_assumed_cm=w,
                ratio_width_height_assumed=w / h, area_assumed_cm2=h * w)


def run_eda(project_root):
    root = Path(project_root).resolve()
    raw = pd.read_csv(root / "data/raw/art_catalog.csv", dtype=str, keep_default_na=False)
    x = pd.read_csv(root / "data/interim/paintings.csv", keep_default_na=False)
    if x.empty or not x["FORM"].eq("painting").all():
        raise ValueError("Un catalogue nettoyé non vide de peintures est attendu")
    metadata = ["AUTHOR", "BORN-DIED", "TITLE", "DATE", "TECHNIQUE", "LOCATION", "URL", "FORM", "TYPE", "SCHOOL", "TIMEFRAME"]
    x[metadata] = x[metadata].replace("", pd.NA)
    output, figure_dir = root / "data/interim/eda", root / "images/eda"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables, figures = {}, []

    def table(name, frame):
        tables[name] = frame
        frame.to_csv(output / f"{name}.csv", index=False)
        return frame

    def counts(series, name, label):
        frame = series.fillna("Non renseigné").value_counts().rename_axis(label).reset_index(name="nombre")
        frame["pourcentage"] = 100 * frame["nombre"] / len(series) if len(series) else 0
        return table(name, frame)

    def save_plot(name, fig, question, note):
        fig.suptitle(question)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        figures.append({"file": f"{name}.png", "question": question, "interpretation": note})

    def bars(name, frame, label, question, note, top=20):
        data = frame.head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(data[label].astype(str), data["nombre"])
        ax.set_xlabel("Nombre d'entrées")
        save_plot(name, fig, question, note)

    n_raw = int(raw["FORM"].str.strip().str.casefold().eq("painting").sum())
    table("dataset_size", pd.DataFrame({"mesure": ["Catalogue brut", "Peintures brutes", "Peintures nettoyées", "Part de peintures (%)"],
                                       "valeur": [len(raw), n_raw, len(x), 100 * n_raw / len(raw)]}))
    for column, name, number, question, note in [
        ("AUTHOR", "artists", "01", "Quelles attributions dominent ?", "Une attribution n'est pas toujours un artiste identifié."),
        ("SCHOOL", "schools", "03", "Quelles écoles dominent ?", "SCHOOL ne donne ni un mouvement précis ni une nationalité."),
        ("TYPE", "themes", "04", "Quels sujets dominent ?", "TYPE décrit un sujet, pas un style."),
        ("LOCATION", "locations", "05", "Quels lieux de conservation dominent ?", "Un lieu de conservation n'est pas un pays de création.")]:
        frame = counts(x[column], name, column)
        bars(f"{number}_{name}", frame, column, question, note)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tables["artists"].loc[tables["artists"]["AUTHOR"].ne("Non renseigné"), "nombre"], bins=30)
    ax.set_xlabel("Entrées par attribution")
    ax.set_ylabel("Nombre d'attributions")
    save_plot("02_artist_distribution", fig, "La représentation des auteurs est-elle équilibrée ?", "Les fortes fréquences peuvent dominer l'apprentissage.")

    a, b = (pd.to_numeric(x[c], errors="coerce") for c in ["year_start", "year_end"])
    eligible = x["date_kind"].isin(["year", "approximate_year", "range", "decade"]) & (b - a).between(0, 50)
    x["year_representative"] = ((a + b) / 2).where(eligible)
    dates = counts(x["date_kind"], "date_coverage", "date_kind")
    bars("06_date_coverage", dates, "date_kind", "Quelle précision des dates est disponible ?", "Les dates non interprétées sont exclues des distributions chronologiques.")
    for name, series, number in [
        ("decades", (x["year_representative"] // 10 * 10).dropna(), "07"),
        ("centuries", ((x["year_representative"] - 1) // 100 + 1).dropna(), "08")]:
        frame = series.astype(int).value_counts().sort_index().rename_axis(name).reset_index(name="nombre")
        table(name, frame)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(frame[name], frame["nombre"], width=8 if name == "decades" else 0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("Nombre d'entrées")
        save_plot(f"{number}_{name}", fig, "Comment le corpus se répartit-il dans le temps ?", "Milieu des intervalles de 50 ans maximum : dates représentatives, pas exactes.")
    missing = pd.DataFrame({"variable": metadata, "nombre": x[metadata].isna().sum().values})
    missing["pourcentage"] = 100 * missing["nombre"] / len(x)
    table("missing_values", missing)
    bars("09_missing", missing.sort_values("nombre", ascending=False), "variable", "Quelles informations manquent ?", "Les absences limitent certaines analyses.")

    dimensions = pd.DataFrame(x["TECHNIQUE"].map(extract_dimensions).tolist(), index=x.index)
    derived = pd.concat([x[["source_record", "AUTHOR", "SCHOOL", "year_representative"]], dimensions], axis=1)
    table("derived_metadata", derived)
    valid = dimensions["height_assumed_cm"].notna()
    table("dimension_coverage", pd.DataFrame({"mesure": ["Paires extraites", "Non extraites"], "nombre": [int(valid.sum()), int((~valid).sum())]}))
    if valid.any():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dimensions.loc[valid, "ratio_width_height_assumed"], bins=30)
        ax.set_xlabel("Largeur / hauteur physiques supposées")
        save_plot("10_ratios", fig, "Les formats physiques sont-ils variés ?", "Les dimensions du tableau ne donnent pas la résolution de sa photographie.")
    century = ((x["year_representative"] - 1) // 100 + 1).astype("Int64")
    cross = pd.crosstab(x["SCHOOL"], century)
    table("school_century_counts", cross.reset_index())
    if not cross.empty:
        percentages = cross.div(cross.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(10, 7))
        heat = ax.imshow(percentages, aspect="auto")
        ax.set_yticks(range(len(cross)), cross.index)
        ax.set_xticks(range(len(cross.columns)), cross.columns.astype(str))
        ax.set_xlabel("Siècle représentatif")
        fig.colorbar(heat, ax=ax, label="% au sein de chaque école datée")
        save_plot("11_school_century", fig, "Les écoles couvrent-elles les mêmes périodes ?", "Association descriptive, pas relation causale ; consulter aussi les effectifs.")
    table("artist_periods", derived.groupby("AUTHOR")["year_representative"].agg(["count", "min", "max"]).reset_index())
    paired = derived.dropna(subset=["year_representative", "area_assumed_cm2"])
    correlation = paired[["year_representative", "area_assumed_cm2"]].rank().corr().iloc[0, 1] if len(paired) >= 3 else float("nan")
    if not paired.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(paired["year_representative"], paired["area_assumed_cm2"], s=8, alpha=0.2)
        ax.set_xlabel("Année représentative")
        ax.set_ylabel("Surface physique supposée (cm²)")
        save_plot("12_area_period", fig, "Les dimensions varient-elles avec l'époque ?", "Sous-ensemble avec dimensions extraites et dates interprétables ; biais de disponibilité possible.")
    summary = {"total": len(raw), "paintings": len(x), "artist_labels": int(x["AUTHOR"].nunique()),
               "dated_representative": int(eligible.sum()), "dimension_pairs": int(valid.sum()),
               "dimensions_with_date": len(paired), "spearman_year_area": None if pd.isna(correlation) else float(correlation)}
    report_path = output / "EDA_synthese.md"
    report_path.write_text("# Limites de l'EDA\n\nSCHOOL : école, pas nationalité ni mouvement.\n"
                           "LOCATION : conservation, pas origine.\nDimensions : paires simples, ordre hauteur/largeur supposé.\n"
                           "Dates représentatives : approximations. Interpréter les graphiques avec leurs effectifs.\n", encoding="utf-8")
    for filename, data in [("eda_summary.json", summary), ("figures.json", figures)]:
        (output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"tables": tables, "figures": figures, "summary": summary,
            "report_path": report_path, "figure_dir": figure_dir}