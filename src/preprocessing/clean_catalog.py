"""Nettoyage conservateur des métadonnées ; aucune requête réseau."""
from pathlib import Path
import re
import json
import unicodedata
from datetime import date
from urllib.parse import urlsplit
import pandas as pd

MARKERS = {'', '-', '-.', '?', 'n/a', 'na', 'null', 'none', 'unknown'}

def text_clean(value):
    if pd.isna(value):
        return pd.NA
    value = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(value))).strip()
    return pd.NA if value.casefold() in MARKERS else value

def parse_date(value):
    """Bornes annuelles, sans transformer une approximation en date exacte."""
    result = {'date_kind': 'missing', 'year_start': None, 'year_end': None,
              'date_approximate': False, 'date_uncertain': False}
    if pd.isna(value):
        return result
    s = str(value).lower().replace('–', '-').replace('—', '-')
    result['date_uncertain'] = '?' in s
    s = re.sub(r'\(\?\)|\?', '', s).strip()
    result['date_approximate'] = bool(re.match(r'^(?:c\.|ca\.|circa)\s*', s))
    s = re.sub(r'^(?:c\.|ca\.|circa)\s*', '', s)
    result['date_kind'] = 'unparsed'
    if re.fullmatch(r'\d{3,4}', s):
        y = int(s)
        result.update(date_kind='approximate_year' if result['date_approximate'] else 'year', year_start=y, year_end=y)
    elif m := re.fullmatch(r'(\d{3,4})-(\d{1,4})', s):
        a, suffix = int(m[1]), m[2]
        b = int(suffix)
        if len(suffix) < len(m[1]):
            base = 10**len(suffix)
            b = a//base*base + b
            if b < a:
                b += base
        result.update(date_kind='range', year_start=a, year_end=b)
    elif m := re.fullmatch(r'(\d{3,4})s', s):
        a = int(m[1])
        # Seuls les nombres terminant par 0 décrivent ici une décennie.
        if a % 10 == 0:
            result.update(date_kind='decade', year_start=a, year_end=a+9)
    elif m := re.fullmatch(r'(\d{1,2})(?:st|nd|rd|th) century', s):
        c = int(m[1])
        result.update(date_kind='century', year_start=(c-1)*100+1, year_end=c*100)
    elif m := re.fullmatch(r'(before|after) (\d{3,4})', s):
        y = int(m[2])
        result.update(date_kind=m[1], year_start=y+1 if m[1]=='after' else None,
                      year_end=y-1 if m[1]=='before' else None)
    # Dates narratives, alternatives et périodes multiples restent à revoir.
    return result

def url_check(value):
    if pd.isna(value):
        return 'missing'
    try:
        u = urlsplit(str(value))
        if u.scheme.lower() not in {'http', 'https'} or not u.hostname or not u.path or re.search(r'\s', str(value)):
            return 'invalid_syntax'
        if u.username or u.password:
            return 'credentials_in_url'
        if u.hostname.lower() not in {'www.wga.hu', 'wga.hu'}:
            return 'unexpected_domain'
        return 'syntax_ok_not_fetched'
    except ValueError:
        return 'invalid_syntax'

def clean_catalog(project_root):
    root = Path(project_root).resolve()
    # Le nom adopté aux étapes précédentes reste la référence du brut.
    source = root / 'data/raw/art_catalog.csv'
    if not source.exists():
        raise FileNotFoundError(f'CSV brut attendu : {source}')
    out = root / 'data/interim'
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source, dtype=str, encoding='utf-8', keep_default_na=False)
    required = ['AUTHOR','BORN-DIED','TITLE','DATE','TECHNIQUE','LOCATION','URL','FORM','TYPE','SCHOOL','TIMEFRAME']
    if list(raw.columns) != required:
        raise ValueError('Colonnes inattendues : contrôler le schéma avant nettoyage.')
    mask = raw['FORM'].str.strip().str.casefold().eq('painting')
    selected = raw.loc[mask].copy()
    # Identifiant de traçabilité = position de l’enregistrement dans le CSV (base 1).
    selected.insert(0, 'source_record', selected.index + 1)
    initial_count = len(selected)
    exact_mask = selected.duplicated(subset=required, keep='first')
    exact = selected.loc[exact_mask].copy()
    work = selected.loc[~exact_mask].copy().reset_index(drop=True)
    result = work.copy()
    audit = []
    for c in required:
        result[c + '_raw'] = work[c]
        result[c] = work[c].map(text_clean).astype('string')
        if c in ['FORM','TYPE','SCHOOL']:
            result[c] = result[c].str.casefold()
        changed = work[c].astype('string').ne(result[c].fillna('')).fillna(True)
        audit.append({'variable':c, 'changed_records':int(changed.sum()),
                      'missing_before_empty':int(work[c].str.strip().eq('').sum()),
                      'missing_after':int(result[c].isna().sum())})
    dates = pd.DataFrame([parse_date(v) for v in result['DATE']], index=result.index)
    for c in ['year_start','year_end']:
        dates[c] = pd.array(dates[c], dtype='Int64')
    result = pd.concat([result, dates], axis=1)
    result['url_status'] = result['URL'].map(url_check)
    # Les mêmes URL et titres sont signalés, jamais supprimés sans examen.
    result['duplicate_url'] = result['URL'].notna() & result['URL'].duplicated(keep=False)
    result['duplicate_author_title'] = result[['AUTHOR','TITLE']].notna().all(axis=1) & result.duplicated(['AUTHOR','TITLE'],keep=False)
    anomalies = []
    today = date.today().year
    for _, row in result.iterrows():
        flags = []
        if row['url_status'] != 'syntax_ok_not_fetched': flags.append(row['url_status'])
        if row['duplicate_url']: flags.append('repeated_url')
        for c in ['AUTHOR','TITLE']:
            if pd.isna(row[c]): flags.append('missing_' + c.lower())
        a,b = row['year_start'],row['year_end']
        if pd.notna(a) and pd.notna(b) and a>b: flags.append('reversed_date_range')
        if any(pd.notna(v) and (v<=0 or v>today) for v in [a,b]): flags.append('year_outside_1_to_current_year')
        if row['date_kind']=='unparsed': flags.append('date_requires_review')
        if row['date_kind']=='range' and pd.notna(a) and pd.notna(b) and b-a>100: flags.append('date_range_over_100_years_review')
        if flags: anomalies.append({'source_record':row['source_record'],'flags':';'.join(flags),
                                   'DATE_raw':row['DATE_raw'],'URL_raw':row['URL_raw']})
    quality = pd.DataFrame(audit)
    missing = pd.DataFrame({'variable':required,
        'missing_count':[int(result[c].isna().sum()) for c in required]})
    missing['missing_pct'] = (missing['missing_count']/len(result)*100).round(2)
    candidates = result.loc[result['duplicate_url'] | result['duplicate_author_title'],
                            ['source_record','AUTHOR','TITLE','URL','duplicate_url','duplicate_author_title']]
    anomaly_table = pd.DataFrame(anomalies,columns=['source_record','flags','DATE_raw','URL_raw'])
    unparsed = result.loc[result['date_kind'].eq('unparsed'),'DATE_raw'].value_counts().rename_axis('DATE_raw').reset_index(name='count')
    # Pas de fusion des catégories Dutch/Netherlandish, ni inférence d’une nationalité.
    summary = {'source':str(source.relative_to(root)), 'raw_records':len(raw),
        'painting_records_before_cleaning':initial_count,'exact_duplicates_removed':int(exact_mask.sum()),
        'clean_records':len(result),'missing':missing.set_index('variable')['missing_count'].to_dict(),
        'date_kinds':result['date_kind'].value_counts().to_dict(),
        'url_status':result['url_status'].value_counts().to_dict(),
        'records_flagged':len(anomaly_table),
        'records_in_repeated_author_title':int(result['duplicate_author_title'].sum()),
        'notes':['No network request; URL accessibility not checked.',
                 'Dates are annual bounds, not exact timestamps.',
                 'Flagged records retained. No dimensions inferred.',
                 'Original metadata retained in *_raw columns.']}
    assert len(result) == initial_count - int(exact_mask.sum())
    assert result['FORM'].eq('painting').all()
    exports = {'paintings.csv':result,'cleaning_audit.csv':quality,'paintings_missing_values.csv':missing,
               'paintings_removed_exact_duplicates.csv':exact,'paintings_duplicate_candidates.csv':candidates,
               'paintings_anomalies.csv':anomaly_table,'paintings_unparsed_dates.csv':unparsed}
    for name,table in exports.items(): table.to_csv(out/name,index=False,encoding='utf-8')
    (out/'cleaning_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    reread = pd.read_csv(out/'paintings.csv',dtype=str,keep_default_na=False)
    assert len(reread)==len(result) and reread['FORM'].eq('painting').all()
    return result, summary, quality, missing, anomaly_table
