"""Strict identity/label checks shared by Colab training and legacy scripts."""
import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def column(df, candidates):
    names = {str(c).strip().casefold(): c for c in df.columns}
    return next((names[c.casefold()] for c in candidates if c.casefold() in names), None)


def strict_grade(value):
    text = clean(value)
    match = re.fullmatch(r"(?:(?:ISUP|Grade|Glade)\s*[:=]?\s*)?([0-5])(?:\.0)?", text, re.I)
    if not match:
        raise ValueError(f"Invalid/missing ISUP grade: {text!r}; expected 0..5, not a Gleason sum.")
    return int(match.group(1))


def strict_lens(value):
    match = re.fullmatch(r"(4|10|20|40)(?:\.0)?\s*[xX×]?", clean(value))
    if not match:
        raise ValueError(f"Invalid/missing objective lens: {value!r}")
    return int(match.group(1))


def read_table(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}. Supply Metadata.xlsx or a labelled CSV.")
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.read_excel(path, dtype=str, keep_default_na=False)


def metadata_table(path, patient_columns=None):
    """Only an explicit patient key or user-selected columns can define a patient."""
    raw = read_table(path)
    image_col = column(raw, ['Ten_File', 'image_id', 'slide_id', 'filename', 'file_name'])
    grade_col = column(raw, ['isup_grade', 'isup', 'Grade', 'Glade', 'label'])
    lens_col = column(raw, ['objective_lens', 'Do_Phong_Dai', 'vat_kinh', 'vật kính', 'lens'])
    if image_col is None or grade_col is None or lens_col is None:
        raise ValueError('Metadata requires image/slide ID, ISUP (Grade/Glade/label) and objective lens columns.')
    if patient_columns is None:
        canonical = column(raw, ['patient_id'])
        if canonical is None:
            raise ValueError('Set PATIENT_ID_COLUMNS / --patient_id_columns to the confirmed patient key. '
                             'Ma_Nam + Ma_So is NOT inferred automatically.')
        patient_columns = [canonical]
    selected = [column(raw, [c]) for c in patient_columns]
    if not selected or any(c is None for c in selected):
        raise ValueError(f'Patient ID columns not found: {patient_columns}')
    rows = []
    for _, row in raw.iterrows():
        if not any(clean(v) for v in row):
            continue
        image = Path(clean(row[image_col]).replace('\\', '/')).stem
        keys = [clean(row[c]) for c in selected]
        for i, c in enumerate(selected):
            if str(c).casefold() == 'ma_nam':
                keys[i] = re.sub(r'[\s-]+', '', keys[i]).upper()
        if not image or not all(keys):
            raise ValueError('Missing slide ID / patient identity in metadata.')
        # JSON encoding avoids collisions when composing explicitly selected columns.
        patient = keys[0] if len(keys) == 1 else json.dumps(keys, ensure_ascii=False)
        grade = strict_grade(row[grade_col])
        if str(grade_col).casefold() == 'label' and grade not in [0, 1]:
            raise ValueError('Binary label column must contain only 0 or 1.')
        rows.append(dict(slide_id=image, patient_id=patient, isup=grade,
                         label=int(grade > 0), objective_lens=strict_lens(row[lens_col])))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('Empty metadata.')
    if (df.groupby('slide_id')[['patient_id','isup','objective_lens']].nunique() > 1).any().any():
        raise ValueError('Conflicting metadata rows for the same slide ID.')
    df = df.drop_duplicates('slide_id').reset_index(drop=True)
    df['patient_label'] = df.groupby('patient_id')['label'].transform('max')
    return df


def identity_map(path, patient_columns=None):
    return metadata_table(path, patient_columns).set_index('slide_id')['patient_id'].to_dict()


def validate_frame(df, name='data'):
    required = ['patient_id', 'label']
    if any(c not in df for c in required) or df.empty:
        raise ValueError(f'{name}: empty data or missing patient_id/label.')
    if df.patient_id.map(clean).eq('').any() or df.patient_id.isna().any():
        raise ValueError(f'{name}: missing patient identity.')
    if not df.label.isin([0,1]).all():
        raise ValueError(f'{name}: invalid labels.')
    label_col = 'patient_label' if 'patient_label' in df else 'label'
    patients = df.groupby('patient_id')[label_col].max()
    if set(patients.unique()) != {0,1}:
        raise ValueError(f'{name}: requires both classes at patient level.')
    return patients


def split_patients(df, ratio, seed):
    if not 0 < ratio < 1:
        raise ValueError('Split ratio must be between 0 and 1.')
    patients = validate_frame(df)
    rng = random.Random(seed)
    held = set()
    for label in [0,1]:
        ids = sorted(patients[patients == label].index)
        if len(ids) < 2:
            raise ValueError(f'Need at least two patients of class {label} to split.')
        rng.shuffle(ids)
        count = min(len(ids)-1, max(1, round(len(ids)*ratio)))
        held.update(ids[:count])
    a, b = df[~df.patient_id.isin(held)].copy(), df[df.patient_id.isin(held)].copy()
    assert_disjoint(a, b)
    return a, b


def assert_disjoint(*frames):
    for i, a in enumerate(frames):
        validate_frame(a, f'split {i}')
        for b in frames[i+1:]:
            for key in ['patient_id','slide_id','image_path','sha256']:
                if key in a and key in b and set(a[key]) & set(b[key]):
                    raise ValueError(f'Data leakage: overlapping {key}.')


def patient_predictions(df):
    if df.empty or not np.isfinite(df.y_prob.to_numpy(float)).all():
        raise ValueError('Empty/non-finite predictions.')
    label_col = 'patient_label' if 'patient_label' in df else 'label'
    return df.groupby('patient_id', as_index=False).agg(
        y_true=(label_col,'max'), y_prob=('y_prob','mean'), n_tiles=('y_prob','size'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(temp, path)


def lock_config(folder, config):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'run_config.json'
    normalized = json.loads(json.dumps(config, default=str))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f'Configuration/data changed: {folder}. Use a new run directory.')
    elif list(folder.glob('*.pt')):
        raise ValueError(f'Unversioned checkpoints in {folder}; use a new output directory.')
    else:
        atomic_json(path, normalized)


def verify_against_metadata(df, metadata_path, patient_columns):
    meta = metadata_table(metadata_path, patient_columns).set_index('slide_id')
    out = df.copy()
    key = 'slide_id' if 'slide_id' in out else 'image_id'
    if not out[key].isin(meta.index).all():
        raise ValueError('Table contains slides absent from metadata.')
    grades = out[key].map(meta.isup)
    grade_col = 'isup' if 'isup' in out else 'isup_grade'
    if not out[grade_col].map(strict_grade).eq(grades).all():
        raise ValueError('Saved table labels differ from current metadata; rebuild table, not tiles.')
    lenses = out[key].map(meta.objective_lens)
    if not out.objective_lens.map(strict_lens).eq(lenses).all():
        raise ValueError('Saved table magnifications differ from metadata.')
    out['patient_id'] = out[key].map(meta.patient_id)
    out['patient_label'] = out[key].map(meta.patient_label)
    return out
