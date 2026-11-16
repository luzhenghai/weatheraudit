"""Calendar-preserving TimesFM exploration. Read TIMESFM_NOTE.md first.

This long-horizon script does not reproduce legacy tm_results.json and is not
a fair comparison with short-lead numerical weather forecasts.
"""
import argparse
import csv
from datetime import date, timedelta
import gzip
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parent
CONTEXT, TEST_DATES, HORIZON = 512, 306, 384
SITES = [
    ('Quzhou', 'CHM00058633', '2026-09-22'),
    ('Shanghai', 'CHM00058362', '2026-09-22'),
    ('Osaka', 'JAM00047772', '2026-11-06'),
    ('Hotan', 'CHM00051828', '2026-09-22'),
    ('Chuncheon', 'KSM00047101', '2026-11-06'),
    ('Gangneung', 'KSM00047105', '2026-11-06'),
    ('Pohang', 'KSM00047138', '2026-11-06'),
    ('Busan', 'KSM00047159', '2026-11-06'),
]


def load_station(sid):
    path = ROOT/'cache'/f'station_{sid}_v2.csv.gz'
    if not path.exists():
        with zipfile.ZipFile(ROOT/'cache.zip') as archive:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(archive.read(path.name))
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as stream:
        return {date.fromisoformat(row['date']): float(row['TMAX'])
                for row in csv.DictReader(stream) if row.get('TMAX', '')}


def calendar_context(values, start, n=CONTEXT, min_coverage=.8, max_gap=7):
    """Carry past values forward, preserving dates; reject sparse or stale history."""
    days = [start-timedelta(days=n-i) for i in range(n)]
    coverage = sum(day in values for day in days)/n
    detail = dict(observed_fraction=round(coverage, 3))
    if coverage < min_coverage:
        return None, dict(reason='sparse daily context', **detail)
    filled, previous, gap = [], None, 0
    for day in days:
        if day in values:
            previous, gap = values[day], 0
        else:
            gap += 1
        if previous is None or gap > max_gap:
            return None, dict(reason='missing context start or gap exceeds seven days', **detail)
        filled.append(previous)
    return filled, dict(imputation='past-value forward fill', **detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import torch
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    torch.set_num_threads(2)
    model = TimesFM_2p5_200M_torch.from_pretrained(
        'google/timesfm-2.5-200m-pytorch', torch_compile=False,
        local_files_only=args.local_files_only)
    model.compile(ForecastConfig(max_context=CONTEXT, max_horizon=HORIZON,
                                per_core_batch_size=1, normalize_inputs=True))
    rows = []
    for city, sid, end_string in SITES:
        end = date.fromisoformat(end_string)
        start = end-timedelta(days=TEST_DATES-1)
        values = load_station(sid)
        context, detail = calendar_context(values, start)
        row = dict(city=city, station=sid, start=str(start), end=str(end), **detail)
        if context is None:
            row['status'] = 'skipped'
        else:
            points, _ = model.forecast(HORIZON, [np.asarray(context, dtype=float)])
            test_days = [start+timedelta(days=i) for i in range(TEST_DATES)]
            pairs = [(i, values[day]) for i, day in enumerate(test_days) if day in values]
            row.update(status='exploratory_long_horizon', n=len(pairs),
                       timesfm_mae=round(float(np.mean([abs(points[0][i]-v) for i, v in pairs])), 3))
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = ROOT/'tm_results_repaired.json'
    output.write_text(json.dumps(dict(
        warning='Exploratory long-horizon run; unequal information versus weather forecasts.',
        legacy_results='tm_results.json used the old missing-date-compressing script.',
        rows=rows), indent=2), encoding='utf-8')
    print('Saved', output)


if __name__ == '__main__':
    main()
