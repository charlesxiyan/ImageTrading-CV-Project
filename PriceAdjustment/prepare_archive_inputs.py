#!/usr/bin/env python3
"""Convert a WRDS archive (stock universe or benchmark ETF, CRSP CIZ format) into the inputs of adjust_dividend_bars.py.

The archive written by the downloader keeps raw bars in year=/month= partitions and saves
CRSP's CIZ daily and distribution tables. The adjustment scripts expect the demo layout:
a raw-root manifest, a daily factor table (date, permno, cfacpr, cfacshr), and distribution
inputs (distributions.parquet, daily_references.parquet). This script writes only those small
files to a new directory. Raw bars are referenced by absolute path and checked against the
downloader's SHA-256; nothing in the archive is modified.

Distribution classification follows the price adjustment policy in CLAUDE.md:
- divamt > 0 and facpr = facshr = 0: proportional cash adjustment (`kind = cash`). This covers
  ordinary and special dividends, return of capital, capital gains and non-USD dividends, whose
  CIZ amounts are already in USD (checked against CRSP ret - retx).
- facpr or facshr nonzero: handled by the cumulative factors alone (`kind = corporate`).
- anything else stops the run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adjust_bar_prices import sha256

DEFAULT_ARCHIVE = Path.home()/'Market Data/US Equity/v4/01_raw'


def read_partitions(root, pattern, columns=None):
    files = sorted(root.glob(pattern))
    if not files:
        raise ValueError(f'No files for {pattern}')
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    return pd.concat(frames, ignore_index=True)


def number(series):
    return pd.to_numeric(series, errors='raise').astype('float64')


def build_daily(archive):
    cols = ['permno', 'dlycaldt', 'dlyprc', 'dlycumfacpr', 'dlycumfacshr']
    r = read_partitions(archive, '05_daily_reference/**/daily_reference.parquet', cols)
    r = r.rename(columns={'dlycaldt': 'date', 'dlyprc': 'prc', 'dlycumfacpr': 'cfacpr', 'dlycumfacshr': 'cfacshr'})
    r['date'] = pd.to_datetime(r.date.astype(str), errors='raise').dt.strftime('%Y-%m-%d')
    r['permno'] = r.permno.astype('int64')
    for c in ['prc', 'cfacpr', 'cfacshr']:
        r[c] = number(r[c])
    if r.duplicated(['permno', 'date']).any():
        raise ValueError('Duplicate daily reference rows')
    bad = ~(np.isfinite(r.cfacpr) & r.cfacpr.gt(0) & np.isfinite(r.cfacshr) & r.cfacshr.gt(0))
    # Rows without valid factors are left out; a bar that needs one then fails loudly downstream.
    return r[~bad].sort_values(['permno', 'date']).reset_index(drop=True), int(bad.sum())


def build_events(archive):
    d = read_partitions(archive, '04_corporate_actions/**/distributions.parquet')
    d = d.drop(columns=[c for c in ['year'] if c in d])
    for c in ['disdivamt', 'disfacpr', 'disfacshr']:
        d[c] = number(d[c]).fillna(0.0)
    d['exdt'] = pd.to_datetime(d.disexdt.astype(str), errors='raise').dt.strftime('%Y-%m-%d')
    if d.duplicated(['permno', 'exdt', 'disseqnbr']).any():
        raise ValueError('Duplicate distribution records')
    corporate = d.disfacpr.ne(0) | d.disfacshr.ne(0)
    cash = d.disdivamt.gt(0) & ~corporate
    if not (cash | corporate).all():
        raise ValueError('Distribution with neither a cash amount nor a factor; classify first')
    events = pd.DataFrame({
        'permno': d.permno.astype('int64'), 'exdt': d.exdt, 'disseqnbr': d.disseqnbr.astype('int64'),
        'kind': np.where(cash, 'cash', 'corporate'),
        'divamt': d.disdivamt, 'facpr': d.disfacpr, 'facshr': d.disfacshr,
        'distype': d.distype, 'disdetailtype': d.disdetailtype,
        'dispaymenttype': d.dispaymenttype, 'disorigcurtype': d.disorigcurtype,
    })
    return events.sort_values(['permno', 'exdt', 'disseqnbr']).reset_index(drop=True)


def build_manifest(archive):
    progress = json.loads((archive/'00_metadata/progress.json').read_text())['completed']
    config = json.loads((archive/'00_metadata/config.json').read_text())
    calendar = pd.read_parquet(archive/'00_metadata/exchange_calendar.parquet')
    days = sorted(str(x) for x in calendar.date.astype(str))
    if set(days) != set(progress):
        raise ValueError('Archive is incomplete: calendar and completed sessions differ')
    completed = {}
    for day in days:
        path = archive/'03_raw_bars_5min'/f'year={day[:4]}'/f'month={day[5:7]}'/f'{day}.parquet'
        digest = sha256(path)
        if digest != progress[day]['sha256']:
            raise ValueError(f'Checksum mismatch for {path}')
        completed[day] = {'file': str(path.resolve()), 'sha256': digest, 'rows': progress[day]['grid_rows'],
                          'session_minutes': progress[day]['session_minutes']}
    return {'status': 'complete', 'source': 'wrds_5min_2016_latest_v1 archive (CRSP CIZ)',
            'archive_root': str(archive.resolve()), 'config': config,
            'requested_trading_days': days, 'completed': completed}


def run(archive, output):
    if output.exists():
        raise ValueError('Output already exists; use a new directory')
    daily, dropped = build_daily(archive)
    events = build_events(archive)
    manifest = build_manifest(archive)
    output.mkdir(parents=True)
    daily[['date', 'permno', 'cfacpr', 'cfacshr']].to_parquet(output/'crsp_daily_adjustment_factors.parquet', index=False)
    daily[['permno', 'date', 'prc', 'cfacpr', 'cfacshr']].to_parquet(output/'daily_references.parquet', index=False)
    events.to_parquet(output/'distributions.parquet', index=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    summary = {'output': str(output), 'sessions': len(manifest['completed']),
               'early_close_sessions': sum(v['session_minutes'] != 390 for v in manifest['completed'].values()),
               'daily_rows': len(daily), 'daily_rows_without_factors': dropped,
               'permnos': int(daily.permno.nunique()), 'events': len(events),
               'event_kinds': events.kind.value_counts().to_dict()}
    (output/'inputs_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument('--output', type=Path, default=Path(str(DEFAULT_ARCHIVE)+'_adjustment_inputs'))
    a = p.parse_args()
    run(a.archive, a.output)
