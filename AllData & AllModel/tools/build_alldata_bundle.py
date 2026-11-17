#!/usr/bin/env python3
"""Package the 2016-2025 prepared stock data plus aligned SPY bars into one Colab zip.

    python "AllData & AllModel/tools/build_alldata_bundle.py" \
        --prepared   "~/Market Data/US Equity/cnn78u_5min_2016_2025/prepared" \
        --market     "~/Market Data/US Equity/wrds_5min_2016_latest_v1_spy_dividend_adjusted" \
        --output     "~/Market Data/US Equity/alldata_bundle_v1.zip"

Bundle layout (unzips to prepared/ + chartcnn/):
    prepared/features.npy            (stocks, slots, OHLCV) float32, end-of-data basis (raw * M)
    prepared/multipliers.npy         (stocks, slots, 2) float32: price M, volume V per bar
    prepared/valid.npy               (stocks, slots) bool; early-close sessions are all False
    prepared/market_features.npy     (slots, OHLCV) float32: SPY on the same clock and basis
    prepared/market_multipliers.npy  (slots, 2) float32
    prepared/market_valid.npy        (slots,) bool
    prepared/{train,validation,test}.parquet  windows thinned to end_index % 6 == 5 (stride 6 and
                                     12 are subsets); every audited window otherwise kept
    prepared/manifest.json           us_pipeline manifest + bundle notes
Model inputs are point in time: each window is divided by its last bar's multipliers
(chartcnn.images.point_in_time_window), for the stock and for SPY alike.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
THIN = 6


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def market_arrays(market_root, clock, slots_per_day=78):
    """SPY bars placed on the prepared clock (bar end times)."""
    manifest = json.loads((market_root/'manifest.json').read_text())
    raw = json.loads((Path(manifest['raw_root'])/'manifest.json').read_text())['completed']
    days = list(dict.fromkeys(clock.tz_convert('America/New_York').strftime('%Y-%m-%d')))
    if len(days)*slots_per_day != len(clock):
        raise ValueError('Clock is not 78 slots per session')
    if set(days) != set(manifest['completed']):
        raise ValueError('SPY sessions differ from the stock sessions')
    x = np.full((len(clock), 5), np.nan); m = np.full((len(clock), 2), np.nan)
    ok = np.zeros(len(clock), bool)
    for d, day in enumerate(days):
        f = pd.read_parquet(market_root/manifest['completed'][day]['file'], partitioning=None)
        if len(f.permno.unique()) != 1:
            raise ValueError('Market file must hold one security')
        session = raw[day]['session_minutes']//5
        if len(f) != session:
            raise ValueError(f'{day}: expected {session} SPY bars, got {len(f)}')
        ends = pd.DatetimeIndex(f.bar_end).tz_convert('UTC').as_unit('ns')
        sl = slice(d*slots_per_day, d*slots_per_day+session)
        if not (ends.asi8 == clock[sl].as_unit('ns').asi8).all():
            raise ValueError(f'{day}: SPY bar ends do not match the stock clock')
        v = f[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
        good = (f.status.eq('observed').to_numpy() & np.isfinite(v).all(1) & (v > 0).all(1)
                & (v[:, 1] >= v[:, :4].max(1)) & (v[:, 2] <= v[:, :4].min(1)))
        x[sl] = v; m[sl] = f[['price_adjustment_multiplier', 'volume_adjustment_multiplier']].to_numpy(float)
        ok[sl] = good if session == slots_per_day else False      # early close: excluded like the stocks
    return x, m, ok


def build(prepared, market_root, output):
    prepared, market_root, output = Path(prepared), Path(market_root), Path(output)
    stage = output.with_suffix('.staging')
    if output.exists() or stage.exists():
        raise ValueError(f'{output} or its staging folder exists; choose a new name')
    m = json.loads((prepared/'manifest.json').read_text())
    audit = json.loads((prepared/'validation_report.json').read_text())
    if m['status'] != 'complete' or audit['status'] != 'validated_complete':
        raise ValueError('Run us_pipeline prepare and audit first')
    out = stage/'prepared'; out.mkdir(parents=True)
    clock = pd.to_datetime(m['decision_clock_ns'], unit='ns', utc=True)

    x, mm, ok = market_arrays(market_root, pd.DatetimeIndex(clock))
    np.save(out/'market_features.npy', x.astype(np.float32))
    np.save(out/'market_multipliers.npy', mm.astype(np.float32))
    np.save(out/'market_valid.npy', ok)

    src = np.load(prepared/'features.npy', mmap_mode='r')
    mul = np.load(prepared/'multipliers.npy', mmap_mode='r')
    f32 = np.lib.format.open_memmap(out/'features.npy', 'w+', np.float32, src.shape)
    m32 = np.lib.format.open_memmap(out/'multipliers.npy', 'w+', np.float32, mul.shape)
    for s in range(src.shape[0]):
        f32[s] = src[s]; m32[s] = mul[s]
    f32.flush(); m32.flush(); del f32, m32
    np.save(out/'valid.npy', np.load(prepared/'valid.npy'))

    counts = {}
    for split in ('train', 'validation', 'test'):
        w = pd.read_parquet(prepared/f'{split}.parquet')
        w = w[w.end_index % THIN == THIN-1]
        w.to_parquet(out/f'{split}.parquet', index=False, compression='zstd')
        counts[split] = len(w)
    m['bundle'] = {'name': output.name, 'thinning': f'end_index % {THIN} == {THIN-1}', 'windows': counts,
                   'dtype': 'float32 features and multipliers', 'market': 'SPY (PERMNO 84398), same filters, '
                   'same clock, adjusted with the same CIZ cash-dividend and factor rules',
                   'market_valid_slots': int(ok.sum()), 'source_audit': audit}
    (out/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')

    files = {f'prepared/{p.name}': p for p in sorted(out.iterdir())}
    files.update({f'chartcnn/{p.name}': p for p in sorted((HERE/'chartcnn').glob('*.py'))})
    report = {'format_version': 1, 'bundle': output.name, 'windows': counts, 'files': {}}
    tmp = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for rel, path in files.items():
            report['files'][rel] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
            z.write(path, rel, compress_type=zipfile.ZIP_STORED if path.suffix == '.parquet' else zipfile.ZIP_DEFLATED)
        z.writestr('bundle_manifest.json', json.dumps(report, indent=2)+'\n')
    tmp.replace(output)
    summary = {'zip': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output), 'windows': counts,
               'market_valid_slots': int(ok.sum()), 'slots': len(clock)}
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True, type=lambda s: Path(s).expanduser())
    p.add_argument('--market', required=True, type=lambda s: Path(s).expanduser())
    p.add_argument('--output', required=True, type=lambda s: Path(s).expanduser())
    a = p.parse_args()
    build(a.prepared, a.market, a.output)
