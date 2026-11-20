#!/usr/bin/env python3
"""Download five-minute bars for one benchmark ETF (default SPY) on the stock archive's calendar.

Same trade filter, aggregation SQL, session grid, bar schema and partition layout as
download_us_equity_archive.py, so the benchmark aligns bar for bar with the stock archive.
Sessions are read from the stock archive's 00_metadata/exchange_calendar.parquet. CRSP CIZ daily
records and distributions for the ETF are saved too (SPY pays quarterly cash dividends).
Credentials come from pgpass. Resumable: completed sessions are verified by SHA-256 and skipped.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from download_intraday_bars import aggregate_sql, write_json, write_parquet
from download_us_equity_archive import Database, archive_grid, cached

STOCK_ARCHIVE = Path.home()/'Market Data/US Equity/v3/01_raw'


def resolve_permno(db, symbol, start, end):
    frame = db.raw_sql('''SELECT DISTINCT permno, ticker, tradingsymbol, secinfostartdt, secinfoenddt
        FROM crsp_a_stock.stksecurityinfohist
        WHERE tradingsymbol=%(s)s AND secinfostartdt<=%(end)s AND secinfoenddt>=%(start)s''',
        {'s': symbol, 'start': start, 'end': end})
    if frame.permno.nunique() != 1:
        raise RuntimeError(f'{symbol}: expected one PERMNO over the period, got {frame.to_dict("records")}')
    return int(frame.permno.iloc[0]), frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbol', default='SPY')
    p.add_argument('--stock-archive', type=Path, default=STOCK_ARCHIVE)
    p.add_argument('--output', type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit-days', type=int, default=0)
    a = p.parse_args()
    symbol = a.symbol.upper()
    root = a.output or Path(f'{a.stock_archive}_{symbol.lower()}')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'.download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = root/'00_metadata'; meta.mkdir(exist_ok=True)
    cal = pd.read_parquet(a.stock_archive/'00_metadata/exchange_calendar.parquet')
    stock_config = json.loads((a.stock_archive/'00_metadata/config.json').read_text())
    start, end = cal.date.min(), cal.date.max()
    db = Database()
    permno, history = resolve_permno(db, symbol, start, end)
    write_parquet(history, meta/'source_security_history.parquet')
    write_parquet(cal, meta/'exchange_calendar.parquet')
    config = {'version': 1, 'symbol': symbol, 'permno': permno, 'start': start, 'end': end, 'minutes': 5,
              'aligned_with': str(a.stock_archive), 'policy': stock_config['policy'],
              'mapping': f'Fixed TAQ symbol {symbol} with empty suffix on every session'}
    cfg = meta/'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise RuntimeError('Configuration differs from this archive; use a new directory')
    write_json(cfg, config)
    ref_start = (pd.Timestamp(start)-pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    q = {'p': permno, 'start': ref_start, 'end': end}
    cached(db, root/'04_corporate_actions/distributions.parquet', '''SELECT * FROM crsp_a_stock.stkdistributions
        WHERE permno=%(p)s AND disexdt BETWEEN %(start)s AND %(end)s''', q)
    cached(db, root/'05_daily_reference/daily_reference.parquet', '''SELECT * FROM crsp_a_stock.dsf_v2
        WHERE permno=%(p)s AND dlycaldt BETWEEN %(start)s AND %(end)s''', q)
    db.close()
    print(f'{symbol}: PERMNO {permno}; {len(cal)} sessions {start} to {end}', flush=True)

    progress_path = meta/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'completed': {}}
    progress.update(status='running', requested_sessions=len(cal))
    write_json(progress_path, progress)
    todo = []
    for row in (cal.iloc[:a.limit_days] if a.limit_days else cal).itertuples():
        path = root/'03_raw_bars_5min'/f'year={row.date[:4]}'/f'month={row.date[5:7]}'/f'{row.date}.parquet'
        old = progress['completed'].get(row.date)
        if not (old and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == old['sha256']):
            todo.append((row, path))
    local = threading.local(); guard = threading.Lock()

    def fetch(item):
        row, path = item
        if not hasattr(local, 'db'):
            local.db = Database()
        day = row.date
        universe = pd.DataFrame({'date': [day], 'permno': [permno], 'ticker': [symbol],
                                 'indices': ['BENCHMARK'], 'identifier_status': ['benchmark']})
        mapping = pd.DataFrame({'date': [day], 'permno': [permno], 'ticker': [symbol], 'sym_root': [symbol],
                                'sym_suffix': [''], 'mapping_source': ['fixed_symbol'], 'match_lvl': [float('nan')]})
        sql = aggregate_sql(f'taqm_{day[:4]}.ctm_{day.replace("-", "")}', 5)
        close = row.market_close.tz_convert('America/New_York').strftime('%H:%M:%S')
        sql = sql.replace("'16:00:00'::time", f"'{close}'::time")
        started = time.monotonic()
        bars = local.db.raw_sql(sql, {'roots': [symbol], 'symbols': [symbol+'|']})
        result, quality = archive_grid(day, universe, mapping, bars, 5, row.minutes)
        result = result[result.bar_end <= row.market_close].reset_index(drop=True)
        if len(result) != row.minutes//5:
            raise RuntimeError(f'{day}: invalid session grid size')
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(result, path)
        quality.update(grid_rows=len(result), observed_rows=int(result.status.eq('observed').sum()),
                       empty_rows=int(result.status.ne('observed').sum()), session_minutes=row.minutes,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                       seconds=round(time.monotonic()-started, 2))
        with guard:
            progress['completed'][day] = quality
            progress['updated_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(progress_path, progress)
            print(f'{day}: {quality["observed_rows"]}/{len(result)} observed, {quality["seconds"]} s; '
                  f'{len(progress["completed"])}/{len(cal)} sessions', flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in [pool.submit(fetch, item) for item in todo]:
            future.result()
    progress['status'] = 'complete' if len(progress['completed']) == len(cal) else 'pilot_complete'
    write_json(progress_path, progress)


if __name__ == '__main__':
    main()
