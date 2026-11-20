#!/usr/bin/env python3
"""Resumable WRDS archive. Run from the project environment; credentials use pgpass.
The WRDS user name is $WRDS_USERNAME, or the local user name when it is not set."""
import argparse
import datetime as dt
import fcntl
import getpass
import os
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import psycopg2

from download_intraday_bars import POLICY, aggregate_sql, write_json, write_parquet

VERSION = 1


class Database:
    def __init__(self):
        self.connection = psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737,
            dbname='wrds', user=os.environ.get('WRDS_USERNAME') or getpass.getuser(), sslmode='require', connect_timeout=30)
        self.connection.autocommit = True
        with self.connection.cursor() as cursor:
            cursor.execute("SET statement_timeout='900000'")

    def raw_sql(self, sql, params=None):
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            return pd.DataFrame(cursor.fetchall(), columns=[x.name for x in cursor.description])

    def close(self):
        self.connection.close()


def cached(db, path, sql, params=None):
    if path.exists():
        return pd.read_parquet(path, partitioning=None)
    print(f'Fetching {path.name}', flush=True)
    frame = db.raw_sql(sql, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, path)
    return frame


def schedule(start, end):
    frame = mcal.get_calendar('NYSE').schedule(start_date=start, end_date=end)
    frame['date'] = frame.index.strftime('%Y-%m-%d')
    if frame.empty:
        frame['minutes'] = pd.Series(dtype='int64')
        return frame.reset_index(drop=True)
    frame['minutes'] = ((frame.market_close-frame.market_open).dt.total_seconds()/60).astype(int)
    if not (frame.market_open.dt.tz_convert('America/New_York').dt.strftime('%H:%M') == '09:30').all():
        raise RuntimeError('Unexpected session open; aggregation requires a 09:30 open')
    return frame.reset_index(drop=True)


def coverage(db):
    result = {}
    for key, table, column in [
        ('daily_reference', 'crsp_a_stock.dsf_v2', 'dlycaldt'),
        ('security_history', 'crsp_a_stock.stksecurityinfohist', 'secinfoenddt'),
        ('sp500_membership', 'crsp_a_indexes.dsp500list_v2', 'mbrenddt'),
        ('distributions', 'crsp_a_stock.stkdistributions', 'disexdt')]:
        result[key] = str(db.raw_sql(f'SELECT max({column}) AS last FROM {table}').iloc[0, 0])
    tables = db.raw_sql("""SELECT table_schema,table_name FROM information_schema.tables
        WHERE table_schema ~ '^taqm_[0-9]{4}$' AND table_name ~ '^ctm_[0-9]{8}$'""")
    result['taq_latest'] = pd.Timestamp(max(tables.table_name.str[4:])).strftime('%Y-%m-%d')
    result['taq_dates'] = sorted(pd.to_datetime(tables.table_name.str[4:]).dt.strftime('%Y-%m-%d').unique().tolist())
    result['joint_end'] = min(result[k] for k in ('daily_reference','security_history','sp500_membership','taq_latest'))
    result['checked_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    result['limitation'] = ('Joint end is bounded by subscribed annual CRSP coverage. '
        'TAQ after this date is deferred, not filled with stale members or adjustment factors. '
        'Distribution max is an event date, not a guaranteed coverage endpoint. '
        'Compustat historical membership is date-effective, not an as-published vintage.')
    return result


def universe(db, root, year, days):
    path = root/'01_universe'/f'year={year}'/'daily_universe.parquet'
    if path.exists():
        return pd.read_parquet(path, partitioning=None)
    rows = db.raw_sql('''WITH days AS (SELECT unnest(%(days)s::date[]) AS date), members AS (
        SELECT d.date,'SP500'::text AS index_name,s.permno,
            'permno:'||s.permno::text AS member_id
        FROM days d JOIN crsp_a_indexes.dsp500list_v2 s
        ON d.date BETWEEN s.mbrstartdt AND coalesce(s.mbrenddt,'9999-12-31'::date)
        UNION ALL
        SELECT d.date,CASE WHEN m.gvkeyx='000208' THEN 'NDX100' ELSE 'DJIA30' END,
            l.lpermno,m.gvkey||':'||m.iid
        FROM days d JOIN comp.idxcst_his m ON d.date>=m."from"
            AND (m.thru IS NULL OR d.date<=m.thru) AND m.gvkeyx IN ('000208','000005')
        LEFT JOIN crsp_a_ccm.linkhistory l ON l.gvkey=m.gvkey AND l.liid=m.iid
            AND l.linktype IN ('LC','LU','LS') AND l.lpermno>0
            AND (l.linkdt IS NULL OR d.date>=l.linkdt)
            AND (l.linkenddt IS NULL OR d.date<=l.linkenddt)
        ) SELECT DISTINCT m.*,coalesce(nullif(n.tradingsymbol,''),n.ticker) AS ticker,n.ticker AS crsp_ticker,n.secinfostartdt,n.secinfoenddt
        FROM members m LEFT JOIN crsp_a_stock.stksecurityinfohist n ON n.permno=m.permno
            AND m.date BETWEEN n.secinfostartdt AND n.secinfoenddt
        ORDER BY m.date,m.index_name,m.member_id''', {'days':days})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, path.parent/'source_daily_memberships.parquet')
    inactive = set()
    records = rows.to_dict('records')
    for row in records:
        if pd.isna(row['permno']):
            raise RuntimeError(f'Unlinked member: {row}')
        if pd.isna(row['ticker']):
            prior = db.raw_sql('''SELECT coalesce(nullif(tradingsymbol,''),ticker) AS ticker,secinfoenddt FROM crsp_a_stock.stksecurityinfohist
                WHERE permno=%(p)s AND secinfostartdt<=%(d)s AND ticker IS NOT NULL
                ORDER BY secinfostartdt DESC LIMIT 1''', {'p':int(row['permno']), 'd':row['date']})
            if prior.empty or str(prior.iloc[0].secinfoenddt)[:10]>=str(row['date'])[:10]:
                raise RuntimeError(f'Unresolved historical ticker: {row}')
            row['ticker'] = prior.iloc[0].ticker
            inactive.add((str(row['date'])[:10],int(row['permno'])))
    source = pd.DataFrame(records)
    source['date'] = source.date.astype(str).str[:10]
    source['permno'] = source.permno.astype('int64')
    if source.ticker.isna().any() or source.groupby(['date','index_name','member_id']).permno.nunique().gt(1).any():
        raise RuntimeError('Unresolved or ambiguous historical membership')
    if source.groupby(['date','permno']).ticker.nunique().gt(1).any():
        raise RuntimeError('Conflicting tickers for a security')
    if set(source.date)!=set(days) or source.groupby('date').index_name.nunique().ne(3).any():
        raise RuntimeError('Missing daily index membership')
    frame = source.groupby(['date','permno','ticker'],as_index=False).agg(
        indices=('index_name',lambda x:'|'.join(sorted(set(x)))))
    tickers = {day:sorted(set(group.ticker)) for day,group in frame.groupby('date')}
    frame['identifier_status'] = ['last_known_inactive' if (r.date,r.permno) in inactive
                                   else 'date_effective' for r in frame.itertuples()]
    active = frame[frame.identifier_status.eq('date_effective')]
    if active.duplicated(['date','ticker']).any():
        raise RuntimeError('Same ticker identifies multiple active securities')
    write_json(path.parent/'daily_tickers.json',tickers)
    write_parquet(frame,path)
    return frame


def mapping(db, root, year, members):
    path = root/'02_identifier_mapping'/f'year={year}'/'daily_taq_mapping.parquet'
    if path.exists():
        return pd.read_parquet(path, partitioning=None)
    params = {'start':members.date.min(), 'end':members.date.max(),
              'permnos':sorted(int(x) for x in members.permno.unique())}
    master = cached(db,path.parent/'source_cusip_master.parquet',f'''SELECT m.date,n.permno,m.symbol_15,m.cusip
        FROM taqm_{year}.mastm_{year} m JOIN crsp_a_stock.stksecurityinfohist n
          ON substring(m.cusip,1,8)=n.cusip AND m.date BETWEEN n.secinfostartdt AND n.secinfoenddt
        WHERE m.date BETWEEN %(start)s AND %(end)s AND n.permno=ANY(%(permnos)s)
          AND (m.test_symbol_flag='N' OR m.test_symbol_flag IS NULL) ''',params)
    master['date'] = pd.to_datetime(master.date).dt.strftime('%Y-%m-%d')
    master = master.merge(members[['date','permno','ticker']], on=['date','permno'])
    parts = master.symbol_15.str.strip().str.split()
    if not parts.map(len).isin([1,2]).all():
        raise RuntimeError('Unexpected master symbol format')
    master['sym_root'] = parts.str[0]
    master['sym_suffix'] = parts.map(lambda p:p[1] if len(p)==2 else '')
    master['mapping_source'] = 'ciz_cusip_daily_master'
    master['match_lvl'] = float('nan')
    cols = ['date','permno','ticker','sym_root','sym_suffix','mapping_source','match_lvl']
    master = master[cols].drop_duplicates()
    # Separate regular securities from when-issued/distribution variants sharing CUSIP.
    canonical = lambda series: series.str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    master['matches_trading_symbol'] = canonical(master.sym_root+master.sym_suffix).eq(canonical(master.ticker))
    ambiguous = master.duplicated(['date','permno'],keep=False)
    write_parquet(master[ambiguous],path.parent/'master_symbol_candidates.parquet')
    master = master[~ambiguous | master.matches_trading_symbol].copy()
    if master.duplicated(['date','permno']).any():
        raise RuntimeError(f'{year}: ambiguous master symbols after trading-symbol verification')
    link = cached(db,path.parent/'source_wrds_daily_link.parquet','''SELECT date,permno,sym_root,sym_suffix,ticker,match_lvl
        FROM wrdsapps_link_crsp_taqm.taqmclink WHERE date BETWEEN %(start)s AND %(end)s
        AND permno=ANY(%(permnos)s)''',params)
    link['date'] = pd.to_datetime(link.date).dt.strftime('%Y-%m-%d')
    link['sym_root'] = link.sym_root.str.strip()
    link['sym_suffix'] = link.sym_suffix.fillna('').str.strip()
    link = link.merge(members[['date','permno']],on=['date','permno']).drop_duplicates()
    link['mapping_source'] = 'wrds_daily_link'
    common = link.merge(master,on=['date','permno'],suffixes=('_link','_master'))
    conflict = common[(common.sym_root_link!=common.sym_root_master)|(common.sym_suffix_link!=common.sym_suffix_master)]
    write_parquet(conflict,path.parent/'link_master_disagreements.parquet')
    if not conflict.empty:
        if not conflict.matches_trading_symbol.all():
            raise RuntimeError(f'{year}: unresolvable link/master conflict')
        # A unique CUSIP match independently confirmed by CIZ trading symbol wins.
        keys = pd.MultiIndex.from_frame(conflict[['date','permno']])
        link = link[~pd.MultiIndex.from_frame(link[['date','permno']]).isin(keys)]
    extra = master.merge(link[['date','permno']],on=['date','permno'],how='left',indicator=True)
    result = pd.concat([link[cols],extra.loc[extra['_merge'].eq('left_only'),cols]],ignore_index=True)
    write_parquet(result,path.parent/'resolved_mapping_candidates.parquet')
    status = result.merge(members[['date','permno','identifier_status']],on=['date','permno'],validate='many_to_one')
    collision = status.duplicated(['date','sym_root','sym_suffix'],keep=False)
    removed = status[collision & status.identifier_status.eq('last_known_inactive')]
    write_parquet(removed,path.parent/'inactive_symbol_reuse.parquet')
    result = status.loc[~(collision & status.identifier_status.eq('last_known_inactive')),cols]

    for keys in [['date','permno'],['date','sym_root','sym_suffix']]:
        if result.duplicated(keys).any():
            raise RuntimeError(f'{year}: ambiguous identifier mapping on {keys}')
    audit = members.merge(result[['date','permno','sym_root']],on=['date','permno'],how='left')
    audit = audit[audit.sym_root.isna()]
    write_parquet(audit,path.parent/'unmapped_members.parquet')
    if not audit.empty:
        print(f'{year}: {len(audit)} unmapped security-days retained as missing bars; see audit',flush=True)
    result = result.sort_values(['date','permno']).reset_index(drop=True)
    write_parquet(result,path)
    return result


def symbol_key(root, suffix):
    """Root and suffix joined without a separator: 'GOOG'+'L' and 'GOOGL'+'' are the same security."""
    return (root.fillna('').str.strip()+suffix.fillna('').str.strip()).str.upper()


def symbol_params(daymap):
    """Query parameters for one session's trades.

    The TAQ daily master writes some Nasdaq share classes as one five-letter root ('GOOGL') on some
    dates and as root plus suffix ('GOOG L') on others; in 2016 it switched on 2016-10-24, while the
    WRDS daily link has no rows at all for these classes that year (GOOGL, CMCSA, DISCA, DISCK, LBTYA,
    LVNTA). A five-letter root with no suffix therefore also asks for its split form, and
    archive_grid joins bars on the combined key, so either spelling in the trade table matches.
    """
    roots, symbols = set(daymap.sym_root), set(daymap.sym_root+'|'+daymap.sym_suffix)
    split = daymap[daymap.sym_suffix.eq('') & daymap.sym_root.str.len().eq(5)]
    roots |= set(split.sym_root.str[:4])
    symbols |= set(split.sym_root.str[:4]+'|'+split.sym_root.str[4])
    return {'roots':sorted(roots),'symbols':sorted(symbols)}


def archive_grid(day, universe, mapping, bars, minutes, session_minutes):
    import numpy as np
    import pandas as pd
    members = universe[universe.date.eq(day)].copy()
    symbols = mapping[mapping.date.eq(day)].drop(columns=['date','ticker'])
    members = members.merge(symbols,on='permno',how='left',validate='one_to_one')
    if 'mapping_source' not in members:
        members['mapping_source'] = 'wrds_daily_link'
    missing_map = members.sym_root.isna()
    buckets = pd.DataFrame({'bucket':range(session_minutes//minutes)})
    grid = members.merge(buckets,how='cross')
    bars['sym_suffix'] = bars.sym_suffix.fillna('').str.strip()
    bars['sym_root'] = bars.sym_root.str.strip()
    bars['bucket'] = bars.bucket.astype('int32')
    # Join on the combined key; two spellings of one key in the same session is an error (many_to_one).
    grid['key'] = symbol_key(grid.sym_root,grid.sym_suffix)
    bars['key'] = symbol_key(bars.sym_root,bars.sym_suffix)
    bars = bars.drop(columns=['sym_root','sym_suffix'])
    result = grid.merge(bars,on=['key','bucket'],how='left',validate='many_to_one').drop(columns='key')
    has_trade = result.close.notna()
    result['status'] = np.where(result.sym_root.isna(),'no_symbol_mapping',
                              np.where(has_trade,'observed',
                                       np.where(result.trade_count.notna(),'no_priced_trades','no_eligible_trades')))
    for col in ['volume','trade_count']:
        result[col] = result[col].fillna(0).astype('int64')
    for col in ['open','high','low','close','vwap']:
        result[col] = pd.to_numeric(result[col],errors='raise').astype('float64')
    result['bar_start'] = (pd.Timestamp(day+' 09:30:00',tz='America/New_York')
                           + pd.to_timedelta(result.bucket*minutes,unit='min'))
    result['bar_end'] = result.bar_start + pd.Timedelta(minutes=minutes)
    observed = result[result.status.eq('observed')]
    if not ((observed.high >= observed[['open','close','low']].max(axis=1)).all()
            and (observed.low <= observed[['open','close','high']].min(axis=1)).all()
            and (observed[['open','high','low','close']] > 0).all().all()):
        raise RuntimeError(f'{day}: invalid OHLC values')
    if result.duplicated(['permno','bar_start']).any():
        raise RuntimeError('Duplicate security/bar timestamp')
    columns = ['date','bar_start','bar_end','permno','ticker','indices','sym_root','sym_suffix',
               'identifier_status','mapping_source','match_lvl','open','high','low','close','volume','trade_count','vwap','status']
    quality = {'date':day,'members':len(members),'grid_rows':len(result),
               'observed_rows':len(observed),'empty_rows':int((~has_trade).sum()),
               'unmapped_members':int(missing_map.sum()),
               'members_with_no_eligible_trades':sorted(int(x) for x in
                   set(members.permno)-set(observed.permno)),
               'eligible_trade_count':int(result.trade_count.sum()),
               'eligible_volume':int(result.volume.sum()),
               'match_level_counts':{str(k):int(v) for k,v in members.match_lvl.value_counts().items()}}
    return result[columns].sort_values(['permno','bar_start']).reset_index(drop=True), quality


def select_sessions(cal, limit_days=0, days=None):
    """Sessions to download: all, the first `limit_days`, or the listed dates (a pilot spread over
    several periods). Listed dates must be sessions of this archive's calendar."""
    if days:
        wanted = sorted({d.strip() for d in days.split(',') if d.strip()})
        unknown = sorted(set(wanted)-set(cal.date))
        if unknown:
            raise ValueError(f'Not sessions of this archive: {unknown}')
        return cal.loc[cal.date.isin(wanted)]
    return cal.iloc[:limit_days] if limit_days else cal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',default='2016-01-01')
    parser.add_argument('--limit-days',type=int,default=0,help='Pilot only; zero downloads all days')
    parser.add_argument('--days',default='',help='Pilot only: comma-separated session dates to download; '
                        'the full run later resumes in the same directory and skips them')
    parser.add_argument('--metadata-only',action='store_true')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'.download.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta = root/'00_metadata'; meta.mkdir(exist_ok=True)
    db = Database()
    cp = meta/'coverage.json'
    cov = json.loads(cp.read_text()) if cp.exists() else coverage(db)
    write_json(cp,cov)
    end = cov['joint_end']
    config = {'version':VERSION,'start':args.start,'end':end,'minutes':5,'calendar':'NYSE',
        'policy':{**POLICY,'calendar':'NYSE regular and early-close sessions',
                  'session':'[calendar market_open,calendar market_close)',
                  'mapping':'Date-effective CIZ CUSIP plus daily TAQ master, cross-checked with WRDS daily link',
                  'adjustment':'Unadjusted bars; full CIZ daily and distribution reference saved separately'}}
    cfg = meta/'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise RuntimeError('Configuration differs from archive snapshot; use a new directory')
    write_json(cfg,config)
    cal = schedule(args.start,end)
    missing = set(cal.date)-set(cov['taq_dates'])
    if missing:
        raise RuntimeError(f'Calendar sessions without TAQ tables: {sorted(missing)}')
    write_parquet(cal,meta/'exchange_calendar.parquet')
    print(f'Joint coverage: {args.start} to {end}; TAQ latest {cov["taq_latest"]}; {len(cal)} sessions',flush=True)
    params = {'start':args.start,'end':end}
    cached(db,meta/'source_sp500_membership.parquet','''SELECT * FROM crsp_a_indexes.dsp500list_v2
        WHERE mbrstartdt<=%(end)s AND (mbrenddt IS NULL OR mbrenddt>=%(start)s)''',params)
    cached(db,meta/'source_compustat_membership.parquet','''SELECT * FROM comp.idxcst_his
        WHERE gvkeyx IN ('000208','000005') AND "from"<=%(end)s AND (thru IS NULL OR thru>=%(start)s)''',params)
    cached(db,meta/'source_ccm_links.parquet','''SELECT * FROM crsp_a_ccm.linkhistory
        WHERE gvkey IN (SELECT gvkey FROM comp.idxcst_his WHERE gvkeyx IN ('000208','000005'))
        AND (linkdt IS NULL OR linkdt<=%(end)s) AND (linkenddt IS NULL OR linkenddt>=%(start)s)''',params)
    universes = {}
    for year in sorted(cal.date.str[:4].unique()):
        days = cal.loc[cal.date.str.startswith(year),'date'].tolist()
        universes[year] = universe(db,root,year,days)
        print(f'{year}: {universes[year].permno.nunique()} historical securities',flush=True)
    ids = sorted(set(int(x) for frame in universes.values() for x in frame.permno))
    cached(db,meta/'source_security_history.parquet','''SELECT * FROM crsp_a_stock.stksecurityinfohist
        WHERE permno=ANY(%(ids)s)''',{'ids':ids})
    # Every ever-member's daily/event history is retained even outside index membership.
    for year in universes:
        start = max(args.start,f'{year}-01-01')
        if year == str(pd.Timestamp(args.start).year):
            start = (pd.Timestamp(start)-pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        p = {'start':start,'end':min(end,f'{year}-12-31'),'ids':ids}
        cached(db,root/'04_corporate_actions'/f'year={year}'/'distributions.parquet','''SELECT *
            FROM crsp_a_stock.stkdistributions WHERE permno=ANY(%(ids)s)
            AND disexdt BETWEEN %(start)s AND %(end)s''',p)
        cached(db,root/'05_daily_reference'/f'year={year}'/'daily_reference.parquet','''SELECT *
            FROM crsp_a_stock.dsf_v2 WHERE permno=ANY(%(ids)s)
            AND dlycaldt BETWEEN %(start)s AND %(end)s''',p)
    progress_path = meta/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'completed':{}}
    progress.update(status='running',requested_sessions=len(cal),reference_status='downloaded')
    write_json(progress_path,progress)
    if args.metadata_only:
        print('Membership and corporate-action references downloaded; bars not requested.',flush=True)
        db.close(); return
    selected = select_sessions(cal,args.limit_days,args.days)
    mappings = {}
    for row in selected.itertuples():
        day, year = row.date,row.date[:4]
        path = root/'03_raw_bars_5min'/f'year={year}'/f'month={day[5:7]}'/f'{day}.parquet'
        old = progress['completed'].get(day)
        if old and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==old['sha256']:
            continue
        if year not in mappings:
            mappings[year] = mapping(db,root,year,universes[year])
        daymap = mappings[year].loc[mappings[year].date.eq(day)]
        p = symbol_params(daymap)
        sql = aggregate_sql(f'taqm_{year}.ctm_{day.replace("-", "")}',5)
        close = row.market_close.tz_convert('America/New_York').strftime('%H:%M:%S')
        sql = sql.replace("'16:00:00'::time",f"'{close}'::time")
        started = time.monotonic()
        bars = db.raw_sql(sql,p)
        result, quality = archive_grid(day,universes[year],mappings[year],bars,5,row.minutes)
        result = result[result.bar_end<=row.market_close].reset_index(drop=True)
        if len(result)!=len(universes[year].loc[universes[year].date.eq(day)])*(row.minutes//5):
            raise RuntimeError(f'{day}: invalid session grid size')
        quality.update(grid_rows=len(result),observed_rows=int(result.status.eq('observed').sum()),
                       empty_rows=int(result.status.ne('observed').sum()),session_minutes=row.minutes)
        path.parent.mkdir(parents=True,exist_ok=True)
        write_parquet(result,path)
        quality.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
                       seconds=round(time.monotonic()-started,2))
        progress['completed'][day]=quality
        progress['updated_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(progress_path,progress)
        print(f'{day}: {len(result):,} rows, {quality["seconds"]} s; {len(progress["completed"])}/{len(cal)} sessions',flush=True)
    progress['status']='complete' if len(progress['completed'])==len(cal) else 'pilot_complete'
    write_json(progress_path,progress)
    db.close()


if __name__ == '__main__':
    main()
