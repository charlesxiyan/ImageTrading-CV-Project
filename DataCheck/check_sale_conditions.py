#!/usr/bin/env python3
"""Show how the trade filter treats each TAQ sale condition, per symbol (read-only WRDS query).

For every symbol it prints the share of regular-session volume kept by
  v1: tr_scond IS NOT NULL AND translate(tr_scond,' @EF','') = ''   (2023 H1 demo)
  v2: translate(coalesce(tr_scond,''),' @EF','') = ''               (download_intraday_bars.py now)
and, with --detail, the volume by sale condition. A blank condition (NULL in WRDS) is the regular
sale on Tape A/B; Tape C marks it '@'. Results of 2026-11-19 are in DATA_REVIEW_2026-11-19.md.

    python DataCheck/check_sale_conditions.py --date 2023-01-09 --symbols JPM KO AAPL MSFT SPY
"""
import argparse
import getpass
import os

V1 = "tr_scond IS NOT NULL AND translate(tr_scond,' @EF','') = ''"
V2 = "translate(coalesce(tr_scond,''),' @EF','') = ''"
SESSION = ("sym_root = %(symbol)s AND coalesce(sym_suffix,'') = '' "
           "AND time_m >= '09:30:00'::time AND time_m < '16:00:00'::time "
           "AND price > 0 AND size > 0 AND tr_corr = '00'")


def table_name(date):
    day = date.replace('-', '')
    if len(day) != 8 or not day.isdigit():
        raise ValueError('date must be YYYY-MM-DD')
    return f'taqm_{day[:4]}.ctm_{day}'


def summary(db, date, symbol):
    sql = f'''SELECT sum(size) AS shares, count(*) AS trades,
        sum(CASE WHEN {V1} THEN size ELSE 0 END) AS v1_shares,
        sum(CASE WHEN {V2} THEN size ELSE 0 END) AS v2_shares,
        sum(CASE WHEN {V1} THEN 1 ELSE 0 END) AS v1_trades,
        sum(CASE WHEN {V2} THEN 1 ELSE 0 END) AS v2_trades,
        sum(CASE WHEN tr_scond IS NULL THEN size ELSE 0 END) AS null_shares,
        sum(CASE WHEN tr_scond LIKE '%%I%%' THEN size ELSE 0 END) AS odd_lot_shares
        FROM {table_name(date)} WHERE {SESSION}'''
    return db.raw_sql(sql, params={'symbol': symbol}).iloc[0]


def by_condition(db, date, symbol):
    sql = f'''SELECT coalesce(tr_scond,'NULL') AS tr_scond, ({V1}) AS kept_v1, ({V2}) AS kept_v2,
        count(*) AS trades, sum(size) AS shares
        FROM {table_name(date)} WHERE {SESSION} GROUP BY 1,2,3 ORDER BY shares DESC'''
    frame = db.raw_sql(sql, params={'symbol': symbol})
    frame['share_pct'] = (100*frame.shares/frame.shares.sum()).round(2)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--date', default='2023-01-09')
    parser.add_argument('--symbols', nargs='+', default=['JPM', 'KO', 'XOM', 'BAC', 'SPY', 'AAPL', 'MSFT', 'INTC'])
    parser.add_argument('--detail', action='store_true', help='also print volume by sale condition')
    parser.add_argument('--username', default=os.environ.get('WRDS_USERNAME') or getpass.getuser())
    args = parser.parse_args()
    import wrds
    db = wrds.Connection(wrds_username=args.username)
    try:
        print(f'{args.date}, regular session, price>0, size>0, tr_corr=00; shares of volume')
        print(f'{"symbol":8s}{"v1 kept":>9s}{"v2 kept":>9s}{"NULL":>8s}{"odd lot":>9s}{"v1 trades":>11s}{"v2 trades":>11s}{"all trades":>12s}')
        for symbol in args.symbols:
            r = summary(db, args.date, symbol.upper())
            if not r.trades:
                print(f'{symbol:8s}no trades found (the root has a suffix, or the symbol did not trade)')
                continue
            pct = lambda x: f'{100*x/r.shares:.1f}%'
            print(f'{symbol:8s}{pct(r.v1_shares):>9s}{pct(r.v2_shares):>9s}{pct(r.null_shares):>8s}'
                  f'{pct(r.odd_lot_shares):>9s}{int(r.v1_trades):>11,d}{int(r.v2_trades):>11,d}{int(r.trades):>12,d}')
            if args.detail:
                print(by_condition(db, args.date, symbol.upper()).head(15).to_string(index=False))
    finally:
        db.close()


if __name__ == '__main__':
    main()
