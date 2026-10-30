# ImageTrading-CV-Project

## Dynamic US equity universe

`build_dynamic_universe.py` builds the daily **S&P 500 ∪ NASDAQ 100 ∪ Dow Jones Industrial Average 30** security universe for an inclusive date range. Each date receives the tickers effective on that historical date. Securities appearing in multiple indices are included once; distinct share classes remain separate securities.

### Installation and execution

Use Python 3.9+ on a machine with WRDS access or in WRDS JupyterHub:

```bash
python -m pip install wrds
python build_dynamic_universe.py --username YOUR_WRDS_USERNAME --inspect
python build_dynamic_universe.py --username YOUR_WRDS_USERNAME --start 2020-01-01 --end 2020-12-31
```

The official WRDS Python client handles authentication. Do not store passwords in source files. `--inspect` reads metadata and prints candidate table columns, NASDAQ/Dow index names and IDs, and visible related tables, including TAQ tables. Visibility alone does not establish full subscription access or data quality; actual queries also check read permissions.

If the index name cannot be resolved uniquely, select its `index_id` from the **exact index name** in the inspection report, preserving leading zeros:

```bash
python build_dynamic_universe.py --start 2020-01-01 --end 2020-12-31 --ndx100-id VERIFIED_NDX_GVKEYX --djia30-id VERIFIED_DJIA_GVKEYX
```

These IDs are placeholders and must be replaced. Do not select the NASDAQ Composite, a NASDAQ financial index, or the Dow Jones Transportation Average. The script does not guess fixed IDs or substitute QQQ/DIA holdings for index constituents.

The default includes **every calendar day**, including weekends, using the membership effective on each date. For trading dates only:

```bash
python build_dynamic_universe.py --start 2020-01-01 --end 2020-12-31 --days trading
```

Trading dates come from dates actually present in CRSP `dsi`, rather than a Monday-through-Friday calendar. Requests outside the available calendar range are rejected, including an end date beyond the database's latest date.

### Outputs and Python usage

Outputs default to this project's `data/universe/` directory. Repeated runs overwrite files with the same names; use a separate `--output` directory for each research interval.

- `daily_tickers.json`: a sorted, deduplicated ticker list per date, such as `{"2020-01-02": ["AAPL", "..."], ...}`.
- `daily_universe.csv`: `date,permno,ticker,indices`, with one row per date and PERMNO. Index memberships are separated by `|`.
- `manifest.json`: source tables, index IDs, requested dates, generation time, and daily security counts for each index and their union.

```python
import json
from pathlib import Path

# Run from the project root.
pools = json.loads(Path('data/universe/daily_tickers.json').read_text())
tickers = pools['2020-01-02']
```

All default output paths are within `ImageTrading-CV-Project`. The original `ImageTrading` project requires no changes.

### Sources and limitations

1. **S&P 500**: tries `crsp_a_indexes.dsp500list`, then `crsp.dsp500list`; requires `permno/start/ending`.
2. **NASDAQ 100 and DJIA 30**: tries `comp.idxcst_his`, then `comp_na_daily_all.idxcst_his`; uses `gvkey/iid/gvkeyx/from/thru`. Index names are resolved through `idx_index`, requiring `gvkeyx` (or `gvkey`) and `conm` (or `conmx`). **These are candidate adapters; historical coverage for these indices must be verified in your account.**
3. Compustat constituents are mapped from `gvkey+iid` to PERMNO through the historical CCM `ccmxpf_linktable`, using security-level links, effective dates, and link types `LC/LU/LS`. Ambiguous mappings cause an error; the script does not arbitrarily select a link or substitute the company's current primary security.
4. PERMNO is mapped to the **historical ticker** through CRSP `dsenames` / `msenames` and its `namedt/nameendt` intervals. Current `comp.security.tic` values are not used to backfill history. Punctuation and share-class notation are preserved.
5. Date intervals are inclusive. NULL end dates are treated as ongoing; NULL membership start dates are not backfilled. Other interval conventions or schemas require an adapter based on the relevant data dictionary.

Override table names with `--sp500-table`, `--members-table`, `--headers-table`, `--links-table`, `--names-table`, or `--calendar-table`. Names must use the `schema.table` format, and required columns are validated. If your account only exposes newer CRSP CIZ tables, the script reports a table or column mismatch. **Automatic CIZ conversion is not implemented**; add an adapter using your account's actual data dictionary rather than merely renaming a table.

Before writing outputs, the script checks that all three indices are nonempty on every requested date, each constituent maps to a historical ticker, and mappings are unambiguous. **Nonempty results do not prove source completeness or freshness.** Historical records may be revised, vendor updates may lag, and NULL end dates may reflect stale records. Verify WRDS update dates and historical coverage, and inspect daily counts and abrupt changes in `manifest.json`. Counts are not fixed at 500/100/30 because company counts and security counts can differ.

A WRDS account does not grant access to every underlying database. WRDS states that historical S&P 500 constituent access requires a CRSP subscription. Compustat's licensing for S&P index constituents has changed, so `idxcst_his` cannot be assumed to cover every index. Missing indices, dates without membership records, missing links, or missing tickers prevent new outputs from being generated. Resolve subscription or coverage issues first. The script does not backfill with current constituents or silently reduce the universe to S&P 500 alone. Failed runs leave existing output files untouched; check the manifest before using an earlier result.

The result reconstructs membership by **historical effective date**. It is not an archive of information as originally published. Backtests must still respect signal timing, order timing, and announcement availability. Membership alone does not establish tradability; suspensions, price availability, and liquidity require separate filters. Do not exclude historical members based on future delisting information.

### High-frequency data integration

This step does not download intraday prices or depend on an unverified TAQ schema. Before integrating high-frequency data, confirm NYSE TAQ subscription access, available years, trades/quotes versus aggregated bars, timestamps, trade conditions, and cancellation/correction fields. **A CRSP historical ticker is not a universal join key for TAQ tables.** Map it by date and share class to the actual TAQ symbol root/suffix or other relevant fields. PERMNO is retained for security identification and auditing; this does not imply that TAQ tables contain PERMNO.

### Validation

```bash
python -m unittest discover -s tests -v
```

Offline tests cover deduplication, index memberships, historical ticker changes, weekend retention, missing indices/mappings, ambiguous links, ticker collisions, output serialization, and SQL identifier validation. **Your WRDS account has not been connected; actual permissions, columns, index IDs, and historical coverage remain unverified, and no live download has been performed.** Start with `--inspect`, then validate a short interval containing a known constituent or ticker change against WRDS records.

References:

- [WRDS: downloading historical S&P 500 constituents and CRSP subscription requirements](https://wrds-www.wharton.upenn.edu/pages/grid-items/downloading-the-sp-500-constituents/)
- [WRDS: historical S&P 500 constituent changes](https://wrds-www.wharton.upenn.edu/classroom/sp500-introduction/over-time/)
- [CRSP: official CCM documentation, hosted by WRDS](https://wrds-www.wharton.upenn.edu/documents/415/CRSPAccess_Software_Guide_CCM_part.pdf)
