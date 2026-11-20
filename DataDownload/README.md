# DataDownload: TAQ bars and reference data from WRDS

Step 1 of the data pipeline (see the root README for the whole order). All scripts read WRDS. The user
name is `$WRDS_USERNAME`, or the local login name; the password comes from pgpass and is never stored here.
Run each script from this folder. They import each other by module name.

| Script | What it writes |
|---|---|
| `download_us_equity_archive.py` | 2016-onward archive: universe, TAQ mappings, raw 5-minute bars (one partition per session), CRSP CIZ corporate actions and daily reference. See `US_EQUITY_ARCHIVE.md`. Single-threaded and resumable: about 30 s per session, about 23 h for 2016-2025. |
| `run_archive_job.py OUTPUT` | Runs the archive downloader unattended and writes `00_metadata/job_exit.json`. |
| `download_benchmark_bars.py` | SPY (or `--symbol`) on the stock archive's calendar, with the same filter and grid. |
| `download_intraday_bars.py` | The trade filter and aggregation SQL (`aggregate_sql`, `POLICY`) that the archive and benchmark import, plus the 2023 H1 demo downloader. See `INTRADAY_DEMO.md`. |
| `build_dynamic_universe.py` | Daily S&P 500 ∪ NASDAQ 100 ∪ DJIA 30 membership by PERMNO. |
| `fetch_distribution_inputs.py`, `inspect_wrds_demo.py` | Demo-only: dividend inputs and a WRDS schema probe. |

**Trade filter, policy version 4:**

```sql
price > 0 AND size > 0 AND tr_corr = '00'
AND translate(coalesce(tr_scond, ''), ' @EFI', '') = ''
```

A blank sale condition is a regular sale. WRDS stores it as NULL, so NULL counts as blank. `I` marks an odd lot.

Every eligible trade counts in `volume` and `trade_count`. Prices (OHLC, VWAP) come from round lots only (no `I`).
A bar with no round lot is priced by exchange odd lots (`ex <> 'D'`). Odd lots reported to a FINRA TRF (`ex = 'D'`)
never set a price; a bar with only those keeps its volume and has status `no_priced_trades`.

Versions 1 to 3 are wrong for model data:
- version 1 dropped NULL;
- version 2 dropped odd lots;
- version 3 let odd lots set prices, and a few TRF odd lots print far from the market (AMZN at 181.22 against
  about 205 on 2025-06-02). One such bar rescales a whole chart image. Evidence:
  `../DataCheck/ODD_LOT_PRICES_2026-11-20.md`.

The evidence is in `../DataCheck/`. Check every new download there before using it.

```bash
cd DataDownload
pip install -r requirements-archive.txt
V4=~/"Market Data/US Equity/v4"

# 1. Pilot: one session in each period where policy v1 failed differently, into the real archive folder.
python -u download_us_equity_archive.py --output "$V4/01_raw" \
       --days 2016-01-04,2017-06-01,2018-06-01,2023-06-01,2025-06-02
python ../DataCheck/check_bar_coverage.py --archive --every 1 --root "$V4/01_raw"
python ../DataCheck/check_sale_conditions.py --date 2016-01-04 --symbols JPM KO AAPL MSFT --detail
# Section 4 of check_bar_coverage.py (bar highs and lows vs CRSP) must pass as well.

# 2. Full archive, same folder: the pilot sessions are verified and skipped. Resumes after interruptions.
python run_archive_job.py "$V4/01_raw"
python download_benchmark_bars.py --stock-archive "$V4/01_raw" --output "$V4/01_raw_spy"
```

The pilot passes when both listing groups keep a similar share of CRSP volume in every one of the five
sessions. Policy v1 on the same sessions kept, for other listings and Nasdaq-listed stocks respectively:
77/27%, 77/78%, 25/73%, 12/52% and 11/48%.
