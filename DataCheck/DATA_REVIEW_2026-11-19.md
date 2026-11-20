# Data review of the 2023 H1 demo — November 19, 2026

Reviewer: Tian Guo. Scope: the raw five-minute bars, the universe and symbol mapping, and the
split / cash-dividend adjustment, checked on the real files (`demo_2023_h1`,
`demo_2023_h1_dividend_adjusted`, `demo_2023_h1_distribution_inputs`), not only by reading the code.

## Verdict

Everything that could be checked is correct, with one exception: **the trade filter (policy version 1)
dropped the regular trades of every symbol that is not Nasdaq-listed.** Prices are still right; volume,
the number of empty bars and therefore the set of usable windows are not comparable across listing venues.
`download_intraday_bars.py` is fixed in this commit (policy version 2). The bars must be downloaded again.

## The finding

Eligible bar volume divided by CRSP daily volume, per security-day (`check_bar_coverage.py`):

| group | securities | volume kept (median) | empty bars | complete 78-bar windows | securities with no window |
|---|---:|---:|---:|---:|---:|
| NASDAQ 100 members (Tape C) | 102 | 55% | 0.1% | 95.5% | 0 |
| all other members (mostly Tape A) | 426 | 13% | 5.4% | 65.6% | 26 |

Price does not explain it: below $50 the two groups keep 70% and 19%. JPM, BAC, KO and XOM keep 17–19%;
AAPL, INTC and CSCO keep 62–76%. Outside the NASDAQ 100 the ratio is bimodal: 74 tickers keep more than 30%,
344 keep 20% or less and only 11 fall in between. The twenty highest are all Nasdaq-listed names that are not
in the NASDAQ 100 (AAL, HBAN, PARA, DISH, VTRS, ...), so the dividing line is the listing venue.

### Cause, confirmed on WRDS

Regular-session volume by sale condition on 2023-01-09 (`check_sale_conditions.py --detail`):

| meaning | JPM (NYSE-listed) | AAPL (Nasdaq-listed) |
|---|---|---|
| regular sale | `NULL` 51.1%, **dropped by v1** | `@` 64.5%, kept |
| intermarket sweep | `F` 18.8%, kept | `@F` 19.3%, kept |
| odd lot | `I`, `F I` 20.1%, dropped | `@ I`, `@F I` 10.7%, dropped |
| **kept by v1** | **18.8%** | **83.8%** |

Tape C marks a regular sale `@`; Tape A/B leaves the condition blank, and WRDS stores a blank `tr_scond` as
NULL. The v1 filter required `tr_scond IS NOT NULL`.

Version 1 against version 2 (`translate(coalesce(tr_scond,''),' @EF','') = ''`), share of session volume:

| symbol | listing | v1, 01-09 | v2, 01-09 | v1, 05-15 | v2, 05-15 | trades kept 01-09, v1 → v2 |
|---|---|---:|---:|---:|---:|---:|
| JPM | NYSE | 18.8% | 69.9% | 20.2% | 74.1% | 9,374 → 27,573 |
| KO | NYSE | 21.8% | 79.3% | 24.4% | 76.2% | 11,046 → 35,804 |
| XOM | NYSE | 18.2% | 74.7% | 19.6% | 77.1% | 20,180 → 63,343 |
| BAC | NYSE | 18.6% | 91.2% | 15.7% | 91.5% | 21,750 → 92,635 |
| SPY | NYSE Arca | 30.5% | 82.3% | 30.2% | 84.4% | 113,813 → 269,317 |
| AAPL | Nasdaq | 83.8% | 83.8% | 79.1% | 79.1% | unchanged |
| MSFT | Nasdaq | 72.6% | 72.6% | 69.2% | 69.2% | unchanged |
| INTC | Nasdaq | 85.1% | 85.1% | 89.5% | 89.5% | unchanged |

### Consequences for results built on version 1

- 26 securities have no usable window at all (NVR, CMG, GWW, TDG, ...); 145 keep fewer than half. Training
  and validation samples lean toward Nasdaq-listed names.
- The volume band of the chart is built from about one eighth of CRSP daily volume for NYSE-listed names,
  against more than half for Nasdaq-listed names.
- 4.4% of observed bars of the "other" group are built from a single trade (0.3% for the NASDAQ 100 group),
  so their high-low range is too narrow.
- Closing prices are unaffected: the median gap to the CRSP close is 0.015% in both groups.
- **Any archive built with the same `aggregate_sql` has the same problem**, including the 2016–2025 archive
  and its SPY benchmark (`download_benchmark_bars.py` imports `aggregate_sql`). The 2023 validation set of
  `AllData & AllModel` has 579,533 `vol_h1` windows against roughly 835,000 possible (69%), close to the
  71% measured on this demo.

### Still open after the fix: odd lots

High-priced names stay sparse under version 2, on any tape, because most of their trades are odd lots:

| symbol | v2 volume kept | odd-lot share of volume | v2 trades per day |
|---|---:|---:|---:|
| AZO | 38.4% | 55.8% | 376 |
| NVR | 47.7% | 49.2% | 343 |
| MTD | 37.4% | 59.4% | 297 |
| BKNG (Nasdaq) | 28.2% | 67.3% | 565 |

Whether to admit `I` is a design decision, not a bug. Decide it before the next download so the archive is
built once.

## What passed

| area | check | result |
|---|---|---|
| files | SHA-256 of 248 bar files and 3 input files against the manifests | all match |
| grid | 64,835 security-days, 78 bars each, New York time, no duplicates, 124 sessions | pass |
| OHLC logic | high/low bound open/close, VWAP inside the range, empty bars carry no price or volume | 0 violations in 4,832,974 bars |
| prices | last bar close against the CRSP close | within 0.1% for 98.4%, within 0.5% for 99.98%, max 0.84% |
| mapping | daily bar return against CRSP daily return, per security | correlation ≥ 0.9895, median 0.9998 |
| share classes | GOOG/GOOGL, BRK.B, BF.B, FOX/FOXA, NWS/NWSA | each matches its own CRSP series |
| universe | daily member counts; index changes (six S&P 500 dates, two NASDAQ 100 dates, none for the DJIA) | 503–504 / 101 / 30; dates as expected |
| halts | SIVB, SBNY, FRC, BIIB (halted all day on 2023-06-09) | kept as empty bars, never filled |
| adjustment | daily multipliers recomputed independently from the CRSP inputs | identical on 64,826 security-days |
| adjustment | dates on which a multiplier changes | 803, all on an ex-date or a CFACPR change |
| returns | intraday bar returns, adjusted against raw | max difference 2.2e-16 |
| ex-dates | adjusted overnight return against (open + dividend) / previous close − 1 | 800 nights, median gap 0.002%, max 0.33% |
| splits | GE spin-off, PCAR 3-for-2, MNST 2-for-1 | raw −20%, −33%, −50%; adjusted +3.0%, −0.03%, −0.3% |

Outliers: 557 bars have a high-low range above 5%; 258 are FRC and most of the rest are other regional banks
in March–May 2023. They look real. There is still no quote-based outlier filter.

## What to do

1. Decide the odd-lot policy.
2. Download one session with the new filter and run `check_sale_conditions.py` and `check_bar_coverage.py`;
   JPM and AAPL should keep a similar share of volume, and the group ratio should be above 0.7.
3. Download the archive again into a new directory (policy version 2 changes the run id, so an old directory
   is refused), then redo adjustment, `us_pipeline prepare`, the bundle and the baselines.

The finding that only volatility is predictable and that HGB beats the image CNN is not expected to change,
because prices were right. Sample counts, per-security coverage and the volume input will.

## Not checked

- Raw TAQ trades behind the aggregated bars, apart from the sale-condition counts above. High and low have no
  independent reference (the saved CRSP file has no daily high/low).
- The prepared bundle (`features.npy`, window lists): code and tests were read, the real files were not available.
- `download_us_equity_archive.py` and `US_EQUITY_ARCHIVE.md`, which are not in this repository.
- Index changes were compared with the reviewer's own record of the 2023 changes, not with an external source.
