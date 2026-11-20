# Odd lots and bar prices: policy version 3 review — November 20, 2026

Reviewer: Tian Guo. Scope: the trade filter of policy version 3 (PR #8), checked on WRDS against raw TAQ
trades and CRSP daily data before the 2016-2025 archive is downloaded again. No v3 archive existed yet, so
the check runs the filter directly on the TAQ trade tables.

## Verdict

- **Volume: v3 passes.** Nasdaq-listed and other stocks now keep a similar share of CRSP volume in every
  period tested.
- **Prices: v3 fails.** Odd lots set bar prices, and a few odd lots reported to a FINRA TRF (`ex = 'D'`)
  print far from the market. They move bar highs and lows by up to 12%, increasingly so in 2024-2025,
  which is the test period.
- **Policy version 4 fixes it** and leaves volume and trade counts unchanged: odd lots still count in volume,
  but prices come from round lots, with exchange odd lots only as a fallback.

## Sample

20 symbols: AAPL, MSFT, INTC, CSCO, AMZN, NVDA, COST (Nasdaq-listed); JPM, KO, XOM, BAC, GE, WMT, PFE, T,
AZO, NVR, MTD, CMG (NYSE-listed; the last four are high-priced and trade thinly); SPY (NYSE Arca).
Sessions: 2016-01-04, 2017-06-01, 2018-06-01, 2019-06-03, 2021-06-01, 2022-06-01, 2023-06-01, 2024-06-03,
2025-03-03, 2025-06-02, 2025-12-01. All queries were read-only, on `taqm_YYYY.ctm_YYYYMMDD` and
`crsp_a_stock.dsf_v2`. The sample is small and over-weights high-priced names on purpose.

## 1. Volume kept (median share of CRSP daily volume)

| Session | v1: Nasdaq / other | v3: Nasdaq / other |
|---|---|---|
| 2016-01-04 | 26% / 79% | 86% / 84% |
| 2017-06-01 | 78% / 79% | 85% / 87% |
| 2018-06-01 | 76% / 23% | 88% / 86% |
| 2023-06-01 | 66% / 16% | 81% / 80% |
| 2025-06-02 | 57% / 14% | 76% / 78% |

The v1 column has the same shape as the v1 archive table in `README.md`. On 2016-01-04 the Nasdaq symbols
were the ones stored with a NULL condition; from 2018 the NYSE-listed ones are. `coalesce` handles both.
The rest of CRSP volume is the opening and closing auctions, extended hours and in-session conditions
outside the whitelist (derivatively priced, contingent and similar trades), all excluded by design. For JPM on
2023-06-01, every TAQ trade of the day minus the official-close (`M`) and duplicate opening (`Q`) reports adds
up to CRSP volume exactly (12,248,094 shares); the closing auction alone is 20% of it.

## 2. The price problem in v3

Regular-session bars where admitting odd lots moves the high or low by more than 1%
(20 symbols, about 1,560 bars per session):

| Session | Bars moved > 1% | Symbols affected |
|---|---|---|
| 2016-01-04 | 3 | 3 |
| 2017-06-01 | 2 | 2 |
| 2018-06-01 | 1 | 1 |
| 2019-06-03 | 3 | 3 |
| 2021-06-01 | 5 | 5 |
| 2022-06-01 | 9 | 6 |
| 2023-06-01 | 7 | 4 |
| 2024-06-03 | 16 | 11 |
| 2025-03-03 | 47 | 17 |
| 2025-06-02 | 38 | 17 |
| 2025-12-01 | 19 | 13 |

Examples (low or high with odd lots, against round lots only, same bar):

| Session | Symbol | Bar | With odd lots | Round lots |
|---|---|---|---|---|
| 2025-06-02 | AMZN | 11:45 | low 181.22 | low 204.95 |
| 2025-06-02 | JPM | 11:40 | low 234.34 | low 263.90 |
| 2025-06-02 | MSFT | 11:10 | low 425.40 | low 460.72 |
| 2017-06-01 | CSCO | 12:50 | high 34.80 | high 31.69 |

Where they come from: on five sessions, 149 odd lots printed more than 1% outside the round-lot range of
their bar; 136 of them were reported to a TRF (`ex = 'D'`).

Why it matters here: each chart image is scaled to the window's highest high and lowest low, so one such bar
squeezes the other 77 into a few pixels. The problem grows from 2024, so it would hit the test period much
more than the training period.

## 3. Policy version 4

```text
volume, trade_count : every eligible trade (as v3)
OHLC, VWAP          : round lots (no 'I'); if the bar has none, exchange odd lots (ex <> 'D');
                      TRF odd lots never set a price
```

`aggregate_sql` was run on WRDS in both versions for the 20 symbols and 11 sessions (17,151 bars):

| | v3 | v4 |
|---|---|---|
| Volume and trade count | — | identical to v3 in every bar |
| Bars with volume but no price (`no_priced_trades`) | — | 1 |
| Security-days with day high or low > 1% from CRSP | 59 of 220 | 0 |
| > 0.5% from CRSP | 68 | 0 |
| > 0.1% from CRSP | 95 | 7 |
| Last close within 0.1% of CRSP close | 93.2% | 93.2% |

The seven remaining days within 0.1-0.5% are all MTD, NVR, CMG and AZO. Their CRSP high or low can come from
an auction print or an odd lot that the filter leaves out. Closes are unchanged. The 93.2% is below the 95% pass mark
in `README.md` for both versions, because a fifth of this sample are thinly traded high-priced names whose
last regular trade can differ from the closing auction price. The full pilot will tell whether it holds
for the whole universe.

The SQL was also tested on synthetic trades in a local PostgreSQL 16 (round lots with outlying odd lots,
a bar priced only by exchange odd lots, a bar with only a TRF odd lot, excluded conditions, a correction
and a 16:00 print): all values matched by hand.

## 4. Also checked: CRSP CIZ adjustment factors

`prepare_archive_inputs.py` reads `dlycumfacpr` and `dlycumfacshr` from `crsp_a_stock.dsf_v2`. For every
S&P 500 member of 2016-2025, 174 security-days have a factor change; on all of them
`(prc/cumfacpr) / (prev_prc/prev_cumfacpr) - 1` equals CRSP's `dlyretx` within 5e-4. Five more days have
a price step without a factor change: GE on 2019-02-26 (Wabtec spin-off, `distype` SP) and three APO and
one BX distribution of return of capital, special dividend and capital gains. All have `divamt > 0` with zero
factors, so `prepare_archive_inputs.py` treats them as cash and adjusts them.

## Not checked

- No archive was downloaded; the pilot (`DataDownload/README.md`) is still the first real test of v4.
- The window function adds a sort per bar. On the 20 symbols v4 took about 10.4 s per session against 9.3 s for
  v3 (two runs each on 2023-06-01 and 2025-06-02); full-universe run time is not measured.
- Quote-based outlier filtering is still not done. TRF round lots can also print away from the market, but none
  of the round-lot highs and lows in this sample was more than 0.5% from CRSP's.
