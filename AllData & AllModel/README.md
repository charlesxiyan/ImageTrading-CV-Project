# AllData & AllModel

Every model on the full 2016–2025 data, with and without the market (SPY).

Copy of the folder in `charlesxiyan/ImageTrading-EC-Project` (PR #4). The data-building scripts named below
(`prepare_archive_inputs.py`, `adjust_dividend_bars.py`, `us_pipeline`, …) live in that repository.

- **Notebook:** `AllData_AllModel.ipynb` (built by `tools/make_notebook.py`). Run on Colab with a GPU (L4 or A100).
- **Data:** `alldata_bundle_v1.zip`, built locally by `tools/build_alldata_bundle.py`. It holds licensed WRDS
  data, so it stays on Drive and is never committed.
- **Code:** `chartcnn/`, a copy of `CNN-Relevant/chartcnn` extended with SPY inputs (version `2026-11-16.1`).

## Data

| | |
|---|---|
| Universe | Daily S&P 500 ∪ NASDAQ 100 ∪ DJIA 30 members, by PERMNO: 775 securities, 516–531 a day |
| Sessions | 2,514 from 2016-01-04 to 2025-12-31; the 21 early-close sessions are excluded from windows |
| Bars | Five-minute OHLCV from TAQ trades, regular session, filters as in `US_EQUITY_ARCHIVE.md` |
| Market | SPY (PERMNO 84398) built with the same filters on the same clock |
| Adjustment | CRSP CIZ factors and cash dividends; each window is put on the basis of its own last bar |
| Split | train 2016–2022, validation 2023, test 2024–2025 |

Build steps, from raw archive to bundle:

```bash
python prepare_archive_inputs.py                      # CIZ -> adjustment inputs (stocks)
python adjust_dividend_bars.py --raw-root ..._adjustment_inputs --inputs ..._adjustment_inputs --output ..._dividend_adjusted
python validate_dividend_bars.py --root ..._dividend_adjusted
python download_benchmark_bars.py                     # CV repository: SPY bars on the same calendar
# same three steps for the SPY archive
python -m us_pipeline prepare --config configs/us_5min_2016_2025.json
python -m us_pipeline audit   --config configs/us_5min_2016_2025.json
python "AllData & AllModel/tools/build_alldata_bundle.py" --prepared ... --market ... --output .../alldata_bundle_v1.zip
```

**Distribution classification (CIZ).** A record with a cash amount and no price or share factor is a cash
dividend and gets the proportional chart adjustment. This covers ordinary, special, return of capital, capital
gains and non-USD dividends. A record with a factor is covered by the cumulative factors alone. 19,374 of
19,384 cash ex-date groups match CRSP's own `ret - retx` within 1%. That includes the 8 non-USD dividends of
CCEP and FER, whose CIZ amounts are already in USD; all 8 match to four decimals. The 10 that do not match are
return-of-capital / capital-gain mixes and one security exchange, where CRSP's `retx` treats part of the
payment differently. The price still drops by the full amount.

## Models per cell

| name | input | parameters |
|---|---|---|
| `plain` | stock chart 234×234 | 776,450 |
| `plain_spy` | stock chart over SPY chart, 468×234, bar columns aligned | 936,194 |
| `series1d` | 5 scaled OHLCV series | 720,770 |
| `series1d_spy` | 10 series: stock and SPY | 723,970 |
| HGB hand features | 20 features of the stock window | tree nodes, printed per cell |
| HGB hand + SPY | plus 18 SPY and stock-versus-SPY features | tree nodes, printed per cell |

Targets: `dir`, `vol` and `xsdir` at 5, 15, 30 and 60 minutes, stride 12.

## Results: `run_20261116_044159` (validation 2023)

Full run on an L4 (`QUICK = False`): 12 cells × 4 networks + 2 HGBs, 549 minutes. CNNs trained on 300,000
random training windows per cell; HGB on 1,000,000; every cell scored on all 247 validation days of 2023
(~570,000 windows). Files: `results/run_20261116_044159/` and the executed `AllData_AllModel.ipynb`.
The 2024–2025 test section has not been run.

Validation AUC (95% day-bootstrap intervals in `grid_table.csv`):

| model | vol 5 / 15 / 30 / 60 min | xsdir 5 / 15 / 30 / 60 min | dir 5 / 15 / 30 / 60 min |
|---|---|---|---|
| HGB hand + SPY | **0.690 / 0.672 / 0.668 / 0.680** | **0.520 / 0.512 / 0.510 / 0.508** | 0.516 / 0.497 / 0.498 / 0.507 |
| HGB hand features | 0.684 / 0.665 / 0.659 / 0.672 | 0.518 / 0.511 / 0.509 / 0.506 | 0.506 / 0.501 / 0.501 / 0.506 |
| 1D CNN + SPY | 0.664 / 0.646 / 0.634 / 0.654 | 0.508 / 0.500 / 0.502 / 0.500 | 0.499 / 0.501 / 0.490 / 0.502 |
| image CNN + SPY panel | 0.662 / 0.646 / 0.634 / 0.648 | 0.508 / 0.500 / 0.501 / 0.502 | 0.498 / 0.496 / 0.494 / 0.503 |
| image CNN | 0.658 / 0.639 / 0.637 / 0.649 | 0.507 / 0.500 / 0.501 / 0.500 | 0.500 / 0.498 / 0.499 / 0.503 |
| 1D CNN | 0.657 / 0.641 / 0.633 / 0.650 | 0.511 / 0.503 / 0.504 / 0.502 | 0.503 / 0.503 / 0.495 / 0.505 |

- **Volatility is predictable at every horizon; direction barely is.** vol AUC 0.63–0.69 for every model;
  xsdir 0.50–0.52 (only HGB's intervals clear 0.5 at all horizons); dir is at 0.50 except HGB + SPY at 5 min (0.516).
- **HGB beats the image CNN** in 8 of 12 cells and ties in 4 (paired day bootstrap). The image CNN beats the
  1D CNN in 1 cell, loses in 5, ties in 6: the picture adds nothing over the same numbers.
- **SPY helps a little.** HGB with SPY features beats HGB without in 6 cells and never loses; the SPY panel
  helps the image CNN in 3 cells (never hurts); for the 1D CNN it helps in 3 and hurts in 2.
- The CNNs stop early: the best epoch averaged 1.5–1.9 of 8, so more epochs on the same 300,000 windows
  would not help; more training windows or a higher learning rate might.
