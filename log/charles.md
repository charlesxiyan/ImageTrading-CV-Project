# Work log: charles

No entries yet. Charles's Claude appends here.

## 2026-11-13
- **Asked:** diagnose why the CNN78u demo trains poorly; rebuild the baseline cleanly; add a
  ResNet-style residual variant; publish the CNN work so jeffery can see it.
- **Done:** wrote `CNN-Relevant/` (PR #1, branch `cnn-relevant-residual`): `chartcnn` package and
  `ChartCNN_Baseline.ipynb`. Plain JKX-style CNN (776,450 params) and a residual variant
  (2,047,362 params, blocks 2-3 are ResNet basic blocks). First conv has stride 3 so each 3-px bar
  is one feature column; height-only pooling to 4 rows keeps head fan-in about 80k (flattening
  600k features made Adam unstable). Zero-init head. Trains `vol_h1` and `xsdir_h12` on both,
  with HGB and rule baselines, day-bootstrap AUC intervals and calibration.
- **Found:** next-bar direction collapses to the prior (AUC 0.50); HGB on hand features also
  about 0.50-0.51 for h = 1, 6, 12. Volatility is learnable (CNN 0.687, HGB 0.705).
  Frame correlation: aligned pixel correlation 0.94 at lag 1 and 0.75 at lag 6; vol label ICC
  0.134 across about 360 stocks, so effective N is about 3k at stride 6; xsdir_h12 label
  autocorrelation 0.32 at lag 6, about 0 at lag 12, hence stride 12.
- **Mistakes:** a CPU dummy forward in `parameter_table` crashed on GPU; fixed, and the model
  now moves inputs to its own device.
- **Not verified:** the GPU run of this notebook version has not finished. The bundle's price
  adjustment has not been checked against the policy in `CLAUDE.md`.

## 2026-11-13 (later)
- **Asked:** compare the image CNN with a model fed the raw OHLCV numbers.
- **Done:** added `series1d` to `CNN-Relevant` (PR #1): a 1D CNN on the same 78-bar windows as a
  (5, 78) sequence, scaled exactly like the chart (per DECISIONS: same per-window scaling as the
  images). Conv1d 128/256/384, 720,770 params vs 776,450 for the plain image CNN. The notebook
  trains all three architectures on `vol_h1` and `xsdir_h12`; the analysis reports the paired
  day-bootstrap AUC difference between the best image CNN and `series1d`.
- **Not verified:** no GPU results yet; CPU smoke test only.

## 2026-11-14
- **Asked:** push the finished Colab runs of ChartCNN and the pretrained-model comparison to
  GitHub for jeffery; then also make sure the EC repository has the universe, data processing and
  model-test code, latest versions.
- **Done:** EC PR #2 and CV PR #2 (branch `claude/project-thread-0s23rf`). Both carry the latest
  `CNN-Relevant/` and `Pretrained Model Comparison/` code, executed notebooks with outputs, and
  small result files for CNN run `run_20261114_020923` and pretrained run `run_20261114_012926`.
  EC PR #2 also adds `colab/` (earlier CNN78u baseline, repair and scheduler tests), top-level
  `diagnostics/` and three protocol tests. Universe and data-processing scripts on EC `main`
  already matched charles's local copies apart from removed local paths. CV PR #2 contains CV PR #1.
- **Found:** charles's adjustment scripts and the `CLAUDE.md` policy cover the same events
  (`cfacpr`, cash dividends from `dsedist`, `cfacshr` for volume). The difference is the basis:
  the scripts rebase every price to the last date of the data, the policy to each window's anchor.
  Ratios inside a window are identical, so current models (all inputs are scaled per window,
  hand features are log returns and volume ratios) are unaffected. Any feature that uses an
  absolute price level would see future splits under the scripts.
- **Not verified:** the 8 tests that need `torch` or `wrds` were not run. The pretrained run's
  CNN rows come from CNN run `run_20261113_185632`, not `run_20261114_020923`.
- **Correction (same day):** charles says the EC repository has nothing to do with jeffery; no
  review request there. Only CV PR #2 is for jeffery.

## 2026-11-14 (later)
- **Asked:** switch model inputs to point-in-time price adjustment (anchor = window's last bar),
  add absolute-direction targets next to vol_h1 / xsdir_h12, then hand charles a Colab rerun.
- **Found:** the bundle already stores end-of-data-basis prices plus per-bar multipliers, and the
  chart path already divided each window by its anchor's multipliers. That is exactly the
  CLAUDE.md formula, so no data rebuild is needed. About 0.7% of validation windows contain an event.
- **Done:** made it explicit (`chartcnn.images.point_in_time_window`), documented it in EC
  `DATA_PIPELINE.md`, and added EC `tests/test_point_in_time.py`, which checks against raw CRSP
  inputs that the anchor is unchanged and later events cancel. Added `dir_h1` (stride 6) and
  `dir_h12` (stride 12) to both notebooks; zero returns are dropped. Executed notebooks of the
  previous runs moved to `results/run_*/`. Pushed to EC PR #2 and CV PR #2.
- **Not verified:** notebooks not executed (no torch here). The bundle's windows were already
  filtered to a non-zero next-bar return when it was prepared (`zero_return_epsilon` at h=1).

## 2026-11-14 (multi-target grid)
- **Asked:** a new Drive folder "Multi-Target Chart CNN Baseline" (leave CNN-Relevant on Drive
  untouched) with 15 / 30 / 60-minute horizons on top of 5 minutes, for absolute direction,
  volatility and relative direction, to show direction carries little signal at high frequency.
- **Done:** `CNN-Relevant/chartcnn/grid.py` (12 cells = {dir, vol, xsdir} × {h1, h3, h6, h12}),
  notebook `Multi-Target Chart CNN Baseline/Multi_Target_Chart_CNN.ipynb`, optional bfloat16
  autocast in `training.py` (default off). Stride 12 for all horizons. Models per cell: plain CNN,
  series1d, HGB, rules (no residual). Each finished cell saves `predictions.parquet` + `done.json`;
  rerun with the same `RUN_ID` skips it. Uploaded to Drive Demo/Multi-Target Chart CNN Baseline
  with its own copy of `chartcnn/`. EC PR #3 and CV PR #3 (stacked on the #2 branches).
- **Found:** chart rendering is ~0.15 ms per image, so the plain CNN epoch time is GPU-bound, not
  rendering-bound; caching images would not help.
- **Not verified:** no GPU run, so no results and no measured runtime (estimate 2.5–3.5 h on L4).
  Labels crossing the overnight gap into an ex-date session use stored (adjusted) closes, not the
  unadjusted return the policy asks for; unchanged from before.

## 2026-11-14 (bundle v2 and unadjusted labels)
- **Asked:** charles said fix both open data issues: the window list dropping zero next-bar
  returns, and labels crossing an overnight ex-date using adjusted returns.
- **Done:** `us_pipeline` `drop_zero_returns` option + `rescreen` command, `configs/us_5min_v2.json`,
  `colab/build_colab_bundle.py --version v2`, EC `tests/test_bundle_v2.py`. v2 = 1,796,575 /
  537,272 / 1,248,499 windows (v1 + 163,566 zero-return windows; bars byte-identical; audit passes).
  `chartcnn.data.build_split` (2026-11-14.3) now uses unadjusted label returns and drops labels
  whose horizon contains an ex-date (0.02% at H=1, 0.2% at H=12; includes a 2:1 split).
  `stage_bundle` uses `cnn78u_colab_bundle_v2.zip` if on Drive, else derives v2 windows from the v1
  zip at start-up (row-identical). Drive Multi-Target folder updated. Pushed to EC #3 / CV #3.
- **Found:** the Drive connector refuses uploads over 10 MB, so the 109 MB v2 zip is not on Drive;
  the start-up conversion makes that unnecessary.
- **Not verified:** no GPU run. Drive's CNN-Relevant and Pretrained folders still have the old
  chartcnn (v1 windows, adjusted labels) until charles copies the new chartcnn there.

## 2026-11-15 (multi-target grid results)
- **Asked:** push the finished Colab run of the multi-target grid with this update.
- **Done:** pulled executed notebook + grid_table/labels/takeaways/auc_by_horizon/settings of
  `run_20261114_223246` from Drive into `Multi-Target Chart CNN Baseline/results/` (EC #3, CV #3),
  results table in that README.
- **Found:** L4 + bf16, 68 min for 12 cells. vol AUC 0.64–0.70 at every horizon (HGB best, CNN
  ~0.015 lower). dir and xsdir: CNN AUC within 0.02 of 0.5, log loss worse than prior; only HGB
  dir_h6 0.543 [0.525, 0.561], HGB xsdir_h6 0.513 and series1d xsdir_h1 have CIs above 0.5, about
  chance level for 24 tries. Image CNN ties 1D CNN in 10/12 cells, never beats HGB.
- **Not verified:** per-cell predictions/histories stay on Drive and were not read.

## 2026-11-15
- EC PR #2 and CV PR #2 merged. Waiting on charles's Colab rerun of the `*_dir_targets` notebooks.

## 2026-11-15 (merge)
- **Asked:** merge. **Done:** merged #3 into #2's branch, then #2 into `main`, in both repositories;
  CV #1 shows merged (contained in #2); EC #1 closed as superseded (its 1D CNN is in `main`).

## 2026-11-16 (full 2016-2025 preprocessing, started)
- **Asked:** run the full preprocessing pipeline (adjustment, cleaning, images) on the new
  2016-2025 WRDS archive (`US Equity/wrds_5min_2016_latest_v1` under the Market Data folder).
- **Found:** archive complete, 2,514 sessions (21 early closes), 102.1M bar slots, 2.8% empty,
  775 PERMNOs, 516-531 a day (S&P 500 ∪ NDX 100 ∪ DJIA 30). Bars have the demo schema. CRSP
  reference tables are in CIZ format (`dlycumfacpr`, `distype/disdetailtype`), not SIZ `distcd`.
  19,374 of 19,384 cash ex-date groups match CRSP `ret - retx` within 1%, incl. the 8 non-USD
  (FX) dividends, so CIZ `disdivamt` is in USD. The 10 misses are ROC/CAPG mixes and one SECEX.
- **Done (EC repo, branch `claude/project-thread-5acfxn`, not pushed):** `prepare_archive_inputs.py`
  (CIZ -> adjustment inputs; kind=cash if divamt>0 and facpr=facshr=0, else corporate);
  `adjust_dividend_bars.py` accepts `kind` and is faster; `us_pipeline/data.py` accepts any period
  and marks early-close sessions all-invalid; `configs/us_5min_2016_2025.json` (train <=2022,
  val 2023, test 2024-2025, pending charles's confirmation). All 62 tests pass.
- **Not verified yet:** full adjustment run and validation are in progress.
- **SPY benchmark (same day):** charles asked for SPY 5-min bars aligned with the stock archive.
  Wrote `download_benchmark_bars.py` (CV working tree, untracked like the archive downloader):
  same filter/SQL/grid/schema, PERMNO 84398, plus CIZ daily + 41 distributions. Pilot of 3
  sessions OK (open 200.49 = CRSP daily open on 2016-01-04). Full download blocked by auto-mode
  permissions; waiting for charles's explicit approval.
