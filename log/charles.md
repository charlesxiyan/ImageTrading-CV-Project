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
- **Full build done (same day):** stock adjustment 102,142,206 rows, 19,380 cash ex-date groups,
  validated (all files). SPY: 2,514 sessions, 9 empty bars, 41 dividends, validated. us_pipeline
  prepare+audit: 79.5M windows (train 59.6M / val 7.0M / test 12.9M), 21 early closes excluded.
  Fixed a bug: `decision_clock_ns` was written in microseconds under pandas 3 (`as_unit('ns')`).
  Bundle `alldata_bundle_v1.zip` 1.19 GB (float32, windows thinned to end_index % 6 == 5, SPY on
  the same clock), outside the repo next to the archive. New folder `AllData & AllModel` (EC branch):
  chartcnn copy 2026-11-16.1 with SPY panel (468x234), 1D+SPY, HGB+SPY; notebook generator; QUICK switch.
- **Delivered (same day):** notebook uploaded to Drive `Demo/AllData & AllModel/`; local smoke
  run of the whole notebook (incl. test section) passed on CPU. Waiting for charles to upload the
  1.19 GB bundle and run QUICK, then the full grid. Not verified: any GPU run or real numbers;
  RAM use on a standard Colab runtime (estimated ~6 GB for label arrays).
- charles: hold push/PR of EC branch `claude/project-thread-5acfxn` until he has run the
  AllData & AllModel notebook; then push code and results together.

## 2026-11-16 (AllData run pushed; ViT added)
- **Asked:** push yesterday's finished AllData & AllModel run to GitHub; add the missing ViT.
- **Done:** pulled executed notebook + `run_20261116_044159` summary files from Drive into
  `AllData & AllModel/results/`; EC PR #4 (branch `claude/project-thread-5acfxn`, includes the
  data-pipeline commit). ViT (`chartcnn/vit.py`, T2T-style soft splits + 8x96 backbone, 864,002 /
  880,226 params with SPY panel) wired in as `vit`/`vit_spy`, own Adam lr 1e-4; notebook variant
  runs only the ViTs and merges the CNN/HGB scores of `run_20261116_044159` (rows checked equal).
  ViT is NOT in PR #4.
- **Found (validation 2023, L4, 549 min):** vol AUC 0.63-0.69 (best HGB+SPY 0.690 at 5 min);
  xsdir 0.50-0.52; dir ~0.50 (HGB+SPY 0.516 at 5 min). HGB > image CNN in 8/12 cells, ties 4;
  image CNN vs 1D CNN 1 win / 5 losses / 6 ties; SPY features help HGB in 6 cells, never hurt;
  SPY panel helps image CNN in 3. CNN best epoch 1.5-1.9 of 8.
- **Not verified:** per-cell prediction files stayed on Drive; test 2024-2025 not run.
- Also mirrored the folder (commit 77e07bb state) to CV PR #4, branch `claude/project-thread-5acfxn`, reviewers faintcicado + charlesxiyan.
- Added with-vs-without-SPY table to the README in EC #4 and CV #4 (mean AUC change: HGB +0.0034, 10/12 up; image CNN +0.0002; 1D CNN -0.0005). Per-cell paired CIs not computed: predictions are 30 MB each, Drive connector caps downloads at 10 MB.
- charles prefers plain-language conclusions in result READMEs (key findings + how to read each chart), not just tables. Added to EC #4 / CV #4.
- ViT ready: EC local branch `vit-t2t` commit 860ca06 (not pushed). Drive: `AllData & AllModel/AllData_AllModel_ViT.ipynb` + `chartcnn/` (2026-11-16.2). Local smoke passed incl. merge with previous run and test section. Upload trick: Drive connector file references work only from the scratchpad path (not paths with spaces/&).

## 2026-11-20 — TAQ filter bug (EC #6), repo reorganization, CV mirror
- **Asked:** review teammate PR on data cleaning (EC #6); confirm the old data is wrong before re-downloading;
  reorganize code into one folder per step; bring the CV repo up to date with EC.
- **Found:** policy v1 filter `tr_scond IS NOT NULL AND translate(tr_scond,' @EF','')=''` dropped the
  blank (NULL in WRDS) regular-sale condition. Confirmed on the v1 bundle (2023 ADV vs public: AAPL 67%,
  MSFT 57%, JPM 18%, KO 17%, BAC 21%, XOM 18%, SPY 31%) and on the v1 archive by CRSP primary exchange,
  one session per year: other listings 77% (2016-01), 77% (2017-06), 25% (2018-06) down to 11% (2025-06);
  Nasdaq-listed 27%, 78%, 73% down to 48%. Bias is time-varying; closes are fine (97.8% within 0.1%).
  The archive downloader lived only in an untracked local working tree and imported the old filter.
- **Done:** merged EC #4, #5, #6 (Charles's go). EC #8 (was #7): step folders DataDownload/DataCheck/
  PriceAdjustment/Pipeline, archive downloader added (WRDS user from $WRDS_USERNAME), filter policy v3
  (odd lots kept), CRSP acceptance check (`check_bar_coverage.py --archive`), v3 config and paths.
  CV #6 (was #5; branch data/step-folders-v3, per charles no "claude" in branch names) mirrors EC #8 (supersedes CV #4), reviewer faintcicado. 69 tests pass in both.
- **Not verified:** any download with policy v3. Charles must confirm before the full re-download
  (about 23 h; the last one took 22.7 h at ~30 s a session). Pilot should cover 2016, 2017, 2018, 2023.

## 2026-11-20 v3 pilot result
- Pilot (2016-01-04, 2017-06-01, 2018-06-01, 2023-06-01, 2025-06-02) with filter v3 passed the balance check: other/Nasdaq volume kept 87/89, 88/88, 85/86, 77/78, 74/76 percent of CRSP (v1: 77/27, 77/78, 25/73, 12/52, 11/48).
- Not verified by me: the close-accuracy line of the check output. Full re-download waits for Charles's go.

## 2026-11-20 Dual-class symbol fix and policy v4
- Asked: verify teoguo's report that 2016 dual-class stocks had no data. Confirmed: WRDS daily link has no 2016 rows for GOOGL, CMCSA, DISCA, DISCK, LBTYA, LVNTA (and BATRA/BATRK); the TAQ master spells them as five letters until 2016-10-21 while trades use root plus suffix (checked on WRDS), so 1,224 security-days in v1 had no trades.
- Fix: EC #11 (join on root+suffix key), merged; EC #10 from teoguo (policy v4, odd lots in volume, not prices) was already merged. CV #8 mirrors both, merged.
- v4 pilot (5 days) passed: no zero-trade members, closes 97.9% within 0.1%, highs/lows 0 of 2,612 more than 1% from CRSP, volume balance 0.99.
- Not verified: the full v4 download (about 35 h, to be started by Charles).

## 2026-11-20 (roadmap notes)
Asked: add 4 notes to roadmap artifact. Did: added CNN vs ViT results table to step 5 (from AllData grid_table.csv and probe run) and next-step cards for image-validity ablation, CNN vs ViT (LoRA), efficient compute (ViT LoRA / CNN INT8 PTQ), CNN filter/channel analysis. Not verified: none of these experiments run yet.

## 2026-11-21 (full download start)
Asked: start full v4 download. Checked local EC main = a3e71a3 (#11), v4/01_raw config matches full run so pilot days resume. Gave Charles the run_archive_job.py command; he runs it. Not verified: whether it started.

## 2026-11-22 (v4 full download acceptance)
Asked: data re-download finished. Did: read progress.json (2514/2514 complete), staged every 21st session + daily reference to container, ran DataCheck/check_bar_coverage.py --archive. Found: PASS on all four checks (closes 97.5% within 0.1%, volume ratio 0.99, H/L 99.8% within 0.5%, 0.01% empty bars). Outliers: VSM 2016-10-03 close 8.9% off (spin-off first day), 36 H/L >1% mostly CRSP H/L from excluded auctions. Not verified: check_sale_conditions.py on WRDS; sessions not in the sample.

## 2026-11-23 (SPY/VIXY check, adjustment runner)
Asked: SPY and VIXY downloaded. Did: confirmed both complete (2514 sessions), sampled 40 sessions each vs CRSP daily (SPY closes 100% within 0.1%, VIXY max 0.3%, no empty bars). Wrote a local runner script in the v4 data folder (not in git) chaining PriceAdjustment for stocks/SPY/VIXY, Pipeline prepare+audit with the v4 config, and the bundle build, logging to chain.log. Not verified: the chain has not run on v4 yet.

## 2026-11-23 (v4 chain failure, PR #13)
Asked: v4 chain errored. Found: adjust_dividend_bars stopped on bars with volume and no factor: 114 member-days after delisting where CRSP has no row; policy v4 TRF odd lots give volume-only bars. Did: EC PR #13 zeroes adjusted volume for factorless unpriced rows (volume_raw kept), adds a test; 74 tests pass. Not verified: full chain rerun.

## 2026-11-23 (PR #13 merged)
Merged EC #13 on Charles request. He reruns the v4 chain. Not verified: rerun result.

## 2026-11-23 (PR #14)
v4 chain rerun: stock and SPY adjustment validated; VIXY validator crashed because VIXY has no cash dividends (empty table). Opened EC PR #14 guarding the report. Not verified: prepare/audit/bundle.

## 2026-11-23 (PR #14 merged)
Merged EC #14 on Charles request; he reruns the chain. Not verified: prepare/audit/bundle.

## 2026-11-23 (v4 bundle built)
The v4 chain finished: all adjustments validated, us_pipeline prepare+audit passed (100.3M windows, +26% vs v1), bundle alldata_bundle_v4.zip 1.34 GB (train/val/test thinned 11.75M/1.67M/3.30M). Not verified: why 34% of candidate windows are invalid_input (v1 48%); bundle not yet uploaded or used in Colab.

## 2026-11-23 (rerun plan, feature selection idea)
Proposed v4 rerun tiers (grid + indicator FULL must; ViT via LoRA; skip daily JKX/demo) and an HGB feature-selection notebook (permutation, group ablation, forward selection) for the 38 hand features. Nothing built yet.

## 2026-11-23 (MindFlow notebooks)
Asked: redo the comparison experiments as a step-by-step argument (1 daily + pretrained no signal, 2 intraday direction no signal incl. ViT with parameter counts, 3 volatility signal, 4 add image info until CNN reaches HGB) in Drive Demo/MindFlow. Built nine notebooks (EC PR #15, branch mindflow/step-notebooks, folder MindFlow/): 1a daily JKX, 1b daily pretrained (frozen DINOv2/CLIP probes + Chronos-2/TimesFM/Moirai/Kronos zero-shot), 2a/3a grids (CNN, SPY panel, ResNet, 1D CNN, ViT, HGB), 2b/3b pretrained, 4a ladder + features-only control, 4b HGB feature selection (chosen on 2022, confirmed on 2023), 5 summary. Code and notebooks uploaded to Drive MindFlow (v4 bundle already there). Synthetic smoke test passes for all nine. Not verified: anything on real data, foundation-model installs on Colab, run times.

## 2026-11-23 (later) MindFlow first results checked
- Asked: are 1a, 2a, 3a, 4b done. All four finished on v4 (Drive MindFlow/runs).
- 1a: daily JKX CNN and HGB AUC 0.47-0.52 in all 24 val/test cells, no signal.
- 2a: direction 5 min best HGB+SPY xsdir 0.526, dir 0.516 (CI above 0.5, log-loss gain ~0.1%, L/S ~2 bps gross); 60 min ~0.50.
- 3a: vol 5/60 min HGB+SPY 0.689/0.679, CNN 0.660/0.648, 1D CNN+SPY 0.664/0.655, ResNet 0.657/0.643.
- Scratch ViT collapsed in all cells (AUC 0.500, prob_std ~1e-5) even on vol: training failure, not evidence. Proposed fixing ViT training (warmup, lower lr) and rerunning only ViT cells; awaiting Charles.
- 4b: forward selection (2022) picks time_of_day, mean_abs_r12, spy_mean_abs_r12; on 2023 -0.005 (5m) / -0.008 (60m) vs all 38; top 10 ties at 5m; dropping volume+time group costs 0.034 at 60m.
- Not verified: run-folder names (mapped by content).

## 2026-11-23 (evening) Scratch ViT fix + retrain-only mode; Demo cleanup list
- Cause of ViT collapse: std-0.02 init made tokenizer image signal ~4x smaller than position embedding (token std across images 0.004 -> 0.48 after fix); near-constant logits rounded in bf16 -> AUC exactly 0.500.
- Fix (chartcnn 2026-11-23.2, EC PR #15 branch mindflow/step-notebooks): fan-in init + LayerNorm in tokenizer, mean pooling, AdamW lr 3e-4 wd 0.05, 1000-step warmup, grad clip 1.0, fp32 head, collapse warning. CNNs unchanged.
- New RETRAIN_ARCHS in Step2a/3a: retrain only listed networks of a finished run on identical (asserted) rows; old net folders renamed <arch>_before_<version>. Smoke (all 9 notebooks + retrain) passes.
- Uploaded to Drive; old run notebooks renamed *_run1_outputs.ipynb; old grid.py renamed grid_old_2026-11-23_1.py.txt (trash was blocked).
- Run ids: 2a grid_20261123_083042, 3a grid_20261123_083117. Charles to rerun 3a then 2a with RETRAIN_ARCHS=["vit","vit_spy"].
- Not verified: the fix on real data (synthetic check only).
- Demo cleanup suggestion: alldata_bundle_v1.zip (1.19 GB) + ~3.7 GB v1 .parquet/.pt/.npy outside MindFlow + old folders; awaiting Charles.
