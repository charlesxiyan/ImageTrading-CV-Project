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
- Demo cleanup done at Charles's request (to Drive trash, ~4.9 GB): v1 bundle zip, AllData runs/probe_runs/indicator_runs, Daily runs, old folders Multi-Target/Pretrained Model Comparison/CNN-Relevant/chartcnn, 4 old root notebooks. Kept MindFlow, notebooks and .py in AllData/Daily, CNN_Paper_Style_Targets, wrds_demo_schema.json. v1 run outputs (run_20261116_044159, probe_20261118_031436, indicator runs) are now only in trash.
- 22:10Z check: ViT fix WORKS on real data (3a retrain, early-stop subset AUC): vol 5m vit 0.660 / vit_spy 0.660, 60m vit 0.650 (CNN 0.660/0.648, HGB+SPY 0.689/0.679). Stuck ~0.5 for 1-2 warmup epochs then learns. 1b done: all daily pretrained/TSFM AUC 0.47-0.53 (DINOv2 linear I5R5 val 0.524 borderline, test 0.510). 4a still running. Told Charles to start 2a ViT retrain now, 3b after 3a finishes.

## 2026-11-24 (status check, ~8 PM ET 11-23)
- Asked: status of MindFlow runs.
- Found (Drive outputs): 3a done 6:10 PM ET with fixed ViT (5m/60m 0.662/0.645 vs CNN 0.660/0.648, HGB+SPY 0.689/0.679). 2a ViT retrain 3/4 cells done, dir_h1 left, ~1.5-2 h. 4a still running (5m cell done). 1b done (all 0.47-0.53).
- Next: 3b now, 2b after 2a, Step5 last (CPU). EC PR #15 awaits merge go.
- Not verified: 2a/4a final tables (not written yet).

## 2026-11-24 (status check, ~9:50 PM ET 11-23)
- 4a done (8:52 PM ET): ladder vol 5m/60m: plain 0.660/0.647, +SPY 0.660/0.648, +EWSD 0.678/0.664, +MA 0.677/0.664, +38 tab 0.685/0.675, tab-only control 0.685/0.671, HGB+SPY 0.687/0.677. EWSD +0.018 signif.; MA nothing; image adds ~+0.002 over features-only.
- 2a ViT retrain done 9:08 PM ET: all 4 direction cells AUC 0.49-0.52, prob std ~1e-7 (constant output; expected with no signal; 3a proves the code learns). grid_table cell still running when checked.
- 3b running (started 8:09 PM ET): frozen probes done (CLIP ~0.64), TS foundation models in progress.

## 2026-11-24 (~10:50 PM ET 11-23)
- 2a fully done 10:21 PM ET (grid_table rebuilt): dir best 5m 0.516 / 60m 0.508; xsdir 0.526 / 0.509 (HGB+SPY); ViT vs CNN worse 2, tied 2. Told Charles to start 2b now.
- 3b probes done 10:22 PM ET (DINOv2 vs random +0.04; vs scratch CNN -0.006; vs HGB -0.034); Chronos-2/TimesFM/Moirai done, Kronos running.

## 2026-11-24 (~11:30 PM ET 11-23)
- 3b: only Kronos h12 left (~done 11:30 PM ET). Note: TimesFM 3.0 download 401 -> code fell back to google/timesfm-2.5-200m (231M params).
- 2b started 10:49 PM ET (probe_20261124_034956), feature extraction phase; ETA ~1 AM ET. Step5 after 2b.

## 2026-11-24 (7:30 AM ET)
- 3b done 12:31 AM ET; 2b done 5:28 AM ET (probe_20261124_034956).
- 3b TSFM zero-shot vol AUC 5m/60m: Kronos 0.638/0.621 (25 days; HGB same rows 0.681/0.684), Chronos-2 0.612/0.560, TimesFM 2.5 0.593/0.576, Moirai2 0.601/0.552, random walk 0.558/0.563 (49 days; HGB 0.688/0.679).
- 2b direction: TSFM 0.497-0.515; probes pretrained ~= random (+0.0006); best still HGB+SPY xsdir 5m 0.526.
- Next: Step5 summary on CPU.

## 2026-11-24 (8:50 AM ET)
- Step5 ran 7:50 AM ET. Found foundation-model rows with point AUC outside the CI.
- Cause: frozen_probe.table (and indicator_cnn.table) computed AUC on scores clipped to [0,1]; foundation-model scores are not probabilities (vol ratio mostly >1, direction returns half negative) -> ties -> AUC biased to 0.5. CI used raw scores (correct). Only 2b/3b foundation-model AUCs affected.
- Fix pushed to EC PR #15 (table AUC on raw scores; pretrained_ts.rebuild_tsfm_table; Step5 rescores 2b/3b once, old table kept). Smoke passes. Drive code and a new Step5 notebook uploaded; old copies renamed.
- Not verified on real data yet: Charles to rerun the new Step5 on CPU. Foundation-model AUCs reported last night are understated.

## 2026-11-24 (1:05 PM ET)
- New Step5 finished 12:44 PM ET; 2b/3b foundation-model tables rescored, points now inside CIs.
- 3b vol 5m/60m corrected: Kronos 0.644/0.623, Chronos-2 0.622/0.571, TimesFM 2.5 0.623/0.585, Moirai2 0.621/0.579, random walk 0.580/0.592 (CNN 0.661/0.646, HGB+SPY 0.688/0.679). At 60m only Kronos beats random walk.
- 2b direction corrected: all 0.496-0.518 (Kronos dir 5m 0.518).
- MindFlow chain complete on v4. PR #15 still awaiting merge go.

## 2026-11-24 (1:10 PM ET)
- Asked: next step per plan -> Charles chose VIX test first.
- Built MindFlow Step6a_VIX_Regime (EC PR #15, code/regime.py): HGB vol, 38 base vs +VIX (prev-session close log, chg1, chg5, rel20; Cboe public CSV cached in MindFlow/data/) vs +multi-day RV (stock rv5d/rv20d/ratio, SPY rv5d/rv20d from intraday returns of completed sessions) vs both vs regime-only; paired day bootstrap on 2023, AUC by VIX tercile, permutation importance. CPU High-RAM ~50 min.
- Smoke passes. Not verified: Cboe download (blocked in this container); fallback is a WRDS cboe.cboe export at MindFlow/data/VIX_History.csv.

## 2026-11-24 (2:00 PM ET)
- Charles started Step6a (1:29 PM ET) and asked for CNN+VIXY / ViT+VIXY.
- Built (EC PR #15, chartcnn 2026-11-24.1): PANELS per arch (spy, vixy); archs plain_vixy, plain_spy_vixy, vit_vixy, vit_spy_vixy (702x234 for spy+vixy). tools/build_panel_bundle.py packs VIXY on the stock clock into vixy_panel_v4.zip (missing full-session bars filled flat); mindflow.stage_panels unpacks it. Step3a: RUN_ID=3a run + RETRAIN_ARCHS=VIXY_ARCHS adds them on identical rows. Smoke passes.
- Waiting on Charles: run build_panel_bundle on the Mac (VIXY adjusted folder name assumed 02_adjusted_vixy, unverified), upload zip, run new Step3a (L4, ~2 h).
- Not added: HGB with VIXY numeric features.

## 2026-11-25 (8:05 PM ET 11-24)
- build_panel_bundle failed: wrong --prepared path; correct is v4/03_prepared/prepared (us_pipeline writes artifact_root/prepared). Docstring fixed. VIXY adjusted folder name still unverified.

## 2026-11-25 (8:10 PM ET 11-24)
- Charles built vixy_panel_v4.zip on the Mac; told him to upload it and run new Step3a with RUN_ID=grid_20261123_083117, RETRAIN_ARCHS=VIXY_ARCHS.
- Step6a done 2:14 PM ET 11-24. HGB vol 2023, 5m / 60m: base 0.6888/0.6788; +VIX 0.6897/0.6787 (tied); +multi-day RV 0.6914/0.6811 (better, +0.0026/+0.0023 significant); both 0.6915/0.6805 (better); regime-only 0.559/0.562.
- By VIX tercile: positive share 0.43 -> 0.53; AUC falls 0.693 -> 0.669 (5m).

## 2026-11-25 (8:55 PM ET 11-24)
- Charles started Step3a VIXY retrain (~8:45 PM ET; plain_spy_vixy, vit_spy_vixy on 3a run grid_20261123_083117).
- Added HGB + VIXY intraday features to Step6a (regime 2026-11-25.1: 11 vixy_* features over the 78-bar window, like SPY's; subsets "+ VIXY intraday" and "+ VIXY intraday + multi-day RV"). Smoke passes. New Step6a notebook uploaded (old one renamed run1_outputs). Charles to run it on CPU, ~1 h.

## 2026-11-25 00:10 ET
- Asked: confirm Step3a VIXY retrain finished. Found: on Drive only plain_spy_vixy cell 1 epoch 1 written (val AUC 0.652, 8:22 PM ET 11-24), nothing after for 3.5+ h; no vit_spy_vixy, no grid_table/takeaways. Likely Colab disconnect. Told Charles to check notebook / rerun same cell. Step6a VIXY-feature run not started yet. Not verified: actual notebook error.
- 00:29 ET: Step3a VIXY retrain restarted on L4 (resumes plain_spy_vixy from epoch 1 last.pt). Note: T4 has no bf16 -> fp32; eval_batch 1024 on 702x234 ViT may OOM on T4 (lower eval batch if T4 ever used).
- 00:55 ET: Wrote MindFlow Step7a_Efficient_Computing (EC PR #15 commit 4d15f2c; code/efficiency.py, tools/cells_efficiency.py). Charles: run it only AFTER the final model is frozen. SOURCE = any 3a grid net or 4a indicator variant; float32/bf16 GPU + INT8 CPU (static FX PTQ for CNN, dynamic for ViT), size/latency/memory, paired day-bootstrap AUC; DINOv2 ViT-S/14 + LoRA (r8 q/v) fine-tune on same inputs/rows, training-cost table (CNN, scratch T2T-ViT, LoRA, full FT, frozen). Smoke passed on both source kinds (CPU only; GPU paths and real timings unverified; time estimate ~1.5-2 h on L4 not measured). Uploaded efficiency.py, mindflow.py (old renamed *_old_20261124.py.txt), Step7a notebook to Drive.
- 11:05 ET: 3a VIXY retrain (T4 fp32 after L4 bf16 epochs 1-2; eval batch 256). vol_h1 ES-day results (every 4th day, 204k rows, best epoch): CNN+SPY 0.655/LL 0.645 vs CNN+SPY+VIXY 0.650/0.654 (1,095,938 params); ViT+SPY 0.660 vs ViT+SPY+VIXY 0.654 at epoch 5 (still training, 73 min/epoch on T4; 896,770 params). VIXY panel does not help. Suggested L4 or stop after vol_h1 with HORIZONS=[1]. Full-year grid_table not yet available (predictions.parquet 57 MB too big for Drive connector).
- 15:45 ET: 3a VIXY retrain DONE (vol_h1 only; HORIZONS=[1]). Full 2023 vol 5m AUC: CNN+SPY 0.660, CNN+SPY+VIXY 0.661 (1,095,938 params, best ep 2 = bf16), ViT+SPY 0.661, ViT+SPY+VIXY 0.663 (896,770, best ep 5, T4 fp32), HGB+SPY 0.689. VIXY vs without: tied for both. Recommended closing VIXY line; next: final model = 4a top rung + rv5d/rv20d in tab branch (awaiting Charles's go).

## 2026-11-25 16:15 ET
- Asked: write the final-model notebook (Step6b); tell Charles when Step6 and later are ready, he runs them.
- Checked 4b: forward selection (3 features) was worse than all 38, so all 38 kept + 5 multi-day RV (rv5d, rv20d, rv_ratio, spy_rv5d, spy_rv20d) from 6a.
- Wrote MindFlow Step6b_Final_Model (EC PR #15 commit b82dcf4; code/final_model.py, tools/cells_final.py). Reuses newest Step 4a run's windows + EWSD scale (identical rows), same CNN settings. Variants ma_ewsd_tab_rv, tab_only_rv, ewsd_tab_rv; cells vol_h1 + vol_h12; HGB 38 and 43 refit on same train windows; Step 4a scores joined; paired day-bootstrap checks; auto-pick FINAL (best image variant at 5m on 2023) -> final_choice.json; test 2024-25 section behind RUN_TEST. Step7a can now take SOURCE step "6b".
- Smoke (CPU) passed for Step6b, Step7a (4a source) and Step7a (6b source). GPU timing not measured (estimate ~4 h on L4).
- Uploaded to Drive: final_model.py, new efficiency.py and mindflow.py (old ones trashed), Step6b notebook, new Step7a notebook (old trashed, never run).
- Not verified: real-data run, RAM for RV tables (~6 GB estimate).

## 2026-11-25 17:25 ET
- Charles asked whether Step6a needs running: no. Skipped the 6a VIXY-feature rerun (reason in DECISIONS). 6b is independent of 6a runs.
- 17:25 ET: Charles started Step6b on Colab (browser closed, Pro+ background).
- 17:45 ET: Pushed MindFlow results for 1a-6a (summary files only, run used by Step 5; Drive paths stripped from settings; example price-chart images and predictions not committed) + results/README.md. Charles: merge only finished steps, no executed notebooks. Reverted 6b, 7a and the unrun 6a VIXY-feature commit on PR #15, merged EC PR #15 (3ee614a). 6b/7a code now on branch mindflow/final-and-efficiency (main + revert of reverts, same code as b82dcf4); open its PR when 6b has results.
- 18:25 ET: Charles idea: single-factor strategy metrics (Sharpe, MDD, IC, IR) from the vol forecast. Proposed Step6c after 6b: IC/ICIR vs realized |r|, vol-sorted L/S Sharpe/MDD (expected ~0, costs 0/1 bp), vol-targeting Sharpe/MDD; 2023 first, test after unlock. Awaiting his go.
- 18:30 ET: Charles approved Step6c. Wrote Step6c_Volatility_Use (code/vol_use.py, tools/cells_vol_use.py; EC branch mindflow/final-and-efficiency commit 9dd3f30). Reads 6b predictions (no training, CPU): IC/ICIR (score vs |r|/own median; implied vol s=m/(-ln p) vs raw |r|; vs signed return), low-minus-high predicted-vol quintile L/S Sharpe/MDD at 0/1/2 bp (4 one-way trades per period), long-only sizing (equal weight, inverse median vol, risk parity, vol target) Sharpe/MDD/vol-of-daily-vol, time-series risk IC. Test split read only if 6b test_predictions exist. Smoke OK (synthetic). Uploaded vol_use.py, new mindflow.py (old trashed), Step6c notebook to Drive.
- 18:50 ET: 6b run final_20261125_222625 progress: vol_h1 done 23:13Z (46 min, ~4.2 min/epoch, best epochs 2/2/3); vol_h12 ma+tab_only done, ewsd_tab_rv epoch 3 at 23:43Z (likely early-stopped). ES-subset AUC 5m/60m: ma_ewsd_tab_rv 0.6912/0.6888, ewsd_tab_rv 0.6917/0.6888, tab_only_rv 0.6930/0.6883 (subset only, not final). Params CNN 783,556, tab_only 7,106.
- 19:28 ET: 6b DONE (run final_20261125_222625, 00:14Z). FINAL = ewsd_tab_rv (783,556 params). Val AUC 5m/60m: HGB43 0.6894/0.6780, final CNN 0.6881/0.6760, tab_only_rv 0.6878/0.6739, ma_ewsd_tab_rv 0.6873/0.6758, HGB38 0.6863/0.6760, 4a CNN 0.6848/0.6748. Checks: RV->CNN +0.0026 5m better, 60m tied; RV->HGB better both; no-MA tied; final CNN vs HGB43 5m tied (-0.0013), 60m worse (-0.0020); image vs numbers-only 5m tied, 60m +0.0020 better (no MA). Told Charles next: 6b RUN_TEST once, then 6c (CPU), then 7a SOURCE=dict(step="6b", cell="vol_h1", model="ewsd_tab_rv").
- 20:15 ET: Verified 6b test procedure: no extra 6b run on Drive; resume (RUN_ID="final_20261125_222625", RUN_TEST=True, Run all, GPU T4+High-RAM) tested on synthetic data: loads done cells, scores test, writes test_table/test_checks, no new folder. 4a ma_ewsd_tab checkpoints present on Drive. Sent Charles exact steps + checks.
- 21:10 ET: Charles asked whether run-length encoding suits CNN efficiency; approved exploring it. Added Step7a section "6b. Data pipeline" (EC branch mindflow/final-and-efficiency commit 4f32b95): chart storage per encoding, drawing vs packed-cache loader rate vs GPU train rate, lossless check (model scores drawn vs cached), repay epochs. Charts have 3 pixel values (0/180/255) so 1-bit is lossy; cache uses 2-bit palette bit planes, decoded on GPU. Smoke (CPU, synthetic) per chart 312x234: float32 292 KB, uint8 73 KB, RLE rows 43 KB, RLE columns 2.3 KB, 2-bit packed 18.7 KB, packed+zlib 1.0 KB; cached scores identical (diff 0). RLE does not speed up convolution itself (dense conv needs dense input); benefit is storage/IO only. Real rates (GPU) not measured yet. Uploaded new efficiency.py + Step7a notebook to Drive (old trashed).

## 2026-11-25 23:50 ET
- 6b TEST done (run final_20261125_222625, test 2024-25, 485 days, ~1.52M rows/cell, files 04:13Z). AUC 5m/60m: HGB43 0.6983/0.6850, final CNN ewsd_tab_rv 0.6974/0.6827, HGB38 0.6973/0.6835, ma_ewsd_tab_rv 0.6970/0.6822, 4a CNN 0.6969/0.6809, MLP43 0.6960/0.6816. Checks: final CNN vs HGB43 5m tied (-0.0009), 60m worse (-0.0023); image vs numbers-only (no MA) better both (+0.0014/+0.0011); RV->HGB better both; RV->CNN 5m tied, 60m better (+0.0012); drop MA tied/better. Same story as 2023 val. Next: Step6c (CPU), Step7a SOURCE 6b ewsd_tab_rv.

## 2026-11-26 00:58 ET
- Step6c DONE (Drive runs/step6c_volatility_use, finished 04:58Z, ~3 min CPU). Final CNN ewsd_tab_rv vs HGB43, val 2023 / test 2024-25:
  - IC vs |r|/own median 5m: CNN 0.117/0.129, HGB43 0.124/0.137 (ICIR ~1.5, >90% times positive); 60m CNN 0.103/0.114, HGB43 0.113/0.124. IC vs raw |r| ~0.23-0.24 vs no-forecast median 0.17-0.19. IC vs signed return ~0 (5m test +0.012 t3.5 only).
  - Low-minus-high predicted-vol L/S: gross Sharpe -0.1 to -1.7 (no low-vol anomaly intraday), <= -3.5 at 1 bp. Dead.
  - Long-only sizing: risk parity lowers vol slightly (5m test 0.076->0.067), Sharpe same; vol target halves vol-of-daily-vol (1.18->0.48) and cuts 60m test MDD 1909->1019 bp, but lowers Sharpe (60m 0.95->0.56; 5m ~ -0.1).
  - Time-series risk forecast: Spearman(pred book vol, |EW book return|) 0.42-0.45 for all models vs ~0 for no-forecast.

## 2026-11-26 01:20 ET
- Charles: 7a to be run by a teammate (shared Demo folder; she must have Editor access and set HERE in her copy to her shortcut path; told him only MindFlow/ is needed). Charles more interested in distillation; I noted expected gain is small (MLP 7,106 params already 0.696 vs CNN 0.697).
- Wrote MindFlow Step7b_Distillation (code/distill.py, tools/cells_distill.py; EC branch mindflow/final-and-efficiency commits efb0769, 206ab6b). Teacher = 6b final checkpoint (reads 6b run; scores train windows once, cached). Students on identical rows/rule: CNN 16-32-64 (86,020 params, 1/9) full chart and half-height chart (2x1 max pool), each on labels (alpha 0) and distilled (alpha 0.7, T 1); distilled 43-number MLP (twin = 6b tab_only_rv). Paired checks, params/MB/GPU latency frontier (frontier.csv/png), RUN_TEST section uses 6b test windows/scores. HORIZONS=[1] default. Smoke OK (new + resume). Est. 2.5-3 h on L4, High-RAM. Uploaded distill.py, new mindflow.py (old trashed), Step7b notebook to Drive. Not run on real data.
- 02:05 ET: Charles said merge 6b+6c ("推一次...然后合并"). New branch mindflow/final-results from main with only 6b/6c code (no 7a/7b, no unrun 6a VIXY-feature) + results (results/step6b_final_model/final_20261125_222625, results/step6c_volatility_use/voluse_20261126_045556; summaries/figures only) + README rows. Smoke 6b+6c OK. EC PR #16 merged (c4a43c4). 7a/7b stay on mindflow/final-and-efficiency (unrun).

## 2026-11-26 21:00 ET
- Step7b DONE on val 2023 (run distill_20261126_232557, ~55 min on GPU, 5m only). AUC: teacher 0.6881, HGB43 0.6894, MLP labels (6b) 0.6878, MLP distilled 0.6887, small CNN labels 0.6875 / distilled 0.6885, half-height labels 0.6875 / distilled 0.6886. Distillation better for all 3 students (+0.0008..+0.0010, significant); distilled students tie teacher; half-height = full. GPU bf16 batch 256: teacher 56 ms, small 12.6 ms (4.5x), half 6.7 ms (8.3x), MLP 0.29 ms (192x). Params 783,556 / 86,020 / 7,106. Test section not run yet.

## 2026-11-27 — Roadmap artifact updated to v4 + steps 6a–7b
- Asked: update the 心路历程 roadmap page with everything done since 2026-11-20.
- Did: republished the same artifact (version 6). All numbers now v4 (val 2023, test 2024–25). Stages 3–6 refreshed (ViT collapse noted as fixed training bug, ViT = CNN 0.662; ablation image over features +0.002; feature selection kept 38). Added stage 7 (regime: VIX flat, multi-day RV +0.0025, VIXY no; final CNN 783,556 params val/test vs HGB43/HGB38/MLP43), stage 8 (6c IC/ICIR, long-short negative, vol targeting halves vol-of-vol, max DD 19.1%→10.2%), stage 9 (7b distillation table + speeds; 7a pending teammate with pipeline storage numbers). New "next" cards: 7b test, 7a run, filter analysis/Grad-CAM, cleanup/report.
- Not verified: page checked only by headless render (no horizontal scroll at 400/1200 px, no JS errors); figures copied from earlier results, not recomputed.

## 2026-11-27 — Step 6d pure-image model (written, not run)
- Asked: a CNN that sees only pictures (no numbers input) as the project's original idea; draw the key features into the image; update the roadmap with a sample once run. Charles chose 5-minute only first.
- Did: EC branch mindflow/final-and-efficiency 268d927: code/pure_image.py + Step6d_Pure_Image notebook. Variants pure_key (423x234: 6b chart+EWSD, grey session-open line for time of day, SPY EWSD band, volume band log(vol/mean prev 78), 4 multi-day RV bars stock/SPY 5d/20d on one log scale) and pure_all (+ strip of all 43 standardised numbers as bars, 466x234). Strip scales fitted on train only. Same CNN/rows/training as 6b; joins 6b + 4a scores for paired checks; saves sample PNGs. Smoke (synthetic) passes. Uploaded pure_image.py, mindflow.py (6d registered) and the notebook to Drive.
- Found: Drive code/ had a teammate's pruning.py and a "7c": "step7c_pruning" entry in mindflow.py that is not in git; kept it in the Drive copy, not added to the repo.
- Not verified: no real-data run; L4 time estimate (~2.5-3 h for both variants at 5 min) is extrapolated from 6b.

## 2026-11-27 — Step 6d run done; roadmap updated
- Run pure_20261127_023749 (A100, ~35 min, 5 min only, val 2023, same rows as 6b): original chart 0.6599; chart+EWSD 0.6776; pure_key 0.6866 (+0.0089 signif; tied with final 0.6881; -0.0028 vs HGB43 signif); pure_all 0.6894 (+0.0027 over key signif; tied with HGB43 0.6894, final, MLP43). 776,450 params each. Test and 60m not run.
- Results committed to EC branch mindflow/final-and-efficiency (500c827, MindFlow/results/step6d_pure_image). Roadmap artifact v7: new stage 8 PURE IMAGE with two real 2023-01-12 samples; USE -> 9, EFFICIENCY -> 10.

## 2026-11-27 — Step 7b+ (pure-image distillation) and Step 6e (pure image, 60 min) written
- Asked: distill the pure-image line first (for tomorrow's advisor meeting), and a 6e notebook for the 60-min pure-image run after distillation.
- Did: EC a9a65d2 on mindflow/final-and-efficiency: code/pure_distill.py + Step7b_plus_Pure_Image_Distillation (teacher 6d pure_all; students small 16-32-64 hard/kd 78,914 params, half-height hard/kd, tiny 8-16-32 kd 29,858; alpha 0.7 T 1; speed/size frontier; test section). Step6e_Pure_Image_60min = 6d cells with HORIZONS [12], VARIANTS [pure_all]. Step keys "6e", "7b+". Smoke OK. Uploaded pure_distill.py, mindflow.py (kept teammate's 7c line) and both notebooks to Drive; 6d notebook on Drive left as run.
- Not verified: no real-data run; time estimate ~1-1.5 h on A100 is a guess.

## 2026-11-27 05:20Z — weight analysis of the 6d pure-image CNN (weights only, no data)
- Asked: what did the model learn; look at weight distributions.
- Did: reconstructed init (seed 69) for 6d pure_all / pure_key, compared W with W0; figures kept in the project files (weights/), script figs.py there too. Not in git.
- Found: conv layers moved only ~5% from init (5.1/4.7/5.7% pure_all), BN gamma 0.993-1.007 (lr 1e-5, best epoch 2-3), so raw histograms look like init. Layer-1 dW concentrates on the LEFT column of each 3-px bar (open tick + session-open marker line): |dW| ~2x the other columns, 61-64% of channels positive. Head |w| near uniform over 4 vertical bins (22-27%) and channels; recent 5-8 bars weighted 1.37x (all) / 1.64x (key) of mean; left-edge peak is mostly zero-padding.
- Not verified: weight size is not importance. Needs data: strip/marker occlusion, channel ablation, Grad-CAM (not written yet).
- 7b+ running on Charles's L4 since ~00:17 ET; self check-in scheduled 08:30 ET to read results and prep advisor summary (meeting 13:30 ET).

## 2026-11-27 06:10Z — Step 8a notebook (data-based "what the pure-image CNN learned")
- Asked: go beyond weights (activation size matters); how to analyse what the model learned.
- Did: EC branch mindflow/final-and-efficiency 6e81eda: code/inside_cnn.py + tools/cells_inside.py -> Step8a_Inside_Pure_CNN (step key "8a" -> step8a_inside_pure_cnn). Fixed 6d models, 20k val-2023 rows (~80/day), no training: region occlusion (mean/black fill), erase (session line, open/high-low/close ticks), 6-bar time blocks, per-number zeroing (pure_all), activation x weight std maps + channel ablation (top-k vs random), Grad-CAM + grad x input region shares, ridge probing of 43 numbers + time of day (GroupKFold by day). Smoke OK (scores match 6d saved scores, corr 1.0). Uploaded to Drive (mindflow.py re-uploaded WITH the 7c line).
- Not verified: real run; GPU time estimate 20-40 min on L4.
- 7b+ (puredistill_20261127_051847) running on L4: small_kd ~2.7 min/epoch, es AUC 0.694 at epoch 7.

## 2026-11-27 13:55Z — morning check: 8a results, 7b+ stall
- 8a FULL done (inside_20261127_063450, 19,762 val rows, 241 d): erase session-open line -0.017 (all) / -0.024 (key) AUC = biggest single element; whole chart -0.052/-0.087; price area -0.011/-0.031; number strip -0.008 (each number <=0.0006, rv_ratio top); EWSD -0.003/-0.013; RV -0.002/-0.003; volume band -0.001/-0.003; SPY band, chart volume bars, open/close ticks ~0. Time: oldest 6 bars -0.018/-0.021 (unexplained: marker position or edge), newest 6 -0.005/-0.018, middle ~0. Top-64 channel zeroing only -0.002 (distributed). Probe R2: rv 0.95, rel_volume 0.90, realized_vol 0.82-0.85, time_of_day 0.64-0.68, single returns ~0. Results in EC results/step8a_inside_pure_cnn/ (d56e9fa); roadmap v9 stage 11.
- 7b+ two runs (puredistill_20261127_051847, _063830) both stopped after small_kd finished 8 epochs (last.pt deleted -> fit returned), no valid_prob.npy: failed/hung while scoring 2023 validation. Cause unknown (no error seen). Fix 313ec9f: finished students are scored only; progress prints. Uploaded to Drive. Resume: RUN_ID="puredistill_20261127_063830".
