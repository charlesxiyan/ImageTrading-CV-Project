# Status

Last updated 2026-11-20 by charles.

## Project
CV course research project: do pretrained vision models help on intraday price charts?
Planned comparison: ResNet-18 from scratch against ImageNet-pretrained, sparse OHLC against
candlestick rendering, a DINOv2 probe, a 1D baseline, and a synthetic diagnostic.
Inputs are 78 five-minute bars. The proposal's label is the sign of the next 30-minute return.
The same data is reused for a Big Data course project. Presentation Dec 10, report Dec 18.

## Data pipeline (jeffery)
- **Universe:** built for 2010-2024. Union of S&P 500, Nasdaq 100 and DJIA 30 by date:
  3,774 trading days, 877 PERMNOs, 518-531 securities a day. Checked against WRDS source tables.
- **5-minute bars from TAQ:** built and validated for 2023 only: 250 dates, 10.2M bars.
  Open, high and low match CRSP within 1 bp on about 99.9% of security-days.
- **Set aside on 2026-11-14:** charles's pipeline is the data source. No full build from this one.
- **Code:** `taq_bars.py`, `examples.py`, `validate_bars.py`, `validate_universe.py`,
  `build_taq_symbol_map.py` and changes to `build_dynamic_universe.py` exist only in jeffery's
  working tree on `test/mock-wrds`. **Not committed or pushed.**
- **Known gap:** `examples.py` window code does not yet follow the price adjustment policy in
  `CLAUDE.md`. It ignores cash dividends and divides by absolute `cfacpr`.

## Modeling (charles)
- Colab runs on 2023 H1 data, train to 2023-03-31, validate on April. See `EXPERIMENTS.md`.
- Direction targets show no signal so far, AUC 0.50. A volatility target reaches AUC 0.69.
- **GPU results are in.** CNN `run_20261114_020923` and pretrained `run_20261114_012926`: see
  `EXPERIMENTS.md`. Code, executed notebooks and small results are in CV PR #2 (contains PR #1)
  and EC PR #2, both merged 2026-11-15. Next: Colab rerun with dir_h1 / dir_h12 (`*_dir_targets.ipynb`).
- Charles's data preparation (`us_pipeline`, universe, TAQ bars, adjustment scripts) is in
  `charlesxiyan/ImageTrading-EC-Project`, not in this repository. The EC repository is charles's
  own (for another course); jeffery does not review it.
- To be confirmed by charles: how his bars were built and adjusted, and his current plan.

- **Multi-target grid (2026-11-14):** {dir, vol, xsdir} × {5, 15, 30, 60 min}, stride 12, plain CNN
  + series1d + HGB. Code merged to `main` in both repos (2026-11-15, with the #2 PRs) and on Drive `Demo/Multi-Target Chart CNN Baseline/`.
  Not run yet; charles runs it on Colab.
- **Bundle v2 (2026-11-14):** window lists no longer drop zero next-bar returns; labels use
  unadjusted prices and drop horizons containing an ex-date (chartcnn 2026-11-14.3). See DECISIONS.

## Full 2016-2025 data build (charles, done 2026-11-16)
- Archive 2016-01-04..2025-12-31: 2,514 sessions, 775 PERMNOs, CRSP CIZ. Adjusted + validated (all files).
  SPY (PERMNO 84398) on the same calendar/grid, adjusted + validated. us_pipeline prepare+audit:
  79.5M windows (train 2016-22 59.6M / val 2023 7.0M / test 2024-25 12.9M); 21 early closes excluded.
- Bundle `alldata_bundle_v1.zip` (1.19 GB, float32, windows end_index % 6 == 5, SPY arrays) lives
  next to the archive locally; charles must upload it to Drive `Demo/AllData & AllModel/` himself
  (the Drive connector caps uploads at 10 MB). Notebook `AllData_AllModel.ipynb` already there.
- Notebook: 12 cells (dir/vol/xsdir x 5/15/30/60 min, stride 12) x {image CNN, image CNN + SPY panel
  (468x234), 1D CNN, 1D CNN + SPY, HGB, HGB + 18 SPY features}; QUICK switch; test section opt-in.
  Smoke-tested end to end locally on CPU. Not run on GPU yet.
- Code: EC branch `claude/project-thread-5acfxn` commit fedd666, local only (not pushed).

- **AllData (2026-11-16):** 2016-2025 bars from charles's pipeline, train <=2022, val 2023,
  test 2024-2025 not run. Run `run_20261116_044159` in CV PR #4 (open, jeffery to review).
  A ViT variant is ready on Drive, not run.

- **Big Data course (jeffery):** Spark pipeline in its own private repository: WRDS JDBC
  ingestion, profiling, cleaning and 5-minute bars for 2023-03-06 to 03-17. Built and validated
  (bars identical to the pandas pipeline); report ready. Due 2026-11-20.

## Open questions
- Does the WRDS license allow TAQ-derived bars on shared servers and Colab? Not yet confirmed.

## Next steps
1. CV PRs #1 to #3 are merged (2026-11-15). Pretrained vision models (ImageNet ResNet-18,
   DINOv2) and OHLC versus candlestick rendering, the proposal's core comparison, are not run yet.
2. charles: merge his memory here; push training code; confirm the points above.
3. Both: agree on one dataset, one label and one evaluation protocol before more training.

## Data rebuild (2026-11-20, charles)
- The 2016-2025 v1 archive and bundle (`alldata_bundle_v1`) have biased volume: the TAQ filter dropped
  blank (NULL) regular-sale conditions, hitting Nasdaq names in early 2016 and other listings from 2018.
  Prices are fine. All AllData results so far use v1.
- Fix: trade-filter policy v3 (NULL read as blank, odd lots kept), in EC #8 / CV #6 with the code
  reorganized into DataDownload/, DataCheck/, PriceAdjustment/, Pipeline/.
- 2026-11-20: policy v4 (teoguo, EC #10): odd lots count in volume but not in prices. EC #11 fixes 2016
  dual-class symbols (GOOGL, CMCSA, DISCA, DISCK, LBTYA, LVNTA, BATRA, BATRK had no trades for parts
  of 2016). Both mirrored in CV #8. v4 pilot (5 sessions) passed every acceptance check.
- 2026-11-23: v4 bundle built (`v4/04_bundle/alldata_bundle_v4.zip`, 1.34 GB; 100.3M windows, +26% vs v1; all validations and audit pass). Next: charles uploads it to Drive `Demo/AllData & AllModel/`, then
  re-run of the main experiments.
- 2026-11-23: MindFlow (EC PR #15, Drive Demo/MindFlow with v4 bundle + code + 9 notebooks Step1a..Step5) replaces the ad-hoc v4 rerun plan. Next: charles runs them in order on Colab; smoke-tested only.
- 2026-11-24: MindFlow complete on v4 (Step5 rerun 12:44 PM ET with corrected foundation-model AUCs). Chain: daily null, intraday direction null for all models, vol signal (HGB+SPY 0.689/0.679 best; foundation models 0.58-0.64), CNN+EWSD+features ties HGB. EC PR #15 awaiting merge.
- 2026-11-24: Step6a done: daily VIX adds nothing to HGB; multi-day RV (rv5d/rv20d) +0.0025 AUC significant. VIXY panel: zip built, Step3a VIXY retrain pending (Charles).
- 2026-11-25: running: Step3a VIXY retrain (L4) and Step6a rerun with HGB+VIXY intraday features (CPU). Compare both when done.
- 2026-11-25 00:30 ET: Step3a VIXY retrain restarted on L4 (first try died after 1 epoch). Step6a VIXY-feature rerun not yet started. Plan to freeze: 3a VIXY result -> 6a VIXY -> final features (add rv5d/rv20d, VIXY if useful) into CNN tab branch, one 4a rung rerun -> test 2024-25 once -> then Step7a_Efficient_Computing (written, smoke-tested; Charles runs it only after the freeze, setting SOURCE to the final model).
- 2026-11-25 15:45 ET: Step3a VIXY retrain done (5m only): VIXY panel tied for CNN (0.661) and ViT (0.663) vs SPY-only; HGB+SPY 0.689. VIXY line closed (proposed). Next: freeze final model (4a top rung + rv5d/rv20d), then test 2024-25, then Step7a.
- 2026-11-25 16:15 ET: Step6b_Final_Model written + smoke-tested, on Drive, DONE 19:14 ET 11-25 (run final_20261125_222625, FINAL=ewsd_tab_rv; val 5m CNN 0.6881 vs HGB43 0.6894 tied). Then: check FINAL, set RUN_TEST=True once (test 2024-25), then Step7a with SOURCE step "6b". EC PR #15 merged 11-25 (steps 1a-6a code + results). 6b/6c/7a code on EC branch mindflow/final-and-efficiency, PR after 6b results. Step6c (vol factor IC/L-S/sizing, CPU, minutes) runs after 6b. 6b TEST 2024-25 DONE: 5m/60m HGB43 0.6983/0.6850, final CNN 0.6974/0.6827 (5m tied, 60m -0.002), image vs numbers-only +0.001 both. Step7a now also has a data-pipeline section (drawing vs 2-bit packed chart cache; RLE sizes), smoke OK, GPU rates unmeasured.
