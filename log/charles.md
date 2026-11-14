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
