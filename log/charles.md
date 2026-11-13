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
