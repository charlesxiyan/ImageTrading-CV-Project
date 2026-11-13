# Experiments

One row per run. Artifacts stay in Drive; never commit data or weights. Rows marked `?` were
filled in by jeffery from summary files and need charles to confirm or correct them.

| Date | Owner | Run | Data | Target | Model and settings | Validation result | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-12 | charles | `cnn78u_h1_baseline` | 2023 H1, train to 03-31, val April, 1.72M / 510k samples | next-bar direction | CNN78u, 655k params, lr 1e-3, wd 1e-4, batch 64, 3 epochs | AUC 0.500, log loss 0.6930 | ? Predicts up for every sample from epoch 2 |
| 2026-11-12 | charles | `cnn78u_repair_20pct` | 20% of the same train set | next-bar direction | wd 1e-4 against wd 0, same init | AUC 0.498 and 0.501 | ? No test evaluation |
| 2026-11-12 | charles | `cnn78u_scheduler_20pct` | 20% of the same train set | next-bar direction | wd 0, ReduceLROnPlateau from 1e-3, 8 epochs | AUC 0.496-0.500 | ? Early stopped |
| 2026-11-12 | charles | `paper_cnn` `vol_h1` | 2023 H1, stride 6, 85k val samples | next-bar volatility, high or low | channels 64/128/256, lr 1e-5, wd 0, batch 128 | AUC 0.687; HGB on hand features 0.705 | ? Shows the pipeline can learn |
| 2026-11-12 | charles | `paper_cnn` `xsdir_h12` | same | 12-bar cross-sectional direction | same | AUC 0.500; HGB 0.503; decile spread -0.4 bp a day | ? No signal |

## Reference results
Jiang, Kelly and Xiu (2023), daily US stocks, test 2001-2019: equal-weight decile spread Sharpe
7.15 for 5-day images and 5-day returns, 1.02 when restricted to the 500 largest stocks.
Out-of-sample McFadden R2 of 0.09-0.2% for the CNN forecast alone.
