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
| 2026-11-14 | charles | CNN `run_20261114_020923` | 2023 H1, val April, all val samples | `vol_h1` | plain 776,450 / residual 2,047,362 / series1d 720,770 params, lr 1e-5, batch 128 | AUC 0.687 / 0.685 / 0.690; HGB 0.705 | Image adds nothing over raw OHLCV |
| 2026-11-14 | charles | CNN `run_20261114_020923` | same, stride 12 | `xsdir_h12` | same | AUC 0.508 / 0.499 / 0.495; HGB 0.503 | No signal; spread t = -0.20 |
| 2026-11-14 | charles | Pretrained `run_20261114_012926` | 80 val timestamps, about 27k samples | `vol_h1` | Kronos-small 28.7M, Chronos-2 119M (+LoRA), Moirai2-small 11.4M, TimesFM 331M, Bolt-base 205M, zero-shot | AUC 0.699 / 0.615 (0.611) / 0.609 / 0.605 / 0.594; HGB 0.734, CNNs 0.718-0.722 | CNN rows from run_20261113_185632 |
| 2026-11-14 | charles | Pretrained `run_20261114_012926` | same | `xsdir_h12` | same | all 0.48-0.51 | No signal |
