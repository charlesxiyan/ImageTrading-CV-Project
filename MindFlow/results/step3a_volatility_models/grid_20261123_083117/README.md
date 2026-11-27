# Step 3a, volatility target: grid_20261123_083117

Validation 2023 (247 days, 822,653 rows per cell), v4 bundle. Summary files only; per-row predictions stay on Drive.

History of this run:
- 2026-11-23: all networks and HGB on `vol_h1` (5 min) and `vol_h12` (60 min).
- 2026-11-23: ViTs retrained after the tokenizer fix (chartcnn 2026-11-23.2).
- 2026-11-25: VIXY panel added (`plain_spy_vixy`, `vit_spy_vixy`, chartcnn 2026-11-24.1) on `vol_h1` only
  (`HORIZONS = [1]`), so these tables now list the 5-minute cell only. The 60-minute cell is unchanged on Drive
  (`vol_h12/predictions.parquet`) and in Step 5's summary.
  CNN + SPY + VIXY: best epoch 2 (L4, bfloat16); ViT + SPY + VIXY: epochs 3-7 on a T4 (float32), best epoch 5.

Why the VIXY networks were not trained on the 60-minute cell:
1. **No expected gain.** At 5 minutes the VIXY panel tied for both networks (below), consistent with Step 6a, where
   daily VIX added nothing to HGB once the stock's own data and SPY were in. Every 60-minute result so far has
   followed the 5-minute ranking (Steps 3a, 4a), so a tie at 5 minutes gives no reason to expect a gain at 60.
2. **Compute limit.** Colab assigned only T4 GPUs at the time (A100/L4/H100 requests all fell back to T4). On a T4
   the ViT with three stacked panels took 73 min per epoch (about 6 min on an L4), so the 60-minute cell would have
   needed roughly 12 more GPU hours for a comparison not expected to change the conclusion.
The 60-minute cell can still be added later with the same `RUN_ID`, `HORIZONS = [1, 12]` and
`RETRAIN_ARCHS = VIXY_ARCHS`.

Result (5 min): VIXY panel vs without, paired day bootstrap: tied for the CNN (0.661 vs 0.660) and the ViT
(0.663 vs 0.661). HGB hand + SPY 0.689 stays best.
