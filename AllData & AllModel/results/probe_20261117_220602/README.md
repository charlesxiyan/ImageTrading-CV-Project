# probe_20261117_220602 (QUICK pass)

`FrozenEncoder_Probe.ipynb` with `QUICK = True`: 40,000 training windows, every 5th validation day of 2023
(49 days, ~115,000 rows per cell), horizons 60 and 5 minutes, encoders DINOv2-S and random ViT-S only.
Every model is scored on the same rows; CNN/HGB scores come from `run_20261116_044159`.

Validation AUC, vol (60 min / 5 min): HGB + SPY 0.682 / 0.690; image CNN (scratch) 0.651 / 0.658;
DINOv2-S linear probe 0.632 / 0.645; DINOv2-S MLP probe 0.632 / 0.644; random ViT-S MLP probe 0.596 / 0.606.

- Pretraining helps: DINOv2 beats the random encoder by about 0.04 on vol (significant in both vol cells; tied in the 4 direction cells).
- The frozen probes stay 0.01–0.02 below the scratch image CNN (worse in 2 of 6 cells, tied in 4) and about 0.05 below HGB + SPY.
- MLP probe = linear probe (tied in 6 of 6); SPY features add nothing; dir and xsdir are ~0.50 for every model.
- Caveat: the probes had 40,000 training rows, the CNN 300,000 and HGB 1,000,000. The full run is still to do.

Parameters: DINOv2-S encoder frozen; trainable probe 769 (linear) and 394,241 (MLP), doubled input with SPY.
Files: `probe_table.csv` (all cells and models) and `takeaways.md`, also on Drive in `probe_runs/probe_20261117_220602/`.
