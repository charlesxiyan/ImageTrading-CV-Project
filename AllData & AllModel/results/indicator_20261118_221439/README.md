# indicator_20261118_221439 (QUICK)

`IndicatorChart_CNN.ipynb` with `QUICK = True`:
- target: vol at 5 minutes;
- 40,000 training windows, 3 epochs;
- 49 validation days (every 5th day of 2023).

Training ran at about 1,100 images per second on Colab, about 2.5 minutes per variant.

| model | validation AUC (95% day-bootstrap CI) |
|---|---|
| CNN, original chart (control) | 0.641 (0.618–0.660) |
| CNN, chart + MA | 0.643 |
| CNN, chart + EWSD band | **0.667** (0.647–0.685) |
| CNN, chart + MA + EWSD | 0.667 |
| CNN, original chart + 38 hand/SPY features | 0.679 |
| CNN, chart + MA + EWSD + hand/SPY features | 0.678 |
| image CNN, reference run (300k windows) | 0.648 |
| HGB hand + SPY, reference run | 0.683 |

- **EWSD band:** +0.026 over the original chart, which is significant (paired day bootstrap). The original chart leaves out the absolute volatility level.
- **MA:** adds nothing for volatility (+0.002).
- **Gap to HGB + SPY:** 0.042 → 0.016 with the EWSD band, and 0.005 with the hand/SPY features too. HGB is still not beaten.

These are rough numbers: a small training set and only 3 epochs, and several models were still improving at epoch 3.
