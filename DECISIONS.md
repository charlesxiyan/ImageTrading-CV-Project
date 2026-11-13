# Decisions

Each entry: the decision, then why. Add the date and who decided.

## Data
- **Identity is CRSP PERMNO, never the ticker.** Tickers change and TAQ spells them differently.
  A date-level map from PERMNO to TAQ symbol root and suffix is built first. 2026-11-01, jeffery.
- **Sample is 2010-2024.** TAQ read access covers 2010 onward and CRSP ends 2024-12-31. 2026-11-01, jeffery.
- **Universe keeps every index member, including REITs and foreign-incorporated firms.**
  Filter by share code later if needed. 2026-11-02, jeffery.
- **Bars are half-open five-minute intervals from 09:30 to 16:00, 78 per session.** The closing
  cross is in no bar, so a model never sees the auction print. 2026-11-01, jeffery.
- **Odd lots set no price but count toward volume.** They hold the worst off-market prints.
  Cost: very high-priced stocks get sparse bars. 2026-11-02, jeffery.
- **Bars on disk are raw, unadjusted prices.** Adjustment is applied per window. See the price
  adjustment policy in `CLAUDE.md`: adjust history only, never the future. 2026-11-02, jeffery.
- **Early-close sessions are dropped when building examples**, as is 2023-01-24, the NYSE
  opening failure. Bars for those dates are still built. 2026-11-02, jeffery.
- **Licensed WRDS data never enters the repository.** `data/` is gitignored.

## Modeling and evaluation
- **Learning rate 1e-5 with Adam**, as in Jiang, Kelly and Xiu (2023). Runs at 1e-3 collapsed to a
  constant prediction. 2026-11-12, jeffery, proposed; charles to confirm.
- **Do not benchmark against `RichardS0268/CNN-for-Trading`.** Its reported accuracy is validation
  accuracy on a random split of overlapping windows. Compare with the original paper, whose
  out-of-sample McFadden R2 is only 0.1-0.4%. 2026-11-12, jeffery.
- **Report more than accuracy:** AUC, log loss against the training prior, and decile long-short
  returns, with uncertainty clustered by date. Samples on one day are strongly dependent: the
  daily share of up labels ranged from 31% to 65% in 2023. 2026-11-12, jeffery, proposed.
- **A 1D baseline must use the same per-window scaling as the images.** In the paper a 1D CNN on
  image-scaled data matches the 2D CNN, so scaling explains much of the gain. 2026-11-12, jeffery, proposed.

## Process
- **Shared context lives in this branch**, not in each machine's local Claude memory. 2026-11-12.
