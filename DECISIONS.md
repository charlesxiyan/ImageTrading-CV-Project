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
- **Charles's data pipeline is the source of bars for all modeling.** Jeffery's TAQ bar pipeline
  is set aside: no full 2010-2024 build from it, and it is not being pushed. 2026-11-14, jeffery.

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
- **Model inputs use point-in-time adjustment: each window is rebased to its last input bar.**
  Stored data stay on one basis with per-bar multipliers; nothing else is stored twice.
  2026-11-14, charles (agrees with the policy in `CLAUDE.md`).
- **Absolute direction (`dir_h1`, `dir_h12`) is reported next to `vol_h1` and `xsdir_h12`.**
  Zero returns are dropped; stride 6 for h=1 and 12 for h=12. 2026-11-14, charles.

## 2026-11-14: window screening and labels (charles)
- Window lists never filter on the label or future returns. Bundle v2 keeps zero next-bar returns;
  direction targets drop zeros per target and horizon at label time.
- Labels are unadjusted traded-price returns. A label whose horizon contains an ex-date (only across
  the overnight gap) is dropped rather than computed, since its unadjusted return holds the event jump.

## 2026-11-16: full data (charles)
- **Sample is 2016-01-04 to 2025-12-31 for modeling; split train 2016-2022, validation 2023, test
  2024-2025.** charles accepted the proposal. 2023 H1 was already used for tuning, so it is validation.
- **CIZ distributions:** cash adjustment iff divamt > 0 and facpr = facshr = 0 (incl. ROC, capital
  gains, special and non-USD dividends, whose CIZ amounts are already USD); any nonzero factor is left
  to the cumulative factors. Checked against CRSP ret - retx: 19,374 / 19,384 groups within 1%.
- **Charts are rendered per sample, never stored.** SPY is drawn as its own panel below the stock
  chart (charles's choice), each panel on its own window-local axis.

## 2026-11-20 — TAQ trade filter policy v3 (charles)
Keep sale conditions that are blank/NULL or only spaces, `@`, `E`, `F`, `I`:
`translate(coalesce(tr_scond,''),' @EFI','') = ''`. NULL is the regular sale on Tape A/B; odd lots (`I`)
are real trades whose share grew over 2016-2025. Every new archive must pass the CRSP acceptance
check (`DataCheck/check_bar_coverage.py --archive`) before it is adjusted or bundled.
- 2026-11-25 (charles): VIXY panel networks are evaluated on the 5-min cell only. Reasons to state in reports: (1) tied at 5 min for CNN and ViT, consistent with Step 6a (daily VIX adds nothing); 60-min results have followed the 5-min ranking; (2) compute limit: Colab gave only T4 (73 min/epoch for the 3-panel ViT), ~12 GPU-hours more for a comparison not expected to change the conclusion. Charles wants both reasons stated explicitly whenever something is skipped.
- 2026-11-25: Step6a rerun with HGB + VIXY intraday features skipped. Reason: no expected gain (VIXY panel tied in 3a for CNN and ViT; daily VIX tied in 6a, subsumed by stock RV). Step6b computes the RV features itself and does not need a 6a run.
- 2026-11-27 (Charles): report more than AUC: train/validation loss curves, log loss / calibration, and statistical diagnostics (QQ-plots etc.). Every run must save per-row predictions with join keys (security_index, end_index, decision_date) so later analyses need no retraining.
