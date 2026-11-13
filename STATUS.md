# Status

Last updated 2026-11-13 by charles.

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
- **Full 2010-2024 build:** not started. Planned on a university compute server, about 58 hours.
- **Code:** `taq_bars.py`, `examples.py`, `validate_bars.py`, `validate_universe.py`,
  `build_taq_symbol_map.py` and changes to `build_dynamic_universe.py` exist only in jeffery's
  working tree on `test/mock-wrds`. **Not committed or pushed.**
- **Known gap:** `examples.py` window code does not yet follow the price adjustment policy in
  `CLAUDE.md`. It ignores cash dividends and divides by absolute `cfacpr`.

## Modeling (charles)
- Colab runs on 2023 H1 data, train to 2023-03-31, validate on April. See `EXPERIMENTS.md`.
- Direction targets show no signal so far, AUC 0.50. A volatility target reaches AUC 0.69.
- **CNN code is in PR #1** (`CNN-Relevant/`, branch `cnn-relevant-residual`): plain JKX-style
  CNN and a residual (ResNet) variant, run on `vol_h1` and `xsdir_h12`. GPU run pending.
- Data preparation (`us_pipeline`) is still not in the repository.
- To be confirmed by charles: how his bars were built and adjusted, and his current plan.

## Open questions
- Jeffery's and Charles's bar data come from different pipelines. Which one is the source?
- The demo predicts the next bar or 12 bars ahead. The proposal says 30 minutes, 6 bars.
- Does the WRDS license allow TAQ-derived bars on shared servers and Colab? Not yet confirmed.

## Next steps
1. jeffery: commit and push the bar pipeline; fix window adjustment; run the full build.
2. charles: merge his memory here; push training code; confirm the points above.
3. Both: agree on one dataset, one label and one evaluation protocol before more training.
