# Status

Last updated 2026-11-14 by jeffery.

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
- **Set aside on 2026-11-14:** charles's pipeline is the data source. No full build from this one.
- **Code:** `taq_bars.py`, `examples.py`, `validate_bars.py`, `validate_universe.py`,
  `build_taq_symbol_map.py` and changes to `build_dynamic_universe.py` exist only in jeffery's
  working tree on `test/mock-wrds`. **Not committed or pushed.**
- **Known gap:** `examples.py` window code does not yet follow the price adjustment policy in
  `CLAUDE.md`. It ignores cash dividends and divides by absolute `cfacpr`.

## Modeling (charles)
- Colab runs on 2023 H1 data, train to 2023-03-31, validate on April. See `EXPERIMENTS.md`.
- Direction targets show no signal so far, AUC 0.50. A volatility target reaches AUC 0.69.
- **GPU results are in.** CNN `run_20261114_020923` and pretrained `run_20261114_012926`: see
  `EXPERIMENTS.md`. Code, executed notebooks and small results are in CV PR #2 (contains PR #1)
  and EC PR #2, both merged 2026-11-15. Next: Colab rerun with dir_h1 / dir_h12 (`*_dir_targets.ipynb`).
- Charles's data preparation (`us_pipeline`, universe, TAQ bars, adjustment scripts) is in
  `charlesxiyan/ImageTrading-EC-Project`, not in this repository. The EC repository is charles's
  own (for another course); jeffery does not review it.
- To be confirmed by charles: how his bars were built and adjusted, and his current plan.

- **Multi-target grid (2026-11-14):** {dir, vol, xsdir} × {5, 15, 30, 60 min}, stride 12, plain CNN
  + series1d + HGB. Code in CV PR #3 / EC PR #3 and on Drive `Demo/Multi-Target Chart CNN Baseline/`.
  Not run yet; charles runs it on Colab.
- **Bundle v2 (2026-11-14):** window lists no longer drop zero next-bar returns; labels use
  unadjusted prices and drop horizons containing an ex-date (chartcnn 2026-11-14.3). See DECISIONS.

## Open questions
- The demo predicts the next bar or 12 bars ahead. The proposal says 30 minutes, 6 bars.
- Does the WRDS license allow TAQ-derived bars on shared servers and Colab? Not yet confirmed.

## Next steps
1. jeffery: review CV PRs #1 to #3.
2. charles: merge his memory here; push training code; confirm the points above.
3. Both: agree on one dataset, one label and one evaluation protocol before more training.
