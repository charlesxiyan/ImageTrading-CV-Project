# ImageTrading-CV-Project

Chart-image models for five-minute US equity bars, and the data pipeline that feeds them.
Code and documentation only: data, prepared samples, run artifacts and model weights stay
local and are ignored by git.

This repository mirrors the code and results of ImageTrading-EC-Project. Rules for both people and their
Claude Code sessions are in `CLAUDE.md`, and shared project state is on the `memory` branch (`memory/`).
The universe builder that used to sit at the root is now `DataDownload/build_dynamic_universe.py`;
its documentation is in `DATA_PIPELINE.md` ("Dynamic US equity universe").

## Layout

Data steps, in the order they run (each folder has a README with its commands):

| Step | Folder | What it does |
|---|---|---|
| 1 | `DataDownload/` | TAQ five-minute bars, index universe and CRSP reference data from WRDS (stocks 2016-2025, SPY). The trade filter is policy version 3: blank (NULL) sale conditions and odd lots are kept. |
| 2 | `DataCheck/` | Acceptance check of every download against CRSP daily data: prices, volume kept by listing group and year, and coverage. A failing archive is not used. |
| 3 | `PriceAdjustment/` | Split, cash-dividend and share-volume adjustment into a new folder; raw bars are never changed. |
| 4 | `Pipeline/` | `us_pipeline`: audited 78-bar windows and the train, validation and test splits (`configs/`). |
| 5 | `AllData & AllModel/tools/build_alldata_bundle.py` | Packages the prepared data, SPY and `chartcnn` into the Colab zip. |

Local data layout for the current build (licensed data, never committed):

```
~/Market Data/US Equity/v3/
  01_raw/  01_raw_spy/                 step 1
  02_adjust_inputs/  02_adjusted/      step 3 (plus the same two with _spy)
  03_prepared/                         step 4
  04_bundle/alldata_bundle_v3.zip      step 5
```

Experiments and other folders:

| Path | What it is |
|---|---|
| `AllData & AllModel/` | 2016-2025 grid (image CNN, 1D CNN, HGB, with and without SPY), frozen DINOv2/CLIP probes, indicator-chart CNN. Results so far use the v1 bundle. |
| `Daily Chart Replication/` | The paper's daily I5/I20 image CNN on the same stocks. |
| `CNN-Relevant/` | Chart CNN baseline (plain and residual/ResNet variants) and a 1D CNN on raw OHLCV, executed notebook, diagnostics and run results. See its README. |
| `Multi-Target Chart CNN Baseline/` | Direction, volatility and cross-sectional direction at 5-60 minutes on the 2023 H1 demo. |
| `Pretrained Model Comparison/` | Pretrained time-series foundation models (Chronos-2, TimesFM, Moirai, Kronos) against the chart CNN on the same samples, executed notebook and results. See its README. |
| `colab/` | Earlier CNN78u Colab tests: baseline, training-repair and LR-scheduler notebooks with their runners and validators. The data bundle (`cnn78u_colab_bundle.zip`) stays local. |
| `diagnostics/` | Diagnosis of the CNN78u next-bar collapse (`BASELINE_DIAGNOSIS.md`, `check_baseline.py`). |
| `tests/` | Offline unit tests (`tests/paths.py` puts the step folders on the import path). |

Documentation:
- [DATA_PIPELINE.md](DATA_PIPELINE.md) covers the universe, the bar standard, the acceptance check and price adjustment.
- [DataDownload/US_EQUITY_ARCHIVE.md](DataDownload/US_EQUITY_ARCHIVE.md) covers the 2016-2025 archive.
- [DataDownload/INTRADAY_DEMO.md](DataDownload/INTRADAY_DEMO.md) covers the 2023 H1 demo.
- [Pipeline/US_PIPELINE.md](Pipeline/US_PIPELINE.md) covers the CNN78u pipeline.

## Latest results (2023 H1 data, April 2023 validation)

| model | parameters | `vol_h1` AUC | `xsdir_h12` AUC |
|---|---|---|---|
| Chart CNN, plain | 776,450 | 0.687 | 0.508 |
| Chart CNN, residual | 2,047,362 | 0.685 | 0.499 |
| 1D CNN on raw OHLCV (`series1d`) | 720,770 | 0.690 | 0.495 |
| HGB on 20 hand features | 14,762-18,300 tree nodes | 0.705 | 0.503 |

Volatility (`vol_h1`) is learnable, but the chart image adds nothing over the raw numbers and
hand features do best. Cross-sectional direction (`xsdir_h12`) has no signal. Zero-shot
foundation models (11M-331M parameters) do worse than the small trained models on `vol_h1`;
Kronos, at 0.699 on a validation subsample, is the closest. Details, intervals and run IDs are in
[CNN-Relevant/README.md](CNN-Relevant/README.md) and
[Pretrained Model Comparison/README.md](Pretrained%20Model%20Comparison/README.md).

## Setup

```bash
python -m pip install -r Pipeline/requirements-pipeline.txt -r DataDownload/requirements-archive.txt
python -m unittest discover -s tests -v
```

Data defaults live under `~/Market Data/`. WRDS scripts take the username from
`WRDS_USERNAME` (else the local login name); credentials are handled by the WRDS client and
are never stored in this repository.
