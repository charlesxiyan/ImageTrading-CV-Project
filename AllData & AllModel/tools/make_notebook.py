"""Builds AllData_AllModel.ipynb.  Usage: python tools/make_notebook.py <output.ipynb>"""
import json, sys
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# AllData & AllModel: 2016–2025 five-minute charts, every model, with and without SPY

**Data.** S&P 500 ∪ NASDAQ 100 ∪ DJIA 30 members by date (775 PERMNOs, 516–531 a day), 2,514 sessions
from 2016-01-04 to 2025-12-31, five-minute bars from TAQ trades. SPY bars are built with the same filters on
the same clock. Prices are adjusted **point in time**: every window is put on the basis of its last bar
(splits, stock dividends and cash dividends before it are adjusted, nothing after it). Labels use traded
(unadjusted) prices; a label whose horizon contains an ex-date is dropped. Early-close sessions are excluded.

**Split.** Train 2016–2022 · validation 2023 (all model choices) · test 2024–2025 (section 8, run once at the end).

**Grid.** `dir` (return > 0), `vol` (|return| > the stock's training median), `xsdir` (return > cross-sectional
median) × 5 / 15 / 30 / 60 minutes, one sample per stock per hour (stride 12) for every horizon.

**Models per cell** (every model prints its parameter count; for HGB the count is tree nodes):

| model | input |
|---|---|
| `plain` image CNN | 234×234 chart of the stock's 78 bars |
| `plain_spy` image CNN + SPY panel | 468×234: the stock chart with SPY's chart in its own panel below, bar columns aligned |
| `series1d` 1D CNN | the 5 OHLCV series, scaled exactly as the chart is |
| `series1d_spy` 1D CNN + SPY | 10 series: the stock's 5 and SPY's 5 |
| HGB hand features | 20 hand features of the stock window |
| HGB hand + SPY | the same plus 18 SPY / stock-vs-SPY features (SPY returns, realized vol, beta, correlation, idiosyncratic vol, relative return) |

**Resumable**: finished cells are saved on Drive and skipped on a rerun; an unfinished CNN resumes from its last epoch.
""")

md("## 1. Setup\nCopies the data bundle (1.2 GB) from Drive to the Colab disk and unzips it (4.5 GB); a few minutes.")
code("""
import os, sys, json, shutil, tempfile, zipfile, hashlib
from pathlib import Path

LOCAL_TEST = os.environ.get("ALLDATA_LOCAL_NOTEBOOK_TEST") == "1"     # repository QA only
if LOCAL_TEST:
    HERE = Path(os.environ["ALLDATA_LOCAL_HERE"])
else:
    from google.colab import drive
    drive.mount("/content/drive")
    HERE = Path("/content/drive/MyDrive/Colab Notebooks/EC/Project/Demo/AllData & AllModel")
BUNDLE = HERE / "alldata_bundle_v1.zip"
assert BUNDLE.is_file(), f"Put alldata_bundle_v1.zip in {HERE}"

LOCAL = Path(tempfile.mkdtemp(prefix="alldata_", dir=None if LOCAL_TEST else "/content"))
STAGE = LOCAL / "bundle"
STAGE.mkdir(parents=True)
shutil.copy2(BUNDLE, LOCAL / "bundle.zip")
with zipfile.ZipFile(LOCAL / "bundle.zip") as z:
    for info in z.infolist():                       # refuse paths escaping the folder
        assert (STAGE / info.filename).resolve().is_relative_to(STAGE.resolve())
    z.extractall(STAGE)
(LOCAL / "bundle.zip").unlink()
def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
for rel, d in json.loads((STAGE / "bundle_manifest.json").read_text())["files"].items():
    assert _sha(STAGE / rel) == d["sha256"], f"checksum mismatch: {rel}"
PREPARED = STAGE / "prepared"

# chartcnn/ next to this notebook on Drive wins (easy to edit); otherwise the copy inside the bundle.
CODE = HERE if (HERE / "chartcnn").is_dir() else STAGE
sys.path.insert(0, str(CODE))
import numpy as np, pandas as pd, torch
from IPython.display import display, Markdown
import chartcnn
from chartcnn import grid
from chartcnn.utils import save_json
assert chartcnn.__version__ >= "2026-11-16.1", f"stale chartcnn {chartcnn.__version__} in {CODE}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("data:", PREPARED, "| code:", CODE, "| device:", DEVICE, torch.cuda.get_device_name() if DEVICE.type == "cuda" else "")
""")

md("""
## 2. Configuration
Change settings here only. **To resume after a disconnect**, set `RUN_ID` to the id printed by the first run and Run all again.

**Data on Colab.** The bundle (`alldata_bundle_v1.zip`, 1.2 GB) is copied from Drive to the Colab disk and unzipped
(4.5 GB). Bars are opened as memory maps and charts are drawn per sample, so no images are stored. The label arrays
take about 6 GB of RAM, which fits the standard runtime; a High-RAM runtime is safer with 8 data-loading workers.
**`QUICK = True`** runs a small first pass (fewer rows, every 5th validation day, horizons 60 and 5 minutes) to check
the whole notebook before the full run; its numbers are rough.

Time budget: the CNNs train on a random `max_train_nn` subset of the training windows (whole years do not fit a Colab
session); HGB uses up to `max_train_hgb`. Early stopping watches every 4th validation day; final validation scores use the
whole of 2023. Rough guess on an L4, not measured on this data: 10–20 min per image CNN and 3–6 min per 1D CNN per cell,
so 6–9 hours for all 12 cells × 4 networks; reruns resume. To get the answers that matter first, the cells run
in the order of `HORIZONS` (the `vol` and `xsdir` cells at 60 and 5 minutes first).
""")
code("""
from datetime import datetime, timezone
SMOKE = LOCAL_TEST                                # tiny code check; never a result
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"

RUN_ID = None                                     # None = new run; or an old RUN_ID to resume it
QUICK = False                                     # True: small first pass (~30-60 min) to check everything runs
SETTINGS = dict(
    learning_rate=1e-5, weight_decay=0.0,         # Adam, as in Jiang-Kelly-Xiu and earlier runs
    batch_size=4 if SMOKE else 128, eval_batch_size=8 if SMOKE else 1024,
    epochs=1 if SMOKE else 8, patience=2, min_delta=1e-5,       # early stop on validation log loss
    channels=[64, 128, 256], head_height=4, dropout=0.5,
    max_train_nn=300_000, max_train_hgb=1_000_000, early_stop_every_kth_day=4,
    amp=True,                                     # bfloat16 autocast on A100/L4/H100; ignored on T4/CPU
    num_workers=0 if SMOKE else min(8, os.cpu_count() or 2),
    seed=69,
)
if QUICK:                                         # fewer rows and days, 2 horizons; numbers are rough
    SETTINGS.update(max_train_nn=40_000, max_train_hgb=200_000, validation_every_kth_day=5, epochs=3)
HORIZONS = [12, 1] if QUICK else [12, 1, 6, 3]    # bars of 5 minutes: 60, 5, 30, 15 minutes (run order)
KINDS = ["vol", "xsdir", "dir"]                   # volatility, relative direction, absolute direction
STRIDE = 12                                       # one sample per stock per hour, for every horizon
ARCHS = ["plain", "plain_spy", "series1d", "series1d_spy"]
CELLS = grid.grid_cells(HORIZONS, KINDS, STRIDE)

RESUME = RUN_ID is not None
RUN_ID = RUN_ID or ("local_smoke" if LOCAL_TEST else datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"))
RUN_DRIVE, RUN_LOCAL = HERE / "runs" / RUN_ID, LOCAL / "run"
if RESUME:
    assert RUN_DRIVE.exists(), f"{RUN_DRIVE} not found"
else:
    assert LOCAL_TEST or not RUN_DRIVE.exists(), "RUN_ID exists: set RUN_ID to resume it"
RUN_DRIVE.mkdir(parents=True, exist_ok=True); RUN_LOCAL.mkdir(parents=True, exist_ok=True)
save_json(RUN_LOCAL / ("settings_resume.json" if RESUME else "settings.json"),
          dict(settings=SETTINGS, cells=CELLS, archs=ARCHS, quick=QUICK, chartcnn=chartcnn.__version__), RUN_DRIVE)
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS))
""")

md("## 3. Data overview\nSessions, securities and windows per split, as built and audited locally (`manifest.json`).")
code("""
store = chartcnn.Store(PREPARED)
M = json.loads((PREPARED / "manifest.json").read_text())
print("stocks:", len(M["permnos"]), "| slots:", store.T, "| sessions:", store.T // 78,
      "| early-close sessions excluded:", len(M["config"].get("early_close_sessions_excluded", [])))
print("SPY in bundle:", store.has_market, "| SPY valid slots:", M["bundle"]["market_valid_slots"])
print("windows in bundle (every 6th bar):", M["bundle"]["windows"])
display(pd.DataFrame(M["splits"]).T[["samples", "decision_start", "decision_end"]])
""")

md("""
## 4. What the models see
A few training samples rendered by the **same dataset code the models use**: the stock chart alone, and the stock chart with
SPY's panel below it. Each panel has its own price axis (lowest low to highest high of its 78 bars) and volume band.
""")
code("""
import matplotlib.pyplot as plt
sample, _ = chartcnn.build_split(store, "train", "vol", 1, STRIDE)
show = sample.sample(4, random_state=1).reset_index(drop=True)
plain, with_spy = chartcnn.make_dataset(PREPARED, show, "plain"), chartcnn.make_dataset(PREPARED, show, "plain_spy")
fig, axes = plt.subplots(2, 4, figsize=(16, 9), gridspec_kw=dict(height_ratios=[1, 2]))
for i in range(4):
    axes[0, i].imshow(plain.images(i), cmap="gray", aspect="auto")
    axes[1, i].imshow(with_spy.images(i), cmap="gray", aspect="auto")
    axes[0, i].set_title(f"PERMNO {show.permno[i]}  {store.dates[show.end_index[i]]}")
    for ax in axes[:, i]: ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel("plain (234x234)"); axes[1, 0].set_ylabel("plain_spy (468x234): stock / SPY")
fig.tight_layout(); fig.savefig(RUN_LOCAL / "what_models_see.png", dpi=110); plt.show()
shutil.copy2(RUN_LOCAL / "what_models_see.png", RUN_DRIVE / "what_models_see.png")
""")

md("## 5. Models\nLayer-by-layer output shapes and parameter counts. Output layers start at zero, so every initial prediction is exactly 0.5.")
code("""
for arch in ARCHS:
    data = chartcnn.make_dataset(PREPARED, show, arch)
    x = torch.stack([data[i][0] for i in range(4)])
    model = chartcnn.build_model(dict(SETTINGS, arch=arch))
    table, total, trainable = chartcnn.parameter_table(model)
    print(f"{arch} ({grid.ARCH_LABEL[arch]}): input {tuple(x.shape[1:])}, {total:,} parameters ({trainable:,} trainable)")
    display(table)
    print("initial P(y=1):", model.eval()(x).softmax(1)[:, 1].tolist(), "\\n")
del model
""")

md("""
## 6. Labels
Sample counts and positive shares per cell (train 2016–2022, validation 2023). `vol` uses each stock's training-period median,
so its validation share drifts with market volatility; AUC is not affected, log loss is.
""")
code("""
OVERVIEW = grid.label_overview(store, CELLS)
display(OVERVIEW.round(3)); OVERVIEW.to_csv(RUN_LOCAL / "labels.csv")
""")

md("""
## 7. Train the grid (validation 2023)
Per cell: labels → rules and both HGBs → the four networks → validation predictions in `runs/<RUN_ID>/<cell>/`.
Watch `val AUC`, the log-loss gain vs prior (> 0 for a useful probability) and `prob std` (≈ 0 means a collapsed CNN).
""")
code("""
import time
RESULTS, t0 = {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    RESULTS[cell["name"]] = grid.run_cell(PREPARED, cell, SETTINGS, ARCHS, DEVICE, RUN_LOCAL / cell["name"],
                                          RUN_DRIVE / cell["name"], smoke=SMOKE)
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
""")

md("""
### Validation analysis
AUC per cell and model with 95% intervals from resampling whole validation days. The takeaways include paired
with-SPY minus without-SPY comparisons for HGB, the image CNN and the 1D CNN.
""")
code("""
TABLE, BOOTS = grid.grid_table(RESULTS, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_LOCAL / "grid_table.csv")
for model in ["CNN (plain)", "CNN (plain_spy)", "CNN (series1d)", "CNN (series1d_spy)", "HGB hand features", "HGB hand + SPY"]:
    if model in TABLE.index.get_level_values("model"):
        print(model); display(grid.auc_matrix(TABLE, model))
display(TABLE.drop(columns=["kind", "horizon"]).round(4))
grid.plot_grid(TABLE, RUN_LOCAL / "auc_by_horizon.png")
TAKEAWAYS = grid.takeaways(TABLE, BOOTS)
(RUN_LOCAL / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
for name in ("labels.csv", "grid_table.csv", "auc_by_horizon.png", "takeaways.md"):
    shutil.copy2(RUN_LOCAL / name, RUN_DRIVE / name)
print("saved to", RUN_DRIVE)
""")

md("""
## 8. Held-out test 2024–2025 (run once, at the end)
Only after every choice above is final. Each network is scored from its best validation checkpoint; HGB is refit on the
same training rows. Set `RUN_TEST = True` and run this cell. Do not change anything after looking at it.
""")
code("""
RUN_TEST = SMOKE                                  # set True once, at the very end
if RUN_TEST:
    TEST = {}
    for cell in CELLS:
        folder = (RUN_LOCAL if SMOKE else RUN_DRIVE) / cell["name"]
        TEST[cell["name"]] = grid.evaluate_cell_test(PREPARED, cell, SETTINGS, ARCHS, DEVICE, folder, smoke=SMOKE)
    TEST_TABLE, TEST_BOOTS = grid.grid_table(TEST, n_boot=50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_LOCAL / "test_grid_table.csv")
    display(TEST_TABLE.drop(columns=["kind", "horizon"]).round(4))
    display(Markdown(grid.takeaways(TEST_TABLE, TEST_BOOTS)))
    shutil.copy2(RUN_LOCAL / "test_grid_table.csv", RUN_DRIVE / "test_grid_table.csv")
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}}
for i, c in enumerate(cells):
    c["id"] = f"c{i:02d}"
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", sys.argv[1])
