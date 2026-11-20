"""Builds IndicatorChart_CNN.ipynb.  Usage: python tools/make_indicator_notebook.py <output.ipynb>"""
import json, sys
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Charts with technical indicators: MA line and EW standard deviation

**Question.** The CNN trained from scratch loses to HGB on volatility (validation AUC 0.64–0.66 vs 0.67–0.69). One
reason is visible in the chart itself: prices are scaled to the window's own low and high, so a quiet stock and a wild
stock draw almost the same picture. The **level** of volatility is not in the image, while HGB gets it as a number.
Does the CNN improve when the picture carries more of the past?

| variant | picture | what it answers |
|---|---|---|
| `plain` | original chart (control, retrained here) | baseline on the same rows and canvas |
| `ma` | + 78-bar moving average of the close (uses the previous session), as in Jiang, Kelly & Xiu | does trend context help? |
| `ewsd` | + band of EW standard deviation of 5-min returns, on **one fixed log scale** for all stocks | does an absolute volatility level help? |
| `ma_ewsd` | both | together |
| `plain_tab` | original chart, plus the 38 hand + SPY numbers HGB uses as a second input | is the CNN short of information or of skill? |
| `ma_ewsd_tab` | both indicators plus the numbers | everything |

Every variant uses the **same CNN** as `run_20261116_044159` (image CNN, 3 blocks 64/128/256), the same settings
(Adam, learning rate 1e-5, batch 128, up to 8 epochs, early stopping on every 4th validation day) and the **same rows**.
All pictures have the same size (234×234 chart + 78-row band, black when EWSD is off), so the CNN part has the same
parameter count in every variant. The scores are joined row by row with the reference run's HGB and CNN.

**Leakage controls.**
- Labels are purged at the split boundaries (`chartcnn.build_split`).
- An input now covers two sessions (previous session + window), so **2 whole sessions** are embargoed on each side
  of every boundary. Checks below fail if any validation input reaches into the training period.
- **Fit on train only**: the vol threshold, the EWSD band's log scale and the standardisation of the hand/SPY
  numbers.
- Every price comes from the point-in-time window: all bars are on the anchor bar's basis, so no later split or
  dividend reaches the picture.

**Advisor rule.** Each variant prints its parameter count (CNN + feature MLP).

**Resumable.** Finished variants and cells are saved on Drive and skipped. After a disconnect, set `RUN_ID` and
Run all. The 2024–25 test period is not touched here.
""")

md("## 1. Setup\nCopies the data bundle (1.2 GB) from Drive to the Colab disk and unzips it; a few minutes.")
code("""
import os, sys, json, shutil, tempfile, zipfile, hashlib, time
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
assert (HERE / "indicator_cnn.py").is_file(), f"Put indicator_cnn.py in {HERE}"

LOCAL = Path(tempfile.mkdtemp(prefix="indicator_", dir=None if LOCAL_TEST else "/content"))
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

CODE = HERE if (HERE / "chartcnn").is_dir() else STAGE
sys.path[:0] = [str(HERE), str(CODE)]
import numpy as np, pandas as pd, torch
from IPython.display import display, Markdown
import chartcnn
import indicator_cnn as ic
assert chartcnn.__version__ >= "2026-11-16.1", f"stale chartcnn {chartcnn.__version__} in {CODE}"
print("indicator_cnn", ic.__version__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("data:", PREPARED, "| code:", CODE, "| device:", DEVICE, torch.cuda.get_device_name() if DEVICE.type == "cuda" else "")
""")

md("""
## 2. Configuration
Change settings here only. **To resume after a disconnect**, set `RUN_ID` to the id printed by the first run.

**Start with `QUICK = True`** (40,000 training windows, 3 epochs, every 5th validation day, 5-minute vol only, all six
variants). It checks that everything runs and prints how many images per second your runtime trains, so you can
judge the full run. Its AUCs are rough.

The full run trains six CNNs per cell on 300,000 windows each, the same budget as the reference CNN. The default
cells are vol at 5 and 60 minutes (12 CNNs). Add `3, 6` to `HORIZONS` later with the same `RUN_ID`, and only the new
cells are trained. Direction targets are off: every model so far is at AUC 0.50 there.
""")
code("""
from datetime import datetime, timezone
SMOKE = LOCAL_TEST                                 # tiny code check; never a result
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"

RUN_ID = None                                      # None = new run; or an old RUN_ID to resume / extend it
QUICK = True
REFERENCE_RUN = "run_20261116_044159"
VARIANTS = ["plain", "ma", "ewsd", "ma_ewsd", "plain_tab", "ma_ewsd_tab"]
HORIZONS = [1, 12]                                 # bars of 5 minutes; add 3, 6 for 15 and 30 minutes
KINDS = ["vol"]                                    # "xsdir", "dir" possible, but no model has found signal there
SETTINGS = dict(
    stride=12, embargo_sessions=2,                 # an input spans 2 sessions -> embargo 2 sessions per side
    max_train_windows=300_000,                     # same training budget as the reference CNN (max_train_nn)
    validation_every_kth_day=1, early_stop_every_kth_day=4,
    halflife=39,                                   # EWSD half-life in bars (half a session)
    # reference CNN settings (run_20261116_044159)
    learning_rate=1e-5, weight_decay=0.0, batch_size=128, eval_batch_size=1024, epochs=8, patience=2,
    min_delta=1e-5, channels=[64, 128, 256], head_height=4, dropout=0.5, amp=True,
    tab_hidden=64, tab_learning_rate=1e-3,         # small MLP on the 38 hand/SPY numbers (tab variants only)
    num_workers=0 if SMOKE else min(12, os.cpu_count() or 2), seed=69,
)
if QUICK:
    SETTINGS.update(max_train_windows=40_000, validation_every_kth_day=5, epochs=3)
    HORIZONS = [1]
if SMOKE:
    SETTINGS.update(max_train_windows=400, batch_size=32, eval_batch_size=64, epochs=2, embargo_sessions=2,
                    early_stop_every_kth_day=1)
    HORIZONS, KINDS = [12, 1], ["vol", "xsdir"]
CELLS = [dict(name=f"{k}_h{h}", kind=k, target=ic.KINDS[k], horizon=h, minutes=5 * h, stride=SETTINGS["stride"])
         for h in HORIZONS for k in KINDS]

RESUME = RUN_ID is not None
RUN_ID = RUN_ID or ("local_smoke" if LOCAL_TEST else datetime.now(timezone.utc).strftime("indicator_%Y%m%d_%H%M%S"))
RUN_DRIVE, RUN_LOCAL = HERE / "indicator_runs" / RUN_ID, LOCAL / "run"
if RESUME:
    assert RUN_DRIVE.exists(), f"{RUN_DRIVE} not found"
else:
    assert LOCAL_TEST or not RUN_DRIVE.exists(), "RUN_ID exists: set RUN_ID to resume it"
RUN_DRIVE.mkdir(parents=True, exist_ok=True); RUN_LOCAL.mkdir(parents=True, exist_ok=True)
(RUN_DRIVE / ("settings_resume.json" if RESUME else "settings.json")).write_text(json.dumps(dict(
    settings=SETTINGS, variants=VARIANTS, cells=CELLS, quick=QUICK, reference=REFERENCE_RUN,
    chartcnn=chartcnn.__version__, indicator_cnn=ic.__version__), indent=2))
REF_DIR = HERE / "runs" / REFERENCE_RUN
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("reference run found:", REF_DIR.is_dir(), REF_DIR)
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS), "|", len(VARIANTS), "variants:", ", ".join(VARIANTS))
""")

md("""
## 3. Windows, purge and embargo
One fixed set of windows per split, shared by every cell and variant (saved on Drive so a resume uses the same rows).
A window is kept only if its previous session is complete too, since MA and EWSD start there.
""")
code("""
store = chartcnn.Store(PREPARED)
cfg = store.config
EMBARGO = ic.embargo_dates(store, SETTINGS["embargo_sessions"])
days = ic.sessions(store)
for b in (cfg["train_end"], cfg["validation_end"]):
    k = np.searchsorted(days, b, side="right")
    print(f"boundary after {b}: embargoed sessions", [d for d in days[max(0, k - 3):k + 3] if d in EMBARGO])

W = {}
for split, kw in (("train", dict(max_rows=SETTINGS["max_train_windows"])),
                  ("valid", dict(every_kth_day=SETTINGS["validation_every_kth_day"]))):
    path = RUN_DRIVE / f"windows_{split}.parquet"
    if path.exists():
        W[split] = pd.read_parquet(path)
    else:
        W[split] = ic.decision_windows(store, "train" if split == "train" else "validation", SETTINGS["stride"],
                                       EMBARGO, seed=SETTINGS["seed"], **kw)
        W[split].to_parquet(path, index=False)

# Leakage checks: the first bar any validation input uses (previous session included) is after the training period,
# and the last bar any training label uses is inside it.
first_valid_input = store.dates[W["valid"].end_index.min() - ic.LOOKBACK - ic.HISTORY + 1]
last_train_label = store.dates[W["train"].end_index.max() + max(HORIZONS)]
assert first_valid_input > cfg["train_end"], "a validation input reaches into the training period"
assert last_train_label <= cfg["train_end"], "a training label ends after the training period"
print(f"train windows {len(W['train']):,} ({W['train'].decision_date.min()} – {W['train'].decision_date.max()}), "
      f"last training label bar on {last_train_label}")
print(f"validation windows {len(W['valid']):,} ({W['valid'].decision_date.min()} – {W['valid'].decision_date.max()}, "
      f"{W['valid'].decision_date.nunique()} days), first validation input bar on {first_valid_input}")
""")

md("""
## 4. Preprocessing fitted on train only, and what the CNN sees
The EWSD band's scale is the 0.5%–99.5% range of log EWSD over training windows; the 38 hand/SPY numbers are
standardised with training means and standard deviations. The figure shows a quiet and a volatile training window in
every variant: in the original chart they look alike, in the EWSD band they do not.
""")
code("""
import matplotlib.pyplot as plt
SCALE_PATH = RUN_DRIVE / "preprocessing.json"
if SCALE_PATH.exists():
    PRE = json.loads(SCALE_PATH.read_text())
    SCALE = tuple(PRE["ewsd_log_scale"])
else:
    SCALE = ic.fit_ewsd_scale(PREPARED, W["train"], SETTINGS["halflife"], n=500 if SMOKE else 20_000, seed=SETTINGS["seed"])
TAB_RAW = {s: ic.tab_features(store, W[s]) for s in W}
TAB_SCALER = ic.TabScaler(TAB_RAW["train"])                      # train rows only
TAB = {s: TAB_SCALER(TAB_RAW[s]) for s in W}
if not SCALE_PATH.exists():
    SCALE_PATH.write_text(json.dumps(dict(ewsd_log_scale=SCALE, halflife=SETTINGS["halflife"], tab=TAB_SCALER.state())))
print(f"EWSD band: log scale {SCALE[0]:.2f} .. {SCALE[1]:.2f}  (5-min sigma {np.exp(SCALE[0]):.5f} .. {np.exp(SCALE[1]):.5f})")
print(f"hand/SPY numbers: {TAB['train'].shape[1]} columns:", ", ".join(TAB_SCALER.columns))

# a quiet and a volatile training window (by realized vol of the 78 bars)
rv = TAB_RAW["train"]["realized_vol"].to_numpy()
pick = W["train"].iloc[[int(np.nanargmin(np.abs(rv - np.nanpercentile(rv, q)))) for q in (5, 95)]].reset_index(drop=True)
show = ["plain", "ma", "ewsd", "ma_ewsd"]
fig, axes = plt.subplots(2, len(show), figsize=(3.2 * len(show), 7.2))
for r in range(2):
    for c, v in enumerate(show):
        img = ic.Charts(PREPARED, pick, v, SCALE, SETTINGS["halflife"]).image(r)
        axes[r, c].imshow(img, cmap="gray", aspect="auto"); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, c].set_title(f"{v}\\n{'quiet' if r == 0 else 'volatile'} (PERMNO {pick.permno[r]})", fontsize=9)
fig.tight_layout(); fig.savefig(RUN_LOCAL / "what_the_cnn_sees.png", dpi=110); plt.show()
shutil.copy2(RUN_LOCAL / "what_the_cnn_sees.png", RUN_DRIVE)

# Same picture as the reference run when no indicator is drawn
w0 = ic.Charts(PREPARED, pick, "plain", SCALE, SETTINGS["halflife"]).long_window(0)
assert (ic.draw_chart(w0[-ic.LOOKBACK:]) == chartcnn.images.render_window(w0[-ic.LOOKBACK:])).all()

for v in VARIANTS:
    m = ic.IndicatorCNN(SETTINGS, TAB["train"].shape[1] if ic.VARIANTS[v][2] else 0)
    n = ic.count(m)
    print(f"{v:12s} {n['total']:>9,} parameters (CNN {n['cnn']:,} + hand/SPY MLP {n['tab']:,})")
del m
""")

md("""
## 5. Train every variant (validation 2023)
Per cell, the six CNNs in turn. Each prints its parameter count, the loss per epoch and its validation AUC. A cell
that is already finished on Drive is loaded instead.
""")
code("""
RESULTS, t0 = {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    tr, median = ic.cell_rows(store, cell, "train", W["train"])
    va, _ = ic.cell_rows(store, cell, "validation", W["valid"], median)       # vol threshold from train only
    assert (store.dates[tr.end_index + cell["horizon"]] <= cfg["train_end"]).all()          # purge
    assert (store.dates[va.end_index + cell["horizon"]] <= cfg["validation_end"]).all()
    es_days = np.sort(va.decision_date.unique())[::SETTINGS["early_stop_every_kth_day"]]
    es = va[va.decision_date.isin(es_days)].reset_index(drop=True)
    data = dict(train=tr, es=es, valid=va, tab_train=TAB["train"], tab_valid=TAB["valid"], scale=SCALE,
                prior=float(tr.label.mean()))
    RESULTS[cell["name"]] = ic.run_cell(PREPARED, cell, data, VARIANTS, SETTINGS, DEVICE,
                                        RUN_LOCAL / cell["name"], RUN_DRIVE / cell["name"])
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
""")

md("""
## 6. Compare on identical rows
The reference run's scores (HGB, HGB + SPY, image CNN, 1D CNN) are joined on (stock, decision bar); only rows present
in both are kept. AUC with 95% intervals from resampling whole validation days; *paired* comparisons use the same
resampled days for both models. "better"/"worse" means the 95% interval of the difference excludes zero.
""")
code("""
JOINED = {}
for name, r in RESULTS.items():
    pred, share = ic.join_reference(r["predictions"], REF_DIR / name)
    print(f"{name}: {len(pred):,} rows" + ("  (reference not found)" if share is None
                                          else f", {share:.1%} of this run's rows matched in the reference run"))
    JOINED[name] = dict(info=r["info"], predictions=pred)
TABLE, BOOTS = ic.table(JOINED, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_LOCAL / "indicator_table.csv")

MAIN = [ic.pretty("ind_" + v) for v in VARIANTS] + ["image CNN (ref run)", "HGB hand + SPY (ref)"]
for m in MAIN:
    if m in set(TABLE.index.get_level_values("model")):
        print(m); display(ic.auc_matrix(TABLE, m))
ic.plot(TABLE, MAIN, RUN_LOCAL / "indicator_auc_by_horizon.png")
for a, b, label in ic.CHECKS:
    p = ic.paired(BOOTS, a, b)
    if len(p):
        print(label); display(p.round(4))
TAKEAWAYS = ic.takeaways(TABLE, BOOTS)
(RUN_LOCAL / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
for name in ("indicator_table.csv", "indicator_auc_by_horizon.png", "takeaways.md"):
    shutil.copy2(RUN_LOCAL / name, RUN_DRIVE / name)
print("saved to", RUN_DRIVE)
""")

md("""
### How to read it
- **EWSD > original**: the picture was missing the volatility level; drawing it on a fixed scale helps the CNN.
- **MA ≈ original**: trend context adds nothing for volatility (it may matter for direction, where nothing works yet).
- **`plain_tab` ≫ every chart-only variant**: the CNN is short of information that is easier to give as numbers;
  the fair final model is the hybrid.
- **Any CNN ≥ HGB hand + SPY**: the picture adds something the 38 numbers do not.
- **Control** (`plain` here vs the reference run's image CNN) should be tied; the rows, embargo and canvas differ
  slightly, and the training sample is a fresh draw.
""")

md("## 7. Full results table")
code("""
display(TABLE.drop(columns=["raw"]).round(4))
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}}
for i, c in enumerate(cells):
    c["id"] = f"c{i:02d}"
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", sys.argv[1])
