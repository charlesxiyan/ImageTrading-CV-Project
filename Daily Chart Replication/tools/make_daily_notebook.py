"""Builds Daily_JKX_Replication.ipynb.  Usage: python tools/make_daily_notebook.py <output.ipynb>"""
import json, sys
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Daily chart CNN, Jiang–Kelly–Xiu style: does the image signal live at low frequency?

**Hypothesis.** The chart-image signal of Jiang, Kelly & Xiu (2023, *(Re-)Imag(in)ing Price Trends*) works at the daily
frequency but not at 5 minutes. Our 5-minute experiments find no direction signal (AUC ≈ 0.50 at 5–60 minutes for
image CNNs, 1D CNNs, HGB and frozen DINOv2/CLIP). Here the same stocks and years are used at the **daily** frequency,
with the **original JKX images, network and training rule**. Only the frequency changes.

| | JKX (2023) | this notebook |
|---|---|---|
| data | CRSP daily, all US stocks, 1993–2019 | daily bars built from our 5-minute bars: same ~500 large caps, 2016–2025 |
| images | I5 (32×15), I20 (64×60): OHLC, moving average, volume | same |
| labels | sign of the next 5 / 20-day return | same (`dir`), plus the cross-sectional version (`xsdir`) |
| CNN | 2 / 3 blocks, 64-128-256, 5×3 conv, dropout 0.5, Adam 1e-5, batch 128 | same (I20: 708,866 parameters) |
| training | random 70/30 split of the training years, early stop after 2 bad epochs, 5 runs averaged | same |
| split | train 1993–2000, test 2001–2019 | train 2016–2022, validation 2023, test 2024–2025 (as every other run here) |
| evaluation | accuracy, decile long-short Sharpe | AUC with date-bootstrap CI, accuracy, log-loss gain, decile long-short |

Benchmarks on identical rows: HGB on daily hand features (past returns, volatility, volume, range) and the classic
rules (1-day and 1-week reversal, 1- and 3-month momentum). The last figure puts the daily AUCs next to the 5-minute
AUCs of `run_20261116_044159`, so the frequency question is answered in one picture.

**Leakage controls.**
- Inputs are point in time. Each window is divided by its anchor day's multipliers, so no later split or dividend
  reaches the picture.
- Labels are h-day split- and dividend-adjusted returns. These are total returns, like CRSP `ret` in JKX. Over
  several days, the unadjusted price would turn a split into a return.
- Training labels must end inside the training period (purge).
- Validation and test anchors start h sessions after the boundary (embargo).
- Evaluation uses non-overlapping labels: one anchor every h days, on the same days for every stock.

**Known difference from JKX.** Our universe is large caps. JKX find the strongest effect in small stocks. A null
result here therefore says "not in large caps at daily frequency either". A clear signal here, against a null at 5
minutes, supports the low-frequency hypothesis.

**Advisor rule.** The CNN prints its parameter count; HGB reports its tree nodes.
""")

md("## 1. Setup\nThe data bundle lives in `AllData & AllModel`; this notebook's code and runs live in this folder.")
code("""
import os, sys, json, shutil, tempfile, zipfile, hashlib, time
from pathlib import Path

LOCAL_TEST = os.environ.get("DAILY_LOCAL_NOTEBOOK_TEST") == "1"        # repository QA only
if LOCAL_TEST:
    HERE, DATA = Path(os.environ["DAILY_LOCAL_HERE"]), Path(os.environ["DAILY_LOCAL_DATA"])
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DEMO = Path("/content/drive/MyDrive/Colab Notebooks/EC/Project/Demo")
    HERE, DATA = DEMO / "Daily Chart Replication", DEMO / "AllData & AllModel"
BUNDLE = DATA / "alldata_bundle_v1.zip"
assert BUNDLE.is_file(), f"alldata_bundle_v1.zip not found in {DATA}"
assert (HERE / "daily_jkx.py").is_file(), f"Put daily_jkx.py in {HERE}"

LOCAL = Path(tempfile.mkdtemp(prefix="daily_", dir=None if LOCAL_TEST else "/content"))
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

sys.path[:0] = [str(HERE)]
import numpy as np, pandas as pd, torch
from IPython.display import display, Markdown
import daily_jkx as dj
print("daily_jkx", dj.__version__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("data:", PREPARED, "| device:", DEVICE, torch.cuda.get_device_name() if DEVICE.type == "cuda" else "")
""")

md("""
## 2. Configuration
Change settings here only. **To resume after a disconnect**, set `RUN_ID` to the id printed by the first run.

`QUICK = True` is a first pass: one CNN instead of five, at most 60,000 training images, `I20/R5` only. It takes
about 10–15 minutes.

The full run trains 5 CNNs on each of 6 cells: {I5/R5, I20/R5, I20/R20} × {`dir`, `xsdir`}. A rough estimate is
1–2 hours on an L4; the images are small and kept on the GPU.

`RUN_TEST`: the design is fixed in advance (JKX hyper-parameters, nothing tuned on 2023), so the test years can be
scored in the same run. A daily strategy needs years to judge, and 2023 alone has only 50 weekly or 12 monthly
rebalances. Set it to `False` to keep 2024–25 untouched.
""")
code("""
from datetime import datetime, timezone
SMOKE = LOCAL_TEST
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"

RUN_ID = None                                       # None = new run; or an old RUN_ID to resume it
QUICK = False
RUN_TEST = True
REFERENCE_RUN = "run_20261116_044159"               # 5-minute run in AllData & AllModel/runs
CONFIGS = [(5, 5), (20, 5), (20, 20)]               # (image days, label horizon days): I5/R5, I20/R5, I20/R20
TARGETS = ["dir", "xsdir"]
SETTINGS = dict(
    train_stride=5,                                 # one anchor every 5 sessions (same days for every stock)
    min_bars=60,                                    # a session needs its 16:00 bar and >= 60 of 78 bars
    seeds=5, learning_rate=1e-5, batch_size=128, max_epochs=30, patience=2, min_delta=0.0,
    dropout=0.5, holdout_share=0.3, max_train=None, seed=69,
)
if QUICK:
    SETTINGS.update(seeds=1, max_train=60_000)
    CONFIGS = [(20, 5)]
if SMOKE:
    SETTINGS.update(seeds=2, max_epochs=2, batch_size=64)
CELLS = [dict(name=f"{t}_I{L}R{h}", image_days=L, horizon=h, target=t) for L, h in CONFIGS for t in TARGETS]

RESUME = RUN_ID is not None
RUN_ID = RUN_ID or ("local_smoke" if LOCAL_TEST else datetime.now(timezone.utc).strftime("daily_%Y%m%d_%H%M%S"))
RUN_DIR = HERE / "runs" / RUN_ID
if RESUME:
    assert RUN_DIR.exists(), f"{RUN_DIR} not found"
else:
    assert LOCAL_TEST or not RUN_DIR.exists(), "RUN_ID exists: set RUN_ID to resume it"
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / ("settings_resume.json" if RESUME else "settings.json")).write_text(json.dumps(dict(
    settings=SETTINGS, cells=CELLS, quick=QUICK, run_test=RUN_TEST, reference=REFERENCE_RUN,
    daily_jkx=dj.__version__), indent=2))
REF_DIR = DATA / "runs" / REFERENCE_RUN
print("RUN_ID:", RUN_ID, "->", RUN_DIR)
print("5-minute reference run found:", REF_DIR.is_dir())
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS))
""")

md("""
## 3. Daily bars from the 5-minute bars
Checks: sessions per year, share of stock-days present, and one stock's daily close next to its 16:00 5-minute close.
""")
code("""
t0 = time.perf_counter()
store = dj.DailyStore(PREPARED, min_bars=SETTINGS["min_bars"])
cfg = store.config
print(f"{store.S} stocks x {store.D} sessions ({store.dates[0]} – {store.dates[-1]}) in {time.perf_counter() - t0:.0f}s")
years = pd.Series(store.dates).str[:4]
print("stock-days present by year:", (pd.Series(store.valid.mean(0)).groupby(years.values).mean().round(3)).to_dict())
print("splits: train ≤", cfg["train_end"], "| validation ≤", cfg["validation_end"], "| test after")
s0 = int(np.argmax(store.valid.sum(1)))
feats = np.load(PREPARED / "features.npy", mmap_mode="r")
last_bar_close = np.asarray(feats[s0, dj.SLOTS - 1::dj.SLOTS, 3], float)
ok = store.valid[s0]
assert np.allclose(store.ohlcv[s0, ok, 3], last_bar_close[ok]), "daily close is not the 16:00 bar's close"
print(f"stock {s0}: daily close equals the 16:00 bar close on all {ok.sum()} sessions")
""")

md("""
## 4. Anchors and images
One anchor every 5 sessions for every stock; each image length is rendered once and shared by every horizon and
target. The images stay on the GPU (uint8).
""")
code("""
import matplotlib.pyplot as plt
FRAMES, IMAGES = {}, {}
for L in sorted({L for L, _ in CONFIGS}):
    t0 = time.perf_counter()
    FRAMES[L] = dj.anchors(store, L, SETTINGS["train_stride"])
    imgs = dj.render_all(store, FRAMES[L], L)
    IMAGES[L] = torch.from_numpy(imgs).to(DEVICE)
    print(f"I{L}: {len(FRAMES[L]):,} images {imgs.shape[1:]} in {time.perf_counter() - t0:.0f}s "
          f"({imgs.nbytes / 1e9:.2f} GB)")
    ex = FRAMES[L].sample(4, random_state=1)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, (i, r) in zip(axes, ex.iterrows()):
        ax.imshow(imgs[i], cmap="gray", aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"I{L}  stock {r.security_index}  {r.date}", fontsize=8)
    fig.tight_layout(); fig.savefig(RUN_DIR / f"examples_I{L}.png", dpi=110); plt.show()
""")

md("""
## 5. Train (CNN × seeds, HGB, rules) per cell
Each cell prints its sample sizes, the CNN's parameter count and every epoch. Finished cells are saved to
`runs/<RUN_ID>/<cell>/` and skipped on a rerun.
""")
code("""
RESULTS, t0 = {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    RESULTS[cell["name"]] = dj.run_cell(store, cell, FRAMES[cell["image_days"]], IMAGES[cell["image_days"]],
                                        SETTINGS, DEVICE, RUN_DIR / cell["name"])
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
if not RUN_TEST:
    for r in RESULTS.values():
        r["predictions"] = r["predictions"][r["predictions"].split != "test"]
""")

md("""
## 6. Results
AUC with 95% intervals from resampling whole rebalance dates. Accuracy and the log-loss gain over the training prior
are reported for the CNN and HGB. `ls_*` is the equal-weighted top-minus-bottom decile return over the next h days:
mean in basis points per period, t-statistic, and annualised Sharpe ratio.
""")
code("""
TABLE = dj.evaluate(RESULTS, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_DIR / "daily_table.csv", index=False)
cols = ["image", "horizon_days", "model", "auc", "auc_low", "auc_high", "accuracy", "log_loss_gain_pct",
        "ls_mean_bps", "ls_t", "ls_sharpe_ann", "dates", "rows"]
for split in ("validation", "test"):
    for tgt in TARGETS:
        t = TABLE[(TABLE.split == split) & (TABLE.target == tgt)]
        if len(t):
            print(f"{split} — {tgt}: {dj.TARGETS[tgt]}")
            display(t[[c for c in cols if c in t]].round(4))
""")

md("""
## 7. Daily vs 5-minute
Left of the grey line are the 5-minute AUCs (validation 2023) of `run_20261116_044159`. Right of it are the daily
AUCs from this run. Filled markers are 5-minute models, open markers daily models.
""")
code("""
INTRADAY = dj.intraday_reference(REF_DIR, kinds=TARGETS, n_boot=50 if SMOKE else 300)
if len(INTRADAY):
    INTRADAY.to_csv(RUN_DIR / "intraday_reference.csv", index=False)
for split in ("validation",) + (("test",) if RUN_TEST else ()):
    dj.frequency_plot(TABLE, INTRADAY, split, RUN_DIR / f"frequency_{split}.png")
TAKEAWAYS = dj.takeaways(TABLE)
(RUN_DIR / "takeaways.md").write_text(TAKEAWAYS + "\\n")
display(Markdown(TAKEAWAYS))
print("saved to", RUN_DIR)
""")

md("""
### How to read it
- **Daily CNN AUC clearly above 0.5, with a positive long-short, against 0.50 at 5 minutes**: supports the hypothesis.
  The image signal is a low-frequency phenomenon.
- **Daily ≈ 0.50 as well**: no image signal in these large caps at either frequency. The next step would be the CRSP
  all-stock version, where JKX find most of the effect, in small stocks.
- **Daily CNN ≈ HGB or the reversal / momentum rules**: the picture recovers known price patterns rather than
  something new. JKX report that their CNN beats these rules.
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}}
for i, c in enumerate(cells):
    c["id"] = f"c{i:02d}"
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", sys.argv[1])
