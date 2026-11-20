"""Builds FrozenEncoder_Probe.ipynb.  Usage: python tools/make_probe_notebook.py <output.ipynb>"""
import json, sys
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Frozen pretrained encoders + probes vs CNNs trained from scratch

**Question.** Do general-purpose visual features carry information about the next 5–60 minutes that the image CNN
trained from scratch does not find? The encoder is **frozen** (never trained); only a small head (the *probe*) is fit on
its features. This is the cheap version of the scratch-versus-pretrained comparison.

**Same data as `AllData_AllModel.ipynb`** (`run_20261116_044159`): 2016–2025 five-minute charts of the S&P 500 ∪ NASDAQ 100
∪ DJIA members, rendered on the fly from point-in-time windows (234×234, exactly the image the CNN saw), split train
2016–2022 · validation 2023 · test 2024–2025 (section 9, run once at the end). Targets `vol`, `xsdir`, `dir` at 5 / 15 / 30 /
60 minutes, one window per stock per hour (stride 12).

| group | model | what it answers |
|---|---|---|
| probes | **DINOv2 ViT-S/14 frozen + linear probe** (logistic regression) | the core experiment |
| | DINOv2 frozen + 2-layer MLP probe | does a more expressive head help? |
| | CLIP ViT-B/16 frozen + the same probes | is it specific to DINO? |
| | each probe also with SPY's chart features appended | same SPY question as before |
| floor | **random ViT-S/14** (DINOv2 architecture, untrained, frozen) + the same probes | does *pretraining* matter, or just a big random projection? |
| | training prior (majority) | AUC 0.5 |
| reference | image CNN, image CNN + SPY, 1D CNN, HGB, HGB + SPY | **loaded** from `runs/run_20261116_044159`, not retrained |

Every model is scored on **identical validation rows**: the probes' predictions are joined row by row with the saved
predictions of the reference run.

**Leakage controls** (Boyang's points):
- **Purge.** A label is kept only if its whole horizon ends inside its own split (`chartcnn.build_split`); labels are
  intraday, so no label crosses a split boundary.
- **Embargo.** An input window is 78 bars, exactly one session, so windows of the first session after a boundary still
  contain bars of the session before it. `embargo_sessions` whole sessions are removed on **both sides** of every boundary
  (train | validation and validation | test), so no input or label overlaps across splits, with a gap to spare.
- **Preprocessing fit on train only.** The `vol` threshold (each stock's median |return|) and the probes' feature
  standardisation (mean, std) are computed from training rows only and applied unchanged to validation and test.
  Encoders are frozen with their published statistics. Probe hyper-parameters (L2 strength, weight decay, epochs) are
  chosen on validation days, as the CNNs' early stopping was; the test period is never used.

**Advisor rule.** Every model prints its parameter count: the encoder's (all frozen, 0 trainable) and the probe's
(trainable) separately.

**Resumable**: features are saved to Drive in chunks and finished cells are skipped; after a disconnect set `RUN_ID`
and Run all.
""")

md("## 1. Setup\nCopies the data bundle (1.2 GB) from Drive to the Colab disk and unzips it (4.5 GB); a few minutes.")
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
assert (HERE / "frozen_probe.py").is_file(), f"Put frozen_probe.py in {HERE}"

LOCAL = Path(tempfile.mkdtemp(prefix="probe_", dir=None if LOCAL_TEST else "/content"))
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
sys.path[:0] = [str(HERE), str(CODE)]
import numpy as np, pandas as pd, torch
from IPython.display import display, Markdown
import chartcnn
import frozen_probe as fp
assert chartcnn.__version__ >= "2026-11-16.1", f"stale chartcnn {chartcnn.__version__} in {CODE}"
print("frozen_probe", fp.__version__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("data:", PREPARED, "| code:", CODE, "| device:", DEVICE, torch.cuda.get_device_name() if DEVICE.type == "cuda" else "")
""")

md("""
## 2. Configuration
Change settings here only. **To resume after a disconnect**, set `RUN_ID` to the id printed by the first run and Run all again.

`QUICK = True` is a small first pass (40,000 training windows, every 5th validation day, 60 and 5 minutes, DINOv2 and
the random encoder only), about 20–30 minutes on an L4, to check everything runs; its numbers are rough.

**Time on an L4 (estimate, not measured on this data).** The full run renders ~300,000 training and ~570,000 validation
charts once and feeds each to every encoder: roughly 20–40 minutes for DINOv2-S + random ViT-S, plus 30–60 minutes for
CLIP ViT-B/16. Chart drawing runs on the CPU, so more CPU cores (High-RAM runtime) help. The probes then take 1–3 minutes
per cell, about 30 minutes for all 12 cells. Features are cached on the Colab disk (float16, about 5 GB for the full
run); set `FEATURES_ON_DRIVE = True` to keep them on Drive too, so a resume or a new probe does not redo the encoders.
""")
code("""
from datetime import datetime, timezone
SMOKE = LOCAL_TEST                                 # tiny code check with untrained tiny encoders; never a result
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime"

RUN_ID = None                                      # None = new run; or an old RUN_ID to resume it
QUICK = False
REFERENCE_RUN = "run_20261116_044159"              # AllData & AllModel run whose CNN/HGB scores are compared
ENCODER_NAMES = ["dinov2_s", "dinov2_s_random", "clip_b16"]   # add "dinov2_b" for DINOv2 ViT-B/14
PROBES = ["linear", "mlp"]
FEATURES_ON_DRIVE = False                          # True: also copy the feature cache (~5 GB) to Drive for resuming
SETTINGS = dict(
    stride=12, embargo_sessions=2,                 # sessions removed on each side of each split boundary
    max_train_windows=300_000,                     # same training budget as the CNNs (max_train_nn)
    validation_every_kth_day=1,                    # 1 = all of 2023
    early_stop_every_kth_day=4,                    # probe selection on every 4th validation day, as for the CNNs
    extract_batch=256, chunk=50_000, amp=True,     # bfloat16 encoders on L4/A100
    num_workers=0 if SMOKE else min(12, os.cpu_count() or 2),
    probe_lr=1e-3, probe_batch=4096, probe_epochs=30, probe_patience=3,
    linear_l2_grid=[1e-5, 1e-4, 1e-3],             # logistic regression ridge penalty, picked on validation days
    mlp_hidden=512, mlp_dropout=0.2, mlp_weight_decay_grid=[1e-2, 1e-1],
    seed=69,
)
HORIZONS = [12, 1, 6, 3]                           # bars of 5 minutes: 60, 5, 30, 15 minutes (run order)
KINDS = ["vol", "xsdir", "dir"]
if QUICK:
    SETTINGS.update(max_train_windows=40_000, validation_every_kth_day=5)
    HORIZONS, ENCODER_NAMES = [12, 1], ["dinov2_s", "dinov2_s_random"]
if SMOKE:
    SETTINGS.update(max_train_windows=300, extract_batch=16, chunk=128, probe_epochs=3, probe_batch=64,
                    embargo_sessions=1, early_stop_every_kth_day=1, linear_l2_grid=[1e-4], mlp_weight_decay_grid=[1e-2])
    HORIZONS = [12, 1]
CELLS = [dict(name=f"{k}_h{h}", kind=k, target=fp.KINDS[k], horizon=h, minutes=5 * h, stride=SETTINGS["stride"])
         for h in HORIZONS for k in KINDS]

RESUME = RUN_ID is not None
RUN_ID = RUN_ID or ("local_smoke" if LOCAL_TEST else datetime.now(timezone.utc).strftime("probe_%Y%m%d_%H%M%S"))
RUN_DRIVE, RUN_LOCAL = HERE / "probe_runs" / RUN_ID, LOCAL / "run"
if RESUME:
    assert RUN_DRIVE.exists(), f"{RUN_DRIVE} not found"
else:
    assert LOCAL_TEST or not RUN_DRIVE.exists(), "RUN_ID exists: set RUN_ID to resume it"
RUN_DRIVE.mkdir(parents=True, exist_ok=True); RUN_LOCAL.mkdir(parents=True, exist_ok=True)
(RUN_DRIVE / ("settings_resume.json" if RESUME else "settings.json")).write_text(json.dumps(dict(
    settings=SETTINGS, encoders=ENCODER_NAMES, probes=PROBES, cells=CELLS, quick=QUICK, reference=REFERENCE_RUN,
    chartcnn=chartcnn.__version__, frozen_probe=fp.__version__), indent=2))
REF_DIR = HERE / "runs" / REFERENCE_RUN
print("RUN_ID:", RUN_ID, "->", RUN_DRIVE)
print("reference run found:", REF_DIR.is_dir(), REF_DIR)
print(len(CELLS), "cells:", ", ".join(c["name"] for c in CELLS))
""")

md("""
## 3. Windows, purge and embargo
One fixed set of windows per split, shared by every cell, encoder and probe (saved on Drive so a resume uses the same rows).
The checks below fail loudly if any validation input reaches back into the training period or any training label ends
after it.
""")
code("""
store = chartcnn.Store(PREPARED)
cfg = store.config
EMBARGO = fp.embargo_dates(store, SETTINGS["embargo_sessions"])
days = fp.sessions(store)
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
        W[split] = fp.decision_windows(store, "train" if split == "train" else "validation", SETTINGS["stride"],
                                       EMBARGO, seed=SETTINGS["seed"], **kw)
        W[split].to_parquet(path, index=False)

# Leakage checks (inputs and labels never cross a split boundary)
first_valid_input = store.dates[W["valid"].end_index.min() - fp.LOOKBACK + 1]
last_train_label = store.dates[W["train"].end_index.max() + max(HORIZONS)]
assert first_valid_input > cfg["train_end"], "a validation input window reaches into the training period"
assert last_train_label <= cfg["train_end"], "a training label ends after the training period"
print(f"train windows {len(W['train']):,} ({W['train'].decision_date.min()} – {W['train'].decision_date.max()}), "
      f"last training label bar on {last_train_label}")
print(f"validation windows {len(W['valid']):,} ({W['valid'].decision_date.min()} – {W['valid'].decision_date.max()}, "
      f"{W['valid'].decision_date.nunique()} days), first validation input bar on {first_valid_input}")

# SPY's chart depends only on the time slot: encode it once per slot.
SLOTS = np.unique(np.concatenate([W["train"].end_index, W["valid"].end_index]))
SLOT_POS = np.full(store.T, -1, np.int64); SLOT_POS[SLOTS] = np.arange(len(SLOTS))
print(f"distinct SPY slots: {len(SLOTS):,}")
""")

md("""
## 4. Encoders and what they see
Each encoder prints its parameter count (all frozen). The chart is resized from 234×234 to 224×224, copied to three
channels and normalised with the encoder's own pretraining statistics; the feature is the CLS token next to the mean of
the patch tokens.
""")
code("""
import matplotlib.pyplot as plt
ENCODERS = fp.build_encoders(ENCODER_NAMES, DEVICE, tiny=SMOKE, seed=SETTINGS["seed"])
ENCODER_PARAMS = {e: enc.parameters_count()[0] for e, enc in ENCODERS.items()}
example = W["train"].sample(3, random_state=1).reset_index(drop=True)
charts = fp.StockCharts(PREPARED, example)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i in range(3):
    img = charts[i]
    axes[0, i].imshow(img, cmap="gray"); axes[0, i].set_title(f"PERMNO {example.permno[i]} {example.decision_date[i]}")
    small = torch.nn.functional.interpolate(img[None, None].float(), size=(224, 224), mode="bilinear", antialias=True)[0, 0]
    axes[1, i].imshow(small, cmap="gray"); axes[1, i].set_title("encoder input (224x224)")
    for ax in axes[:, i]: ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); plt.show()
""")

md("""
## 5. Extract features (once, cached on Drive)
Each chart is drawn once and passed through every encoder. Saved per chunk on the Colab disk; with
`FEATURES_ON_DRIVE = True` also to `probe_runs/<RUN_ID>/features/` on Drive (~5 GB), so a rerun after a disconnect skips
finished chunks. With the default (`False`) a disconnect means the features are extracted again; results, predictions
and probe weights are always saved to Drive.
""")
code("""
FEAT_LOCAL, FEAT_DRIVE = LOCAL / "features", RUN_DRIVE / "features"
FEATS = {e: {} for e in ENCODER_NAMES}
kw = dict(batch_size=SETTINGS["extract_batch"], workers=SETTINGS["num_workers"], chunk=SETTINGS["chunk"],
          amp=SETTINGS["amp"], drive_dir=FEAT_DRIVE if FEATURES_ON_DRIVE else None)
t0 = time.perf_counter()
for split in ("train", "valid"):
    out = fp.extract(fp.StockCharts(PREPARED, W[split]), ENCODERS, FEAT_LOCAL, split, DEVICE, **kw)
    for e in out: FEATS[e][split] = out[e]
SPY = fp.extract(fp.SpyCharts(PREPARED, SLOTS), ENCODERS, FEAT_LOCAL, "spy", DEVICE, **kw)
print(f"features ready in {(time.perf_counter() - t0) / 60:.1f} min:",
      {e: {s: FEATS[e][s].shape for s in FEATS[e]} for e in FEATS}, "SPY", {e: SPY[e].shape for e in SPY})
del ENCODERS
if DEVICE.type == "cuda": torch.cuda.empty_cache()
""")

md("""
## 6. Fit the probes (validation 2023)
Per cell and encoder: linear and MLP probe on the stock's features, then the same with SPY's features appended. Each line
prints the probe's trainable parameters, the chosen regularisation and the validation AUC.
""")
code("""
RESULTS, t0 = {}, time.perf_counter()
for i, cell in enumerate(CELLS, 1):
    print("\\n" + "=" * 20, f"[{i}/{len(CELLS)}]", cell["name"], "=" * 20)
    tr, median = fp.cell_rows(store, cell, "train", W["train"])
    va, _ = fp.cell_rows(store, cell, "validation", W["valid"], median)       # vol threshold from train only
    assert (store.dates[tr.end_index + cell["horizon"]] <= cfg["train_end"]).all()          # purge
    assert (store.dates[va.end_index + cell["horizon"]] <= cfg["validation_end"]).all()
    data = dict(train=tr, valid=va, feats=FEATS, spy=SPY, slot_pos=SLOT_POS)
    RESULTS[cell["name"]] = fp.run_cell(cell, data, ENCODER_NAMES, PROBES, SETTINGS, DEVICE,
                                        RUN_LOCAL / cell["name"], RUN_DRIVE / cell["name"], smoke=SMOKE)
    if cell["target"] == "vol":
        median.to_csv(RUN_LOCAL / cell["name"] / "train_abs_return_median.csv")
        shutil.copy2(RUN_LOCAL / cell["name"] / "train_abs_return_median.csv", RUN_DRIVE / cell["name"])
    print(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
""")

md("""
## 7. Compare with the CNNs, HGB and the random encoder
The reference scores of `run_20261116_044159` are joined on (stock, decision bar). Only rows present in both are kept,
so every number below is on identical rows (the embargo removes the first and last two sessions of 2023).
AUC with 95% intervals from resampling whole validation days; *paired* comparisons use the same resampled days for both
models.
""")
code("""
JOINED = {}
for name, r in RESULTS.items():
    pred, share = fp.join_reference(r["predictions"], REF_DIR / name)
    print(f"{name}: {len(pred):,} rows" + ("  (reference not found: probes only)" if share is None
                                          else f", {share:.1%} of probe rows matched in the reference run"))
    JOINED[name] = dict(info=r["info"], predictions=pred)
TABLE, BOOTS = fp.table(JOINED, n_boot=50 if SMOKE else 300)
TABLE.to_csv(RUN_LOCAL / "probe_table.csv")

# Parameter counts (advisor rule): frozen encoder vs trainable head
ref_params = {}
if (REF_DIR / CELLS[0]["name"] / "done.json").exists():
    ref_params = json.loads((REF_DIR / CELLS[0]["name"] / "done.json").read_text())["parameters"]
rows = []
for raw in dict.fromkeys(TABLE.raw):
    enc = next((e for e in sorted(fp.ENCODERS, key=len, reverse=True) if raw.startswith(e + "_")), None)
    if enc:
        p = TABLE[TABLE.raw == raw].probe_parameters.iloc[0]
        rows.append(dict(model=fp.pretty(raw), frozen_encoder=ENCODER_PARAMS[enc], trainable=int(p), total=ENCODER_PARAMS[enc] + int(p)))
    elif raw.startswith(("cnn_", "hgb_")):
        n = ref_params.get(raw[4:] if raw.startswith("cnn_") else raw)
        rows.append(dict(model=fp.pretty(raw), frozen_encoder=0, trainable=n, total=n))
PARAMS = pd.DataFrame(rows).set_index("model")
print("Parameters (HGB: tree nodes of the first cell; probes: trainable head only, encoder frozen):")
display(PARAMS)

MAIN = ["DINOv2-S linear probe", "DINOv2-S MLP probe", "random ViT-S linear probe", "random ViT-S MLP probe",
        "CLIP-B/16 MLP probe", "image CNN (scratch)", "HGB hand + SPY"]
for m in MAIN + ["DINOv2-S + SPY MLP probe", "image CNN + SPY panel (scratch)", "1D CNN (scratch)", "HGB hand features"]:
    if m in set(TABLE.index.get_level_values("model")):
        print(m); display(fp.auc_matrix(TABLE, m))
fp.plot(TABLE, MAIN, RUN_LOCAL / "probe_auc_by_horizon.png")
TAKEAWAYS = fp.takeaways(TABLE, BOOTS)
(RUN_LOCAL / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
for name in ("probe_table.csv", "probe_auc_by_horizon.png", "takeaways.md"):
    shutil.copy2(RUN_LOCAL / name, RUN_DRIVE / name)
print("saved to", RUN_DRIVE)
""")

md("""
### How to read it
- **Probe > image CNN from scratch**: pretrained visual features help; worth trying fine-tuning, a bigger probe head or
  other encoders.
- **Probe > random encoder but < CNN**: the pretrained features hold some information, but less than training on charts.
- **Probe ≈ random encoder**: general visual pretraining does not help on these charts; any signal comes from the
  architecture/projection, not from what DINO learned.
""")
code("""
for a, b in (("dinov2_s_linear", "dinov2_s_random_linear"), ("dinov2_s_mlp", "dinov2_s_random_mlp"),
             ("dinov2_s_mlp", "cnn_plain"), ("dinov2_s_spy_mlp", "hgb_hand_spy")):
    p = fp.paired(BOOTS, a, b)
    if len(p):
        print(f"{fp.pretty(a)} − {fp.pretty(b)}"); display(p.round(4))
""")

md("""
## 8. Full results table
""")
code("""
display(TABLE.drop(columns=["raw"]).round(4))
""")

md("""
## 9. Held-out test 2024–2025 (run once, at the end)
Only after every choice above is final. Test windows get the same embargo; each saved probe is applied with its own
training-set standardisation. The reference run has no test scores yet, so this section reports the probes only.
""")
code("""
RUN_TEST = SMOKE                                  # set True once, at the very end
if RUN_TEST:
    W["test"] = fp.decision_windows(store, "test", SETTINGS["stride"], EMBARGO, seed=SETTINGS["seed"])
    tslots = np.unique(W["test"].end_index)
    tslot_pos = np.full(store.T, -1, np.int64); tslot_pos[tslots] = np.arange(len(tslots))
    ENCODERS = fp.build_encoders(ENCODER_NAMES, DEVICE, tiny=SMOKE, seed=SETTINGS["seed"])
    TF = fp.extract(fp.StockCharts(PREPARED, W["test"]), ENCODERS, FEAT_LOCAL, "test", DEVICE, **kw)
    TSPY = fp.extract(fp.SpyCharts(PREPARED, tslots), ENCODERS, FEAT_LOCAL, "spytest", DEVICE, **kw)
    del ENCODERS
    TEST = {}
    for cell in CELLS:
        folder = (RUN_LOCAL if SMOKE else RUN_DRIVE) / cell["name"]
        info = json.loads((folder / "done.json").read_text())
        median = None
        if cell["target"] == "vol":
            median = pd.read_csv(folder / "train_abs_return_median.csv", index_col=0).iloc[:, 0]
        te, _ = fp.cell_rows(store, cell, "test", W["test"], median)
        out = te[["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]].copy()
        out["score_training_prior"] = info["prior"]
        for name in info["probes"]:
            ck = folder / f"{name}.pt"
            enc = torch.load(ck, map_location="cpu", weights_only=False)
            X = fp.gpu_matrix(TF[enc["encoder"]], te.row.to_numpy(), TSPY[enc["encoder"]] if enc["with_spy"] else None,
                              tslot_pos[te.end_index.to_numpy()], DEVICE)
            out[f"score_{name}"] = fp.score_saved(ck, X, DEVICE)
        TEST[cell["name"]] = dict(info=info, predictions=out)
    TEST_TABLE, _ = fp.table(TEST, n_boot=50 if SMOKE else 300)
    TEST_TABLE.to_csv(RUN_LOCAL / "test_probe_table.csv"); shutil.copy2(RUN_LOCAL / "test_probe_table.csv", RUN_DRIVE)
    display(TEST_TABLE.drop(columns=["raw"]).round(4))
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}}
for i, c in enumerate(cells):
    c["id"] = f"c{i:02d}"
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", sys.argv[1])
