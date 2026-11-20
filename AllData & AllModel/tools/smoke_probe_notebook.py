"""Code check for FrozenEncoder_Probe.ipynb on a tiny synthetic bundle (no WRDS data, no downloads).

Builds a fake `alldata_bundle_v1.zip` (24 stocks + SPY, 16 sessions), a fake reference run, then executes the
whole notebook with tiny untrained encoders, including the test section. Numbers are meaningless.

    python "AllData & AllModel/tools/smoke_probe_notebook.py" <notebook.ipynb> <work dir>
"""
import hashlib, json, os, shutil, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE_SRC = Path(__file__).resolve().parents[1]
NB, WORK = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
STOCKS, DAYS, SLOTS = 24, 16, 78       # xsdir needs >= 20 stocks per slot


def fake_prepared(folder):
    rng = np.random.default_rng(0)
    T = DAYS * SLOTS
    def bars(n):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, (n, T)), axis=1))
        f = np.repeat(close[:, :, None], 5, axis=2)
        f[:, :, 0] *= np.exp(rng.normal(0, 5e-4, (n, T)))
        f[:, :, 1] = f[:, :, :4].max(axis=2) * 1.001; f[:, :, 2] = f[:, :, :4].min(axis=2) * 0.999
        f[:, :, 4] = rng.integers(500, 5000, (n, T))
        return f
    features = bars(STOCKS)
    mult = np.ones((STOCKS, T, 2)); mult[0, :9 * SLOTS, 0] = 0.5          # a 2:1 split on stock 0 at day 9
    valid = np.ones((STOCKS, T), bool); valid[1, 300:310] = False
    market = bars(1)[0]
    days = pd.bdate_range("2022-12-20", periods=DAYS)
    clock = pd.DatetimeIndex(np.concatenate([
        pd.date_range(pd.Timestamp(f"{d.date()} 09:35", tz="America/New_York"), periods=SLOTS, freq="5min") for d in days]))
    dates = clock.tz_convert("America/New_York").strftime("%Y-%m-%d").to_numpy()
    train_end, valid_end = str(days[6].date()), str(days[11].date())
    folder.mkdir(parents=True)
    np.save(folder / "features.npy", features); np.save(folder / "multipliers.npy", mult)
    np.save(folder / "valid.npy", valid)
    np.save(folder / "market_features.npy", market); np.save(folder / "market_multipliers.npy", np.ones((T, 2)))
    np.save(folder / "market_valid.npy", np.ones(T, bool))
    rows = []
    for s in range(STOCKS):
        for t in range(SLOTS - 1, T):
            if valid[s, t - SLOTS + 1:t + 1].all():
                rows.append(dict(security_index=s, permno=10001 + s, end_index=t, decision_time=clock[t]))
    w = pd.DataFrame(rows)
    d = dates[w.end_index]
    splits = {"train": w[d <= train_end], "validation": w[(d > train_end) & (d <= valid_end)], "test": w[d > valid_end]}
    for name, f in splits.items():
        f.reset_index(drop=True).to_parquet(folder / f"{name}.parquet", index=False)
    manifest = dict(config=dict(lookback=78, bar_minutes=5, train_end=train_end, validation_end=valid_end),
                    permnos=[10001 + s for s in range(STOCKS)], decision_clock_ns=clock.as_unit("ns").asi8.tolist(),
                    splits={k: dict(samples=len(v), decision_start="", decision_end="") for k, v in splits.items()},
                    bundle=dict(market_valid_slots=T, windows=len(w)))
    (folder / "manifest.json").write_text(json.dumps(manifest))


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    here = WORK / "drive"; here.mkdir(parents=True)
    staging = WORK / "staging"
    fake_prepared(staging / "prepared")
    files = {str(p.relative_to(staging)): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in (staging / "prepared").iterdir()}
    (staging / "bundle_manifest.json").write_text(json.dumps(dict(files=files)))
    with zipfile.ZipFile(here / "alldata_bundle_v1.zip", "w") as z:
        for p in staging.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(staging))
    shutil.copytree(HERE_SRC / "chartcnn", here / "chartcnn")
    shutil.copy2(HERE_SRC / "frozen_probe.py", here / "frozen_probe.py")

    # Fake reference run: random scores on every validation row of each cell (the join must keep only shared rows).
    sys.path[:0] = [str(here)]
    import chartcnn
    store = chartcnn.Store(staging / "prepared")
    rng = np.random.default_rng(1)
    for h in (12, 1):
        for kind, target in (("vol", "vol"), ("xsdir", "xs_dir"), ("dir", "dir")):
            tr, med = chartcnn.build_split(store, "train", target, h, 12)
            va, _ = chartcnn.build_split(store, "validation", target, h, 12, med)
            out = va[["security_index", "end_index", "label"]].copy()
            for m in ("hgb_hand_spy", "hgb_hand_features", "cnn_plain", "cnn_plain_spy", "cnn_series1d"):
                out[f"score_{m}"] = rng.random(len(out))
            d = here / "runs" / "run_20261116_044159" / f"{kind}_h{h}"; d.mkdir(parents=True)
            out.to_parquet(d / "predictions.parquet", index=False)
            (d / "done.json").write_text(json.dumps(dict(parameters=dict(plain=776450, plain_spy=936194, series1d=720770,
                                                                        hgb_hand_spy=18300, hgb_hand_features=16653))))

    import nbformat
    from nbclient import NotebookClient
    os.environ.update(ALLDATA_LOCAL_NOTEBOOK_TEST="1", ALLDATA_LOCAL_HERE=str(here), MPLBACKEND="Agg")
    nb = nbformat.read(NB, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(WORK)}}).execute()
    nbformat.write(nb, WORK / "executed.ipynb")
    print("smoke run OK ->", WORK / "executed.ipynb")


if __name__ == "__main__":
    main()
