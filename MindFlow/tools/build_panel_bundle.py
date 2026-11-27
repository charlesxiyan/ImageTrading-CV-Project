#!/usr/bin/env python3
"""Package one more market series (e.g. VIXY) as a small add-on to the MindFlow v4 bundle.

    python MindFlow/tools/build_panel_bundle.py \\
        --prepared "$V4/03_prepared/prepared" --market "$V4/02_adjusted_vixy" --name vixy \\
        --output   "$V4/04_bundle/vixy_panel_v4.zip"

Upload the zip next to alldata_bundle_v4.zip in Drive Demo/MindFlow/. The notebooks unpack it into the
same prepared/ folder (mindflow.stage_panels), where it adds

    prepared/<name>_features.npy     (slots, OHLCV) float32, end-of-data basis like SPY's
    prepared/<name>_multipliers.npy  (slots, 2) float32: price and volume multiplier per bar
    prepared/<name>_valid.npy        (slots,) bool: bar observed and well formed (before filling)

The bars are placed on the stock clock with the same checks as SPY (build_alldata_bundle.market_arrays).
The rows of every model are fixed by the stock and SPY windows, so a missing bar of this series must not
drop a row. Missing bars of a full session are filled with a flat bar at the last observed close
(volume 1 share, multipliers carried forward); the fill share is printed and kept in the manifest.
Early-close sessions stay empty: no stock window uses them.
"""
import argparse
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SLOTS = 78


def _bundle_tools():
    path = REPO / "AllData & AllModel" / "tools" / "build_alldata_bundle.py"
    spec = importlib.util.spec_from_file_location("build_alldata_bundle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fill_missing(x, m, ok, full_session):
    """Flat bars at the last observed close for unobserved slots of full sessions."""
    x, m = x.copy(), m.copy()
    filled = 0
    last = None
    for t in range(len(x)):
        if ok[t]:
            last = t
        elif full_session[t] and last is not None:
            c = x[last, 3]
            x[t] = (c, c, c, c, 1.0)
            m[t] = m[last]
            filled += 1
    return x, m, filled


def build(prepared, market_root, name, output):
    prepared, market_root, output = Path(prepared), Path(market_root), Path(output)
    if output.exists():
        raise ValueError(f"{output} exists; choose a new name")
    manifest = json.loads((prepared / "manifest.json").read_text())
    clock = pd.DatetimeIndex(pd.to_datetime(manifest["decision_clock_ns"], unit="ns", utc=True))
    tools = _bundle_tools()
    x, m, ok = tools.market_arrays(market_root, clock, SLOTS)
    day_ok = ok.reshape(-1, SLOTS)
    stock_valid = np.load(prepared / "valid.npy", mmap_mode="r")
    # a session is "full" when any stock has its last (16:00) bar: early-close sessions have none
    full_session = np.repeat(np.asarray(stock_valid[:, SLOTS - 1::SLOTS]).any(0), SLOTS)
    xf, mf, filled = fill_missing(x, m, ok, full_session)
    still = full_session & ~(np.isfinite(xf).all(1) & (xf > 0).all(1))
    report = dict(series=name, slots=len(clock), observed=int(ok.sum()), full_session_slots=int(full_session.sum()),
                  filled_slots=int(filled), filled_share=round(filled / max(1, full_session.sum()), 6),
                  unfillable_slots=int(still.sum()), sessions_without_bars=int((~day_ok.any(1)).sum()))
    print(json.dumps(report, indent=2))
    stage = output.with_suffix(".staging")
    stage.mkdir(parents=True)
    files = {}
    for suffix, arr in (("features", xf.astype(np.float32)), ("multipliers", mf.astype(np.float32)), ("valid", ok)):
        p = stage / f"{name}_{suffix}.npy"
        np.save(p, arr)
        files[f"prepared/{p.name}"] = p
    meta = stage / f"{name}_panel.json"
    meta.write_text(json.dumps(dict(report, source=market_root.name, stock_bundle_clock_slots=len(clock)), indent=2) + "\n")
    files[f"prepared/{meta.name}"] = meta
    sums = {rel: hashlib.sha256(p.read_bytes()).hexdigest() for rel, p in files.items()}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, p in files.items():
            z.write(p, rel)
        z.writestr("panel_manifest.json", json.dumps(dict(name=name, files={k: dict(sha256=v) for k, v in sums.items()}), indent=2))
    for p in files.values():
        p.unlink()
    stage.rmdir()
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--prepared", required=True, type=lambda s: Path(s).expanduser())
    a.add_argument("--market", required=True, type=lambda s: Path(s).expanduser())
    a.add_argument("--name", default="vixy")
    a.add_argument("--output", required=True, type=lambda s: Path(s).expanduser())
    args = a.parse_args()
    build(args.prepared, args.market, args.name, args.output)
