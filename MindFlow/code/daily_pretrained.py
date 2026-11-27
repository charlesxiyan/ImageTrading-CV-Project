"""Step 1b: pretrained models on the daily data of Step 1a, scored on the same rows.

  frozen image encoders  DINOv2-S, CLIP-B/16 and a random ViT-S (floor) read the JKX images of Step 1a
                         (I5 32x15, I20 64x60, resized to 224x224) -> [CLS, mean patch] features ->
                         linear and MLP probes trained on the training years. As for the JKX CNN, the
                         probe's early stopping and its regularisation use a random 30% of the training
                         sample, never 2023.
  time-series models     Chronos-2, TimesFM, Moirai 2, Kronos forecast the next h daily closes from the
                         last `context` sessions (zero-shot, see pretrained_ts.py).

Rows: dj.labelled(...) with the same anchors, horizons, targets, purge, embargo and evaluation stride
as Step 1a, so validation and test rows are identical to Step 1a's and its CNN / HGB / rule scores
can be joined row by row.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import daily_jkx as dj
import frozen_probe as fp
import pretrained_ts as pt

__version__ = "2026-11-23.1"

KEYS = ["security_index", "t", "split"]


@torch.no_grad()
def encode(images, encoders, out_dir, tag, device, batch=512, amp=True):
    """Features of uint8 images (N, H, W) for every encoder, cached as float16 .npy in out_dir."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, enc in encoders.items():
        path = out_dir / f"{tag}_{name}.npy"
        if not path.exists():
            t0 = time.perf_counter()
            parts = []
            with fp._autocast(device, amp):
                for i in range(0, len(images), batch):
                    x = torch.as_tensor(np.asarray(images[i:i + batch]))
                    parts.append(enc(x).half().cpu())
            arr = torch.cat(parts).numpy()
            assert np.isfinite(arr).all(), f"non-finite features from {name}"
            np.save(path, arr)
            print(f"  {tag} {name}: {arr.shape} in {(time.perf_counter() - t0) / 60:.1f} min")
        out[name] = np.load(path, mmap_mode="r")
    return out


def cell_rows(store, frame, cell, settings):
    """Training sample (with a random 30% early-stopping part) and evaluation rows, as in dj.run_cell."""
    L, h, target = cell["image_days"], cell["horizon"], cell["target"]
    f = dj.labelled(store, frame.assign(row=np.arange(len(frame))), h, target, eval_stride=h)
    tr = f[f.split == "train"]
    if settings.get("max_train") and len(tr) > settings["max_train"]:
        tr = tr.sample(settings["max_train"], random_state=settings["seed"])
    ev = f[f.split != "train"].reset_index(drop=True)
    rng = np.random.default_rng(settings["seed"])
    is_es = rng.random(len(tr)) < settings["holdout_share"]
    return tr.reset_index(drop=True), is_es, ev


def probe_cell(cell, tr, is_es, ev, feats, encoder_names, probes, settings, device, out_dir):
    """Linear / MLP probes on frozen features for one cell; returns (scores dict, probe info)."""
    scores, info = {}, {}
    ytr = tr.label.to_numpy()
    prior = float(ytr.mean())
    for enc in encoder_names:
        X_all = fp.gpu_matrix(feats[enc], tr.row.to_numpy(), device=device)
        X_ev = fp.gpu_matrix(feats[enc], ev.row.to_numpy(), device=device)
        fit_rows = torch.from_numpy(np.flatnonzero(~is_es)).to(device)
        es_rows = torch.from_numpy(np.flatnonzero(is_es)).to(device)
        Xfit, Xes = X_all[fit_rows], X_all[es_rows]
        scaler = fp.Standardiser(Xfit)                      # training rows only
        Xfit, Xes, X_ev = scaler(Xfit), scaler(Xes), scaler(X_ev)
        for kind in probes:
            name = f"{enc}_{kind}"
            model, pinfo = fp.fit_probe(kind, Xfit, ytr[~is_es], Xes, ytr[is_es], prior, settings, device,
                                        settings["seed"])
            scores[name] = fp._predict(model, X_ev, device)
            info[name] = pinfo
            torch.save(dict(state={k: v.half().cpu() for k, v in model.state_dict().items()},
                            mean=scaler.mean.cpu(), std=scaler.std.cpu(), kind=kind, encoder=enc, **pinfo),
                       Path(out_dir) / f"{name}.pt")
            va = ev.split.to_numpy() == "validation"
            auc = dj._auc(ev.label.to_numpy()[va], scores[name][va])
            print(f"  {name:28s} {pinfo['parameters']:>8,} trainable probe params | reg {pinfo['regularisation']:g} "
                  f"| best epoch {pinfo['best_epoch']} | validation AUC {auc:.4f}")
        del X_all, X_ev, Xfit, Xes
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return scores, info


def run_cell(store, cell, frame, feats, encoder_names, probes, tsfm, settings, device, out_dir, step1a_dir=None):
    """Probes + time-series scores for one cell, joined with Step 1a's scores on identical rows.

    tsfm: {model: {h: dict(keys=frame, q=(n, 9))}} from forecast_daily (may be empty)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "done.json").exists():
        print(f"{cell['name']}: already done, loading")
        return dj.load_cell(out_dir)
    t0 = time.perf_counter()
    tr, is_es, ev = cell_rows(store, frame, cell, settings)
    print(f"{cell['name']}: train {len(tr):,} ({(~is_es).sum():,} fit / {is_es.sum():,} early stop) | validation "
          f"{(ev.split == 'validation').sum():,} | test {(ev.split == 'test').sum():,}")
    prior = float(tr.label.mean())
    out = ev[["security_index", "t", "date", "split", "forward_return", "label"]].copy()
    out["score_training_prior"] = prior
    scores, pinfo = probe_cell(cell, tr, is_es, ev, feats, encoder_names, probes, settings, device, out_dir)
    for n, s in scores.items():
        out[f"score_{n}"] = s
    h = cell["horizon"]
    for name, by_h in tsfm.items():
        d = by_h[h]
        k = d["keys"][["security_index", "t"]].copy()
        k[f"score_tsfm_{name}"] = pt.scores(d["q"], cell["target"], d["keys"], "t")
        out = out.merge(k, on=["security_index", "t"], how="left")       # NaN where not forecast
    matched = None
    if step1a_dir is not None and (Path(step1a_dir) / cell["name"] / "predictions.parquet").exists():
        ref = pd.read_parquet(Path(step1a_dir) / cell["name"] / "predictions.parquet")
        cols = [c for c in ref.columns if c.startswith("score_") and c != "score_training_prior" and "_seed" not in c]
        ref = ref[KEYS + ["label"] + cols].rename(columns={"label": "ref_label"})
        n0 = len(out)
        out = out.merge(ref, on=KEYS, how="inner")
        assert (out.label == out.ref_label).all(), "Step 1a labels differ: not the same cell definition"
        out = out.drop(columns="ref_label")
        matched = len(out) / max(1, n0)
        print(f"  joined Step 1a scores on {len(out):,} rows ({matched:.1%} of this cell's rows)")
    info = dict(cell=cell, prior=prior, probes=pinfo, train_rows=int(len(tr)), step1a_matched=matched,
                minutes=(time.perf_counter() - t0) / 60, version=__version__)
    out.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "done.json").write_text(json.dumps(info, default=float, indent=2))
    return dict(info=info, predictions=out)


def forecast_daily(store, frames, cells, names, settings, device, out_dir, splits=("validation", "test")):
    """Zero-shot forecasts for every evaluation anchor of the cells (one set per horizon).

    The rows are the union over cells of a horizon's evaluation anchors in `splits`."""
    out = {}
    params = {}
    for h in sorted({c["horizon"] for c in cells}):
        keys = []
        for c in cells:
            if c["horizon"] != h:
                continue
            f = dj.labelled(store, frames[c["image_days"]], h, c["target"], eval_stride=h)
            keys.append(f[f.split.isin(splits)][["security_index", "t", "date"]])
        keys = pd.concat(keys).drop_duplicates(["security_index", "t"]).sort_values(["t", "security_index"])
        keys = keys.reset_index(drop=True)
        if settings.get("tsfm_max_rows") and len(keys) > settings["tsfm_max_rows"]:
            dates = np.sort(keys.date.unique())
            k = int(np.ceil(len(keys) / settings["tsfm_max_rows"]))
            keys = keys[keys.date.isin(dates[::k])].reset_index(drop=True)
        print(f"h={h}: forecasting {len(keys):,} daily series ({keys.date.nunique()} dates)")
        q, p = pt.forecast_all(names, lambda hh: pt.daily_contexts(store, keys, settings["tsfm_context"], hh),
                               keys, [h], Path(out_dir) / f"h{h}", device)
        params.update(p)
        for name in names:
            out.setdefault(name, {})[h] = dict(keys=keys, q=q[name][h])
    return out, params


MODEL_LABEL = {
    "dinov2_s_linear": "DINOv2-S linear probe", "dinov2_s_mlp": "DINOv2-S MLP probe",
    "dinov2_s_random_linear": "random ViT-S linear probe", "dinov2_s_random_mlp": "random ViT-S MLP probe",
    "dinov2_b_linear": "DINOv2-B linear probe", "dinov2_b_mlp": "DINOv2-B MLP probe",
    "clip_b16_linear": "CLIP-B/16 linear probe", "clip_b16_mlp": "CLIP-B/16 MLP probe",
    **{f"tsfm_{k}": pt.label(k) for k in pt.MODELS},
}
dj.MODEL_LABEL.update(MODEL_LABEL)
