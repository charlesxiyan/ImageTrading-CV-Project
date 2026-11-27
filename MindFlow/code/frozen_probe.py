"""Frozen pretrained image encoders + probes on the AllData & AllModel chart windows.

Question: do general-purpose visual features (DINOv2, CLIP) carry information about the next
5-60 minutes that a CNN trained from scratch on the same charts does not find?

Pipeline (FrozenEncoder_Probe.ipynb):
  1. windows   one fixed set of decision windows per split (stride 12, as in run_20261116_044159),
               with an embargo of whole sessions at every split boundary;
  2. features  every window's 234x234 chart is rendered once and pushed through every frozen
               encoder; SPY's chart depends only on the time slot, so it is encoded once per slot;
  3. probes    per cell (target kind x horizon): a linear probe (logistic regression) and a 2-layer
               MLP on [stock] or [stock, SPY] features, standardised with training statistics only;
  4. compare   the probes' validation scores are joined row by row with the saved predictions of
               run_20261116_044159 (image CNN, 1D CNN, HGB), so every model is scored on identical rows.

Nothing in this file changes the chart, the labels or the price adjustment: windows are rendered
by chartcnn.images.render_window from point-in-time windows (chartcnn.images.point_in_time_window)
and labels come from chartcnn.build_split.
"""
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from chartcnn import build_split
from chartcnn.images import point_in_time_window, render_window

__version__ = "2026-11-24.1"   # table(): AUC on raw scores (clipping tied unbounded scores)
LOOKBACK = 78
SESSION_BARS = 78
KINDS = {"dir": "dir", "vol": "vol", "xsdir": "xs_dir"}
KIND_LABEL = {"dir": "absolute direction", "vol": "volatility", "xsdir": "relative direction"}

# name -> how to build it. "random" encoders have the same architecture as their pretrained
# twin, randomly initialised (fixed seed) and frozen: the floor that says whether pretraining matters.
ENCODERS = {
    "dinov2_s": dict(family="dinov2", hub="facebook/dinov2-small", pretrained=True, label="DINOv2 ViT-S/14"),
    "dinov2_s_random": dict(family="dinov2", hub="facebook/dinov2-small", pretrained=False,
                            label="random ViT-S/14 (DINOv2 architecture, untrained)"),
    "dinov2_b": dict(family="dinov2", hub="facebook/dinov2-base", pretrained=True, label="DINOv2 ViT-B/14"),
    "clip_b16": dict(family="clip", hub="openai/clip-vit-base-patch16", pretrained=True, label="CLIP ViT-B/16"),
}
NORMALISE = {"dinov2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),            # ImageNet
             "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))}


# ----------------------------------------------------------------------------- windows and embargo

def sessions(store):
    """Sorted unique session dates (YYYY-MM-DD) of the bundle."""
    return np.array(sorted(set(store.dates[::SESSION_BARS])))


def embargo_dates(store, n_sessions):
    """Dates removed by the embargo: `n_sessions` whole sessions on each side of each split boundary.

    An input window is 78 bars, exactly one session, so the windows of the first session after a
    boundary still contain bars of the session before it. Removing at least one session on the
    later side gives inputs that never overlap across the split; removing the same number on the
    earlier side keeps a clean gap between the last training input/label and the first
    validation input. Labels themselves are purged in chartcnn.build_split: a sample is kept only
    if its whole label horizon ends inside its own split."""
    days = sessions(store)
    out = set()
    for boundary in (store.config["train_end"], store.config["validation_end"]):
        k = np.searchsorted(days, boundary, side="right")          # first session after the boundary
        out.update(days[max(0, k - n_sessions):k])                 # last sessions before it
        out.update(days[k:k + n_sessions])                         # first sessions after it
    return out


def decision_windows(store, split, stride, embargo, max_rows=None, every_kth_day=1, seed=69):
    """One fixed set of windows per split, shared by every cell, encoder and probe.

    Same rule as run_20261116_044159: stride-thinned windows of the bundle whose SPY input window is
    complete; minus the embargo sessions; optionally a random subset (train) or every k-th day."""
    f = store.windows(split)
    f = f[f.end_index % stride == stride - 1]
    t = f.end_index.to_numpy(np.int64)
    f = f[store.market_complete(t - LOOKBACK + 1, t + 1)].copy()
    f["decision_date"] = store.dates[f.end_index.to_numpy()]
    f = f[~f.decision_date.isin(embargo)]
    if every_kth_day > 1:
        f = f[f.decision_date.isin(np.sort(f.decision_date.unique())[::every_kth_day])]
    if max_rows is not None and len(f) > max_rows:
        f = f.sample(max_rows, random_state=seed)
    return f.sort_values(["end_index", "security_index"]).reset_index(drop=True)


def cell_rows(store, cell, split, windows, train_median=None):
    """Labelled rows of one cell restricted to the cached windows.

    Returns (frame with a `row` column = position in `windows`, vol median fitted on train only)."""
    f, median = build_split(store, split, cell["target"], cell["horizon"], cell["stride"], train_median)
    key = windows[["security_index", "end_index"]].reset_index(drop=True).reset_index().rename(columns={"index": "row"})
    f = f.merge(key, on=["security_index", "end_index"], how="inner")
    return f.sort_values("row").reset_index(drop=True), median


# ----------------------------------------------------------------------------- rendering

class StockCharts(Dataset):
    """uint8 234x234 chart of each window, rendered exactly as the image CNN sees it."""

    def __init__(self, prepared, frame):
        self.root = Path(prepared)
        self.s = frame.security_index.to_numpy(np.int64)
        self.t = frame.end_index.to_numpy(np.int64)
        self._x = self._m = None

    def __len__(self):
        return len(self.t)

    def __getstate__(self):
        d = self.__dict__.copy(); d["_x"] = d["_m"] = None
        return d

    def __getitem__(self, i):
        if self._x is None:
            self._x = np.load(self.root / "features.npy", mmap_mode="r")
            self._m = np.load(self.root / "multipliers.npy", mmap_mode="r")
        w = point_in_time_window(self._x[self.s[i]], self._m[self.s[i]], self.t[i], LOOKBACK)
        return torch.from_numpy(render_window(w))


class SpyCharts(StockCharts):
    """SPY's chart for each distinct decision slot (it does not depend on the stock)."""

    def __init__(self, prepared, slots):
        self.root = Path(prepared)
        self.t = np.asarray(slots, np.int64)
        self._x = self._m = None

    def __getitem__(self, i):
        if self._x is None:
            self._x = np.load(self.root / "market_features.npy", mmap_mode="r")
            self._m = np.load(self.root / "market_multipliers.npy", mmap_mode="r")
        return torch.from_numpy(render_window(point_in_time_window(self._x, self._m, self.t[i], LOOKBACK)))


# ----------------------------------------------------------------------------- encoders

class FrozenEncoder(nn.Module):
    """A frozen ViT that maps a uint8 chart batch [B, 234, 234] to [CLS token, mean patch token].

    The chart is resized to 224x224 (bilinear, antialiased), copied to 3 channels and normalised
    with the encoder's own pretraining statistics. Nothing here is trained."""

    def __init__(self, name, tiny=False, seed=0):
        super().__init__()
        spec = ENCODERS[name]
        self.name, self.spec = name, spec
        mean, std = NORMALISE[spec["family"]]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        torch.manual_seed(seed)
        if spec["family"] == "dinov2":
            from transformers import AutoConfig, Dinov2Config, Dinov2Model
            if tiny:                                    # offline code check only (no download)
                self.net = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                                    intermediate_size=64, image_size=224, patch_size=14))
            elif spec["pretrained"]:
                self.net = Dinov2Model.from_pretrained(spec["hub"])
            else:
                self.net = Dinov2Model(AutoConfig.from_pretrained(spec["hub"]))
        else:
            from transformers import AutoConfig, CLIPVisionConfig, CLIPVisionModel
            if tiny:
                self.net = CLIPVisionModel(CLIPVisionConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                                            intermediate_size=64, image_size=224, patch_size=16))
            elif spec["pretrained"]:
                self.net = CLIPVisionModel.from_pretrained(spec["hub"])
            else:
                self.net = CLIPVisionModel(AutoConfig.from_pretrained(spec["hub"]).vision_config)
        self.net.requires_grad_(False).eval()
        self.dim = 2 * self.net.config.hidden_size

    def train(self, mode=True):                         # always eval: no dropout, frozen statistics
        return super().train(False)

    @torch.no_grad()
    def forward(self, images_uint8):
        x = images_uint8.to(self.mean.device, non_blocking=True).float().div_(255.0)[:, None]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        out = self.net(pixel_values=x)
        h = out.last_hidden_state                                      # [B, 1 + patches, hidden]
        cls, patches = h[:, 0], h[:, 1:]
        if self.spec["family"] == "clip":                              # CLIP's CLS after its final layer norm
            cls = out.pooler_output
        return torch.cat([cls, patches.mean(1)], dim=1).float()

    def parameters_count(self):
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return total, trainable


def build_encoders(names, device, tiny=False, seed=0):
    encoders = {}
    for n in names:
        offset = 0 if ENCODERS[n]["pretrained"] else 1         # random twins get their own seed
        e = FrozenEncoder(n, tiny=tiny, seed=seed + offset).to(device)
        total, trainable = e.parameters_count()
        print(f"{n} = {ENCODERS[n]['label']}: {total:,} parameters ({trainable:,} trainable, frozen), "
              f"feature size {e.dim} (CLS + mean patch)")
        encoders[n] = e
    return encoders


def _autocast(device, amp):
    if amp and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def extract(dataset, encoders, out_dir, tag, device, batch_size=256, workers=8, chunk=50_000, amp=True,
            drive_dir=None):
    """Render each chart once and feed it to every encoder; saves float16 features per chunk.

    Chunks already on Drive (or local disk) are skipped, so a disconnect costs at most one chunk.
    Returns {encoder: array [len(dataset), dim] float16}."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    n_chunks = max(1, math.ceil(n / chunk))
    t0 = time.perf_counter()
    for c in range(n_chunks):
        names = {e: f"{tag}_{e}_{c:03d}.npy" for e in encoders}
        if drive_dir is not None:                           # restore chunks a previous session saved
            for f in names.values():
                if (Path(drive_dir) / f).exists() and not (out_dir / f).exists():
                    shutil.copy2(Path(drive_dir) / f, out_dir / f)
        if all((out_dir / f).exists() for f in names.values()):
            continue
        lo, hi = c * chunk, min(n, (c + 1) * chunk)
        sub = torch.utils.data.Subset(dataset, range(lo, hi))
        loader = DataLoader(sub, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda",
                            persistent_workers=False, prefetch_factor=4 if workers else None)
        feats = {e: [] for e in encoders}
        with _autocast(device, amp):
            for x in loader:
                x = x.to(device, non_blocking=True)
                for e, enc in encoders.items():
                    feats[e].append(enc(x).half().cpu())
        for e, f in names.items():
            arr = torch.cat(feats[e]).numpy()
            assert len(arr) == hi - lo and np.isfinite(arr).all(), f"bad features {f}"
            np.save(out_dir / f, arr)
            if drive_dir is not None:
                Path(drive_dir).mkdir(parents=True, exist_ok=True)
                tmp = Path(drive_dir) / (f + ".copying")
                shutil.copy2(out_dir / f, tmp); os.replace(tmp, Path(drive_dir) / f)
        rate = hi / (time.perf_counter() - t0 + 1e-9)
        print(f"  {tag}: chunk {c + 1}/{n_chunks} ({hi:,}/{n:,} charts, ~{rate:,.0f} charts/s this session)")
    out = {}
    for e in encoders:                      # one memory-mapped file per encoder: little RAM needed
        target = out_dir / f"{tag}_{e}.npy"
        if not target.exists():
            parts = [np.load(out_dir / f"{tag}_{e}_{c:03d}.npy", mmap_mode="r") for c in range(n_chunks)]
            m = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16, shape=(n, parts[0].shape[1]))
            pos = 0
            for part in parts:
                m[pos:pos + len(part)] = part; pos += len(part)
            m.flush(); del m
        out[e] = np.load(target, mmap_mode="r")
    return out


# ----------------------------------------------------------------------------- probes

def gpu_matrix(feats, rows, spy_feats=None, spy_pos=None, device="cpu", chunk=65536):
    """Rows of a (memory-mapped) feature array, plus SPY's features for each row's slot if given,
    as one float16 tensor on `device`. Built chunk by chunk so CPU memory stays small."""
    d = feats.shape[1] + (spy_feats.shape[1] if spy_feats is not None else 0)
    out = torch.empty((len(rows), d), dtype=torch.float16, device=device)
    for i in range(0, len(rows), chunk):
        r = np.asarray(rows[i:i + chunk])
        order = np.argsort(r)                               # sorted reads from the memmap are faster
        x = np.empty((len(r), feats.shape[1]), np.float16); x[order] = feats[r[order]]
        if spy_feats is not None:
            x = np.concatenate([x, np.asarray(spy_feats)[spy_pos[i:i + chunk]]], axis=1)
        out[i:i + len(r)] = torch.from_numpy(x).to(device)
    return out


class Standardiser:
    """Mean/std from the TRAINING features only, applied unchanged (in place) to validation and test."""

    def __init__(self, X, chunk=65536):
        s = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device); s2 = torch.zeros_like(s)
        for i in range(0, len(X), chunk):
            x = X[i:i + chunk].double(); s += x.sum(0); s2 += (x * x).sum(0)
        n = len(X)
        self.mean = (s / n).float()
        self.std = ((s2 / n - (s / n) ** 2).clamp_min(0) * n / max(1, n - 1)).sqrt().float().clamp_min(1e-6)

    def __call__(self, X, chunk=65536):
        for i in range(0, len(X), chunk):
            X[i:i + chunk] = ((X[i:i + chunk].float() - self.mean.to(X.device)) / self.std.to(X.device)).half()
        return X


def make_probe(kind, dim, hidden=512, dropout=0.2):
    if kind == "linear":                            # logistic regression on the standardised features
        return nn.Linear(dim, 1)
    if kind == "mlp":                               # 2 layers: dim -> hidden -> 1
        return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    raise ValueError(kind)


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _batches(n, size, shuffle, gen):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


@torch.no_grad()
def _predict(model, X, device, batch=65536):
    model.eval()
    out = [torch.sigmoid(model(X[i:i + batch].to(device).float()).squeeze(1)).cpu() for i in range(0, len(X), batch)]
    return torch.cat(out).numpy().astype(np.float64)


def fit_probe(kind, Xtr, ytr, Xes, yes, prior, settings, device, seed=69):
    """Train one probe with Adam; L2 strength (linear) or weight decay (MLP) picked from a small grid
    by log loss on the early-stopping days of validation, as the CNNs were. Returns (model, info)."""
    grid = settings["linear_l2_grid"] if kind == "linear" else settings["mlp_weight_decay_grid"]
    Xtr_d = Xtr; ytr_d = torch.as_tensor(ytr, dtype=torch.float32, device=Xtr.device)
    bias0 = math.log(prior / (1 - prior))
    best = None
    for reg in grid:
        torch.manual_seed(seed)
        model = make_probe(kind, Xtr.shape[1], settings["mlp_hidden"], settings["mlp_dropout"]).to(device)
        last = model if kind == "linear" else model[-1]
        nn.init.zeros_(last.weight); nn.init.constant_(last.bias, bias0)      # starts at the training prior
        wd = 0.0 if kind == "linear" else reg
        opt = torch.optim.AdamW(model.parameters(), lr=settings["probe_lr"], weight_decay=wd)
        gen = torch.Generator().manual_seed(seed)
        best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
        for epoch in range(1, settings["probe_epochs"] + 1):
            model.train()
            for b in _batches(len(Xtr_d), settings["probe_batch"], True, gen):
                b = b.to(Xtr_d.device)
                logit = model(Xtr_d[b].float()).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logit, ytr_d[b])
                if kind == "linear":
                    loss = loss + reg * model.weight.pow(2).sum()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            p = np.clip(_predict(model, Xes, device), 1e-7, 1 - 1e-7)
            val = log_loss(yes, p, labels=[0, 1])
            if val < best_loss - 1e-6:
                best_loss, best_epoch, bad = val, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= settings["probe_patience"]:
                    break
        if best is None or best_loss < best["loss"]:
            best = dict(loss=best_loss, reg=reg, epoch=best_epoch, state=best_state)
    model = make_probe(kind, Xtr.shape[1], settings["mlp_hidden"], settings["mlp_dropout"]).to(device)
    model.load_state_dict(best["state"])
    return model, dict(regularisation=best["reg"], best_epoch=best["epoch"], es_log_loss=best["loss"],
                       parameters=count(model))


def run_cell(cell, data, encoders, probes, settings, device, out_dir, drive_dir=None, smoke=False):
    """All probes of one cell. data: dict with train/valid frames (cell rows), feature arrays,
    SPY features and slot index. Writes predictions.parquet + done.json; skips if done."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    done = (Path(drive_dir) if drive_dir else out_dir) / "done.json"
    if done.exists():
        print(f"{cell['name']}: already done, loading")
        return load_cell(done.parent)
    t0 = time.perf_counter()
    tr, va = data["train"], data["valid"]
    assert len(tr) and len(va) and 0 < tr.label.mean() < 1, f"{cell['name']}: no usable rows"
    prior = float(tr.label.mean())
    k = max(1, int(settings["early_stop_every_kth_day"]))
    es_mask = np.array(va.decision_date.isin(np.sort(va.decision_date.unique())[::k]), dtype=bool)
    print(f"{cell['name']} ({KIND_LABEL[cell['kind']]}, next {cell['minutes']} min): {len(tr):,} train / "
          f"{len(va):,} validation rows ({es_mask.sum():,} on early-stopping days); positive share "
          f"{prior:.3f} / {va.label.mean():.3f}")
    scores = {"training_prior": np.full(len(va), prior)}
    info = dict(cell=cell, prior=prior, train_rows=len(tr), validation_rows=len(va), probes={})
    for enc in encoders:
        for with_spy in (False, True):
            spy = data["spy"][enc] if with_spy else None
            Xtr = gpu_matrix(data["feats"][enc]["train"], tr.row.to_numpy(), spy,
                             data["slot_pos"][tr.end_index.to_numpy()], device)
            Xva = gpu_matrix(data["feats"][enc]["valid"], va.row.to_numpy(), spy,
                             data["slot_pos"][va.end_index.to_numpy()], device)
            scaler = Standardiser(Xtr)                          # fitted on training rows only
            Xtr, Xva = scaler(Xtr), scaler(Xva)
            for kind in probes:
                name = f"{enc}{'_spy' if with_spy else ''}_{kind}"
                model, pinfo = fit_probe(kind, Xtr, tr.label.to_numpy(), Xva[torch.from_numpy(es_mask).to(Xva.device)],
                                         va.label.to_numpy()[es_mask], prior, settings, device, settings["seed"])
                scores[name] = _predict(model, Xva, device)
                torch.save(dict(state={k: v.half().cpu() for k, v in model.state_dict().items()},
                                mean=scaler.mean.cpu(), std=scaler.std.cpu(), kind=kind, with_spy=with_spy,
                                encoder=enc, dim=Xtr.shape[1], settings=settings, **pinfo), out_dir / f"{name}.pt")
                if drive_dir is not None:
                    Path(drive_dir).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(out_dir / f"{name}.pt", Path(drive_dir) / f"{name}.pt")
                auc = roc_auc_score(va.label, scores[name]) if 0 < va.label.mean() < 1 else float("nan")
                print(f"  {name:32s} {pinfo['parameters']:>9,} trainable probe params | reg {pinfo['regularisation']:g} "
                      f"| best epoch {pinfo['best_epoch']} | val AUC {auc:.4f}")
                info["probes"][name] = pinfo
            del Xtr, Xva
            if device.type == "cuda":
                torch.cuda.empty_cache()
    keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
    out = va[keep].copy()
    for n, p in scores.items():
        out[f"score_{n}"] = np.asarray(p, float)
    out.to_parquet(out_dir / "predictions.parquet", index=False)
    info["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    info["smoke"] = smoke
    (out_dir / "done.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    if drive_dir is not None:
        for f in ("predictions.parquet", "done.json"):                 # done.json last: marks the cell finished
            shutil.copy2(out_dir / f, Path(drive_dir) / f)
    print(f"{cell['name']}: done in {info['minutes']} min")
    return load_cell(out_dir)


def score_saved(path, X, device):
    """Apply a saved probe (its own training-set standardisation included) to new features X."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["settings"]
    model = make_probe(ck["kind"], ck["dim"], st["mlp_hidden"], st["mlp_dropout"])
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    model.to(device)
    sc = Standardiser.__new__(Standardiser); sc.mean, sc.std = ck["mean"].to(X.device), ck["std"].to(X.device)
    return _predict(model, sc(X), device)


def load_cell(folder):
    folder = Path(folder)
    return dict(info=json.loads((folder / "done.json").read_text()),
                predictions=pd.read_parquet(folder / "predictions.parquet"))


# ----------------------------------------------------------------------------- comparison

REFERENCE_MODELS = {"hgb_hand_spy": "HGB hand + SPY", "hgb_hand_features": "HGB hand features",
                    "cnn_plain": "image CNN (scratch)", "cnn_plain_spy": "image CNN + SPY panel (scratch)",
                    "cnn_series1d": "1D CNN (scratch)", "cnn_series1d_spy": "1D CNN + SPY (scratch)",
                    "cnn_residual": "residual image CNN (scratch)",
                    "cnn_vit": "image ViT (scratch)", "cnn_vit_spy": "image ViT + SPY panel (scratch)"}


def pretty(name):
    if name in REFERENCE_MODELS:
        return REFERENCE_MODELS[name]
    if name == "training_prior":
        return "training prior (majority)"
    for enc in sorted(ENCODERS, key=len, reverse=True):
        if name.startswith(enc + "_"):
            rest = name[len(enc) + 1:]
            spy = rest.startswith("spy_")
            kind = rest.split("_")[-1]
            short = {"dinov2_s": "DINOv2-S", "dinov2_s_random": "random ViT-S", "dinov2_b": "DINOv2-B",
                     "clip_b16": "CLIP-B/16"}[enc]
            return f"{short} {'+ SPY ' if spy else ''}{'linear' if kind == 'linear' else 'MLP'} probe"
    return name


def join_reference(pred, reference_cell_dir):
    """Add the saved scores of run_20261116_044159 for the same rows. Rows the reference does not
    have are dropped, so every model is scored on identical rows. Returns (frame, matched share)."""
    path = Path(reference_cell_dir) / "predictions.parquet"
    if not path.exists():
        return pred, None
    ref = pd.read_parquet(path)
    cols = [c for c in ref.columns if c.startswith("score_") and c[6:] in REFERENCE_MODELS]
    ref = ref[["security_index", "end_index", "label"] + cols].rename(columns={"label": "ref_label"})
    out = pred.merge(ref, on=["security_index", "end_index"], how="inner")
    assert (out.label == out.ref_label).all(), "reference labels differ: not the same cell definition"
    return out.drop(columns="ref_label"), len(out) / max(1, len(pred))


def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def day_bootstrap(pred, names, n_boot=300, seed=0):
    """AUC draws with whole validation days resampled; the same draws for every model (paired)."""
    rng = np.random.default_rng(seed)
    y = pred.label.to_numpy()
    codes, _ = pd.factorize(pred.decision_date)
    days = [np.flatnonzero(codes == d) for d in range(codes.max() + 1)]
    cols = {n: pred[f"score_{n}"].to_numpy(float) for n in names}
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([days[i] for i in rng.integers(0, len(days), len(days))])
        draws.append({n: _auc(y[idx], cols[n][idx]) for n in names})
    return pd.DataFrame(draws)


def table(results, n_boot=300):
    """One row per (cell, model): AUC with 95% day-bootstrap CI, log-loss gain vs the prior, parameters."""
    rows, boots = [], {}
    for cname, r in results.items():
        pred, info = r["predictions"], r["info"]
        cell = info["cell"]
        names = [c[6:] for c in pred.columns if c.startswith("score_") and c != "score_training_prior"]
        boot = day_bootstrap(pred, names, n_boot)
        boots[cname] = boot
        y = pred.label.to_numpy()
        base = log_loss(y, np.full(len(y), info["prior"]), labels=[0, 1])
        for n in names:
            s = pred[f"score_{n}"].to_numpy(float)          # raw score: AUC needs only its order
            is_prob = bool(np.nanmin(s) >= 0 and np.nanmax(s) <= 1)
            p = np.clip(s, 1e-7, 1 - 1e-7)                   # clipped copy for log loss only
            lo, hi = np.nanpercentile(boot[n], [2.5, 97.5])
            pinfo = info["probes"].get(n, {})
            rows.append(dict(cell=cname, kind=cell["kind"], minutes=cell["minutes"], model=pretty(n), raw=n,
                             roc_auc=_auc(y, s), auc_ci_low=lo, auc_ci_high=hi,
                             log_loss_gain_vs_prior_pct=100 * (base - log_loss(y, p, labels=[0, 1])) / base if is_prob else np.nan,
                             prob_std=float(p.std()) if is_prob else np.nan, probe_parameters=pinfo.get("parameters", np.nan),
                             regularisation=pinfo.get("regularisation", np.nan),
                             best_epoch=pinfo.get("best_epoch", np.nan), rows=len(y),
                             days=pred.decision_date.nunique()))
    return pd.DataFrame(rows).set_index(["cell", "model"]), boots


def paired(boots, a, b):
    """Across cells: in how many is a − b above zero, below zero or tied (95% paired day bootstrap)?"""
    out = []
    for cname, boot in boots.items():
        if a in boot and b in boot:
            d = (boot[a] - boot[b]).to_numpy()
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            out.append(dict(cell=cname, diff=float(np.nanmean(d)), low=lo, high=hi,
                            verdict="better" if lo > 0 else "worse" if hi < 0 else "tied"))
    return pd.DataFrame(out)


def auc_matrix(tab, model):
    t = tab.reset_index()
    t = t[t.model == model]
    t = t.assign(text=t.apply(lambda r: f"{r.roc_auc:.3f} [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}]", axis=1))
    m = t.pivot(index="kind", columns="minutes", values="text")
    m.columns = [f"{c} min" for c in m.columns]
    return m.reindex([k for k in KINDS if k in m.index])


def plot(tab, models, path=None):
    """AUC (95% CI) by horizon, one panel per target kind, one line per model."""
    import matplotlib.pyplot as plt
    t = tab.reset_index()
    kinds = [k for k in KINDS if k in set(t.kind)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.6 * len(kinds), 4.4), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        s = t[t.kind == kind]
        shown = [m for m in models if m in set(s.model)]
        for j, m in enumerate(shown):
            d = s[s.model == m].sort_values("minutes")
            x = d.minutes * (1 + 0.025 * (j - len(shown) / 2))
            ax.errorbar(x, d.roc_auc, yerr=[d.roc_auc - d.auc_ci_low, d.auc_ci_high - d.roc_auc], fmt="o",
                        capsize=3, label=m, ls="--" if "random" in m or "scratch" in m else "-")
        ax.axhline(0.5, ls=":", c="gray", lw=1)
        ax.set_xscale("log"); ax.minorticks_off()
        mins = sorted(t.minutes.unique())
        ax.set_xticks(mins, [f"{m} min" for m in mins])
        ax.set(title=f"{kind}: {KIND_LABEL[kind]}", xlabel="horizon", ylabel="validation AUC")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()


def takeaways(tab, boots, main_encoder="dinov2_s"):
    """Plain-language reading of the comparison, generated from the numbers."""
    lines = []
    t = tab.reset_index()
    for kind in [k for k in KINDS if k in set(t.kind)]:
        s = t[(t.kind == kind)]
        best = s.loc[s.groupby("cell").roc_auc.idxmax()].sort_values("minutes")
        lines.append(f"- **{kind}**: best model by horizon " + ", ".join(
            f"{r.minutes} min {r.roc_auc:.3f} ({r.model})" for _, r in best.iterrows()) + ".")
    e = main_encoder
    checks = [(f"{e}_linear", f"{e}_random_linear" if f"{e}_random" in ENCODERS else None,
               "pretrained vs random encoder, linear probe"),
              (f"{e}_mlp", f"{e}_random_mlp" if f"{e}_random" in ENCODERS else None,
               "pretrained vs random encoder, MLP probe"),
              (f"{e}_mlp", f"{e}_linear", "MLP probe vs linear probe"),
              (f"{e}_linear", "cnn_plain", "frozen linear probe vs image CNN from scratch"),
              (f"{e}_mlp", "cnn_plain", "frozen MLP probe vs image CNN from scratch"),
              (f"{e}_spy_mlp", "cnn_plain_spy", "frozen MLP probe + SPY vs image CNN + SPY panel"),
              (f"{e}_spy_mlp", "hgb_hand_spy", "frozen MLP probe + SPY vs HGB hand + SPY"),
              (f"{e}_spy_mlp", f"{e}_mlp", "with SPY features vs without (MLP probe)"),
              ("clip_b16_mlp", f"{e}_mlp", "CLIP vs DINOv2 (MLP probe)")]
    for a, b, label in checks:
        if b is None:
            continue
        p = paired(boots, a, b)
        if len(p):
            v = p.verdict.value_counts()
            lines.append(f"- {label} ({pretty(a)} − {pretty(b)}), paired day bootstrap over {len(p)} cells: "
                         f"better in {v.get('better', 0)}, worse in {v.get('worse', 0)}, tied in {v.get('tied', 0)}; "
                         f"mean AUC difference {p['diff'].mean():+.4f}.")
    return "\n".join(lines)
