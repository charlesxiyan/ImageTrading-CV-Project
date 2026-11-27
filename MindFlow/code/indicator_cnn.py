"""Charts with technical indicators: does the scratch CNN improve when the picture carries more of the past?

The original chart (chartcnn.images.render_window) scales prices to the window's own low/high and
volume to the window's own maximum. A quiet stock and a wild stock therefore draw almost the same
picture: the *level* of volatility is not in the image, while HGB gets it as a number
(`realized_vol`). This module adds two indicators, in the spirit of Jiang, Kelly & Xiu (2023):

  MA     a moving average of the close over the previous 78 bars (one session), drawn as a line over
         the price bars. At the first bar of the window it already uses the whole previous session,
         so the picture carries information from before the window. The price axis is widened to
         include the line, as in the paper.
  EWSD   the exponentially weighted standard deviation of 5-minute log returns (half-life
         `halflife` bars, started on the previous session), drawn as bars in a band under the chart
         on ONE FIXED log scale for every stock and window. This is the only part of the picture
         with an absolute scale. The scale (lowest and highest log EWSD) is fitted on training
         windows only.

Every variant is drawn on the same canvas (234 x 234 chart + an 78-row band, black when EWSD is off)
and trained with the same CNN, settings and rows, so only the content of the picture changes.
Two more variants add the 38 hand + SPY features that HGB uses as a second input (a small MLP whose
logit is added to the CNN's): if those win over the indicator charts, the CNN lacks information,
not the ability to read pictures.

Point in time: the long window (previous session + window) comes from
chartcnn.images.point_in_time_window, so every bar is on the anchor bar's basis and no later split
or dividend reaches it. Labels come from chartcnn.build_split (purged at split boundaries).
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import lfilter
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from chartcnn import build_split
from chartcnn.baselines import hand_features, market_features
from chartcnn.images import PIXELS_PER_BAR, point_in_time_window, render_window
from chartcnn.models import ChartCNN
from chartcnn.training import autocast

__version__ = "2026-11-24.1"   # MindFlow: + tab_only; table() AUC on raw scores

LOOKBACK = 78                       # bars in the drawn window (one session)
HISTORY = 78                        # extra bars before the window used by MA and EWSD (one more session)
BAND = 78                           # rows of the EWSD band under the chart
SIZE = PIXELS_PER_BAR * LOOKBACK    # 234
KINDS = {"dir": "dir", "vol": "vol", "xsdir": "xs_dir"}
KIND_LABEL = {"dir": "absolute direction", "vol": "volatility", "xsdir": "relative direction"}

VARIANTS = {   # name: (draw MA, draw EWSD band, hand + SPY features as second input)
    "plain": (False, False, False),
    "ma": (True, False, False),
    "ewsd": (False, True, False),
    "ma_ewsd": (True, True, False),
    "plain_tab": (False, False, True),
    "ma_ewsd_tab": (True, True, True),
    "tab_only": (False, False, True),          # control: the same feature MLP, no image at all
}
IMAGE_FREE = {"tab_only"}                      # variants whose model has no CNN
VARIANT_LABEL = {
    "plain": "CNN, original chart (this run)", "ma": "CNN, chart + MA", "ewsd": "CNN, chart + EWSD band",
    "ma_ewsd": "CNN, chart + MA + EWSD", "plain_tab": "CNN, original chart + hand/SPY features",
    "ma_ewsd_tab": "CNN, chart + MA + EWSD + hand/SPY features",
    "tab_only": "features only (MLP, no image)",
}
REFERENCE_MODELS = {"hgb_hand_spy": "HGB hand + SPY (ref)", "hgb_hand_features": "HGB hand features (ref)",
                    "cnn_plain": "image CNN (ref run)", "cnn_plain_spy": "image CNN + SPY panel (ref run)",
                    "cnn_series1d": "1D CNN (ref run)", "cnn_series1d_spy": "1D CNN + SPY (ref run)",
                    "cnn_vit": "image ViT (ref run)", "cnn_vit_spy": "image ViT + SPY panel (ref run)"}


def pretty(name):
    if name.startswith("ind_"):
        return VARIANT_LABEL.get(name[4:], name)
    return REFERENCE_MODELS.get(name, name)


# ----------------------------------------------------------------------------- indicators and drawing

def ew_sigma(close, halflife):
    """EW standard deviation of 1-bar log returns, one value per bar from the second bar on.
    Zero-mean (RiskMetrics) form, started at the mean squared return of the first session."""
    r = np.diff(np.log(np.asarray(close, dtype=np.float64)))
    a = 1.0 - 0.5 ** (1.0 / halflife)
    v0 = float(np.mean(r[:LOOKBACK] ** 2))
    var = lfilter([a], [1.0, -(1.0 - a)], r * r, zi=[(1.0 - a) * v0])[0]
    return np.sqrt(np.maximum(var, 1e-16))


def indicators(long_window, halflife):
    """(MA of the close over the previous 78 bars, EWSD) for each of the last 78 bars."""
    close = long_window[:, 3]
    c = np.cumsum(np.r_[0.0, close])
    ma = (c[LOOKBACK:] - c[:-LOOKBACK]) / LOOKBACK            # ma[j] = mean(close[j : j + 78])
    return ma[-LOOKBACK:], ew_sigma(close, halflife)[-LOOKBACK:]


def draw_chart(window, ma=None, include_volume=True):
    """Same geometry as chartcnn.images.render_window; with `ma` the price axis also spans the MA
    and the MA is drawn as a connected line through each bar's middle column.
    With ma=None the result is identical to render_window (tested in the smoke run)."""
    if ma is None:
        return render_window(window, include_volume)
    x = np.asarray(window, dtype=float)
    length = len(x)
    size = PIXELS_PER_BAR * length
    volume_height = max(1, size // 5) if include_volume else 0
    price_height = size - volume_height - (1 if include_volume else 0)
    image = np.zeros((size, size), dtype=np.uint8)
    low, high = min(float(x[:, 2].min()), float(ma.min())), max(float(x[:, 1].max()), float(ma.max()))

    def to_row(v):
        if high == low:
            return np.full(np.shape(v), price_height // 2, dtype=float)
        return price_height - 1 - (np.asarray(v, float) - low) / (high - low) * (price_height - 1)

    rows = np.rint(to_row(x[:, :4])).astype(int)
    max_volume = x[:, 4].max()
    for i, (o, h, l, c) in enumerate(rows):
        image[o, 3 * i] = 255
        image[h:l + 1, 3 * i + 1] = 255
        image[c, 3 * i + 2] = 255
        if include_volume:
            bar = int(np.rint(x[i, 4] / max_volume * volume_height))
            if bar:
                image[size - bar:, 3 * i + 1] = 180
    centres = 3 * np.arange(length) + 1
    cols = np.arange(centres[0], centres[-1] + 1)
    line = np.rint(np.interp(cols, centres, to_row(ma))).astype(int)
    prev = line[0]
    for c, r in zip(cols, line):                              # connect neighbouring columns
        a, b = min(prev, r), max(prev, r)
        image[a:b + 1, c] = 255
        prev = r
    return image


def draw_band(sigma, scale):
    """EWSD bars on a fixed log scale: log(sigma) = scale[0] -> 1 pixel, scale[1] -> BAND pixels."""
    band = np.zeros((BAND, SIZE), dtype=np.uint8)
    lo, hi = scale
    z = np.clip((np.log(sigma) - lo) / (hi - lo), 0.0, 1.0)
    heights = 1 + np.rint(z * (BAND - 1)).astype(int)
    for i, h in enumerate(heights):
        band[BAND - h:, 3 * i + 1] = 255
    return band


def render(long_window, variant, scale, halflife):
    """uint8 (234 + BAND, 234) picture of one window for one variant."""
    draw_ma, draw_sd, _ = VARIANTS[variant]
    window = long_window[-LOOKBACK:]
    ma, sigma = indicators(long_window, halflife)
    chart = draw_chart(window, ma if draw_ma else None)
    band = draw_band(sigma, scale) if draw_sd else np.zeros((BAND, SIZE), np.uint8)
    return np.vstack([chart, band])


# ----------------------------------------------------------------------------- rows

def sessions(store):
    return np.unique(store.dates)


def embargo_dates(store, n_sessions):
    """`n_sessions` whole sessions removed on each side of each split boundary. An input here spans
    two sessions (previous session + window), so n_sessions must be >= 2 for inputs never to overlap
    across a boundary."""
    days = sessions(store)
    out = set()
    for boundary in (store.config["train_end"], store.config["validation_end"]):
        k = np.searchsorted(days, boundary, side="right")
        out.update(days[max(0, k - n_sessions):k])
        out.update(days[k:k + n_sessions])
    return out


def decision_windows(store, split, stride, embargo, max_rows=None, every_kth_day=1, seed=69):
    """Stride-thinned windows of the bundle whose SPY window is complete (as in the reference run),
    whose previous session is complete too (needed by MA and EWSD), minus the embargo sessions."""
    f = store.windows(split)
    f = f[f.end_index % stride == stride - 1]
    s, t = f.security_index.to_numpy(np.int64), f.end_index.to_numpy(np.int64)
    start = t - LOOKBACK - HISTORY + 1
    ok = start >= 0
    ok[ok] = store.complete(s[ok], start[ok], t[ok] + 1)
    ok &= store.market_complete(t - LOOKBACK + 1, t + 1)
    f = f[ok].copy()
    f["decision_date"] = store.dates[f.end_index.to_numpy()]
    f = f[~f.decision_date.isin(embargo)]
    if every_kth_day > 1:
        f = f[f.decision_date.isin(np.sort(f.decision_date.unique())[::every_kth_day])]
    if max_rows is not None and len(f) > max_rows:
        f = f.sample(max_rows, random_state=seed)
    return f.sort_values(["end_index", "security_index"]).reset_index(drop=True)


def cell_rows(store, cell, split, windows, train_median=None):
    """Labelled rows of one cell restricted to `windows`; `row` = position in `windows`."""
    f, median = build_split(store, split, cell["target"], cell["horizon"], cell["stride"], train_median)
    key = windows[["security_index", "end_index"]].reset_index(drop=True).reset_index().rename(columns={"index": "row"})
    f = f.merge(key, on=["security_index", "end_index"], how="inner")
    return f.sort_values("row").reset_index(drop=True), median


# ----------------------------------------------------------------------------- preprocessing fitted on train

def fit_ewsd_scale(prepared, train_windows, halflife, n=20_000, q=(0.5, 99.5), seed=69):
    """Log-EWSD range of the band, from training windows only (every bar of `n` random windows)."""
    sample = train_windows.sample(min(n, len(train_windows)), random_state=seed).reset_index(drop=True)
    ds = Charts(prepared, sample, "plain", (0.0, 1.0), halflife)
    logs = np.concatenate([np.log(indicators(ds.long_window(i), halflife)[1]) for i in range(len(sample))])
    lo, hi = np.percentile(logs, q)
    return float(lo), float(hi)


def tab_features(store, windows):
    """The 38 numbers HGB + SPY uses (hand features of the stock + SPY features), per window."""
    return pd.concat([hand_features(store, windows), market_features(store, windows)], axis=1)


class TabScaler:
    """Mean/std from training rows only; NaN -> 0 after scaling; clipped to +-5."""

    def __init__(self, frame):
        x = frame.to_numpy(np.float64)
        self.columns = list(frame.columns)
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[~(self.std > 0)] = 1.0

    def __call__(self, frame):
        x = (frame[self.columns].to_numpy(np.float64) - self.mean) / self.std
        return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -5, 5).astype(np.float32)

    def state(self):
        return dict(columns=self.columns, mean=self.mean.tolist(), std=self.std.tolist())


# ----------------------------------------------------------------------------- dataset and model

class Charts(Dataset):
    """Renders one variant's picture on the fly; returns (image [1, 312, 234] in [0, 1], tab, label, i)."""

    def __init__(self, prepared, frame, variant, scale, halflife, tab=None):
        self.root = Path(prepared)
        self.s = frame.security_index.to_numpy(np.int64)
        self.t = frame.end_index.to_numpy(np.int64)
        self.y = frame.label.to_numpy(np.int64) if "label" in frame else np.zeros(len(frame), np.int64)
        self.variant, self.scale, self.halflife = variant, tuple(scale), halflife
        self.tab = np.zeros((len(frame), 0), np.float32) if tab is None else np.asarray(tab, np.float32)
        assert len(self.tab) == len(self.s)
        self._x = self._m = None

    def __len__(self):
        return len(self.s)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_x"] = state["_m"] = None
        return state

    def long_window(self, i):
        """(156, 5) OHLCV: previous session + window, all on the anchor bar's basis (point in time)."""
        if self._x is None:
            self._x = np.load(self.root / "features.npy", mmap_mode="r")
            self._m = np.load(self.root / "multipliers.npy", mmap_mode="r")
        s = self.s[i]
        return point_in_time_window(self._x[s], self._m[s], self.t[i], LOOKBACK + HISTORY)

    def image(self, i):
        return render(self.long_window(i), self.variant, self.scale, self.halflife)

    def __getitem__(self, i):
        if self.variant in IMAGE_FREE:          # nothing to draw: a 1x1 placeholder keeps the batch shape simple
            return torch.zeros(1, 1, 1), torch.from_numpy(self.tab[i]), int(self.y[i]), int(i)
        x = torch.from_numpy(self.image(i).astype(np.float32)[None] / 255.0)
        return x, torch.from_numpy(self.tab[i]), int(self.y[i]), int(i)


class IndicatorCNN(nn.Module):
    """The reference image CNN (chartcnn.ChartCNN, arch 'plain') on the taller canvas; with n_tab > 0
    a 2-layer MLP on the hand/SPY features adds its own logit to the CNN's (it starts at zero, so the
    model starts exactly where the CNN alone does)."""

    def __init__(self, settings, n_tab=0, image=True):
        super().__init__()
        assert image or n_tab, "a model needs the image, the features or both"
        self.cnn = (ChartCNN(channels=tuple(settings["channels"]), dropout=settings["dropout"],
                             head_height=settings["head_height"], arch="plain", input_hw=(SIZE + BAND, SIZE))
                    if image else None)
        self.input_shape = self.cnn.input_shape if image else None
        self.n_tab = n_tab
        if n_tab:
            h = settings["tab_hidden"]
            self.tab = nn.Sequential(nn.Linear(n_tab, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2))
            nn.init.zeros_(self.tab[-1].weight); nn.init.zeros_(self.tab[-1].bias)

    def forward(self, x, t=None):
        if self.cnn is None:
            return self.tab(t.to(x.device, non_blocking=True))
        logits = self.cnn(x)
        if self.n_tab:
            logits = logits + self.tab(t.to(logits.device, non_blocking=True))
        return logits


def count(model):
    total = sum(p.numel() for p in model.parameters())
    cnn = sum(p.numel() for p in model.cnn.parameters()) if model.cnn is not None else 0
    return dict(total=total, cnn=cnn, tab=total - cnn)


def make_loader(dataset, batch_size, workers, shuffle, seed=0):
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
              generator=torch.Generator().manual_seed(seed))
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, **kw)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out = np.empty(len(loader.dataset))
    for x, t, _, idx in loader:
        with autocast(device, amp):
            logits = model(x.to(device, non_blocking=True), t)
        out[idx.numpy()] = logits.float().softmax(1)[:, 1].cpu().numpy()
    return out


def _mirror(path, target_dir):
    if target_dir is None:
        return
    target_dir = Path(target_dir); target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target_dir / (Path(path).name + ".copying")
    shutil.copy2(path, tmp); os.replace(tmp, target_dir / Path(path).name)


def fit(model, train_loader, es_loader, y_es, settings, device, out_dir, drive_dir=None):
    """Adam, early stopping on log loss of the early-stopping days; same rule as the reference run.
    The CNN uses `learning_rate` (1e-5, as in the reference); the small tab MLP uses `tab_learning_rate`.
    Resumes from last.pt (local, else Drive); last.pt is deleted once the variant is finished."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    groups = [dict(params=model.cnn.parameters(), lr=settings["learning_rate"])] if model.cnn is not None else []
    if model.n_tab:
        groups.append(dict(params=model.tab.parameters(), lr=settings["tab_learning_rate"]))
    optimizer = torch.optim.Adam(groups, weight_decay=settings["weight_decay"])
    amp = settings["amp"]
    history, best, bad, first = [], np.inf, 0, 1
    for src in (out_dir, drive_dir):
        if src is not None and (Path(src) / "last.pt").exists():
            ck = torch.load(Path(src) / "last.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            history, best, bad, first = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
            if drive_dir is not None and (Path(drive_dir) / "best.pt").exists() and not (out_dir / "best.pt").exists():
                shutil.copy2(Path(drive_dir) / "best.pt", out_dir / "best.pt")
            print(f"  resuming after epoch {ck['epoch']}")
            if bad >= settings["patience"]:                   # it had already stopped early
                first = settings["epochs"] + 1
            break
    for epoch in range(first, settings["epochs"] + 1):
        train_loader.generator.manual_seed(settings["seed"] + epoch)
        model.train()
        seen = loss_sum = 0
        start = time.perf_counter()
        n_batches = len(train_loader)
        for b, (x, t, y, _) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device, amp):
                logits = model(x, t)
            loss = nn.functional.cross_entropy(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += len(y); loss_sum += loss.item() * len(y)
            if b % max(1, n_batches // 4) == 0:
                print(f"    batch {b}/{n_batches} loss {loss_sum / seen:.4f} "
                      f"{seen / (time.perf_counter() - start):.0f} img/s", flush=True)
        p = predict(model, es_loader, device, amp)
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_seconds=time.perf_counter() - start,
                   es_log_loss=log_loss(y_es, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
                   es_auc=roc_auc_score(y_es, p) if 0 < y_es.mean() < 1 else np.nan)
        history.append(row)
        print(f"  epoch {epoch}: train loss {row['train_loss']:.4f} | early-stop days: AUC {row['es_auc']:.4f}, "
              f"log loss {row['es_log_loss']:.5f} | {row['train_seconds'] / 60:.1f} min")
        if row["es_log_loss"] < best:
            torch.save(dict(model=model.state_dict(), epoch=epoch), out_dir / "best.pt")
            _mirror(out_dir / "best.pt", drive_dir)
        bad = 0 if row["es_log_loss"] < best - settings["min_delta"] else bad + 1
        best = min(best, row["es_log_loss"])
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), history=history, best=best,
                        bad=bad, epoch=epoch), out_dir / "last.pt")
        _mirror(out_dir / "last.pt", drive_dir)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        _mirror(out_dir / "history.csv", drive_dir)
        if bad >= settings["patience"]:
            print(f"  early stop: {bad} epochs without improvement")
            break
    for d in (out_dir, drive_dir):
        if d is not None and (Path(d) / "last.pt").exists():
            (Path(d) / "last.pt").unlink()
    return pd.DataFrame(history)


def run_cell(prepared, cell, data, variants, settings, device, local_dir, drive_dir=None):
    """Train every variant on one cell; save validation scores. Finished variants are skipped.

    data: train / es / valid frames (with `row`), tab_train / tab_valid arrays indexed by `row`,
    scale, prior."""
    local_dir, drive_dir = Path(local_dir), (Path(drive_dir) if drive_dir is not None else None)
    home = drive_dir or local_dir
    local_dir.mkdir(parents=True, exist_ok=True); home.mkdir(parents=True, exist_ok=True)
    if (home / "done.json").exists():
        print(f"{cell['name']}: already done, loading saved predictions")
        return load_cell(home)
    train, es, valid = data["train"], data["es"], data["valid"]
    print(f"{cell['name']} ({KIND_LABEL[cell['kind']]}, next {cell['minutes']} min): {len(train):,} train / "
          f"{len(valid):,} validation ({len(es):,} on early-stopping days); positive share "
          f"{train.label.mean():.3f} / {valid.label.mean():.3f}")
    keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
    out = valid[keep].copy()
    out["score_training_prior"] = data["prior"]
    info = dict(cell=cell, prior=data["prior"], parameters={}, best_epoch={}, minutes_trained={})
    w, bs, ebs = settings["num_workers"], settings["batch_size"], settings["eval_batch_size"]
    for v in variants:
        sub_local, sub_drive = local_dir / v, (drive_dir / v if drive_dir else None)
        sub_home = sub_drive or sub_local
        uses_tab = VARIANTS[v][2]

        def ds(frame, tab):
            return Charts(prepared, frame, v, data["scale"], settings["halflife"],
                          tab[frame.row.to_numpy()] if uses_tab else None)

        torch.manual_seed(settings["seed"]); np.random.seed(settings["seed"])
        model = IndicatorCNN(settings, data["tab_train"].shape[1] if uses_tab else 0, image=v not in IMAGE_FREE).to(device)
        n = count(model)
        info["parameters"][v] = n
        if (sub_home / "valid_prob.npy").exists() and (sub_home / "variant.json").exists():
            out[f"score_ind_{v}"] = np.load(sub_home / "valid_prob.npy")
            vi = json.loads((sub_home / "variant.json").read_text())
            info["best_epoch"][v], info["minutes_trained"][v] = vi["best_epoch"], vi["minutes"]
            print(f"  {v}: already trained, loaded")
            continue
        print(f"  --- {VARIANT_LABEL[v]} ({v}): {n['total']:,} parameters "
              f"(CNN {n['cnn']:,} + tab MLP {n['tab']:,}), all trainable")
        t0 = time.perf_counter()
        history = fit(model, make_loader(ds(train, data["tab_train"]), bs, w, True, settings["seed"]),
                      make_loader(ds(es, data["tab_valid"]), ebs, w, False), es.label.to_numpy(),
                      settings, device, sub_local, sub_drive)
        model.load_state_dict(torch.load(sub_local / "best.pt", map_location=device, weights_only=False)["model"])
        p = predict(model, make_loader(ds(valid, data["tab_valid"]), ebs, w, False), device, settings["amp"])
        out[f"score_ind_{v}"] = p
        best_epoch = int(history.loc[history.es_log_loss.idxmin(), "epoch"])
        minutes = (time.perf_counter() - t0) / 60
        info["best_epoch"][v], info["minutes_trained"][v] = best_epoch, minutes
        np.save(sub_local / "valid_prob.npy", p)
        (sub_local / "variant.json").write_text(json.dumps(dict(best_epoch=best_epoch, minutes=minutes, parameters=n)))
        for name in ("valid_prob.npy", "variant.json"):
            _mirror(sub_local / name, sub_drive)
        print(f"  {v}: validation AUC {roc_auc_score(valid.label, p):.4f}, best epoch {best_epoch}, {minutes:.1f} min")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out.to_parquet(local_dir / "predictions.parquet", index=False)
    (local_dir / "done.json").write_text(json.dumps(info, default=float))
    for name in ("predictions.parquet", "done.json"):
        _mirror(local_dir / name, drive_dir)
    return dict(info=info, predictions=out)


def load_cell(folder):
    folder = Path(folder)
    return dict(info=json.loads((folder / "done.json").read_text()),
                predictions=pd.read_parquet(folder / "predictions.parquet"))


# ----------------------------------------------------------------------------- comparison

def join_reference(pred, reference_cell_dir):
    """Add run_20261116_044159's scores for the same rows; rows it lacks are dropped (identical rows)."""
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
            v = n[4:] if n.startswith("ind_") else None
            rows.append(dict(cell=cname, kind=cell["kind"], minutes=cell["minutes"], model=pretty(n), raw=n,
                             roc_auc=_auc(y, s), auc_ci_low=lo, auc_ci_high=hi,
                             log_loss_gain_vs_prior_pct=100 * (base - log_loss(y, p, labels=[0, 1])) / base if is_prob else np.nan,
                             prob_std=float(p.std()) if is_prob else np.nan,
                             parameters=info["parameters"].get(v, {}).get("total", np.nan) if v else np.nan,
                             best_epoch=info["best_epoch"].get(v, np.nan) if v else np.nan,
                             rows=len(y), days=pred.decision_date.nunique()))
    return pd.DataFrame(rows).set_index(["cell", "model"]), boots


def paired(boots, a, b):
    """Per cell: AUC(a) - AUC(b) with a 95% paired day-bootstrap interval."""
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
    return t.pivot_table(index="kind", columns="minutes", values="roc_auc").round(4)


def plot(tab, models, path=None):
    """Validation AUC (95% CI) by horizon, one panel per target kind, one line per model."""
    import matplotlib.pyplot as plt
    t = tab.reset_index()
    kinds = [k for k in KINDS if k in set(t.kind)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.5 * len(kinds), 4.8), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        s = t[t.kind == kind]
        shown = [m for m in models if m in set(s.model)]
        for j, m in enumerate(shown):
            d = s[s.model == m].sort_values("minutes")
            x = d.minutes * (1 + 0.02 * (j - len(shown) / 2))
            ax.errorbar(x, d.roc_auc, yerr=[d.roc_auc - d.auc_ci_low, d.auc_ci_high - d.roc_auc], fmt="o",
                        capsize=3, label=m, ls="--" if "(ref" in m else "-")
        ax.set_xscale("log"); ax.minorticks_off()
        mins = sorted(s.minutes.unique())
        ax.set_xticks(mins, [f"{m} min" for m in mins])
        ax.set(title=f"{kind}: {KIND_LABEL[kind]}", xlabel="horizon", ylabel="validation AUC")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()


CHECKS = [("ind_ma", "ind_plain", "MA line vs original chart"),
          ("ind_ewsd", "ind_plain", "EWSD band vs original chart"),
          ("ind_ma_ewsd", "ind_plain", "MA + EWSD vs original chart"),
          ("ind_ma_ewsd", "ind_ewsd", "adding MA on top of EWSD"),
          ("ind_plain_tab", "ind_plain", "hand/SPY features added to the original chart CNN"),
          ("ind_ma_ewsd_tab", "ind_plain_tab", "indicators when hand/SPY features are already in"),
          ("ind_ma_ewsd", "hgb_hand_spy", "best chart-only CNN vs HGB hand + SPY"),
          ("ind_ma_ewsd_tab", "hgb_hand_spy", "CNN with everything vs HGB hand + SPY"),
          ("ind_plain_tab", "ind_tab_only", "image validity: original chart + features vs features only"),
          ("ind_ma_ewsd_tab", "ind_tab_only", "image validity: chart + MA + EWSD + features vs features only"),
          ("ind_tab_only", "hgb_hand_spy", "features-only MLP vs HGB on the same 38 features"),
          ("cnn_plain_spy", "cnn_plain", "SPY panel vs original chart (reference run)"),
          ("ind_plain", "cnn_plain", "control: this run's original-chart CNN vs the reference run's")]


def takeaways(tab, boots):
    lines = []
    t = tab.reset_index()
    for kind in [k for k in KINDS if k in set(t.kind)]:
        s = t[t.kind == kind]
        best = s.loc[s.groupby("cell").roc_auc.idxmax()].sort_values("minutes")
        lines.append(f"- **{kind}**: best model by horizon " + ", ".join(
            f"{r.minutes} min {r.roc_auc:.3f} ({r.model})" for _, r in best.iterrows()) + ".")
    for a, b, label in CHECKS:
        p = paired(boots, a, b)
        if len(p):
            v = p.verdict.value_counts()
            lines.append(f"- {label} ({pretty(a)} − {pretty(b)}), paired day bootstrap over {len(p)} cells: "
                         f"better in {v.get('better', 0)}, worse in {v.get('worse', 0)}, tied in {v.get('tied', 0)}; "
                         f"mean AUC difference {p['diff'].mean():+.4f}.")
    return "\n".join(lines)
