"""Samples and labels built from the prepared CNN78u bundle (`prepared/`).

The bundle already contains audited 78-bar input windows (`train/validation/test.parquet`).
Here we keep those windows, optionally thin them with a stride, and attach a new label:

    vol     1 if |r(t -> t+H)| > that stock's TRAINING-period median |r|     (volatility)
    xs_dir  1 if r(t -> t+H) > cross-sectional median at the same timestamp  (relative direction)
    dir     1 if r(t -> t+H) > 0, zero returns dropped                       (raw direction)

r is the return on unadjusted (traded) prices, as the price-adjustment policy in CLAUDE.md
requires for labels. A label whose horizon contains an ex-date (a split, stock dividend or cash
dividend between bar t and bar t+H, which can only happen across the overnight gap) is dropped:
its unadjusted return would contain the event's price jump, not a market move. This affects about
0.02% (H=1) to 0.2% (H=12) of windows.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .images import point_in_time_window, render_window

LOOKBACK = 78          # bars per image (one trading day of 5-minute bars)
TARGETS = ("vol", "xs_dir", "dir")


class Store:
    """In-memory arrays needed to build labels (close, volume, validity, calendar)."""

    def __init__(self, prepared):
        self.root = Path(prepared)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.config = self.manifest["config"]
        assert self.config["lookback"] == LOOKBACK and self.config["bar_minutes"] == 5

        # Stored prices are on the end-of-data basis. Only ratios between two bars are used from
        # these arrays (returns, labels, volume ratios), and those equal the point-in-time ratios.
        # Absolute levels must come from point_in_time_window (see images.prediction_basis).
        features = np.load(self.root / "features.npy", mmap_mode="r")   # (stocks, slots, OHLCV)
        self.close = np.array(features[:, :, 3], dtype=np.float64)
        self.volume = np.array(features[:, :, 4], dtype=np.float64)
        self.log_close = np.log(self.close)
        self.T = self.close.shape[1]
        # Per-bar price multiplier M[d] (stored = raw * M[d]). raw[t+H] / raw[t] =
        # stored[t+H] / stored[t] * M[t] / M[t+H]; M changes only on an ex-date.
        self.price_mult = np.array(np.load(self.root / "multipliers.npy", mmap_mode="r")[:, :, 0], dtype=np.float64)

        # bad_cumsum[s, j] = number of missing slots before slot j  ->  O(1) "is range complete?"
        valid = np.load(self.root / "valid.npy")
        self.bad_cumsum = np.concatenate(
            [np.zeros((valid.shape[0], 1), np.int32), np.cumsum(~valid, axis=1, dtype=np.int32)], axis=1)

        clock = pd.to_datetime(self.manifest["decision_clock_ns"], unit="ns", utc=True)
        self.dates = np.array(clock.tz_convert("America/New_York").strftime("%Y-%m-%d"))

        # Market (SPY) bars on the same clock, if the bundle has them (AllData bundle).
        self.has_market = (self.root / "market_features.npy").exists()
        if self.has_market:
            mf = np.load(self.root / "market_features.npy").astype(np.float64)       # (slots, OHLCV)
            self.market_close = mf[:, 3]
            self.market_log_close = np.log(self.market_close)
            self.market_volume = mf[:, 4]
            mvalid = np.load(self.root / "market_valid.npy")
            self.market_bad_cumsum = np.r_[0, np.cumsum(~mvalid, dtype=np.int32)]

    def complete(self, s, start, stop):
        """True where slots [start, stop) of stock s are all observed."""
        return (self.bad_cumsum[s, stop] - self.bad_cumsum[s, start]) == 0

    def market_complete(self, start, stop):
        """True where market slots [start, stop) are all observed (always True without SPY)."""
        if not self.has_market:
            return np.ones(np.shape(start), dtype=bool)
        return (self.market_bad_cumsum[stop] - self.market_bad_cumsum[start]) == 0

    def windows(self, split):
        cols = ["security_index", "permno", "end_index", "decision_time"]
        return pd.read_parquet(self.root / f"{split}.parquet", columns=cols)


def build_split(store, split, target, horizon, stride, train_abs_median=None):
    """Labelled samples for one split.

    Returns (frame, train_abs_median). For target='vol' pass the training split's
    median into validation/test so thresholds never use future data.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    f = store.windows(split)
    f = f[f.end_index % stride == stride - 1]                  # thin overlapping windows
    s, t, h = f.security_index.to_numpy(np.int64), f.end_index.to_numpy(np.int64), int(horizon)

    # The future horizon must be fully observed and must not leak into the next split.
    keep = t + h < store.T
    s, t, f = s[keep], t[keep], f[keep]
    end_dates = store.dates[t + h]
    boundary = {"train": store.config["train_end"], "validation": store.config["validation_end"]}
    keep = store.complete(s, t + 1, t + h + 1)
    # Same samples for every model: when the bundle has SPY, its 78 input bars must be complete
    # too, so stock-only and stock+SPY models are scored on identical rows.
    keep &= store.market_complete(t - LOOKBACK + 1, t + 1)
    if split in boundary:
        keep &= end_dates <= boundary[split]
    m0, m1 = store.price_mult[s, t], store.price_mult[s, t + h]
    keep &= np.abs(m1 / m0 - 1.0) < 1e-12               # no ex-date inside the label horizon
    f, s, t = f[keep].reset_index(drop=True), s[keep], t[keep]

    # Unadjusted (traded-price) return, per the policy. Equal to the stored ratio on kept rows.
    r = (store.close[s, t + h] / store.price_mult[s, t + h]) / (store.close[s, t] / store.price_mult[s, t]) - 1.0
    f["forward_return"] = r
    f["xs_return"] = r - f.groupby("end_index").forward_return.transform("median").to_numpy()
    f["decision_date"] = store.dates[t]

    if target == "dir":
        f = f[f.forward_return != 0]
        f["label"] = (f.forward_return > 0).astype("int64")
    elif target == "xs_dir":
        n_at_time = f.groupby("end_index").end_index.transform("size")
        f = f[(n_at_time >= 20) & (f.xs_return != 0)]
        f["label"] = (f.xs_return > 0).astype("int64")
    else:  # vol
        abs_r = f.forward_return.abs()
        if train_abs_median is None:
            train_abs_median = abs_r.groupby(f.security_index).median()
        threshold = f.security_index.map(train_abs_median)
        f = f[threshold.notna() & (abs_r != threshold)]
        f["label"] = (f.forward_return.abs() > f.security_index.map(train_abs_median)).astype("int64")
    return f.reset_index(drop=True), train_abs_median


def scale_like_image(window):
    """(L, 5) OHLCV -> (5, L) float32 with exactly the scaling the chart uses:
    prices mapped to [0, 1] by the window's lowest low and highest high, volume divided by the
    window's maximum volume. A 1D model therefore sees the same information as the image,
    minus the drawing (see DECISIONS: a 1D baseline must use the same per-window scaling)."""
    x = np.asarray(window, dtype=np.float64)
    low, high = x[:, 2].min(), x[:, 1].max()
    prices = (x[:, :4] - low) / (high - low) if high > low else np.full((len(x), 4), 0.5)
    volume = x[:, 4:5] / x[:, 4].max()
    return np.concatenate([prices, volume], axis=1).T.astype(np.float32)


class ChartDataset(Dataset):
    """Renders chart images on the fly from the memory-mapped bundle.

    market=True adds SPY as its own panel under the stock chart (468 x 234 image): the same 78
    bars drawn with the same geometry on SPY's own window-local axis, so each bar column of the
    SPY panel lines up with the stock's."""

    PANEL_FILES = {"spy": "market", "vixy": "vixy"}   # <prefix>_features.npy / _multipliers.npy

    def __init__(self, prepared, frame, include_volume=True, market=False, panels=None):
        self.root = Path(prepared)
        self.s = frame.security_index.to_numpy(np.int64)
        self.t = frame.end_index.to_numpy(np.int64)
        self.y = frame.label.to_numpy(np.int64) if "label" in frame else np.zeros(len(frame), np.int64)
        self.include_volume = include_volume
        self.panels = tuple(panels) if panels is not None else (("spy",) if market else ())
        self.market = bool(self.panels)
        for p in self.panels:
            assert (self.root / f"{self.PANEL_FILES[p]}_features.npy").exists(), \
                f"{p} panel files are not in {self.root}: add the {p} panel bundle first"
        self._x = self._m = None                            # opened lazily in each DataLoader worker
        self._px = {}

    def __len__(self):
        return len(self.s)

    def __getstate__(self):                 # do not pickle memmaps into workers
        state = self.__dict__.copy()
        state["_x"] = state["_m"] = None
        state["_px"] = {}
        return state

    def window(self, i):
        """The (78, 5) OHLCV window ending at the decision bar, adjusted to that bar (point in time)."""
        if self._x is None:
            self._x = np.load(self.root / "features.npy", mmap_mode="r")
            self._m = np.load(self.root / "multipliers.npy", mmap_mode="r")
        s, t = self.s[i], self.t[i]
        return point_in_time_window(self._x[s], self._m[s], t, LOOKBACK)

    def panel_window(self, i, panel="spy"):
        """A market series' (78, 5) window ending at the same slot, adjusted to that slot (point in time)."""
        if panel not in self._px:
            prefix = self.PANEL_FILES[panel]
            self._px[panel] = (np.load(self.root / f"{prefix}_features.npy", mmap_mode="r"),
                               np.load(self.root / f"{prefix}_multipliers.npy", mmap_mode="r"))
        x, m = self._px[panel]
        return point_in_time_window(x, m, self.t[i], LOOKBACK)

    def market_window(self, i):
        """SPY's (78, 5) window ending at the same slot, adjusted to that slot (point in time)."""
        return self.panel_window(i, "spy")

    def image(self, i):
        return render_window(self.window(i), self.include_volume)

    def images(self, i):
        """uint8 image: the stock chart (234 x 234), with one panel per market series stacked below."""
        image = self.image(i)
        if self.panels:
            image = np.vstack([image] + [render_window(self.panel_window(i, p), self.include_volume) for p in self.panels])
        return image

    def __getitem__(self, i):
        x = torch.from_numpy(self.images(i).astype(np.float32)[None] / 255.0)  # [1, H, 234] in [0, 1]
        return x, int(self.y[i]), int(i)


class SeriesDataset(ChartDataset):
    """Same samples and windows as ChartDataset, but returns the numbers instead of the
    picture: a (5, 78) tensor of open, high, low, close and volume (see scale_like_image)."""

    def __getitem__(self, i):
        x = scale_like_image(self.window(i))
        for p in self.panels:               # 5 more channels per market series, scaled the same way
            x = np.concatenate([x, scale_like_image(self.panel_window(i, p))])
        return torch.from_numpy(x), int(self.y[i]), int(i)


def make_dataset(prepared, frame, arch):
    """The dataset an architecture expects: chart images or OHLCV sequences, with or without SPY."""
    from .models import INPUT_KIND, PANELS
    cls = SeriesDataset if INPUT_KIND[arch] == "series" else ChartDataset
    return cls(prepared, frame, panels=PANELS[arch])
