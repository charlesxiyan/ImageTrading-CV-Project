"""Daily chart CNN in the original Jiang, Kelly & Xiu (2023) setup, on the AllData & AllModel stocks.

Question: the image signal JKX report is daily (images of 5 or 20 trading days, labels over the next
5 or 20 days). Our 5-minute charts show no direction signal. Is the signal there at the daily
frequency on the same stocks and years? If it is, the result reads as "works at low frequency, not
at high frequency"; if it is not, the same stocks show no signal at either frequency.

Data: daily bars are built from the same 5-minute bars as the intraday experiments
(alldata_bundle_v1.zip): open of the first bar, highest high, lowest low, close of the 16:00 bar, summed
volume. A session counts only if its 16:00 bar and at least `min_bars` of its 78 bars are present
(early-close sessions are therefore missing days, drawn as blank columns, as JKX do for missing data).

Point in time (CLAUDE.md policy): stored bars are on the end-of-data basis, raw * M[d]. Every input
window is divided by the anchor day's multipliers, so all bars are on the anchor's basis and nothing
after the anchor reaches the picture. Labels: the h-day return C[t+h] / C[t] - 1 from the stored ratio,
i.e. split- and dividend-adjusted (a total return, as CRSP `ret` in JKX). Over several days ex-dates
do fall inside the horizon, so the unadjusted price would turn a split or a dividend into a return.

Images and CNN follow JKX exactly:
  I5   32 x 15 pixels, 2 blocks (64, 128 channels)          fully connected fan-in 15,360
  I20  64 x 60 pixels, 3 blocks (64, 128, 256 channels)     fan-in 46,080, 708,866 parameters
  each day 3 pixels wide (open tick, high-low bar, close tick), a moving average over the image length,
  volume in the bottom fifth; block = Conv 5x3 -> BatchNorm -> LeakyReLU -> MaxPool 2x1, the I20 first
  layer with stride (3, 1), dilation (2, 1); dropout 0.5; Xavier init; Adam 1e-5, batch 128; early stop
  after 2 epochs without improvement on a random 30% of the training sample; 5 seeds averaged.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

__version__ = "2026-11-19.1"

SLOTS = 78
IMAGE_HEIGHT = {5: 32, 20: 64}
TARGETS = {"dir": "absolute direction (r > 0)", "xsdir": "relative direction (r > cross-sectional median)"}
HORIZON_LABEL = {1: "1 day", 5: "1 week", 20: "1 month"}


# ----------------------------------------------------------------------------- daily bars

class DailyStore:
    """Daily OHLCV built from the bundle's 5-minute bars, still on the end-of-data basis, with the
    day's multipliers (constant within a session: ex-dates take effect at the open)."""

    def __init__(self, prepared, min_bars=60):
        root = Path(prepared)
        man = json.loads((root / "manifest.json").read_text())
        self.config = man["config"]
        clock = pd.to_datetime(man["decision_clock_ns"], unit="ns", utc=True).tz_convert("America/New_York")
        stamps = np.array(clock.strftime("%Y-%m-%d"))
        assert len(stamps) % SLOTS == 0, "clock is not 78 slots per session"
        D = len(stamps) // SLOTS
        grid = stamps.reshape(D, SLOTS)
        assert (grid == grid[:, :1]).all(), "a session's slots carry different dates"
        self.dates = grid[:, 0]
        self.permnos = np.asarray(man.get("permnos", []))
        feats = np.load(root / "features.npy", mmap_mode="r")
        mult = np.load(root / "multipliers.npy", mmap_mode="r")
        valid = np.load(root / "valid.npy", mmap_mode="r")
        S = feats.shape[0]
        self.S, self.D = S, D
        self.ohlcv = np.full((S, D, 5), np.nan)
        self.mult = np.full((S, D, 2), np.nan)
        self.valid = np.zeros((S, D), bool)
        rows = np.arange(D)
        for s in range(S):
            v = np.asarray(valid[s]).reshape(D, SLOTS)
            x = np.asarray(feats[s], np.float64).reshape(D, SLOTS, 5)
            m = np.asarray(mult[s], np.float64).reshape(D, SLOTS, 2)
            ok = (v.sum(1) >= min_bars) & v[:, -1]
            first = v.argmax(1)
            last = SLOTS - 1 - v[:, ::-1].argmax(1)
            o = x[rows, first, 0]
            h = np.where(v, x[:, :, 1], -np.inf).max(1)
            lo = np.where(v, x[:, :, 2], np.inf).min(1)
            c = x[rows, last, 3]
            vol = np.where(v, x[:, :, 4], 0.0).sum(1)
            day = np.stack([o, h, lo, c, vol], 1)
            ok &= np.isfinite(day).all(1) & (day[:, :4] > 0).all(1) & (vol > 0)
            self.ohlcv[s, ok] = day[ok]
            self.mult[s, ok] = m[rows, last][ok]
            self.valid[s] = ok
        self.log_close = np.log(self.ohlcv[:, :, 3])

    def day_index(self, date):
        """Index of the last session on or before `date`."""
        return int(np.searchsorted(self.dates, date, side="right")) - 1


# ----------------------------------------------------------------------------- images

def long_window(store, s, t, L):
    """(2L - 1, 5) OHLCV ending at day t, on day t's basis (point in time); missing days are NaN.
    The first L - 1 days only feed the moving average."""
    a = t - 2 * L + 2
    x = store.ohlcv[s, a:t + 1].copy()
    price, volume = store.mult[s, t]
    x[:, :4] /= price
    x[:, 4] /= volume
    return x


def draw(window, ma, height):
    """JKX chart: 3 pixels per day, price scaled to the window's (and MA's) low/high, volume in the
    bottom fifth, MA as a connected line. Missing days (NaN rows) stay black."""
    L = len(window)
    width = 3 * L
    vol_h = height // 5
    price_h = height - vol_h - 1
    img = np.zeros((height, width), np.uint8)
    ok = np.isfinite(window).all(1)
    mok = np.isfinite(ma)
    lo = min(window[ok, 2].min(), ma[mok].min() if mok.any() else np.inf)
    hi = max(window[ok, 1].max(), ma[mok].max() if mok.any() else -np.inf)

    def row(v):
        if hi == lo:
            return np.full(np.shape(v), price_h // 2, float)
        return price_h - 1 - (np.asarray(v, float) - lo) / (hi - lo) * (price_h - 1)

    vmax = window[ok, 4].max()
    for i in np.flatnonzero(ok):
        o, h, l, c = np.rint(row(window[i, :4])).astype(int)
        img[o, 3 * i] = 255
        img[h:l + 1, 3 * i + 1] = 255
        img[c, 3 * i + 2] = 255
        bar = int(np.rint(window[i, 4] / vmax * vol_h))
        if bar:
            img[height - bar:, 3 * i + 1] = 255
    idx = np.flatnonzero(mok)
    if len(idx) >= 2:
        centres = 3 * idx + 1
        cols = np.arange(centres[0], centres[-1] + 1)
        line = np.rint(np.interp(cols, centres, row(ma[idx]))).astype(int)
        prev = line[0]
        for c, r in zip(cols, line):
            a, b = min(prev, r), max(prev, r)
            img[a:b + 1, c] = 255
            prev = r
    return img


def image(store, s, t, L):
    x = long_window(store, s, t, L)
    close = x[:, 3]
    ma = np.full(L, np.nan)
    for j in range(L):                        # MA over the L days ending at window day j
        seg = close[j:j + L]
        if np.isfinite(seg).sum() >= (L + 1) // 2:
            ma[j] = np.nanmean(seg)
    return draw(x[-L:], ma, IMAGE_HEIGHT[L])


def anchors(store, L, stride, min_share=0.8):
    """Every (stock, day) whose day index is stride - 1 mod stride (same days for every stock), with
    the anchor day present, at least `min_share` of the L input days present and room for the MA."""
    t = np.arange(2 * L - 2, store.D)
    t = t[t % stride == stride - 1]
    cnt = np.concatenate([np.zeros((store.S, 1), int), np.cumsum(store.valid, 1)], 1)
    rows = []
    for s in range(store.S):
        n_in = cnt[s, t + 1] - cnt[s, t + 1 - L]
        keep = store.valid[s, t] & (n_in >= math.ceil(min_share * L))
        rows.append(pd.DataFrame(dict(security_index=s, t=t[keep])))
    f = pd.concat(rows, ignore_index=True)
    f["date"] = store.dates[f.t.to_numpy()]
    return f.sort_values(["t", "security_index"]).reset_index(drop=True)


def render_all(store, frame, L, log_every=50_000):
    out = np.empty((len(frame), IMAGE_HEIGHT[L], 3 * L), np.uint8)
    s, t = frame.security_index.to_numpy(), frame.t.to_numpy()
    t0 = time.perf_counter()
    for i in range(len(frame)):
        out[i] = image(store, s[i], t[i], L)
        if log_every and (i + 1) % log_every == 0:
            print(f"  rendered {i + 1:,}/{len(frame):,} ({(i + 1) / (time.perf_counter() - t0):.0f}/s)", flush=True)
    return out


# ----------------------------------------------------------------------------- labels and splits

def labelled(store, frame, h, target, eval_stride=None):
    """Attach the h-day forward return and the label; split by date with purge and embargo.

    train      the whole label horizon ends on or before train_end (purged at the boundary)
    validation anchor more than h sessions after train_end, label ends on or before validation_end
    test       anchor more than h sessions after validation_end
    eval_stride keeps only anchors on every eval_stride-th day in validation/test (non-overlapping
    labels for evaluation); training keeps every anchor in `frame`."""
    s, t = frame.security_index.to_numpy(), frame.t.to_numpy()
    end = t + h
    ok = end < store.D
    ok[ok] &= store.valid[s[ok], end[ok]]
    f = frame[ok].copy()
    s, t, end = s[ok], t[ok], end[ok]
    f["forward_return"] = np.exp(store.log_close[s, end] - store.log_close[s, t]) - 1.0
    i_tr, i_va = store.day_index(store.config["train_end"]), store.day_index(store.config["validation_end"])
    split = np.full(len(f), "", object)
    split[end <= i_tr] = "train"
    split[(t > i_tr + h) & (end <= i_va)] = "validation"
    split[t > i_va + h] = "test"
    f["split"] = split
    f = f[f.split != ""]
    if eval_stride:
        f = f[(f.split == "train") | (f.t % eval_stride == eval_stride - 1)]
    if target == "dir":
        f = f[f.forward_return != 0].copy()
        f["label"] = (f.forward_return > 0).astype(np.int64)
    else:
        n = f.groupby("t").t.transform("size")
        med = f.groupby("t").forward_return.transform("median")
        f = f[(n >= 20) & (f.forward_return != med)].copy()
        f["label"] = (f.forward_return > med[f.index]).astype(np.int64)
    return f


# ----------------------------------------------------------------------------- baselines

def hand_features(store, frame):
    """Past returns, volatility, volume and range from daily bars up to day t (no future data)."""
    s, t = frame.security_index.to_numpy(), frame.t.to_numpy()
    lc = store.log_close
    o = store.ohlcv

    def at(k):                                  # value k days back, NaN before the first day
        tk = t - k
        return np.where(tk >= 0, lc[s, np.maximum(tk, 0)], np.nan)

    cols = {f"r{k}": at(k - 1) - at(k) for k in range(1, 6)}
    for w in (5, 20, 60):
        cols[f"cum{w}"] = at(0) - at(w)
    rets = np.stack([at(k - 1) - at(k) for k in range(1, 21)])
    cols["vol20"] = np.nanstd(rets, 0)
    vol = np.stack([np.where(t - k >= 0, o[s, np.maximum(t - k, 0), 4], np.nan) for k in range(1, 21)])
    cols["rel_volume"] = np.log(o[s, t, 4] / np.nanmean(vol, 0))
    rng = np.stack([np.where(t - k >= 0, np.log(o[s, np.maximum(t - k, 0), 1] / o[s, np.maximum(t - k, 0), 2]), np.nan)
                    for k in range(0, 20)])
    cols["range20"] = np.nanmean(rng, 0)
    cols["close_pos"] = (o[s, t, 3] - o[s, t, 2]) / np.maximum(o[s, t, 1] - o[s, t, 2], 1e-12)
    return pd.DataFrame(cols, index=frame.index)


RULES = {"rule_str1": ("r1", -1), "rule_str5": ("cum5", -1), "rule_mom20": ("cum20", 1), "rule_mom60": ("cum60", 1)}


def fit_hgb(train_x, train_y, seed, max_train=1_000_000):
    idx = np.arange(len(train_y))
    if len(idx) > max_train:
        idx = np.random.default_rng(seed).choice(idx, max_train, replace=False)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                       l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                       n_iter_no_change=20, random_state=seed)
    return m.fit(train_x.iloc[idx], train_y[idx])


def tree_nodes(model):
    return int(sum(p.nodes.shape[0] for it in model._predictors for p in it))


# ----------------------------------------------------------------------------- CNN

class JKXCNN(nn.Module):
    def __init__(self, L, dropout=0.5):
        super().__init__()
        H, W = IMAGE_HEIGHT[L], 3 * L
        chans = {5: (64, 128), 20: (64, 128, 256)}[L]
        layers, c_in = [], 1
        for i, c in enumerate(chans):
            if i == 0 and L == 20:
                conv = nn.Conv2d(c_in, c, (5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1))
            else:
                conv = nn.Conv2d(c_in, c, (5, 3), padding=(2, 1))
            layers += [conv, nn.BatchNorm2d(c), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
            c_in = c
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            self.flat = int(self.features(torch.zeros(1, 1, H, W)).numel())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(self.flat, 2))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))


def count(model):
    return sum(p.numel() for p in model.parameters())


def _batches(n, size, gen=None):
    order = torch.randperm(n, generator=gen) if gen is not None else torch.arange(n)
    for i in range(0, n, size):
        yield order[i:i + size]


def _x(images, idx, device):
    return images[idx.to(images.device)].to(device, non_blocking=True).float().div_(255.0).unsqueeze(1)


@torch.no_grad()
def predict(model, images, rows, device, batch=4096):
    """P(up) for image rows `rows` (a LongTensor of positions in `images`)."""
    model.eval()
    out = []
    for b in _batches(len(rows), batch):
        out.append(model(_x(images, rows[b], device)).float().softmax(1)[:, 1].cpu())
    return torch.cat(out).numpy() if out else np.empty(0)


def fit_cnn(L, images, fit_rows, fit_y, es_rows, es_y, settings, device, seed, log=print):
    """One JKX CNN: Adam, cross-entropy, early stopping on log loss of the held-out 30%."""
    torch.manual_seed(seed)
    model = JKXCNN(L, settings["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    gen = torch.Generator().manual_seed(seed)
    y_fit = torch.as_tensor(fit_y, dtype=torch.long)
    best, bad, best_state, history = np.inf, 0, None, []
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        t0, seen, loss_sum = time.perf_counter(), 0, 0.0
        for b in _batches(len(fit_rows), settings["batch_size"], gen):
            x = _x(images, fit_rows[b], device)
            y = y_fit[b].to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            seen += len(b); loss_sum += loss.item() * len(b)
        p = predict(model, images, es_rows, device)
        es_loss = log_loss(es_y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
        history.append(dict(epoch=epoch, train_loss=loss_sum / seen, es_log_loss=es_loss,
                            es_auc=roc_auc_score(es_y, p) if 0 < es_y.mean() < 1 else np.nan,
                            seconds=time.perf_counter() - t0))
        log(f"    seed {seed} epoch {epoch}: train loss {loss_sum / seen:.4f} | held-out log loss {es_loss:.5f} "
            f"AUC {history[-1]['es_auc']:.4f} | {history[-1]['seconds']:.0f}s")
        if es_loss < best - settings["min_delta"]:
            best, bad = es_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def run_cell(store, cell, frame, images, settings, device, out_dir, log=print):
    """One (image length, horizon, target) cell: CNN ensemble + HGB + rules on validation and test.

    frame: anchors of this image length (rows of `images`); images: uint8 tensor (N, H, W)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "done.json").exists():
        log(f"{cell['name']}: already done, loading")
        return load_cell(out_dir)
    L, h, target = cell["image_days"], cell["horizon"], cell["target"]
    f = labelled(store, frame.assign(row=np.arange(len(frame))), h, target, eval_stride=h)
    tr = f[f.split == "train"]
    if settings.get("max_train") and len(tr) > settings["max_train"]:
        tr = tr.sample(settings["max_train"], random_state=settings["seed"])
    ev = f[f.split != "train"].reset_index(drop=True)
    assert len(tr) and (ev.split == "validation").any(), f"{cell['name']}: no training or validation rows"
    log(f"{cell['name']}: train {len(tr):,} ({tr.date.min()} – {tr.date.max()}) | validation "
        f"{(ev.split == 'validation').sum():,} on {ev[ev.split == 'validation'].date.nunique()} dates | test "
        f"{(ev.split == 'test').sum():,} on {ev[ev.split == 'test'].date.nunique()} dates | up share {tr.label.mean():.3f}")
    prior = float(tr.label.mean())
    out = ev[["security_index", "t", "date", "split", "forward_return", "label"]].copy()
    out["score_training_prior"] = prior

    # Rules and HGB on hand features
    X_tr, X_ev = hand_features(store, tr), hand_features(store, ev)
    for name, (col, sign) in RULES.items():
        out[f"score_{name}"] = sign * X_ev[col].to_numpy()
    hgb = fit_hgb(X_tr, tr.label.to_numpy(), settings["seed"])
    out["score_hgb"] = hgb.predict_proba(X_ev)[:, 1]
    params = dict(hgb=tree_nodes(hgb))

    # CNN: random 70/30 split of the training sample (JKX), several seeds averaged
    rng = np.random.default_rng(settings["seed"])
    is_es = rng.random(len(tr)) < settings["holdout_share"]
    rows_fit = torch.as_tensor(tr.row.to_numpy()[~is_es].copy()); rows_es = torch.as_tensor(tr.row.to_numpy()[is_es].copy())
    y_fit, y_es = tr.label.to_numpy()[~is_es], tr.label.to_numpy()[is_es]
    rows_ev = torch.as_tensor(ev.row.to_numpy().copy())
    probs, histories, epochs = [], [], []
    t0 = time.perf_counter()
    for k in range(settings["seeds"]):
        seed = settings["seed"] + k
        model, hist = fit_cnn(L, images, rows_fit, y_fit, rows_es, y_es, settings, device, seed, log)
        if k == 0:
            params["cnn"] = count(model)
            log(f"  JKX CNN I{L}: {params['cnn']:,} parameters (fully connected fan-in {model.flat:,})")
        p = predict(model, images, rows_ev, device)
        out[f"score_cnn_seed{k}"] = p
        probs.append(p)
        hist["seed"] = seed; histories.append(hist)
        epochs.append(int(hist.loc[hist.es_log_loss.idxmin(), "epoch"]))
        torch.save(model.state_dict(), out_dir / f"cnn_seed{k}.pt")
        del model
    out["score_cnn"] = np.mean(probs, 0)
    pd.concat(histories).to_csv(out_dir / "cnn_history.csv", index=False)
    info = dict(cell=cell, prior=prior, parameters=params, best_epochs=epochs, train_rows=int(len(tr)),
                cnn_minutes=(time.perf_counter() - t0) / 60, version=__version__)
    out.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "done.json").write_text(json.dumps(info, default=float))
    return dict(info=info, predictions=out)


def load_cell(folder):
    folder = Path(folder)
    return dict(info=json.loads((folder / "done.json").read_text()),
                predictions=pd.read_parquet(folder / "predictions.parquet"))


# ----------------------------------------------------------------------------- evaluation

MODEL_LABEL = {"cnn": "JKX CNN (ensemble)", "hgb": "HGB hand features", "rule_str1": "rule: 1-day reversal",
               "rule_str5": "rule: 1-week reversal", "rule_mom20": "rule: 1-month momentum",
               "rule_mom60": "rule: 3-month momentum"}


def _auc(y, p):
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    return roc_auc_score(y, p) if len(y) and 0 < y.mean() < 1 else np.nan


def long_short(pred, score, n_groups=10):
    """Equal-weighted top-minus-bottom decile return per date (holding h days)."""
    rows = []
    for d, g in pred.groupby("date"):
        g = g[np.isfinite(g[score])]
        if len(g) < 5 * n_groups:
            continue
        q = pd.qcut(g[score].rank(method="first"), n_groups, labels=False)
        rows.append(g.forward_return[q == n_groups - 1].mean() - g.forward_return[q == 0].mean())
    return np.asarray(rows)


def evaluate(results, n_boot=300, seed=0):
    """Per cell x split x model: AUC (95% date bootstrap), accuracy, log-loss gain, long-short."""
    rng = np.random.default_rng(seed)
    rows = []
    for cname, r in results.items():
        pred, cell = r["predictions"], r["info"]["cell"]
        prior = r["info"]["prior"]
        names = [c[6:] for c in pred.columns if c.startswith("score_") and c != "score_training_prior"
                 and "_seed" not in c]
        for split in ("validation", "test"):
            p = pred[pred.split == split]
            if p.empty:
                continue
            y = p.label.to_numpy()
            codes, _ = pd.factorize(p.date)
            groups = [np.flatnonzero(codes == i) for i in range(codes.max() + 1)]
            boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]
            base = log_loss(y, np.full(len(y), prior), labels=[0, 1])
            for n in names:
                sc = p[f"score_{n}"].to_numpy(float)
                draws = [_auc(y[b], sc[b]) for b in boots]
                lo, hi = np.nanpercentile(draws, [2.5, 97.5])
                row = dict(cell=cname, image=f"I{cell['image_days']}", horizon_days=cell["horizon"],
                           target=cell["target"], split=split, model=MODEL_LABEL.get(n, n), raw=n,
                           auc=_auc(y, sc), auc_low=lo, auc_high=hi, rows=len(y), dates=len(groups))
                if n in ("cnn", "hgb"):
                    pr = np.clip(sc, 1e-7, 1 - 1e-7)
                    row["accuracy"] = float(((pr > 0.5) == y).mean())
                    row["log_loss_gain_pct"] = 100 * (base - log_loss(y, pr, labels=[0, 1])) / base
                ls = long_short(p, f"score_{n}")
                if len(ls) > 1:
                    row["ls_mean_bps"] = 1e4 * ls.mean()
                    row["ls_t"] = ls.mean() / ls.std(ddof=1) * math.sqrt(len(ls))
                    row["ls_sharpe_ann"] = ls.mean() / ls.std(ddof=1) * math.sqrt(252 / cell["horizon"])
                rows.append(row)
    return pd.DataFrame(rows)


def intraday_reference(ref_dir, kinds=("dir", "xsdir"), horizons=(1, 3, 6, 12), models=("cnn_plain", "hgb_hand_spy"),
                       n_boot=300, seed=0):
    """Validation-2023 AUCs of the 5-minute run (run_20261116_044159) for the same targets."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in kinds:
        for h in horizons:
            path = Path(ref_dir) / f"{k}_h{h}" / "predictions.parquet"
            if not path.exists():
                continue
            p = pd.read_parquet(path)
            y = p.label.to_numpy()
            codes, _ = pd.factorize(p.decision_date)
            groups = [np.flatnonzero(codes == i) for i in range(codes.max() + 1)]
            boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]
            for m in models:
                col = f"score_{m}"
                if col not in p:
                    continue
                sc = p[col].to_numpy(float)
                lo, hi = np.nanpercentile([_auc(y[b], sc[b]) for b in boots], [2.5, 97.5])
                rows.append(dict(target=k, horizon=f"{5 * h} min", minutes=5 * h,
                                 model="CNN (5-min chart)" if m.startswith("cnn") else "HGB (5-min features + SPY)",
                                 auc=_auc(y, sc), auc_low=lo, auc_high=hi, rows=len(y)))
    return pd.DataFrame(rows)


def frequency_plot(daily, intraday, split="validation", path=None):
    """AUC from 5 minutes to 1 month, one panel per target: the frequency question in one figure."""
    import matplotlib.pyplot as plt
    targets = [t for t in TARGETS if t in set(daily.target)]
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 4.6), squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        ticks, labels = [], []
        pos = 0
        if len(intraday):
            it = intraday[intraday.target == tgt]
            mins = sorted(it.minutes.unique())
            for model, mk in (("CNN (5-min chart)", "o"), ("HGB (5-min features + SPY)", "s")):
                d = it[it.model == model].set_index("minutes").reindex(mins)
                x = np.arange(len(mins)) + (0.1 if mk == "s" else -0.1)
                ax.errorbar(x, d.auc, yerr=[d.auc - d.auc_low, d.auc_high - d.auc], fmt=mk, capsize=3,
                            color="tab:blue" if mk == "o" else "tab:orange", label=model + ", 2023")
            ticks += list(range(len(mins))); labels += [f"{m} min" for m in mins]
            pos = len(mins) + 0.5
            ax.axvline(pos - 0.5, color="gray", lw=0.8)
        dd = daily[(daily.target == tgt) & (daily.split == split)]
        combos = sorted(map(tuple, dd[["image", "horizon_days"]].drop_duplicates().values),
                        key=lambda c: (c[1], int(c[0][1:])))
        for j, (img, hd) in enumerate(combos):
            x0 = pos + j
            for raw, mk, color, dx in (("cnn", "o", "tab:blue", -0.1), ("hgb", "s", "tab:orange", 0.1)):
                r = dd[(dd.image == img) & (dd.horizon_days == hd) & (dd.raw == raw)]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar([x0 + dx], [r.auc], yerr=[[r.auc - r.auc_low], [r.auc_high - r.auc]], fmt=mk,
                                capsize=3, color=color, mfc="white")
            ticks.append(x0); labels.append(f"{img}/R{hd}\n({HORIZON_LABEL.get(hd, f'{hd} d')})")
        ax.plot([], [], "o", mfc="white", color="tab:blue", label=f"JKX CNN daily, {split}")
        ax.plot([], [], "s", mfc="white", color="tab:orange", label=f"HGB daily, {split}")
        ax.axhline(0.5, ls=":", color="gray")
        ax.set_xticks(ticks, labels, fontsize=8)
        ax.set(title=f"{tgt}: {TARGETS[tgt]}", ylabel="AUC")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()


def takeaways(table):
    lines = []
    for split in ("validation", "test"):
        t = table[(table.split == split) & table.raw.isin(["cnn", "hgb"])]
        if t.empty:
            continue
        lines.append(f"**{split}**")
        for _, r in t.sort_values(["target", "horizon_days", "image", "raw"]).iterrows():
            sig = "above 0.5" if r.auc_low > 0.5 else "below 0.5" if r.auc_high < 0.5 else "not different from 0.5"
            ls = (f"; long-short {r.ls_mean_bps:+.1f} bps per {r.horizon_days} d (t {r.ls_t:+.2f}, Sharpe {r.ls_sharpe_ann:+.2f})"
                  if np.isfinite(r.get("ls_mean_bps", np.nan)) else "")
            lines.append(f"- {r.target} {r.image}/R{r.horizon_days}, {r.model}: AUC {r.auc:.3f} "
                         f"[{r.auc_low:.3f}, {r.auc_high:.3f}] ({sig}){ls}.")
    return "\n".join(lines)
