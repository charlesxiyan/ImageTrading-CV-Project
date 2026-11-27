"""Step 4b: which of the 38 hand + SPY features carry the volatility signal?

All choices are made on a *selection year* that is not the validation year:

    inner train   2016 ... (train_end - 1 year)   HGB fitted here during selection
    selection     the last training year (2022)    permutation importance, group ablation,
                                                   forward selection are scored here
    validation    2023                             the chosen subsets are confirmed here, once,
                                                   each next to the full 38-feature model
                                                   (paired day bootstrap)

The test years are not used. The vol threshold (each stock's median |r|) is recomputed from the
inner-train rows during selection, so the selection year's labels do not use its own returns either.

Analyses
  1. permutation importance  AUC drop on the selection year when one feature is shuffled (n_repeats)
  2. feature groups          retrain without each group (drop) and with that group only (keep-only)
  3. forward selection       greedy: add the feature that raises selection AUC most, stop when the
                             subset is within `tolerance` AUC of the full model (or max_features)
  4. confirmation            full / forward-selected / group models refit on all of 2016-2022 and
                             scored on 2023 with paired bootstrap intervals against the full model
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from chartcnn import build_split
from chartcnn.baselines import hand_features, market_features, tree_size

__version__ = "2026-11-23.1"

GROUPS = {
    "stock returns": [f"r{k}" for k in range(1, 13)] + [f"cum{w}" for w in (6, 12, 39, 77)],
    "stock volatility": ["realized_vol", "mean_abs_r12"],
    "stock volume and time": ["rel_volume", "time_of_day"],
    "SPY": [f"spy_r{k}" for k in (1, 2, 3, 6, 12)] + [f"spy_cum{w}" for w in (6, 12, 39, 77)]
           + ["spy_realized_vol", "spy_mean_abs_r12", "spy_rel_volume"],
    "stock vs SPY": ["beta77", "corr77", "idio_vol77", "vol_ratio77", "rel_cum12", "rel_cum77"],
}


def features(store, frame):
    return pd.concat([hand_features(store, frame), market_features(store, frame)], axis=1)


def group_of(col):
    return next((g for g, cols in GROUPS.items() if col in cols), "other")


def hgb(seed, **kw):
    p = dict(max_iter=300, learning_rate=0.05, min_samples_leaf=500, random_state=seed)   # as the reference HGB
    p.update(kw)
    return HistGradientBoostingClassifier(**p)


def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def relabel_vol(inner, other):
    """Vol labels of both frames with each stock's median |r| over the inner-train rows only."""
    med = inner.forward_return.abs().groupby(inner.security_index).median()
    out = []
    for f in (inner, other):
        thr = f.security_index.map(med)
        f = f[thr.notna() & (f.forward_return.abs() != thr)].copy()
        f["label"] = (f.forward_return.abs() > f.security_index.map(med)).astype("int64")
        out.append(f.reset_index(drop=True))
    return out


def build_data(store, cell, settings):
    """Frames and feature matrices for selection (inner / selection) and confirmation (train / valid)."""
    train, median = build_split(store, "train", cell["target"], cell["horizon"], cell["stride"])
    valid, _ = build_split(store, "validation", cell["target"], cell["horizon"], cell["stride"], median)
    sel_start = settings["selection_start"]                     # first day of the selection year
    gap = np.sort(train.decision_date.unique())
    # purge: inner-train labels must end before the selection year starts (+ embargo sessions)
    k = np.searchsorted(gap, sel_start)
    embargo = set(gap[max(0, k - settings["embargo_sessions"]):k + settings["embargo_sessions"]])
    inner = train[(train.decision_date < sel_start) & ~train.decision_date.isin(embargo)]
    inner = inner[store.dates[inner.end_index.to_numpy() + cell["horizon"]] < sel_start]
    sel = train[(train.decision_date >= sel_start) & ~train.decision_date.isin(embargo)]
    if cell["target"] == "vol":
        inner, sel = relabel_vol(inner, sel)
    rs = settings["seed"]
    inner = inner.sample(min(settings["max_inner"], len(inner)), random_state=rs).reset_index(drop=True)
    sel = sel.sample(min(settings["max_selection"], len(sel)), random_state=rs).reset_index(drop=True)
    train = train.sample(min(settings["max_train"], len(train)), random_state=rs).reset_index(drop=True)
    d = dict(inner=inner, sel=sel, train=train, valid=valid)
    X = {k: features(store, v) for k, v in d.items()}
    print(f"{cell['name']}: inner train {len(inner):,} ({inner.decision_date.min()} – {inner.decision_date.max()}) | "
          f"selection {len(sel):,} ({sel.decision_date.min()} – {sel.decision_date.max()}) | "
          f"train {len(train):,} | validation {len(valid):,}; {X['inner'].shape[1]} features")
    return d, X


def permutation_importance(model, X, y, n_repeats, seed):
    rng = np.random.default_rng(seed)
    base = _auc(y, model.predict_proba(X)[:, 1])
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - _auc(y, model.predict_proba(Xp)[:, 1]))
        rows.append(dict(feature=col, group=group_of(col), auc_drop=float(np.mean(drops)),
                         auc_drop_std=float(np.std(drops))))
    return base, pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


def fit_score(cols, Xa, ya, Xb, yb, seed):
    m = hgb(seed).fit(Xa[cols], ya)
    return m, _auc(yb, m.predict_proba(Xb[cols])[:, 1])


def group_ablation(Xa, ya, Xb, yb, seed, full_auc):
    rows = []
    cols = list(Xa.columns)
    for g, gcols in GROUPS.items():
        gcols = [c for c in gcols if c in cols]
        _, drop = fit_score([c for c in cols if c not in gcols], Xa, ya, Xb, yb, seed)
        _, only = fit_score(gcols, Xa, ya, Xb, yb, seed)
        rows.append(dict(group=g, n_features=len(gcols), auc_without=drop, change_without=drop - full_auc,
                         auc_only=only))
        print(f"  {g:22s} ({len(gcols):2d}): without {drop:.4f} ({drop - full_auc:+.4f}) | only {only:.4f}")
    return pd.DataFrame(rows)


def forward_selection(Xa, ya, Xb, yb, seed, full_auc, tolerance=0.002, max_features=15, candidates=None, log=print):
    """Greedy forward selection on the selection year. `candidates` (default: all) can be pre-ranked
    and cut to the top-k by permutation importance to save time."""
    pool = list(candidates or Xa.columns)
    chosen, path = [], []
    t0 = time.perf_counter()
    while pool and len(chosen) < max_features:
        best = None
        for c in pool:
            _, a = fit_score(chosen + [c], Xa, ya, Xb, yb, seed)
            if best is None or a > best[1]:
                best = (c, a)
        chosen.append(best[0]); pool.remove(best[0])
        path.append(dict(step=len(chosen), added=best[0], group=group_of(best[0]), auc=best[1],
                         gap_to_full=full_auc - best[1]))
        log(f"  +{best[0]:18s} -> selection AUC {best[1]:.4f} (full {full_auc:.4f}, gap {full_auc - best[1]:+.4f}) "
            f"[{(time.perf_counter() - t0) / 60:.1f} min]")
        if full_auc - best[1] <= tolerance:
            break
    return chosen, pd.DataFrame(path)


def day_bootstrap(frame, scores, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    y = frame.label.to_numpy()
    codes, _ = pd.factorize(frame.decision_date)
    days = [np.flatnonzero(codes == d) for d in range(codes.max() + 1)]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([days[i] for i in rng.integers(0, len(days), len(days))])
        draws.append({n: _auc(y[idx], s[idx]) for n, s in scores.items()})
    return pd.DataFrame(draws)


def confirm(subsets, d, X, seed, n_boot=300):
    """Refit each subset on 2016-2022 and score 2023; paired bootstrap against the full model."""
    ytr, yva = d["train"].label.to_numpy(), d["valid"].label.to_numpy()
    scores, rows = {}, []
    for name, cols in subsets.items():
        m = hgb(seed).fit(X["train"][cols], ytr)
        scores[name] = m.predict_proba(X["valid"][cols])[:, 1]
        rows.append(dict(model=name, n_features=len(cols), tree_nodes=tree_size(m), auc=_auc(yva, scores[name])))
    boot = day_bootstrap(d["valid"], scores, n_boot, seed)
    out = pd.DataFrame(rows).set_index("model")
    out["auc_low"], out["auc_high"] = boot.quantile(0.025), boot.quantile(0.975)
    diff = boot.sub(boot["all 38 features"], axis=0)
    out["diff_vs_full"] = diff.mean()
    out["diff_low"], out["diff_high"] = diff.quantile(0.025), diff.quantile(0.975)
    out["verdict"] = np.where(out.diff_low > 0, "better", np.where(out.diff_high < 0, "worse", "tied"))
    out.loc["all 38 features", "verdict"] = "reference"
    return out.reset_index(), scores


def plot_importance(imp, path=None, title=""):
    import matplotlib.pyplot as plt
    colours = dict(zip(GROUPS, ["tab:blue", "tab:red", "tab:gray", "tab:green", "tab:purple"]))
    d = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(d) + 1.2))
    ax.barh(d.feature, d.auc_drop, xerr=d.auc_drop_std, color=[colours.get(g, "black") for g in d.group])
    for g, c in colours.items():
        ax.barh([], [], color=c, label=g)
    ax.axvline(0, c="k", lw=0.8)
    ax.set(xlabel="AUC drop when shuffled (selection year)", title=title)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()
