"""Non-image baselines. The CNN only adds value if it beats these on the same samples."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def hand_features(store, frame):
    """Simple numeric features computed from the same 78-bar input window (no future data)."""
    s = frame.security_index.to_numpy(np.int64)
    t = frame.end_index.to_numpy(np.int64)
    lc = store.log_close

    def ret(k):                                   # k-th most recent 1-bar log return
        return lc[s, t - k + 1] - lc[s, t - k]

    past = np.stack([ret(k) for k in range(1, 78)])
    cols = {f"r{k}": past[k - 1] for k in range(1, 13)}
    cols.update({f"cum{w}": lc[s, t] - lc[s, t - w] for w in (6, 12, 39, 77)})
    cols["realized_vol"] = np.nanstd(past, axis=0)
    cols["mean_abs_r12"] = np.nanmean(np.abs(past[:12]), axis=0)
    past_volume = np.stack([store.volume[s, t - k] for k in range(1, 78)])
    cols["rel_volume"] = np.log(store.volume[s, t] / np.nanmean(past_volume, axis=0))
    cols["time_of_day"] = t % 78
    return pd.DataFrame(cols)


def market_features(store, frame):
    """SPY features over the same 78-bar input window, plus the stock's relation to SPY.
    Only bars up to the decision bar t are used (no future data)."""
    s = frame.security_index.to_numpy(np.int64)
    t = frame.end_index.to_numpy(np.int64)
    lc, mlc = store.log_close, store.market_log_close
    stock = np.stack([lc[s, t - k + 1] - lc[s, t - k] for k in range(1, 78)])     # (77, n)
    spy = np.stack([mlc[t - k + 1] - mlc[t - k] for k in range(1, 78)])
    cols = {f"spy_r{k}": spy[k - 1] for k in (1, 2, 3, 6, 12)}
    cols.update({f"spy_cum{w}": mlc[t] - mlc[t - w] for w in (6, 12, 39, 77)})
    cols["spy_realized_vol"] = spy.std(axis=0)
    cols["spy_mean_abs_r12"] = np.abs(spy[:12]).mean(axis=0)
    past_volume = np.stack([store.market_volume[t - k] for k in range(1, 78)])
    cols["spy_rel_volume"] = np.log(store.market_volume[t] / past_volume.mean(axis=0))
    sc, mc = stock - stock.mean(axis=0), spy - spy.mean(axis=0)
    var = (mc ** 2).mean(axis=0)
    beta = np.where(var > 0, (sc * mc).mean(axis=0) / np.where(var > 0, var, 1), np.nan)
    cols["beta77"] = beta
    cols["corr77"] = (sc * mc).mean(axis=0) / np.sqrt((sc ** 2).mean(axis=0) * var + 1e-30)
    cols["idio_vol77"] = np.nanstd(stock - beta * spy, axis=0)
    cols["vol_ratio77"] = np.nanstd(stock, axis=0) / (cols["spy_realized_vol"] + 1e-12)
    for w in (12, 77):
        cols[f"rel_cum{w}"] = (lc[s, t] - lc[s, t - w]) - (mlc[t] - mlc[t - w])
    return pd.DataFrame(cols)


def tree_size(model):
    """Parameter-count analogue for a boosted tree: number of nodes across all trees."""
    return int(sum(len(p.nodes) for stage in model._predictors for p in stage))


def fit_baselines(store, target, train, valid, seed=69, max_train=300_000):
    """Return ({name: validation scores}, training prior, fitted HGB)."""
    prior = float(train.label.mean())
    Xva = hand_features(store, valid)
    scores = {"training_prior": np.full(len(valid), prior)}
    if target == "vol":
        scores["rule_recent_abs_return"] = Xva.mean_abs_r12.to_numpy()
    else:
        scores["rule_last_bar_return"] = Xva.r1.to_numpy()
        scores["rule_78bar_momentum"] = Xva.cum77.to_numpy()

    sub = train.sample(min(max_train, len(train)), random_state=seed)
    Xtr = hand_features(store, sub)
    models = {}
    variants = [("hgb_hand_features", Xtr, Xva)]
    if getattr(store, "has_market", False):          # same HGB, plus SPY and stock-vs-SPY features
        variants.append(("hgb_hand_spy", pd.concat([Xtr, market_features(store, sub)], axis=1),
                         pd.concat([Xva, market_features(store, valid)], axis=1)))
    for name, A, B in variants:
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=500,
                                             random_state=seed)
        hgb.fit(A, sub.label)
        print(f"{name}: {tree_size(hgb):,} tree nodes ({hgb.n_iter_} trees, {B.shape[1]} features, "
              f"{len(sub):,} training rows)")
        scores[name] = hgb.predict_proba(B)[:, 1]
        models[name] = hgb
    return scores, prior, models
