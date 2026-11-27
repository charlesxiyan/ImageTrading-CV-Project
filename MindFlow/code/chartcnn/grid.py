"""Horizon x target grid: does a signal survive as the horizon grows, and for which target?

One *cell* = one (target kind, horizon) pair, e.g. `dir_h3` = "is the return over the next
15 minutes strictly positive?". Every cell uses the same decision times (one stride for all
horizons), so differences between cells come from the label, not from the sample.

    kind   label                                                           chartcnn target
    dir    r(t -> t+h) > 0, zero returns dropped                (absolute direction)   dir
    vol    |r(t -> t+h)| > that stock's training median |r|     (volatility)           vol
    xsdir  r(t -> t+h) > cross-sectional median at time t       (relative direction)   xs_dir

Each finished cell writes `predictions.parquet` (validation labels, returns and every model's
score) and `done.json` to its folder on Drive. A rerun skips finished cells and resumes an
unfinished CNN from its `last.pt`, so a Colab disconnect costs at most one epoch.
"""
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score

from .baselines import fit_baselines, tree_size
from .data import Store, build_split, make_dataset
from .evaluation import decile_report
from .models import build_model, print_parameter_count
from .training import fit, predict
from .utils import make_loader, mirror, save_json, seed_all

KINDS = {"dir": "dir", "vol": "vol", "xsdir": "xs_dir"}
KIND_LABEL = {"dir": "absolute direction", "vol": "volatility", "xsdir": "relative direction"}
BAR_MINUTES = 5
PROB_MODELS = ("training_prior", "hgb_hand_features", "hgb_hand_spy")      # plus every cnn_<arch>
ARCH_LABEL = {"plain": "image CNN", "plain_spy": "image CNN + SPY panel", "series1d": "1D CNN",
              "series1d_spy": "1D CNN + SPY", "residual": "residual image CNN",
              "vit": "image ViT (T2T-style)", "vit_spy": "image ViT + SPY panel",
              "plain_vixy": "image CNN + VIXY panel", "plain_spy_vixy": "image CNN + SPY + VIXY panels",
              "vit_vixy": "image ViT + VIXY panel", "vit_spy_vixy": "image ViT + SPY + VIXY panels"}


def arch_settings(settings, arch):
    """Per-architecture training settings. Transformers do not train at the CNNs' Adam 1e-5; they
    get the usual ViT recipe: AdamW, their own learning rate and weight decay, linear warmup and
    gradient clipping (vit_learning_rate, vit_weight_decay, vit_warmup_steps, vit_grad_clip)."""
    s = dict(settings, arch=arch)
    if arch.startswith("vit"):
        s.update(learning_rate=settings.get("vit_learning_rate", 3e-4),
                 weight_decay=settings.get("vit_weight_decay", 0.05), optimizer="adamw",
                 warmup_steps=settings.get("vit_warmup_steps", 1000), grad_clip=settings.get("vit_grad_clip", 1.0))
    return s


def grid_cells(horizons=(1, 3, 6, 12), kinds=("dir", "vol", "xsdir"), stride=12):
    """One experiment dict per (kind, horizon), in the order they are run."""
    if stride < max(horizons):
        raise ValueError("stride must be >= the longest horizon, or a stock's labels overlap")
    return [dict(name=f"{k}_h{h}", kind=k, target=KINDS[k], horizon=h, minutes=BAR_MINUTES * h,
                 stride=stride) for h in horizons for k in kinds]


def label_overview(store, cells):
    """Sample counts and positive shares per cell (cheap: no images, no models)."""
    rows = []
    for c in cells:
        tr, med = build_split(store, "train", c["target"], c["horizon"], c["stride"])
        va, _ = build_split(store, "validation", c["target"], c["horizon"], c["stride"], med)
        rows.append(dict(cell=c["name"], kind=c["kind"], minutes=c["minutes"], stride=c["stride"],
                         train=len(tr), validation=len(va), train_pos=tr.label.mean(),
                         validation_pos=va.label.mean(),
                         validation_days=va.decision_date.nunique()))
    return pd.DataFrame(rows).set_index("cell")


def _restore(src, dst):
    """Copy a Drive folder's files to local disk (resume after a disconnect)."""
    if src is not None and Path(src).is_dir():
        Path(dst).mkdir(parents=True, exist_ok=True)
        for f in Path(src).iterdir():
            if f.is_file() and not f.name.endswith(".copying"):
                shutil.copy2(f, Path(dst) / f.name)


def run_cell(prepared, cell, settings, archs, device, local_dir, drive_dir=None, smoke=False, retrain=()):
    """Labels -> baselines -> one CNN per arch -> predictions.parquet. Skips if already done.

    `retrain`: networks of a finished cell to train again (e.g. ["vit", "vit_spy"] after a fix). Only
    those networks are retrained, on the same rows; their scores in predictions.parquet are replaced
    and every other model is kept. A network already retrained with this code version is skipped."""
    local_dir = Path(local_dir)
    drive_dir = Path(drive_dir) if drive_dir is not None else None
    done_at = (drive_dir or local_dir) / "done.json"
    if done_at.exists():
        info = json.loads(done_at.read_text())
        todo = [a for a in retrain if info.get("retrained", {}).get(a) != _code_version()]
        if not todo:
            print(f"{cell['name']}: already done, loading saved predictions")
            return load_cell(done_at.parent)
        return _retrain_cell(prepared, cell, settings, todo, device, local_dir, drive_dir, info, smoke)

    start = time.perf_counter()
    local_dir.mkdir(parents=True, exist_ok=True)
    seed_all(settings["seed"])
    store = Store(prepared)
    train, nn_train, valid, median = _cell_rows(prepared, cell, settings, smoke)
    full_train = len(train)
    # Neural models train on a random subset (whole years of 5-minute windows do not fit a Colab
    # budget); HGB uses its own, larger cap. Validation is always the full validation year.
    print(f"{cell['name']} ({KIND_LABEL[cell['kind']]}, next {cell['minutes']} min, stride {cell['stride']}): "
          f"{full_train:,} train ({len(nn_train):,} used by the CNNs) / {len(valid):,} validation; positive share "
          f"{train.label.mean():.3f} / {valid.label.mean():.3f}")

    scores, prior, hgbs = fit_baselines(store, cell["target"], train, valid, settings["seed"],
                                        max_train=2_000 if smoke else settings.get("max_train_hgb", 1_000_000))
    params = {name: tree_size(m) for name, m in hgbs.items()}   # HGB "parameters" = tree nodes
    epochs = {}
    for arch in archs:
        scores[f"cnn_{arch}"], params[arch], epochs[arch] = _train_arch(
            prepared, cell, settings, arch, device, nn_train, valid, prior, local_dir, drive_dir)

    keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
    out = valid[keep].copy()
    for name, p in scores.items():
        out[f"score_{name}"] = np.asarray(p, float)
    out.to_parquet(local_dir / "predictions.parquet", index=False)
    if median is not None:
        median.to_csv(local_dir / "train_abs_return_median.csv")
        mirror(local_dir / "train_abs_return_median.csv", drive_dir)
    mirror(local_dir / "predictions.parquet", drive_dir)
    info = dict(cell=cell, prior=prior, parameters=params, best_epoch=epochs,
                train_samples=full_train, nn_train_samples=len(nn_train), validation_samples=len(valid),
                minutes=round((time.perf_counter() - start) / 60, 1), smoke=smoke)
    save_json(local_dir / "done.json", info, drive_dir)       # written last: marks the cell finished
    print(f"{cell['name']}: done in {info['minutes']} min")
    return load_cell(local_dir)


def _code_version():
    from . import __version__
    return __version__


def _cell_rows(prepared, cell, settings, smoke):
    """The training subset and validation rows of a cell, exactly as run_cell builds them."""
    store = Store(prepared)
    train, median = build_split(store, "train", cell["target"], cell["horizon"], cell["stride"])
    valid, _ = build_split(store, "validation", cell["target"], cell["horizon"], cell["stride"], median)
    if smoke:
        train = train.sample(256, random_state=0).reset_index(drop=True)
        times = valid.end_index.drop_duplicates().sample(2, random_state=0)
        valid = valid[valid.end_index.isin(times)].reset_index(drop=True)
    kv = max(1, int(settings.get("validation_every_kth_day", 1)))
    if kv > 1:
        valid = valid[valid.decision_date.isin(np.sort(valid.decision_date.unique())[::kv])].reset_index(drop=True)
    nn_train = train.sample(min(settings.get("max_train_nn", len(train)), len(train)),
                            random_state=settings["seed"]).reset_index(drop=True)
    return train, nn_train, valid, median


def _train_arch(prepared, cell, settings, arch, device, nn_train, valid, prior, local_dir, drive_dir):
    """Train one network on a cell; returns (validation scores, parameters, best epoch)."""
    s = arch_settings(settings, arch)
    sub_local, sub_drive = local_dir / arch, (drive_dir / arch if drive_dir else None)
    sub_local.mkdir(parents=True, exist_ok=True)
    _restore(sub_drive, sub_local)
    (sub_local / "code_version.txt").write_text(_code_version())
    mirror(sub_local / "code_version.txt", sub_drive)
    seed_all(settings["seed"])
    model = build_model(s).to(device)
    n_params = print_parameter_count(model, f"{cell['name']} {ARCH_LABEL.get(arch, arch)} ({arch})")
    w = settings["num_workers"]
    train_loader = make_loader(make_dataset(prepared, nn_train, arch), settings["batch_size"], w, True,
                               settings["seed"])
    valid_loader = make_loader(make_dataset(prepared, valid, arch), settings["eval_batch_size"], w, False)
    # Early stopping looks at a fixed subset of whole validation days (every k-th day) to save
    # GPU time; the final scores below use the full validation year.
    k = max(1, int(settings.get("early_stop_every_kth_day", 1)))
    es_days = np.sort(valid.decision_date.unique())[::k]
    es = valid[valid.decision_date.isin(es_days)].reset_index(drop=True)
    es_loader = make_loader(make_dataset(prepared, es, arch), settings["eval_batch_size"], w, False)
    history = fit(model, train_loader, es_loader, es.label.to_numpy(), prior, s, device,
                  sub_local, sub_drive)
    # Score validation with the best checkpoint (works the same after a resume).
    model.load_state_dict(torch.load(sub_local / "best.pt", map_location=device, weights_only=False)["model"])
    p = predict(model, valid_loader, device, settings.get("amp", False))
    del es_loader
    epoch = int(history.loc[history.val_log_loss.idxmin(), "epoch"])
    del model, train_loader, valid_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p, n_params, epoch


def _retrain_cell(prepared, cell, settings, archs, device, local_dir, drive_dir, info, smoke):
    """Retrain some networks of a finished cell on the same rows and replace their scores."""
    start = time.perf_counter()
    src = drive_dir or local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(src / "predictions.parquet")
    seed_all(settings["seed"])
    _, nn_train, valid, _ = _cell_rows(prepared, cell, settings, smoke)
    keys = ["security_index", "end_index"]
    assert len(valid) == len(pred) and (valid[keys].to_numpy() == pred[keys].to_numpy()).all(), \
        "validation rows differ from the finished cell: keep the run's SETTINGS (seed, max_train_nn, ...)"
    assert len(nn_train) == info["nn_train_samples"], "training subset differs from the finished cell"
    print(f"{cell['name']}: retraining {', '.join(archs)} on the same {len(nn_train):,} / {len(valid):,} rows "
          f"(other models kept)")
    for arch in archs:
        for folder in (local_dir / arch, drive_dir / arch if drive_dir else None):
            marker = folder / "code_version.txt" if folder is not None else None
            if folder is not None and folder.is_dir() and \
                    (not marker.exists() or marker.read_text().strip() != _code_version()):
                old = folder.with_name(f"{arch}_before_{_code_version()}")   # old network: set aside
                if old.exists():
                    shutil.rmtree(folder)                    # leftover of an unmarked attempt
                else:
                    folder.rename(old)
        p, n, e = _train_arch(prepared, cell, settings, arch, device, nn_train, valid, info["prior"],
                              local_dir, drive_dir)
        pred[f"score_cnn_{arch}"] = np.asarray(p, float)
        info["parameters"][arch], info["best_epoch"][arch] = n, e
        info.setdefault("retrained", {})[arch] = _code_version()
        pred.to_parquet(local_dir / "predictions.parquet", index=False)
        mirror(local_dir / "predictions.parquet", drive_dir)
        save_json(local_dir / "done.json", info, drive_dir)
    print(f"{cell['name']}: retrained in {(time.perf_counter() - start) / 60:.1f} min")
    return load_cell(local_dir)


def load_cell(folder):
    folder = Path(folder)
    info = json.loads((folder / "done.json").read_text())
    return dict(info=info, predictions=pd.read_parquet(folder / "predictions.parquet"))


# ----------------------------------------------------------------------------- analysis

def _model_names(pred):
    return [c[len("score_"):] for c in pred.columns if c.startswith("score_")]


def _pretty(name):
    if name.startswith("cnn_vit"):
        return f"ViT ({name[4:]})"
    if name.startswith("cnn_"):
        return f"CNN ({name[4:]})"
    return {"hgb_hand_features": "HGB hand features", "hgb_hand_spy": "HGB hand + SPY",
            "training_prior": "training prior"}.get(name, f"rule: {name[5:]}"
                                                                                               if name.startswith("rule_") else name)


def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def day_bootstrap(pred, names, n_boot=300, seed=0):
    """AUC draws when whole validation days are resampled (frames within a day are correlated).
    The same day draw is used for every model, so differences between models are paired."""
    rng = np.random.default_rng(seed)
    y = pred.label.to_numpy()
    days = [np.flatnonzero(m) for m in (pred.decision_date.to_numpy() == d for d in pred.decision_date.unique())]
    cols = {n: pred[f"score_{n}"].to_numpy() for n in names}
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([days[i] for i in rng.integers(0, len(days), len(days))])
        draws.append({n: _auc(y[idx], cols[n][idx]) for n in names})
    return pd.DataFrame(draws)


def grid_table(results, n_boot=300):
    """One row per (cell, model): AUC with 95% day-bootstrap CI, log-loss gain vs the training
    prior (probability models only), decile long-short (direction kinds), parameters."""
    rows, boots = [], {}
    for name, r in results.items():
        pred, info = r["predictions"], r["info"]
        cell = info["cell"]
        models = [m for m in _model_names(pred) if m != "training_prior"]
        boot = day_bootstrap(pred, models, n_boot)
        boots[name] = boot
        y = pred.label.to_numpy()
        prior_loss = log_loss(y, np.full(len(y), info["prior"]), labels=[0, 1])
        for m in models:
            p = pred[f"score_{m}"].to_numpy(float)
            p = np.where(np.isfinite(p), p, np.nanmedian(p))
            lo, hi = np.nanpercentile(boot[m], [2.5, 97.5])
            row = dict(cell=name, kind=cell["kind"], horizon=cell["horizon"], minutes=cell["minutes"],
                       model=_pretty(m), roc_auc=_auc(y, p), auc_ci_low=lo, auc_ci_high=hi,
                       log_loss_gain_vs_prior_pct=np.nan, prob_std=np.nan, parameters=np.nan,
                       best_epoch=np.nan, long_short_bps=np.nan, long_short_t=np.nan,
                       validation_samples=len(y), validation_days=pred.decision_date.nunique())
            probabilistic = m.startswith("cnn_") or m in PROB_MODELS
            if probabilistic:
                pc = np.clip(p, 1e-7, 1 - 1e-7)
                row["log_loss_gain_vs_prior_pct"] = 100 * (prior_loss - log_loss(y, pc, labels=[0, 1])) / prior_loss
                row["prob_std"] = float(pc.std())
            if m.startswith("hgb_"):
                row["parameters"] = info["parameters"].get(m, np.nan)
            if m.startswith("cnn_"):
                row["parameters"] = info["parameters"].get(m[4:], np.nan)
                row["best_epoch"] = info["best_epoch"].get(m[4:], np.nan)
            if probabilistic and cell["kind"] != "vol":
                _, spread = decile_report(pred, p, cell["target"])
                if spread:
                    row["long_short_bps"], row["long_short_t"] = spread["mean_bps"], spread["t_stat"]
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["cell", "model"])
    return table, boots


def best_model_by_cell(table):
    """The best probability model in each cell (CNNs and HGB; rules excluded)."""
    t = table.reset_index()
    t = t[t.model.str.startswith(("CNN", "ViT", "HGB"))]
    return t.loc[t.groupby("cell").roc_auc.idxmax()].set_index("cell")


def plot_grid(table, path=None):
    """Left three panels: AUC (95% day-bootstrap CI) by horizon, one panel per target kind,
    one line per model. Right: the best model's AUC − 0.5 for the three kinds on one axis."""
    t = table.reset_index()
    kinds = [k for k in KINDS if k in set(t.kind)]
    fig, axes = plt.subplots(1, len(kinds) + 1, figsize=(5.2 * (len(kinds) + 1), 4.2))
    models = [m for m in dict.fromkeys(t.model) if not m.startswith("rule")]
    rules = [m for m in dict.fromkeys(t.model) if m.startswith("rule")]
    lo_all = min(0.47, t.auc_ci_low.min() - 0.01)
    hi_all = max(0.55, t.auc_ci_high.max() + 0.01)
    for ax, kind in zip(axes, kinds):
        s = t[t.kind == kind]
        for j, m in enumerate(models + rules):
            d = s[s.model == m].sort_values("minutes")
            if d.empty:
                continue
            x = d.minutes * (1 + 0.03 * (j - len(models) / 2))         # small dodge so CIs do not overlap
            ax.errorbar(x, d.roc_auc, yerr=[d.roc_auc - d.auc_ci_low, d.auc_ci_high - d.roc_auc],
                        fmt="o-" if m in models else "x:", capsize=3, label=m, alpha=1 if m in models else 0.6)
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.set_xscale("log"); ax.minorticks_off()
        ax.set_xticks(sorted(t.minutes.unique()), [f"{m} min" for m in sorted(t.minutes.unique())])
        ax.set_ylim(lo_all, hi_all)
        ax.set(title=f"{kind}: {KIND_LABEL[kind]}", xlabel="horizon", ylabel="validation AUC")
        ax.legend(fontsize=8)
    ax = axes[-1]
    best = best_model_by_cell(table).reset_index()
    for kind in kinds:
        d = best[best.kind == kind].sort_values("minutes")
        ax.errorbar(d.minutes, d.roc_auc - 0.5, yerr=[d.roc_auc - d.auc_ci_low, d.auc_ci_high - d.roc_auc],
                    fmt="o-", capsize=3, label=f"{kind} ({KIND_LABEL[kind]})")
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks(sorted(t.minutes.unique()), [f"{m} min" for m in sorted(t.minutes.unique())])
    ax.set(title="best model per cell: AUC − 0.5", xlabel="horizon", ylabel="AUC − 0.5")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()


def auc_matrix(table, model):
    """kinds x horizons matrix of one model's AUC, formatted 'AUC [lo, hi]'."""
    t = table.reset_index()
    t = t[t.model == model]
    t = t.assign(text=t.apply(lambda r: f"{r.roc_auc:.3f} [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}]", axis=1))
    m = t.pivot(index="kind", columns="minutes", values="text")
    m.columns = [f"{c} min" for c in m.columns]
    return m.reindex([k for k in KINDS if k in m.index])


def takeaways(table, boots):
    """Plain-English conclusions generated from the numbers (no hand-written claims)."""
    best = best_model_by_cell(table)
    lines = []
    for kind in [k for k in KINDS if k in set(best.kind)]:
        b = best[best.kind == kind].sort_values("minutes")
        above = [f"{r.minutes} min" for _, r in b.iterrows() if r.auc_ci_low > 0.5]
        cells = ", ".join(f"{r.minutes} min {r.roc_auc:.3f} [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}] ({r.model})"
                          for _, r in b.iterrows())
        verdict = (f"above 0.5 at {', '.join(above)}" if above else "not distinguishable from 0.5 at any horizon")
        lines.append(f"- **{kind} ({KIND_LABEL[kind]})**: best AUC by horizon {cells}. 95% CI {verdict}.")
    t = table.reset_index()
    for kind in ("dir", "xsdir"):
        d = t[(t.kind == kind) & t.long_short_t.notna()]
        if len(d):
            strong = d[d.long_short_t.abs() >= 2]
            lines.append(f"- {kind} decile long-short (top − bottom, gross, bps per horizon): "
                         + ("no model/horizon with |t| ≥ 2." if strong.empty else
                            "; ".join(f"{r.cell} {r.model} {r.long_short_bps:+.2f} bps (t = {r.long_short_t:.2f})"
                                      for _, r in strong.iterrows())))
    # Paired image-vs-1D and CNN-vs-HGB checks per cell, summarised.
    for a, b_, label in (("CNN (plain)", "CNN (series1d)", "image CNN − 1D CNN"),
                         ("CNN (plain)", "HGB hand features", "image CNN − HGB"),
                         ("HGB hand + SPY", "HGB hand features", "HGB with SPY − HGB without"),
                         ("CNN (plain_spy)", "CNN (plain)", "image CNN with SPY panel − without"),
                         ("CNN (series1d_spy)", "CNN (series1d)", "1D CNN with SPY − without"),
                         ("ViT (vit)", "CNN (plain)", "image ViT − image CNN"),
                         ("CNN (residual)", "CNN (plain)", "residual image CNN − plain image CNN"),
                         ("ViT (vit_spy)", "ViT (vit)", "image ViT with SPY panel − without"),
                         ("CNN (plain_vixy)", "CNN (plain)", "image CNN with VIXY panel − without"),
                         ("CNN (plain_spy_vixy)", "CNN (plain_spy)", "image CNN + SPY with VIXY panel − without"),
                         ("ViT (vit_vixy)", "ViT (vit)", "image ViT with VIXY panel − without"),
                         ("ViT (vit_spy_vixy)", "ViT (vit_spy)", "image ViT + SPY with VIXY panel − without"),
                         ("CNN (plain_spy_vixy)", "HGB hand + SPY", "image CNN + SPY + VIXY − HGB hand + SPY")):
        wins = losses = ties = 0
        for cell, boot in boots.items():
            ca, cb = _raw_name(a), _raw_name(b_)
            if ca in boot and cb in boot:
                lo, hi = np.nanpercentile(boot[ca] - boot[cb], [2.5, 97.5])
                wins += lo > 0; losses += hi < 0; ties += lo <= 0 <= hi
        if wins + losses + ties:
            lines.append(f"- {label} AUC (paired day bootstrap) across {wins + losses + ties} cells: "
                         f"better in {wins}, worse in {losses}, tied in {ties}.")
    return "\n".join(lines)


def _raw_name(pretty):
    if pretty.startswith(("CNN (", "ViT (")):
        return "cnn_" + pretty[5:-1]
    return {"HGB hand features": "hgb_hand_features", "HGB hand + SPY": "hgb_hand_spy"}.get(pretty, pretty)


# ----------------------------------------------------------------------------- held-out test

def evaluate_cell_test(prepared, cell, settings, archs, device, cell_dir, smoke=False):
    """One-shot test-period scores for a finished cell: every CNN from its best.pt, HGB refit on
    the same training rows (same seed). Run only after all choices are frozen on validation."""
    cell_dir = Path(cell_dir)
    info = json.loads((cell_dir / "done.json").read_text())
    store = Store(prepared)
    train, median = build_split(store, "train", cell["target"], cell["horizon"], cell["stride"])
    if cell["target"] == "vol":
        median = pd.read_csv(cell_dir / "train_abs_return_median.csv", index_col=0).iloc[:, 0]
    test, _ = build_split(store, "test", cell["target"], cell["horizon"], cell["stride"], median)
    if smoke:
        train = train.sample(256, random_state=0).reset_index(drop=True)
        test = test[test.end_index.isin(test.end_index.drop_duplicates().sample(2, random_state=0))].reset_index(drop=True)
    scores, _, _ = fit_baselines(store, cell["target"], train, test, settings["seed"],
                                 max_train=2_000 if smoke else settings.get("max_train_hgb", 1_000_000))
    scores["training_prior"] = np.full(len(test), info["prior"])
    for arch in archs:
        model = build_model(arch_settings(settings, arch)).to(device)
        model.load_state_dict(torch.load(cell_dir / arch / "best.pt", map_location=device, weights_only=False)["model"])
        loader = make_loader(make_dataset(prepared, test, arch), settings["eval_batch_size"], settings["num_workers"], False)
        scores[f"cnn_{arch}"] = predict(model, loader, device, settings.get("amp", False))
    keep = ["security_index", "permno", "end_index", "decision_date", "label", "forward_return", "xs_return"]
    out = test[keep].copy()
    for name, p in scores.items():
        out[f"score_{name}"] = np.asarray(p, float)
    out.to_parquet(cell_dir / "test_predictions.parquet", index=False)
    return dict(info=info, predictions=out)


def merge_previous(results, previous_dir, test=False):
    """Add another run's model scores to this run's cells, so new architectures (e.g. the ViTs) are
    compared with models trained earlier without retraining them. Cells must have identical rows
    (same data, cell definition and seed); this is checked, not assumed."""
    keys = ["security_index", "end_index"]
    merged = {}
    for name, r in results.items():
        folder = Path(previous_dir) / name
        if test:
            prev = dict(info=json.loads((folder / "done.json").read_text()),
                        predictions=pd.read_parquet(folder / "test_predictions.parquet"))
        else:
            prev = load_cell(folder)
        a, b = r["predictions"], prev["predictions"]
        if len(a) != len(b) or not (a[keys + ["label"]].to_numpy() == b[keys + ["label"]].to_numpy()).all():
            raise ValueError(f"{name}: rows differ from {folder}; rerun those models instead")
        new = [c for c in b.columns if c.startswith("score_") and c not in a.columns]
        info = dict(r["info"])
        info["parameters"] = {**prev["info"].get("parameters", {}), **info.get("parameters", {})}
        info["best_epoch"] = {**prev["info"].get("best_epoch", {}), **info.get("best_epoch", {})}
        merged[name] = dict(info=info, predictions=pd.concat([a, b[new]], axis=1))
    return merged
