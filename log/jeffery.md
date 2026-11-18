# Work log: jeffery

Entries before 2026-11-12 are condensed from a private log.

## 2026-11-01 — Universe and TAQ symbol map
- Fixed `build_dynamic_universe.py`: the original fails on real WRDS data because an index can
  list a merged company a few days after CRSP's last record. Such members, up to 10 days past
  their last record, are now excluded and listed in the manifest. 136 member-days in 2010-2024.
- Added `build_taq_symbol_map.py`. The WRDS daily CRSP-TAQ link omits every share code other
  than 11, so a CUSIP-based link fills the rest. 8 security-days stay unresolved.
- Before 2016-10-24, link tables write `CMCSA` while trade tables store `CMCS` + `A`. Joining on
  the link spelling returns zero rows with no error. 7,764 security-days were affected.

## 2026-11-01 — Cleaning, bars and labels
- Added `taq_bars.py` and `examples.py`. Condition-code rules came from profiling real trades.
- Outlier filter took three attempts. The first deleted real prints after a trading halt; the
  second flagged steady crashes. Final rule: drop a print only if it is far above, or far below,
  both the 25 trades before and the 25 after.
- Agent errors recorded: a volume check dropped NULL condition codes and falsely showed a gap.

## 2026-11-02 — 2023 pilot
- Built all 250 dates of 2023 with four connections in about 3.9 hours. `validate_bars.py`
  passed with 0 failures.
- Found: 2023-01-24 NYSE opening failure leaves glitch prices marked as valid; 2023-06-05
  opening prints are stamped before 09:30:00 for 122 securities, so bar 1 misses the auction.
- Labels: up share 50.3% overall, but 31% to 65% by day.

## 2026-11-02 — Build environment and universe audit
- The full build will run on a university compute server, not a laptop. WRDS allowed 7
  simultaneous connections and refused the 8th, so the build uses 5. Not started.
- `hpc_envcheck.sh` is on branch `hpc/env-check`.
- Recomputed the universe from WRDS source tables independently of the build SQL: no extra
  rows, and the only missing rows are the excluded ones. Compustat's S&P 500 constituent table
  on WRDS holds current members only, so CRSP is the only usable S&P history.

## 2026-11-02 — Price adjustment policy
- Wrote the policy into `CLAUDE.md`: adjust history to the anchor session, never the future.
- `cfacpr` does not include ordinary cash dividends; those come from `crsp.dsedist`.
- On 2023 ex-dates with a dividend above 0.5%, the mean overnight move is -0.75% raw and
  +0.17% after adjustment. `examples.py` does not implement this yet.

## 2026-11-12 — Reference work and the demo runs
- Read charles's run summaries and the reference repository. Conclusions are in `DECISIONS.md`
  and `EXPERIMENTS.md`. Not read: his notebooks, zips, or data preparation code.
- Read Jiang, Kelly and Xiu (2023) in the published version. It has no accuracy table; the
  working-paper accuracy figure was not verified.

## 2026-11-12 — Shared memory
- Set up this branch, `scripts/memory_sync.sh`, the hooks in `.claude/settings.json` and the
  shared-memory section of `CLAUDE.md`. Seeded every file here from jeffery's side only.
- Not verified: the hooks have been pipe-tested but not yet observed firing in a new session.
- Verified from a second clone: a fresh clone creates `memory/` on `start`, an entry written
  there reaches the first clone, edits to different files merge, and a same-line conflict is
  reported and left for a person or Claude to resolve. A credential-like string blocks the push.
- Added `ONBOARDING.md`, the prompt for connecting charles's Claude. Charles has not run it yet.

## 2026-11-14 — Progress check
- Asked for current progress. Read shared memory, fetched the remote and listed pull requests.
  No code or data changed.
- State: three open pull requests from charles (#1 to #3, stacked), none reviewed by jeffery.
  The bar pipeline is still uncommitted on `test/mock-wrds`; `main` is 2 commits behind
  `origin/main`. The full 2010-2024 build has not started.
- Not verified: the contents of the pull requests were not read.

## 2026-11-14 — Data source decided; model results read
- Jeffery decided: charles's data pipeline is the source. Jeffery's bar pipeline is set aside.
  Recorded in `DECISIONS.md` and `STATUS.md`.
- Read the result summaries on the PR #3 branch (`takeaways.md`, `analysis.csv`, deciles,
  `frame_corr`, pretrained `comparison.csv`). Did not read the model or data code.

## 2026-11-15 — Progress check
- Asked for the latest progress. Synced memory, fetched the remote, read the multi-target grid
  `grid_table.csv` on `origin/main`. No code or data changed.
- Found: CV PRs #1 to #3 are all merged into `main`, so the review step is moot. The grid confirms
  volatility signal at every horizon and no usable direction signal; at the proposal's 30-minute
  horizon (dir_h6) only HGB clears 0.5 (AUC 0.543), with log loss barely above the prior.
- Not verified: per-cell predictions on Drive; whether the `*_dir_targets` rerun has happened.

## 2026-11-17 — Progress check
- Asked for the latest progress. Synced memory and fetched the remote; read charles's log and
  `EXPERIMENTS.md`. No code or data changed.
- Found: charles scaled up to 2016-2025 (train to 2022, validate 2023, test 2024-2025, 79.5M
  windows) with SPY features; run `run_20261116_044159` gives the same picture as the 2023 H1
  runs. CV PR #4 is open with jeffery as a requested reviewer. A ViT is ready but not run.
- Not verified: the PR #4 diff and the per-cell predictions were not read.

## 2026-11-18 — Big Data course ingestion plan
- Asked: list the WRDS data the project uses, and plan the Spark profiling, cleaning and
  ingestion submission for the Big Data course (due 2026-11-20). Planning only; no code or data.
- Found: charles's pipeline aggregates TAQ trades into bars inside WRDS PostgreSQL, so raw
  trades never land; his price filter keeps only conditions ` @EF` and excludes NULL conditions,
  odd lots and auction prints. His 2016-2025 reference data comes from CRSP CIZ tables
  (`dsf_v2`, `stkdistributions`, `stksecurityinfohist`). The laptop has 18 GB free disk.
- Plan: Spark JDBC ingestion of raw TAQ trades for the universe (one month, March 2023) plus
  CRSP and Compustat reference tables into Parquet; profiling, cleaning and bars in Spark;
  validate against the existing 2023 pandas bars.
- Not verified: Spark JDBC against WRDS (Duo, connection limit, pgpass support in the driver).
