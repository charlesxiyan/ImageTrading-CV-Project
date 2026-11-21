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

## 2026-11-19 — Big Data course Spark pipeline (in progress)
- Asked: build the Spark ingestion, profiling and cleaning submission in its own GitHub repo.
  Created a separate private repository (Big Data course only; no data in git).
- Done: Spark JDBC reads WRDS directly, using `~/.pgpass`, with no Duo prompt from the laptop.
  Ingested 11 CRSP, Compustat and TAQ link tables (2016-2024) and raw TAQ trades for the
  universe, 2023-03-06 to 03-17, all hours and conditions (about 24-41M trades a day, about
  550 MB a day). Rebuilt the universe and symbol map in Spark SQL: identical to the pandas
  pipeline row for row (1,185,097 and 130,467 rows). Spark bars for 2023-03-06 equal the
  pandas bars exactly.
- Found: under charles's stricter condition rule, 45% of bars have a different close, session
  volume halves, and the bar 1 open differs on 371 of 523 securities (auction prints dropped).
- Not verified yet: the full ten-day clean, bars and validation; the report is not written.

## 2026-11-19 — Big Data course submission built
- Done: full pipeline ran on 10 days (2023-03-06 to 03-17). 317,208,218 raw trades ingested
  through Spark JDBC (7.2 GB); 309,504,156 after cleaning, 88 outliers; 407,628 bars. Spark
  universe, symbol map and bars are identical to the pandas pipeline: 0 differing values.
- Found: 36% of regular-session trades set prices (odd lots are 64% of trades; 14% have no
  condition string). All 3,449 repeated (symbol, exchange, sequence) keys are correction
  records. Under charles's stricter condition rule, the first bar's open differs on 3,940 of
  5,226 security-days and session volume falls from 28.2B to 13.1B shares.
- Report PDF (9 pages) and code are ready; report and generated tables stay out of git because
  the snippets hold licensed rows. Not verified: any run outside local mode.

## 2026-11-20 — Explained the Spark pipeline
- Asked: explain how Spark is used in the Big Data course pipeline.
- Done: explanation only, from the code: JDBC ingestion with symbol predicates over 4
  connections, date-effective joins for the universe, filter steps plus a grouped
  `applyInPandas` outlier check, bars via `min_by`/`max_by` and a 78-bar grid, CRSP
  reconciliation. No code or data changed. STATUS.md is still correct.

## 2026-11-20 — Big Data report prose rewritten
- Asked: whether the report used the humanizer skill (it had not), and to rewrite it with it.
- Done: rewrote the prose of the Big Data course report; tables, figures, numbers and LaTeX
  structure unchanged. Recompiled: 10 pages, no errors. Left uncommitted in that repository.

## 2026-11-20 — Big Data report cut to 4 pages
- Asked: trim the report to the user's outline (sources, ingestion with a three-layer diagram,
  universe, profiling, three cleaning rules, bars checked against CRSP, snippets); drop
  validation, code, limitations and all mapping and condition-code detail.
- Done: rewrote report.tex to that outline, 4 pages in all; added two generated tables
  (scale without timing, cleaned-trade snippet). Not committed.
- Follow-up: removed the trade-distribution figure; the snippets section now has a candlestick
  figure of AAPL on 2023-03-10, readable and as a 234 x 234 CNN-style image (new
  report/make_chart.py, same geometry as chartcnn/images.py but candlesticks, which the current
  CNN does not use; it uses OHLC ticks). Still 4 pages, not committed.
- Follow-up: removed the universe membership figure too; candlestick figure is now Figure 2. Still 4 pages.
- Follow-up: pushed the 4-page report to a new Overleaf project for manual editing (kept the
  main file name proposal.tex, added only the generated inputs it needs). Local clone is report-overleaf.
- Follow-up: added layer 2 and 3 sizes to the report on Overleaf (on top of the user's own edit) and fixed the
  bar count: bars cover 5,226 stock-days, 5,225 of them have a priced bar and enter the CRSP comparison.
- Follow-up: added table notes under the two Layer 1 scale tables on Overleaf.
- Follow-up: removed Table 2 (reference table row counts) on Overleaf; Table 1 centered.
- Follow-up: removed Table 1 (daily trade counts) on Overleaf too.
- Follow-up: user wants to review the Spark code before packaging; listed files and open questions
  (layer names vs bronze/silver/gold, extra cleaning rules, which scripts to ship). Nothing packaged yet.
- Follow-up: packaged the Spark code (8 modules, tests, run scripts, rewritten README; no
  compare_rules, validate, smoke_jdbc or report/). Trimmed comments. Fresh unzip + new venv +
  full run for 2023-03-10 from WRDS succeeded; bars identical to the earlier run. Repo changes not committed.
- Follow-up: renamed the code package from tick2chart to project_data_ingestion (dir, imports,
  run commands, env var PROJECT_DATA_DIR). Zip is ~/myworkspace/nyu/bdad/project_data_ingestion.zip;
  fresh one-day run from WRDS passed, bars identical. Not committed.
