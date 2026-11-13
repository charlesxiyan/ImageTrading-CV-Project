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
- conflict test line
