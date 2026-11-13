# Project instructions

## Shared memory

Two people work on this repository, each with their own Claude Code. Local Claude memory is not
shared, so everything a teammate's Claude must know lives in `memory/`, a checkout of the
`memory` branch that is the same on every code branch. **This repository is public.**

- **At session start** a hook runs `scripts/memory_sync.sh start`, which pulls `memory/` and
  prints `STATUS.md`, `DECISIONS.md` and the end of each work log. Treat that as current.
  If you did not see it, run that command yourself before doing anything else.
- **Before you end a turn in which you did project work**, append a dated entry to your own log
  `memory/log/NAME.md`, where NAME is `git config memory.author`. Say what was asked, what you
  did, what you found, and what you did not verify. Then make `memory/STATUS.md` true again.
  Add settled choices to `memory/DECISIONS.md` and new training runs to `memory/EXPERIMENTS.md`.
- **A hook pushes `memory/` when your turn ends.** It blocks once per session until your log has
  an entry. If the turn had nothing to do with the project, run `scripts/memory_sync.sh skip`.
  If it reports a conflict, resolve it in `memory/` and run `scripts/memory_sync.sh sync`.
- **Never edit a teammate's log.** `STATUS.md`, `DECISIONS.md` and `EXPERIMENTS.md` are shared.
- **Never write** credentials, account names, NetIDs, emails, personal paths, private links or
  licensed WRDS data into `memory/` or anywhere else in this repository.
- Rules go in this file. Facts, state and history go in `memory/`. See `memory/README.md`.
- If `memory/` is missing, run `scripts/memory_sync.sh start`. If `git config memory.author` is
  empty, ask the user for a short lowercase name and set it.

## Price adjustment policy

This is a hard rule for all data work. Dividends and share changes move the price without any change in value, so they must not look like returns to a model.

**Adjust history only, never the future.** Prices are put on the basis of the anchor session, the session of the last input bar:

- **Before the anchor session:** adjust prices and volumes for every event with an ex-date after that session, up to and including the anchor session.
- **Anchor session:** always unadjusted, so bar t keeps its traded price.
- **After the anchor:** never adjusted. An event with an ex-date after the anchor session must not affect its window, in any way.
- **Labels:** use unadjusted prices. C[t] and C[t+6] are in one session, and ex-dates take effect at the open, so no event falls between them.

For a bar in past session d and anchor session a:

```
price multiplier  = (cfacpr[a] / cfacpr[d]) * product over cash ex-dates e with d < e <= a of (1 - D[e] / P[e-1])
volume multiplier = cfacshr[d] / cfacshr[a]
```

`D[e]` is the cash paid per share and `P[e-1]` is the prior session's CRSP close, as an absolute value.

Rules that follow:

1. **Never use an absolute CRSP-adjusted price.** `prc / cfacpr` puts a price on the basis of the end of the data, so its level reveals splits that had not yet happened. Use only ratios between two sessions, as above.
2. **Compute per window.** Keep bars raw on disk. Adjusted prices depend on the anchor, so they cannot be stored as one series.
3. **Do not count an event twice.** `cfacpr` covers splits, stock dividends and spin-offs but not ordinary cash dividends. Cash adjustment applies only to `crsp.dsedist` rows with `divamt > 0` and `facpr = 0`. A spin-off can have both, as GE did on 2023-01-04; `cfacpr` alone handles it.
4. **Mergers end a series.** A delisted PERMNO has no more windows or labels, and its history is never joined to the acquirer's.

Sources: `crsp.dsf` for `prc`, `cfacpr`, `cfacshr`, `ret` and `retx`; `crsp.dsedist` for `distcd`, `divamt`, `facpr`, `facshr` and `exdt`; `crsp.dsedelist` for `dlstcd`, `dlret`, `nwperm` and `dlamt`. TAQ prices are not adjusted.
