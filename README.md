# Shared memory

This branch holds what every teammate's Claude Code session must know. It is checked out as the
`memory/` folder inside each clone, so it is the same no matter which code branch is in use.
`scripts/memory_sync.sh` on `main` pulls it when a session starts and pushes it when a turn ends.

**This repository is public.** Never write credentials, account names, NetIDs, emails, personal
paths, private links, or licensed WRDS data here. Aggregate statistics are fine.

| File | Holds | How to edit |
|---|---|---|
| `STATUS.md` | What is true now: data, code, experiments, who is doing what, next steps | Rewrite in place. Keep under about 60 lines. |
| `DECISIONS.md` | Decisions that are settled, each with its reason | Add or amend. Do not delete without the team agreeing. |
| `EXPERIMENTS.md` | Index of training runs: setup, data version, result, where artifacts live | One row per run. |
| `log/NAME.md` | One person's work log, newest entry last | Append only, and only your own file. |

A log entry is dated and short: what was asked, what was done, what was found, what was not
verified. Record mistakes too. Rules that Claude must follow belong in `CLAUDE.md` on `main`,
not here.
