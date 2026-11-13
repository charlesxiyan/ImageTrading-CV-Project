#!/usr/bin/env bash
# Keep the shared memory/ folder in sync with GitHub.
#
# memory/ is a git worktree of the orphan branch `memory`, so it is the same on every code
# branch. Claude Code hooks in .claude/settings.json call `start` and `stop`; run `sync` by hand.
#
#   memory_sync.sh start   pull, then print the shared context (SessionStart hook)
#   memory_sync.sh stop    require a log entry, then commit and push (Stop hook)
#   memory_sync.sh sync    pull, commit and push now
#   memory_sync.sh skip    this session changed nothing worth logging

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MEM=$REPO/memory
BRANCH=memory
STATE=$(cd "$REPO" && cd "$(git rev-parse --git-common-dir)" && pwd)/memory-sync
LOG_TAIL=40
mkdir -p "$STATE"

author() { git -C "$REPO" config memory.author; }

json_field() {  # json_field NAME, reading hook JSON from $INPUT
    printf '%s' "$INPUT" | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\).*/\1/p" | head -n 1
}

ensure_worktree() {
    git -C "$MEM" rev-parse --is-inside-work-tree >/dev/null 2>&1 && return 0
    git -C "$REPO" fetch -q origin "$BRANCH" 2>/dev/null
    git -C "$REPO" worktree prune
    if git -C "$REPO" show-ref -q --verify "refs/heads/$BRANCH"; then
        git -C "$REPO" worktree add -q "$MEM" "$BRANCH"
    elif git -C "$REPO" show-ref -q --verify "refs/remotes/origin/$BRANCH"; then
        git -C "$REPO" worktree add -q --track -b "$BRANCH" "$MEM" "origin/$BRANCH"
    else
        echo "memory: branch '$BRANCH' does not exist on origin yet" >&2
        return 1
    fi
}

pull() {
    git -C "$MEM" pull -q --rebase --autostash origin "$BRANCH" 2>&1 && return 0
    git -C "$MEM" rebase --abort 2>/dev/null
    return 1
}

# Refuse to publish anything that looks like a credential: the repository is public.
leak_check() {
    git -C "$MEM" grep -n -I -E 'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|sk-[A-Za-z0-9_-]{20,}|BEGIN [A-Z ]*PRIVATE KEY|(password|passwd|secret|token)[[:space:]]*[:=][[:space:]]*[^[:space:]<]{6,}' -- . 2>/dev/null
}

push() {
    git -C "$MEM" add -A
    if ! git -C "$MEM" diff --cached --quiet; then
        if hits=$(leak_check) && [ -n "$hits" ]; then
            git -C "$MEM" reset -q
            echo "memory: not pushed, possible credential in: $hits" >&2
            return 1
        fi
        git -C "$MEM" commit -q -m "memory: $(author || echo unknown) $(date +%Y-%m-%d)" || return 1
    fi
    [ -z "$(git -C "$MEM" log --oneline "origin/$BRANCH..HEAD" 2>/dev/null)" ] && return 0
    pull || { echo "memory: pull failed or conflicted; resolve in $MEM, then run scripts/memory_sync.sh sync" >&2; return 1; }
    git -C "$MEM" push -q origin "HEAD:$BRANCH" 2>&1 || { echo "memory: push failed" >&2; return 1; }
}

show() {
    local name; name=$(author)
    echo "=== Shared project memory, branch '$BRANCH', synced $(date '+%Y-%m-%d %H:%M') ==="
    if [ -z "$name" ]; then
        echo "memory.author is not set. Ask the user for a short lowercase name, then run:"
        echo "  git -C \"$REPO\" config memory.author NAME"
    else
        echo "You write the log memory/log/$name.md. Other files in memory/log/ belong to teammates."
    fi
    for file in STATUS.md DECISIONS.md; do
        [ -f "$MEM/$file" ] && { echo; echo "----- memory/$file -----"; cat "$MEM/$file"; }
    done
    for file in "$MEM"/log/*.md; do
        [ -f "$file" ] || continue
        echo; echo "----- memory/log/$(basename "$file"), last $LOG_TAIL lines -----"
        tail -n "$LOG_TAIL" "$file"
    done
    echo; echo "Also in memory/: EXPERIMENTS.md and README.md. Read them when relevant."
}

INPUT=
case ${1:-} in
    start)
        [ -t 0 ] || INPUT=$(cat)
        session=$(json_field session_id)
        ensure_worktree || { echo "Shared memory is unavailable: see scripts/memory_sync.sh."; exit 0; }
        pull >/dev/null || echo "WARNING: could not pull memory; what follows may be stale."
        [ -n "$session" ] && touch "$STATE/$session.start"
        find "$STATE" -name '*.start' -mtime +14 -delete 2>/dev/null
        show
        ;;
    stop)
        [ -t 0 ] || INPUT=$(cat)
        session=$(json_field session_id)
        again=$(json_field stop_hook_active)
        ensure_worktree || exit 0
        mark=$STATE/$session.start
        log=$MEM/log/$(author).md
        if [ -f "$STATE/skip" ]; then
            rm -f "$STATE/skip" "$mark"
        elif [ -n "$session" ] && [ -f "$mark" ] && [ "$again" != true ] && [ -n "$(author)" ] && ! [ "$log" -nt "$mark" ]; then
            printf '{"decision":"block","reason":"Shared memory: this session has no log entry yet. Append a dated entry to %s and update memory/STATUS.md if it is now wrong. If nothing project-related happened, run: %s/scripts/memory_sync.sh skip"}\n' "$log" "$REPO"
            exit 0
        fi
        if ! out=$(push 2>&1); then
            [ "$again" = true ] && exit 0
            echo "$out" >&2
            exit 2
        fi
        ;;
    sync)
        ensure_worktree || exit 1
        pull || { echo "memory: pull failed or conflicted in $MEM" >&2; exit 1; }
        push && echo "memory: in sync with origin/$BRANCH"
        ;;
    skip)
        touch "$STATE/skip" && echo "memory: no log entry required for this turn"
        ;;
    *)
        echo "usage: $0 start|stop|sync|skip" >&2
        exit 2
        ;;
esac
