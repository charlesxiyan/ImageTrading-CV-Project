#!/usr/bin/env bash
# Collect what the TAQ bar build needs to know about an NYU HPC cluster.
#
# Run from a login node:  bash hpc_envcheck.sh [--account SLURM_ACCOUNT] [--wrds-user USERNAME]
#
# It submits short CPU jobs, installs wrds, pandas and pyarrow into a venv under $SCRATCH,
# and writes ~/hpc_envcheck_TIMESTAMP.txt. The report holds no passwords, but it does hold
# account names, so keep it out of this public repository.

WRDS_HOST=wrds-pgdata.wharton.upenn.edu
WRDS_PORT=9737
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
WORK=${SCRATCH:-$HOME}/imagetrading-envcheck
VENV=$WORK/venv
MODE=main
ACCOUNT=
WRDS_USER=${WRDS_USER:-}

while [ $# -gt 0 ]; do
    case $1 in
        --account) ACCOUNT=$2; shift 2 ;;
        --wrds-user) WRDS_USER=$2; shift 2 ;;
        --probe|--install|--wrds) MODE=${1#--}; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1

section() { printf '\n===== %s =====\n' "$1"; }

show() {
    printf '$ %s\n' "$*"
    timeout 60 "$@" 2>&1 || printf '[exit %s]\n' "$?"
}

port_state() {
    if timeout 8 bash -c "</dev/tcp/$WRDS_HOST/$WRDS_PORT" 2>/dev/null; then echo open; else echo closed; fi
}

http_state() {
    local code
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "$1" 2>/dev/null)
    echo "${code:-000}"
}

# Prints the checks and leaves the results in NET_WRDS and NET_PYPI.
network() {
    env | grep -i '_proxy=' | sed -E 's#//[^/@]*@#//<hidden>@#'
    getent hosts "$WRDS_HOST" 2>&1 || echo "$WRDS_HOST does not resolve"
    NET_WRDS=$(port_state)
    NET_PYPI=$(http_state https://pypi.org/simple/wrds/)
    echo "wrds_port=$NET_WRDS pypi_http=$NET_PYPI github_http=$(http_state https://github.com)"
}

python_ok() {
    "$1" -c 'import sys; sys.exit(not (3, 9) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null
}

# Sets PY, and records any module it needs so later jobs load the same one.
find_python() {
    PY=
    local candidate name
    [ -f "$WORK/python_module" ] && module load "$(cat "$WORK/python_module")" 2>/dev/null
    for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then PY=$candidate; return 0; fi
    done
    command -v module >/dev/null 2>&1 || return 1
    for name in $(module -t avail python anaconda3 miniconda3 2>&1 | grep -E '^(python|anaconda|miniconda)' | sort -rV); do
        module load "$name" 2>/dev/null
        if command -v python3 >/dev/null 2>&1 && python_ok python3; then
            PY=python3
            echo "$name" > "$WORK/python_module"
            return 0
        fi
        module unload "$name" 2>/dev/null
    done
    return 1
}

install_venv() {
    if ! find_python; then echo "No Python 3.9-3.13 found, on PATH or in modules"; return 1; fi
    echo "Using $(command -v "$PY") ($("$PY" --version 2>&1)), module: $(cat "$WORK/python_module" 2>/dev/null || echo none)"
    [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" || return 1
    "$VENV/bin/pip" install -q wrds pyarrow 'pandas==2.2.3' 2>&1 | tail -5
    "$VENV/bin/python" -c 'import wrds, pandas, pyarrow; print("pandas", pandas.__version__, "pyarrow", pyarrow.__version__, "wrds ok")'
}

write_wrds_test() {
    cat > "$WORK/wrds_test.py" <<'EOF'
import os
import time
import wrds

user = os.environ['WRDS_USER']
start = time.time()
db = wrds.Connection(wrds_username=user)
print('first connection seconds', round(time.time() - start, 1), flush=True)
start = time.time()
rows = db.raw_sql("""SELECT time_m::text AS t, price::float8 AS p, size, tr_scond, tr_corr
    FROM taqmsec.ctm_20231229 WHERE sym_root='AAPL' AND sym_suffix IS NULL""")
elapsed = time.time() - start
print('AAPL 2023-12-29 rows', len(rows), 'seconds', round(elapsed, 1), 'rows per second', int(len(rows) / elapsed), flush=True)
db.close()
held = []
for i in range(8):
    start = time.time()
    try:
        held.append(wrds.Connection(wrds_username=user))
        print('connection', i + 1, 'ok, seconds', round(time.time() - start, 1), flush=True)
    except BaseException as error:
        print('connection', i + 1, 'failed:', type(error).__name__, str(error)[:200], flush=True)
        break
for connection in held:
    connection.close()
print('simultaneous connections held', len(held), flush=True)
EOF
}

run_wrds_test() {
    find_python >/dev/null 2>&1
    hostname
    WRDS_USER=$WRDS_USER timeout 300 "$VENV/bin/python" "$WORK/wrds_test.py" </dev/null 2>&1 | grep -v 'Loading library\|^Done'
    echo "wrds test exit ${PIPESTATUS[0]}"
}

# Modes below run inside Slurm jobs started by the main mode.
case $MODE in
    probe)
        section "Compute node"
        hostname
        echo "partition=$SLURM_JOB_PARTITION qos=$SLURM_JOB_QOS"
        show nproc
        show free -g
        lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread'
        section "Compute node network"
        network
        echo "NET_WRDS=$NET_WRDS NET_PYPI=$NET_PYPI" > "$WORK/compute_net"
        exit 0 ;;
    install) install_venv; exit $? ;;
    wrds) run_wrds_test; exit 0 ;;
esac

compute() {
    local flags=(--cpus-per-task=2 --mem=8G --time=00:20:00 --job-name=envcheck)
    [ -n "$ACCOUNT" ] && flags+=(--account="$ACCOUNT")
    timeout 1500 srun "${flags[@]}" bash "$SELF" "$1" --wrds-user "$WRDS_USER" </dev/null 2>&1
    echo "[srun $1 exit $?]"
}

mkdir -p "$WORK" && chmod 700 "$WORK"
REPORT=$HOME/hpc_envcheck_$(date +%Y%m%d_%H%M%S).txt
exec 3>&1 4>&2
exec > >(tee -a "$REPORT") 2>&1

section "Host"
hostname; date; whoami; echo "shell=$SHELL HOME=$HOME SCRATCH=${SCRATCH:-unset} ARCHIVE=${ARCHIVE:-unset}"
grep -E '^(NAME|VERSION)=' /etc/os-release 2>/dev/null

section "Storage"
show myquota
show ls -ld "$SCRATCH"
show df -h "$SCRATCH"

section "Slurm"
show my_slurm_accounts
show show_slurm_qos
show sinfo -o '%P %a %l %D %c %m %G'
show sacctmgr -nP show assoc user="$USER" format=account,partition,qos,maxwall
if [ -z "$ACCOUNT" ]; then
    ACCOUNT=$(my_slurm_accounts 2>/dev/null | awk 'NR > 1 && $1 !~ /^-/ {print $1; exit}')
fi
echo "Using Slurm account: ${ACCOUNT:-none}"
# Asks the scheduler about the shapes the build could use, without running anything.
for wall in 06:00:00 2-00:00:00 7-00:00:00; do
    show sbatch --test-only ${ACCOUNT:+--account="$ACCOUNT"} --cpus-per-task=8 --mem=32G --time="$wall" --wrap=hostname
done

section "Software"
for tool in python3 python3.13 python3.12 python3.11 python3.10 python3.9 conda singularity apptainer git; do
    printf '%s: %s\n' "$tool" "$(command -v "$tool" || echo missing)"
done
python3 --version 2>&1
if command -v module >/dev/null 2>&1; then
    module -t avail python anaconda3 miniconda3 2>&1 | grep -v ':$' | tail -30
else
    echo "module command not available"
fi

section "Credentials file, passwords hidden"
ls -l "$HOME/.pgpass" 2>&1
if [ -f "$HOME/.pgpass" ]; then
    awk -F: '{print $1 ":" $2 ":" $3 ":" $4 ":<hidden>"}' "$HOME/.pgpass"
    [ -n "$WRDS_USER" ] || WRDS_USER=$(awk -F: -v h="$WRDS_HOST" '$1 == h && $4 != "*" {print $4; exit}' "$HOME/.pgpass")
fi

section "Login node network"
network
LOGIN_WRDS=$NET_WRDS
LOGIN_PYPI=$NET_PYPI

section "Data transfer node network, skipped if it needs an interactive login"
timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 dtn.torch.hpc.nyu.edu \
    "hostname; timeout 8 bash -c '</dev/tcp/$WRDS_HOST/$WRDS_PORT' && echo wrds_port=open || echo wrds_port=closed" 2>&1 | tail -3

section "Compute job"
rm -f "$WORK/compute_net"
compute --probe
NET_WRDS=unknown NET_PYPI=000
[ -f "$WORK/compute_net" ] && source "$WORK/compute_net"
COMPUTE_WRDS=$NET_WRDS
COMPUTE_PYPI=$NET_PYPI

section "Python environment in $VENV"
if [ "$COMPUTE_PYPI" = 200 ]; then
    compute --install
else
    echo "Compute node cannot reach PyPI, installing from the login node"
    install_venv
fi

section "WRDS credentials"
if [ ! -f "$HOME/.pgpass" ] && [ -t 0 ] && [ "$LOGIN_WRDS" = open ] && [ -x "$VENV/bin/python" ]; then
    # This prompt stays out of the report. Answer y when it offers to create .pgpass.
    echo "No ~/.pgpass. Follow the WRDS prompts on screen, and approve a Duo push if one arrives."
    "$VENV/bin/python" -c "import wrds; wrds.Connection().close()" >&3 2>&4
    echo "Interactive connection exit $?"
fi
[ -n "$WRDS_USER" ] || [ ! -f "$HOME/.pgpass" ] || WRDS_USER=$(awk -F: -v h="$WRDS_HOST" '$1 == h && $4 != "*" {print $4; exit}' "$HOME/.pgpass")
ls -l "$HOME/.pgpass" 2>&1

section "WRDS test: approve any Duo push and count how many arrive"
WRDS_WHERE=none
if [ ! -f "$HOME/.pgpass" ] || [ -z "$WRDS_USER" ] || [ ! -x "$VENV/bin/python" ]; then
    echo "Skipped: needs ~/.pgpass, a WRDS username and the venv"
else
    write_wrds_test
    if [ "$COMPUTE_WRDS" = open ]; then
        WRDS_WHERE=compute
        compute --wrds
    elif [ "$LOGIN_WRDS" = open ]; then
        WRDS_WHERE=login
        echo "Compute node cannot reach WRDS, testing from the login node"
        run_wrds_test
    else
        echo "Skipped: no node here reaches WRDS"
    fi
fi

section "Summary"
echo "login:   wrds_port=$LOGIN_WRDS pypi=$LOGIN_PYPI"
echo "compute: wrds_port=$COMPUTE_WRDS pypi=$COMPUTE_PYPI"
echo "account=${ACCOUNT:-none} wrds_test_ran_on=$WRDS_WHERE venv=$([ -x "$VENV/bin/python" ] && echo yes || echo no)"
if [ -t 0 ]; then
    read -r -p "How many Duo pushes arrived during this script, and at which step? " DUO
    echo "duo_pushes: $DUO"
fi
echo "Report: $REPORT"
sleep 1
