#!/usr/bin/env python3
"""Run the archive downloader and persist its exit status for unattended jobs."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parent
output = Path(sys.argv[1]).resolve()
command = [sys.executable, '-u', str(project/'download_us_equity_archive.py'), '--output', str(output)]
result = subprocess.run(command, cwd=project)
status_path = output/'00_metadata'/'job_exit.json'
status_path.write_text(json.dumps({
    'exit_code': result.returncode,
    'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'status': 'finished' if result.returncode == 0 else 'failed',
    'next_step': 'Inspect download.log and progress.json; rerun the same command to resume.'
}, indent=2)+'\n')
sys.exit(result.returncode)
