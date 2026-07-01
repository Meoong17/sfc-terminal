#!/usr/bin/env python3
"""
QLSTM Watchdog — monitors qlstm_daemon.py health and auto-restarts.
===============================================================

Checklist:
  1. Daemon process alive? (PID file check)
  2. Cache fresh? (.qlstm_cache.json < 45 min old)
  3. If either fails → kill stale + restart daemon

Designed to run from cron every 10 minutes or from collect.py's cycle.
"""
import json, os, sys, time, signal, subprocess, logging

SFC_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(SFC_DIR, ".qlstm_daemon.pid")
CACHE_FILE = os.path.join(SFC_DIR, ".qlstm_cache.json")
DAEMON_SCRIPT = os.path.join(SFC_DIR, "qlstm_daemon.py")
LOG_FILE = os.path.join(SFC_DIR, ".qlstm_watchdog.log")

MAX_CACHE_AGE = 2700  # 45 minutes — daemon runs every 30 min, allow 15 min grace

logging.basicConfig(
    level=logging.INFO,
    format="[QLSTM-watchdog] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger(__name__)


def _read_pid():
    """Read PID from file, return int or None."""
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_pid(pid):
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def _pid_alive(pid):
    """Check if process with given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _cache_age():
    """Return age of cache file in seconds, or None if missing."""
    try:
        return time.time() - os.path.getmtime(CACHE_FILE)
    except FileNotFoundError:
        return None


def _stop_stale_daemon(pid):
    """Graceful SIGTERM, then SIGKILL if it won't die."""
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 5s for graceful shutdown
        for _ in range(50):
            time.sleep(0.1)
            if not _pid_alive(pid):
                log.info(f"Stopped daemon PID={pid}")
                return
        # Force kill
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        log.warning(f"Force-killed daemon PID={pid}")
    except (OSError, ProcessLookupError):
        pass  # already dead


def _start_daemon():
    """Start qlstm_daemon.py in background, return PID."""
    # Use the sfc .venv python so it has pennylane + torch
    venv_python = os.path.join(SFC_DIR, ".venv", "bin", "python3")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    try:
        proc = subprocess.Popen(
            [venv_python, DAEMON_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=SFC_DIR,
            start_new_session=True,  # detach from watchdog
        )
        _write_pid(proc.pid)
        log.info(f"Started daemon PID={proc.pid}")
        return proc.pid
    except Exception as e:
        log.error(f"Failed to start daemon: {e}")
        return None


def check_and_restart():
    """
    Main watchdog check. Returns True if daemon is healthy.
    Called by cron or collect.py every 10 minutes.
    """
    pid = _read_pid()
    cache_age = _cache_age()
    problems = []

    # Check 1: PID alive?
    if pid and _pid_alive(pid):
        pass  # daemon process exists
    else:
        if pid:
            problems.append(f"PID={pid} is dead")
        else:
            problems.append("no PID file")
        pid = None

    # Check 2: Cache fresh?
    if cache_age is None:
        problems.append("cache file missing")
    elif cache_age > MAX_CACHE_AGE:
        problems.append(f"cache age={cache_age:.0f}s (max={MAX_CACHE_AGE}s)")

    if not problems:
        # All healthy
        return True

    log.warning(f"Problems detected: {'; '.join(problems)}")

    # Stop any stale process
    if pid:
        _stop_stale_daemon(pid)
        # Clean up stale PID
        try:
            os.remove(PID_FILE)
        except FileNotFoundError:
            pass

    # Start fresh daemon
    new_pid = _start_daemon()
    if new_pid:
        # Give it 10s for first inference cycle then verify cache
        time.sleep(15)
        new_age = _cache_age()
        if new_age is not None and new_age < 120:
            log.info(f"Daemon healthy: cache age={new_age:.0f}s")
            return True
        else:
            log.warning("Daemon started but no fresh cache yet (may need model retrain)")
            return False
    else:
        log.error("Failed to restart daemon")
        return False


if __name__ == "__main__":
    ok = check_and_restart()
    if ok:
        print("OK: QLSTM daemon is healthy")
    else:
        print("WARN: QLSTM daemon issues detected — check log for details")
    sys.exit(0 if ok else 1)
