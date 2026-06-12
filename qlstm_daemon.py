#!/usr/bin/env python3
"""
QLSTM Daemon — runs QLSTM inference in background every 30 minutes.
Writes result to .qlstm_cache.json for collect.py to read instantly.
Uses subprocess to invoke collect.py's exact environment.
"""
import json, os, sys, time, signal, logging, subprocess

SFC_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SFC_DIR, ".qlstm_cache.json")

logging.basicConfig(
    level=logging.INFO,
    format="[QLSTM-daemon] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

running = True

def signal_handler(signum, frame):
    global running
    log.info("Shutting down...")
    running = False

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def run_inference():
    """Run one QLSTM+Hybrid inference cycle via subprocess."""
    code = '''
import sys, os, json, time
sfc2_dir = os.path.join("{sfc_dir}", "..", "sfc2")
venv_path = os.path.join(sfc2_dir, "venv", "lib", "python3.12", "site-packages")
if os.path.isdir(venv_path): sys.path.insert(0, venv_path)
sys.path.insert(0, sfc2_dir)
from qlstm_enhanced import run_enhanced_inference
result = run_enhanced_inference(force=True)
out = {{
    "ts": time.time(),
    "qlstm_pred": result.get("qlstm_pred"),
    "qlstm_pred_0_1": result.get("qlstm_pred", 0) / 100.0,
    "qlstm_ok": result.get("qlstm_ok", False),
    "garch_residual": result.get("garch_residual", 0),
    "garch_volatility": result.get("garch_volatility", 0),
    "proadapt_weight": result.get("proadapt_weight", 0.5),
    "proadapt_final": result.get("proadapt_final", result.get("qlstm_pred", 0)),
    "xai_top_features": result.get("xai_top_features", None),
    "qlstm_error": result.get("error"),
}}
print(json.dumps(out))
'''.format(sfc_dir=SFC_DIR)
    
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
            cwd=SFC_DIR,
        )
        
        if proc.returncode != 0:
            log.warning(f"Inference failed (rc={proc.returncode}): {proc.stderr[-200:]}")
            return None
        
        # Parse JSON output from last line
        for line in reversed(proc.stdout.strip().split('\n')):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            log.warning("No JSON output found")
            return None
        
        if not data.get("qlstm_ok"):
            log.warning(f"Inference failed: {data.get('qlstm_error', 'unknown')}")
            return data  # store failure too
        
        log.info(f"QLSTM pred={data['qlstm_pred']:.1f}% | "
                 f"garch={data.get('garch_residual',0):+.3f} | "
                 f"adapt={data.get('proadapt_weight',0.5):.2f}")
        return data
    
    except subprocess.TimeoutExpired:
        log.error("Inference timed out")
        return None
    except Exception as e:
        log.error(f"Inference error: {e}")
        return None

def write_cache(data):
    """Atomically write cache file."""
    if data is None:
        return
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.rename(tmp, CACHE_FILE)

def main():
    log.info("Starting QLSTM daemon (interval=30min)")
    
    # Run first inference immediately
    data = run_inference()
    write_cache(data)
    if data and data.get("qlstm_ok"):
        log.info("Initial inference cached")
    else:
        log.warning("Initial inference failed, will retry")
    
    interval = 1800  # 30 minutes
    
    while running:
        try:
            for _ in range(interval):
                if not running:
                    break
                time.sleep(1)
            
            if not running:
                break
            
            log.info("Running scheduled inference...")
            data = run_inference()
            write_cache(data)
            if data and data.get("qlstm_ok"):
                log.info("Cached")
        except Exception as e:
            log.error(f"Cycle error: {e}")
    
    log.info("Daemon stopped")

if __name__ == "__main__":
    main()
