#!/usr/bin/env python3
"""
inject_data.py — Inject live data.json into app.js for offline-first serving
Uses placeholder comment marker for reliable injection.
Default: injects into app.js (since JS was extracted from index.html).
Pass --html to inject into index.html instead (legacy mode).
"""

import json, re, sys


def inject_data_into_html(data_path, html_path, output_path=None):
    if output_path is None:
        output_path = html_path

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠ Failed to load {data_path}: {e}", file=sys.stderr)
        return False

    # Merge QLSTM training history (train/val loss) for the Research chart.
    # qlstm_history.json lives next to data.json and is NOT a data.json field.
    try:
        import os as _os
        qh_path = _os.path.join(_os.path.dirname(_os.path.abspath(data_path)), 'qlstm_history.json')
        with open(qh_path, 'r') as qf:
            qh = json.load(qf)
        if isinstance(qh, dict):
            if 'train_loss' in qh: data['qlstm_train'] = qh['train_loss']
            if 'val_loss' in qh:   data['qlstm_val']   = qh['val_loss']
    except Exception:
        pass  # QLSTM chart simply stays empty if history is unavailable

    try:
        with open(html_path, 'r') as f:
            html = f.read()
    except Exception as e:
        print(f"⚠ Failed to load {html_path}: {e}", file=sys.stderr)
        return False

    data_json = json.dumps(data)
    new_html = html

    # Strategy 1: Find placeholder comment + null marker (app.js)
    marker = "/* __EMBEDDED_DATA_PLACEHOLDER__ */\nvar __EMBEDDED_DATA = null;"
    if marker in html:
        replacement = f"/* __EMBEDDED_DATA_PLACEHOLDER__ */\nvar __EMBEDDED_DATA = {data_json};"
        new_html = html.replace(marker, replacement)
        print(f"✅ Injected via placeholder (strategy 1) into {output_path}", file=sys.stderr)
        with open(output_path, 'w') as f:
            f.write(new_html)
        print(f"✅ Injected {len(data)} fields", file=sys.stderr)
        return True

    # Strategy 2: Find placeholder comment + any existing data assignment
    pattern = r'(/\* __EMBEDDED_DATA_PLACEHOLDER__ \*/\n)(?:const|var|let) __EMBEDDED_DATA = \{.*?\};'
    match = re.search(pattern, new_html, re.DOTALL)
    if match:
        replacement = f"{match.group(1)}var __EMBEDDED_DATA = {data_json};"
        new_html = new_html[:match.start()] + replacement + new_html[match.end():]
        print(f"✅ Injected via regex (strategy 2) into {output_path}", file=sys.stderr)
        with open(output_path, 'w') as f:
            f.write(new_html)
        print(f"✅ Injected {len(data)} fields", file=sys.stderr)
        return True

    # Strategy 3: Generic fallback — find any `__EMBEDDED_DATA =`
    pattern3 = r'(?:const|var|let) __EMBEDDED_DATA\s*=\s*\{.*?\};'
    match3 = re.search(pattern3, new_html, re.DOTALL)
    if match3:
        new_html = new_html[:match3.start()] + f"var __EMBEDDED_DATA = {data_json};" + new_html[match3.end():]
        print(f"✅ Injected via generic match (strategy 3) into {output_path}", file=sys.stderr)
        with open(output_path, 'w') as f:
            f.write(new_html)
        print(f"✅ Injected {len(data)} fields", file=sys.stderr)
        return True

    print(f"⚠ Could not find __EMBEDDED_DATA in {html_path}", file=sys.stderr)
    return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <data.json> <target.js> [output.js]", file=sys.stderr)
        print(f"  Default target: app.js  (use app.js for new format, index.html for legacy)", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1]
    html_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    success = inject_data_into_html(data_path, html_path, output_path)
    sys.exit(0 if success else 1)
