#!/usr/bin/env python3
"""
inject_data.py — Inject live data.json into index.html for offline-first serving
"""

import json, sys, re

def inject_data_into_html(data_path, html_path, output_path=None):
    if output_path is None:
        output_path = html_path

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠ Failed to load {data_path}: {e}", file=sys.stderr)
        return False

    try:
        with open(html_path, 'r') as f:
            html = f.read()
    except Exception as e:
        print(f"⚠ Failed to load {html_path}: {e}", file=sys.stderr)
        return False

    data_json = json.dumps(data)
    
    # Pattern untuk __EMBEDDED_DATA
    pattern = r'(const __EMBEDDED_DATA = )({[^}]*?_generated_at[^}]*?});'
    new_html = re.sub(pattern, r'\1' + data_json + ';', html, flags=re.DOTALL)
    
    if new_html == html:
        alt_pattern = r'(const __EMBEDDED_DATA = )\{.*?\};'
        new_html = re.sub(alt_pattern, r'\1' + data_json + ';', html, flags=re.DOTALL)

    try:
        with open(output_path, 'w') as f:
            f.write(new_html)
        print(f"✅ Injected {len(data)} fields into {output_path}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"⚠ Failed to write {output_path}: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <data.json> <index.html> [output.html]", file=sys.stderr)
        sys.exit(1)
    
    data_path = sys.argv[1]
    html_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    success = inject_data_into_html(data_path, html_path, output_path)
    sys.exit(0 if success else 1)
