#!/usr/bin/env python3
"""
inject_data.py — Inject live data.json into index.html for offline-first serving
Uses placeholder comment marker for reliable injection.
"""

import json, sys


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

    # Find the placeholder marker and replace the null after it
    marker = "/* __EMBEDDED_DATA_PLACEHOLDER__ */\nconst __EMBEDDED_DATA = null;"
    replacement = f"/* __EMBEDDED_DATA_PLACEHOLDER__ */\nconst __EMBEDDED_DATA = {data_json};"

    new_html = html.replace(marker, replacement)

    if new_html == html:
        # Fallback: try to find any const __EMBEDDED_DATA = null;
        fallback = "const __EMBEDDED_DATA = null;"
        fallback_repl = f"const __EMBEDDED_DATA = {data_json};"
        new_html = html.replace(fallback, fallback_repl)

    if new_html == html:
        print(f"⚠ Could not find __EMBEDDED_DATA placeholder in {html_path}", file=sys.stderr)
        return False

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
