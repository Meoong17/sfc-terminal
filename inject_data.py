#!/usr/bin/env python3
"""
inject_data.py — Inject live data.json into index.html for offline-first serving
Uses placeholder comment marker for reliable injection.
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

    try:
        with open(html_path, 'r') as f:
            html = f.read()
    except Exception as e:
        print(f"⚠ Failed to load {html_path}: {e}", file=sys.stderr)
        return False

    data_json = json.dumps(data)
    new_html = html

    # Strategy 1: Find placeholder comment + null marker
    marker = "/* __EMBEDDED_DATA_PLACEHOLDER__ */\nconst __EMBEDDED_DATA = null;"
    if marker in html:
        replacement = f"/* __EMBEDDED_DATA_PLACEHOLDER__ */\nconst __EMBEDDED_DATA = {data_json};"
        new_html = html.replace(marker, replacement)
        print(f"✅ Injected via placeholder (strategy 1)", file=sys.stderr)
        with open(output_path, 'w') as f:
            f.write(new_html)
        print(f"✅ Injected {len(data)} fields into {output_path}", file=sys.stderr)
        return True

    # Strategy 2: Find placeholder comment + any existing data assignment
    pattern = r'(/\* __EMBEDDED_DATA_PLACEHOLDER__ \*/\nconst __EMBEDDED_DATA = )\{.*?\};'
    match = re.search(pattern, new_html, re.DOTALL)
    if match:
        replacement = f"{match.group(1)}{data_json};"
        new_html = new_html[:match.start()] + replacement + new_html[match.end():]
        print(f"✅ Injected via regex (strategy 2)", file=sys.stderr)
        with open(output_path, 'w') as f:
            f.write(new_html)
        print(f"✅ Injected {len(data)} fields into {output_path}", file=sys.stderr)
        return True

    # Strategy 3: Generic fallback — find any `const __EMBEDDED_DATA =`
    pattern3 = r'const __EMBEDDED_DATA\s*=\s*\{.*?\};'
    match3 = re.search(pattern3, new_html, re.DOTALL)
    if match3:
        new_html = new_html[:match3.start()] + f"const __EMBEDDED_DATA = {data_json};" + new_html[match3.end():]
        print(f"✅ Injected via generic const match (strategy 3)", file=sys.stderr)
        with open(output_path, 'w') as f:
            f.write(new_html)
        print(f"✅ Injected {len(data)} fields into {output_path}", file=sys.stderr)
        return True

    print(f"⚠ Could not find __EMBEDDED_DATA in {html_path}", file=sys.stderr)
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
