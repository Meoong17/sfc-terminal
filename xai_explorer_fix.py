# ── Helper: map LIME range names back to clean feature names ──

def _clean_lime_name(name: str) -> str:
    """Convert 'M20_OBI <= -0.58' or '-0.68 < M6_Regime' back to clean 'M20_OBI' or 'M6_Regime'."""
    import re
    m = re.search(r'([A-Z]\d+_[A-Za-z]+)', name)
    if m:
        return m.group(1)
    return name
