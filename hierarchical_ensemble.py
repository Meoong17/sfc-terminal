"""
hierarchical_ensemble.py — Layer 2: cluster-level meta-feature aggregation
=============================================================================

Solves a specific, verified problem: `collect.py`'s calculate_sfc_ensemble()
and its factor adjustments (`factors["Lt"] += ...`, etc.) currently combine
RAW method scores — but method_independence_analysis.py found real pairs
with |correlation| > 0.9 among them (m2_logit vs m5_qreg: 0.996, m4_ewc vs
m32_mamba: 0.940, etc.). Summing two 0.99-correlated scores with separate
weights doesn't add two independent pieces of evidence — it effectively
double-counts one piece of evidence with combined weight, silently
inflating that signal's influence relative to genuinely independent ones.

This module takes the cluster assignments from method_independence_analysis.py
and collapses each redundant cluster into ONE meta-feature, so downstream
consumers (Layer 3: domain factor building, Layer 4: final ensemble) work
with genuinely independent inputs.

AGGREGATION METHOD — why simple z-scored mean, not PCA:
    PCA's first principal component is a common choice for this, but has
    two practical downsides in an audited, explainable pipeline like this
    one: (1) sign ambiguity — PC1 can come out positively OR negatively
    oriented relative to the original features depending on the data,
    which would require extra logic to detect and correct, or every
    retraining could silently flip a factor's sign; (2) it's harder to
    explain in a dashboard tooltip than "the average of these N
    correlated signals, each standardized first." Z-scored mean is simpler,
    auditable, and — for a cluster of already highly-correlated inputs —
    captures nearly the same signal PC1 would (when correlation exceeds
    ~0.85, the two approaches are numerically very close; the gap that
    matters is at LOW correlation, which is exactly the case being
    excluded by clustering in the first place).

USAGE:
    from method_independence_analysis import analyze_method_independence
    from hierarchical_ensemble import build_cluster_meta_features

    result = analyze_method_independence(methods_history)
    meta_features = build_cluster_meta_features(
        methods_history, result["cluster_map"]
    )
    # meta_features: DataFrame (n_days x n_clusters), one column per
    # cluster, ready to feed into Layer 3 domain-factor building instead
    # of the raw per-method columns.
"""
import sys

import numpy as np
import pandas as pd


def build_cluster_meta_features(
    methods_history: pd.DataFrame,
    cluster_map: pd.DataFrame,
    min_cluster_size_to_aggregate: int = 2,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Collapse each cluster of correlated methods into one meta-feature
    (z-scored mean of member columns). Singleton clusters (methods with
    no strong redundancy) pass through unchanged, just z-scored for
    consistency with the aggregated columns.

    Args:
        methods_history: raw (n_days x n_methods) DataFrame — same input
            given to analyze_method_independence(). Must contain at least
            the methods listed in cluster_map (extra columns not in
            cluster_map are ignored). Methods that were dropped upstream
            (constant-variance, all-NaN) should already be absent from
            both cluster_map and methods_history.
        cluster_map: DataFrame with "method" and "cluster" columns, as
            returned by analyze_method_independence()["cluster_map"].
        min_cluster_size_to_aggregate: clusters with fewer members than
            this are treated as singletons (passed through individually,
            not merged) — defaults to 2, meaning any cluster with 2+
            members gets aggregated. Raise this if you only want to merge
            "obviously" redundant groups (e.g. 3+) and leave 2-member
            pairs as separate signals for finer-grained downstream use.
        standardize: if True (default), ALL output columns are z-scored
            (mean 0, std 1) — including singletons, so downstream
            consumers get uniform scale regardless of cluster size. If
            False, singletons pass through at their RAW value
            (unstandardized) while aggregated clusters are still
            z-scored before averaging. Set to False when the raw unit
            (e.g. "m1_klr score of 8.9") is meaningful and downstream
            consumers expect the original scale.

    Returns:
        DataFrame (n_days x n_clusters) — one column per cluster, named
        either the single method's name (singleton) or
        "cluster_{id}__{n}methods" (aggregated), plus a
        `.attrs["cluster_membership"]` dict mapping output column name to
        the list of original method names it represents, for
        traceability/debugging.
    """
    meta_columns = {}
    membership = {}

    # Validate: warn about cluster_map methods not in history
    missing = [m for m in cluster_map["method"].unique()
               if m not in methods_history.columns]
    if missing:
        print(f"[HierarchicalEnsemble] WARNING: {len(missing)} method(s) in "
              f"cluster_map not found in methods_history and will be skipped: "
              f"{missing}", file=sys.stderr)

    for cluster_id in sorted(cluster_map["cluster"].unique()):
        members = cluster_map[cluster_map["cluster"] == cluster_id]["method"].tolist()
        members = [m for m in members if m in methods_history.columns]
        if not members:
            continue

        if len(members) < min_cluster_size_to_aggregate:
            # Singleton (or below threshold) — pass through, optionally
            # z-scored for consistency with aggregated columns.
            col_name = members[0]
            series = methods_history[col_name].astype(float)
            std = series.std(skipna=True)
            if std < 1e-12 or pd.isna(std):
                print(f"[HierarchicalEnsemble] WARNING: {col_name} has "
                      f"near-zero variance, passing through unstandardized",
                      file=sys.stderr)
                meta_columns[col_name] = series
            elif standardize:
                mean = series.mean(skipna=True)
                meta_columns[col_name] = (series - mean) / std
            else:
                meta_columns[col_name] = series  # raw, no z-score
            membership[col_name] = members
        else:
            # Aggregate: z-score each member independently first (so a
            # member with a wider raw scale, e.g. 0-100 vs 0-1, doesn't
            # dominate the mean purely from unit differences), then average.
            zscored = []
            for m in members:
                series = methods_history[m].astype(float)
                std = series.std(skipna=True)
                mean = series.mean(skipna=True)
                if std < 1e-12 or pd.isna(std):
                    continue  # skip degenerate member, don't let it pull the mean toward 0 artificially
                zscored.append((series - mean) / std)

            if not zscored:
                continue  # all members degenerate somehow — skip this cluster entirely

            col_name = f"cluster_{cluster_id}__{len(members)}methods"
            meta_columns[col_name] = pd.concat(zscored, axis=1).mean(axis=1, skipna=True)
            membership[col_name] = members

    result = pd.DataFrame(meta_columns)
    result.attrs["cluster_membership"] = membership
    return result


def summarize_reduction(methods_history: pd.DataFrame, meta_features: pd.DataFrame) -> str:
    """Human-readable summary of how much dimensionality reduction happened."""
    n_before = methods_history.shape[1]
    n_after = meta_features.shape[1]
    membership = meta_features.attrs.get("cluster_membership", {})
    n_aggregated_clusters = sum(1 for v in membership.values() if len(v) > 1)
    lines = [
        f"Input: {n_before} raw method columns",
        f"Output: {n_after} meta-feature columns "
        f"({n_before - n_after} fewer, {(1 - n_after/n_before)*100:.0f}% reduction)",
        f"  - {n_aggregated_clusters} clusters aggregated (2+ redundant methods merged into 1)",
        f"  - {n_after - n_aggregated_clusters} singleton methods passed through unchanged",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test with synthetic data: verify aggregation collapses a known
    # redundant pair correctly, and that a genuinely independent signal
    # is NOT altered by being lumped into the wrong group.
    print("=== Self-test: hierarchical_ensemble.py ===\n")

    rng = np.random.default_rng(123)
    n_days = 150
    base = rng.normal(0, 1, n_days)

    history = pd.DataFrame({
        "redundant_A": base * 10 + 50 + rng.normal(0, 0.5, n_days),  # different scale on purpose
        "redundant_B": base * 2 - 5 + rng.normal(0, 0.1, n_days),    # different scale, same underlying signal
        "independent_X": rng.normal(0, 1, n_days),
    })

    cluster_map = pd.DataFrame({
        "method": ["redundant_A", "redundant_B", "independent_X"],
        "cluster": [1, 1, 2],  # A and B clustered together, X alone
    })

    meta = build_cluster_meta_features(history, cluster_map)
    print(f"Output columns: {list(meta.columns)}")
    assert "independent_X" in meta.columns, "Singleton should pass through by its own name"
    cluster_col = [c for c in meta.columns if c.startswith("cluster_")]
    assert len(cluster_col) == 1, "redundant_A + redundant_B should merge into exactly 1 column"
    print(f"✅ PASS: redundant pair merged into '{cluster_col[0]}', independent signal kept separate\n")

    # Verify the merged cluster's correlation with the ORIGINAL underlying
    # signal (base) is still very high — i.e. aggregation didn't destroy
    # the real signal, just removed the duplicate counting of it.
    merged_corr_with_base = np.corrcoef(meta[cluster_col[0]], base)[0, 1]
    print(f"Correlation of merged cluster feature vs true underlying signal: "
          f"{merged_corr_with_base:.3f}")
    assert abs(merged_corr_with_base) > 0.95
    print("✅ PASS: aggregated feature still captures the real signal (not destroyed by merging)\n")

    # Verify independent_X in the output is IDENTICAL in relative shape to
    # its z-scored input (only scaling changed, not the underlying pattern).
    x_input_z = (history["independent_X"] - history["independent_X"].mean()) / history["independent_X"].std()
    max_diff = (meta["independent_X"] - x_input_z).abs().max()
    print(f"Max difference between independent_X output and its own z-score: {max_diff:.2e}")
    assert max_diff < 1e-9
    print("✅ PASS: singleton pass-through is exact (just standardized, not altered)\n")

    print(summarize_reduction(history, meta))
    print()

    # ── Test standardize=False: singleton should be RAW, not z-scored ──
    meta_raw = build_cluster_meta_features(history, cluster_map, standardize=False)
    max_diff_raw = (meta_raw["independent_X"] - history["independent_X"]).abs().max()
    print(f"standardize=False: max diff between singleton and raw input: {max_diff_raw:.2e}")
    assert max_diff_raw < 1e-9, "standardize=False should leave singleton unchanged"
    print("✅ PASS: standardize=False preserves raw singleton values\n")

    print("ALL SELF-TESTS PASSED")
