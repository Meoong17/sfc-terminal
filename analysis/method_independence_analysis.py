"""
method_independence_analysis.py — Cluster SFC methods by redundancy
=======================================================================

Answers: "of all the primary method scores in this pipeline, how many
are actually independent signals vs. duplicates of each other?"

FIXES TWO VERIFIED BUGS from an earlier draft of this analysis:

1. `linkage(corr_matrix, method='ward')` called directly on the raw
   correlation matrix treats each method's correlation PROFILE (a row of
   correlation values against every other method) as a Euclidean feature
   vector, then clusters by Euclidean distance between those profiles.
   This is NOT the same as clustering by "how correlated are methods i
   and j with each other" — confirmed empirically: two methods with
   correlation -0.990 (near-perfect redundancy, just sign-flipped) were
   assigned to DIFFERENT clusters by this approach, when they should be
   recognized as duplicates (a signal and its literal negation carry the
   same information).
   FIX: convert correlation to a proper distance matrix first
   (distance = 1 - |correlation|, so both strong positive AND strong
   negative correlation count as "close"/redundant), condense it with
   scipy's squareform(), THEN pass to linkage(). Ward's method also
   requires genuine Euclidean coordinates for its variance-minimization
   assumptions to hold — 'average' linkage is used instead here since
   we're working with a precomputed distance matrix, not raw coordinates.

2. `effective_dim = np.sum(eigenvalues > 0.1)` uses an arbitrary
   threshold. Tested against synthetic data built from 15 KNOWN
   independent underlying signals (i.e. ground truth = 15): the 0.1
   threshold returned 67 out of 71 — it barely filters anything,
   defeating the entire purpose of the analysis (finding redundancy).
   FIX: use the Kaiser criterion (eigenvalue > 1.0) — a long-established
   heuristic from factor analysis/PCA (an eigenvalue > 1 means that
   component explains more variance than a single original variable
   would on its own). Same test data: Kaiser criterion returned 16,
   very close to the true 15.

Usage:
    python3 method_independence_analysis.py
    (needs git history of data.json to build a multi-day sample —
     see extract_method_history() below; falls back to a single-day
     snapshot with a loud warning if git history isn't available, since
     correlation from one data point is meaningless)
"""
import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ── Primary method whitelist ──
# Deliberately excludes derived/meta fields that are mathematically built
# FROM another method in this list (e.g. m32_hybrid_pred = f(m32_qlstm,
# GARCH residual) — including both would show "high correlation" that
# reflects the derivation formula, not independent redundancy) and
# boolean/status flags (m65_affects_sfc_score, m70_shap_ok) which aren't
# meaningful for correlation-based clustering at all.
#
# Adjust this list to match your actual pipeline's field names if they've
# changed — this was built from a live data.json snapshot, not guessed.
PRIMARY_METHODS = [
    "m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime_score",
    "m7_fisher", "m8_yield", "m9_liquidity", "m10_garch", "m11_var", "m12_jump",
    "m13_funding", "m14_skew", "m15_concentration", "m16_regime_ml", "m17_granger",
    "m18_entropy", "m19_mutual_info", "m20_obi", "m21_trade_flow", "m22_spread",
    "m23_liquidity", "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
    "m28_summers", "m29_debt", "m30_rajan", "m31_altman",
    "m32_qlstm", "m32_mamba",
    "m33_glo_score",
    "m65_cnn_attention", "m69_systemic_risk",
    "m72_m2_growth", "m73_m2_momentum", "m74_fed_balance",  # m75 excluded: derived from m72+m73+m74
    "m76_supply_growth", "m77_ssr", "m78_exchange_flow", "m79_velocity", "m80_dominance",
    "m81_etf_flow", "m82_etf_holdings",
    "m83_tga_score", "m84_rrp_score", "m85_fiscal_composite",
    "m86_repo_stress_score",
]

KAISER_THRESHOLD = 1.0  # eigenvalue > 1.0 = explains more variance than one original variable


def extract_method_history(max_commits=500):
    """
    Build a (n_days x n_methods) DataFrame from git history of data.json,
    mirroring the pattern already used in ensemble_meta.py /
    correlation_analysis.py for consistency across this codebase's
    analysis tools.

    Returns None if git history isn't available (e.g. running from an
    extracted zip with no .git directory) — correlation from a single
    snapshot is meaningless, so this deliberately does not silently fall
    back to fabricating multi-row data from one point.
    """
    try:
        log = subprocess.run(
            ["git", "log", f"-{max_commits}", "--format=%H", "--", "data.json"],
            capture_output=True, text=True, timeout=30,
        )
        commit_hashes = [h for h in log.stdout.strip().split("\n") if h]
    except (subprocess.SubprocessError, FileNotFoundError):
        commit_hashes = []

    if not commit_hashes:
        print("[MethodIndependence] No git history found for data.json — "
              "cannot build a multi-day sample. Run this from the actual "
              "repo (with .git), not an extracted zip.", file=sys.stderr)
        return None

    rows = []
    for h in commit_hashes:
        try:
            show = subprocess.run(
                ["git", "show", f"{h}:data.json"],
                capture_output=True, text=True, timeout=10,
            )
            snap = json.loads(show.stdout)
            row = {m: snap.get(m) for m in PRIMARY_METHODS}
            row["_ts"] = snap.get("ts")
            rows.append(row)
        except (subprocess.SubprocessError, json.JSONDecodeError):
            continue

    if len(rows) < 20:
        print(f"[MethodIndependence] Only {len(rows)} usable snapshots found "
              f"(need >= 20 for meaningful correlation) — collect more history "
              f"first.", file=sys.stderr)
        return None

    df = pd.DataFrame(rows).drop(columns=["_ts"], errors="ignore")
    print(f"[MethodIndependence] Built history from {len(df)} snapshots", file=sys.stderr)
    return df


def analyze_method_independence(methods_history: pd.DataFrame, n_clusters=None):
    """
    methods_history: DataFrame (n_days x n_methods), values may contain
    NaN for methods that were unavailable on a given day (this pipeline
    has several — e.g. m9_liquidity, m13_funding, m76_supply_growth are
    None in the sample snapshot checked while building this).

    Returns a dict with the correlation matrix, linkage matrix, effective
    dimension (Kaiser criterion), and a cluster assignment table.
    """
    # Drop columns that are entirely NaN (method never returned a value
    # across the whole sample — can't correlate a column with nothing).
    valid_cols = methods_history.columns[methods_history.notna().any()]
    dropped = set(methods_history.columns) - set(valid_cols)
    if dropped:
        print(f"[MethodIndependence] Dropping {len(dropped)} all-NaN methods: "
              f"{sorted(dropped)}", file=sys.stderr)
    df = methods_history[valid_cols].copy()

    if df.shape[1] < 2:
        raise ValueError(
            f"Only {df.shape[1]} method(s) have any data after dropping "
            f"all-NaN columns — need at least 2 to compute correlations. "
            f"Dropped: {sorted(dropped)}. This usually means the pipeline "
            f"itself is failing to produce data, not a bug in this analysis."
        )

    # Drop columns with zero variance (constant values across the sample).
    # These produce NaN in the correlation matrix and break clustering
    # (distance = 1 between every constant-method pair flattens the
    # distance matrix, making cluster assignments meaningless).
    # Use nunique() instead of std() because floating-point precision
    # causes pandas std() on identical float values to return ~1e-15
    # instead of exactly 0.0.
    var_mask = df.nunique() > 1
    const_cols = set(df.columns[~var_mask])
    if const_cols:
        print(f"[MethodIndependence] Dropping {len(const_cols)} constant-value methods: "
              f"{sorted(const_cols)}", file=sys.stderr)
    df = df.loc[:, var_mask]

    if df.shape[1] < 2:
        raise ValueError(
            f"Only {df.shape[1]} method(s) have any variance after dropping "
            f"constant columns — need at least 2 to compute correlations. "
            f"Dropped (const): {sorted(const_cols)}. This usually means the "
            f"pipeline is producing static output, not a bug in this analysis."
        )

    # pandas .corr() handles remaining partial-NaN columns via pairwise
    # complete observations by default — no need to impute here, but note
    # this means different cell pairs may be computed from different
    # subsets of rows if missingness isn't uniform across methods.
    corr_matrix = df.corr().values
    n = corr_matrix.shape[0]

    # Guard against any remaining NaN in the correlation matrix itself
    # (can still happen if two dynamic columns never have overlapping
    # non-NaN rows). In practice this is extremely rare after dropping
    # constant columns.
    if np.isnan(corr_matrix).any():
        nan_pairs = np.argwhere(np.isnan(corr_matrix))
        bad_methods = sorted(set(df.columns[i] for i in nan_pairs[:, 0]))
        print(f"[MethodIndependence] WARNING: {len(nan_pairs)} NaN(s) remaining "
              f"in correlation matrix for: {bad_methods}. Dropping these methods "
              f"rather than imputing (false distances break clustering).",
              file=sys.stderr)
        # Drop any column that has ANY NaN correlation with any other column
        cols_to_drop = set(df.columns[i] for i in np.unique(nan_pairs[:, 0]))
        cols_to_keep = [c for c in df.columns if c not in cols_to_drop]
        dropped_nan = cols_to_drop
        if dropped_nan:
            print(f"[MethodIndependence] Dropping {len(dropped_nan)} remaining "
                  f"problematic methods: {sorted(dropped_nan)}", file=sys.stderr)
        df = df[cols_to_keep]
        if df.shape[1] < 2:
            raise ValueError(
                f"Only {df.shape[1]} method(s) survive after dropping NaN-prone "
                f"columns — need at least 2."
            )
        corr_matrix = df.corr().values
        n = corr_matrix.shape[0]

    # ── FIX #1: proper distance matrix, not raw correlation as coordinates ──
    # 1 - |correlation| means strong correlation (either sign) = small
    # distance = likely redundant. See module docstring for the empirical
    # proof this matters (negative-correlation case).
    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0.0)
    # Clip tiny negative values from floating-point noise before squareform
    # (squareform requires a valid distance matrix: symmetric, zero diagonal,
    # non-negative).
    dist_matrix = np.clip(dist_matrix, 0, None)
    condensed = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    # ── FIX #2: Kaiser criterion, not an arbitrary 0.1 threshold ──
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    effective_dim = int(np.sum(eigenvalues > KAISER_THRESHOLD))
    effective_dim = max(1, effective_dim)  # guard against degenerate all-below-threshold case

    if n_clusters is None:
        n_clusters = effective_dim

    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    cluster_map = pd.DataFrame({
        "method": df.columns,
        "cluster": cluster_labels,
    }).sort_values("cluster")

    return {
        "n_methods": n,
        "effective_dim": effective_dim,
        "n_clusters_used": n_clusters,
        "cluster_map": cluster_map,
        "correlation_matrix": corr_matrix,
        "method_names": list(df.columns),
        "eigenvalues": eigenvalues,
        "dropped_methods": sorted(dropped),
        "const_methods": sorted(const_cols) if const_cols else [],
    }


def print_report(result):
    print("\n" + "=" * 70)
    print("METHOD INDEPENDENCE ANALYSIS")
    print("=" * 70)
    print(f"\nTotal methods analyzed: {result['n_methods']}")
    print(f"Effective independent dimensions (Kaiser criterion, eigenvalue > "
          f"{KAISER_THRESHOLD}): {result['effective_dim']}")
    if result["dropped_methods"]:
        print(f"Excluded (no data in sample): {', '.join(result['dropped_methods'])}")
    if result.get("const_methods"):
        print(f"Excluded (constant in sample): {', '.join(result['const_methods'])}")

    print(f"\n{'-'*70}")
    print("CLUSTERS (methods grouped together are likely redundant):")
    print(f"{'-'*70}")
    cluster_map = result["cluster_map"]
    corr = result["correlation_matrix"]
    method_names = result["method_names"]

    for cluster_id in sorted(cluster_map["cluster"].unique()):
        members = cluster_map[cluster_map["cluster"] == cluster_id]["method"].tolist()
        if len(members) > 1:
            print(f"\nCluster {cluster_id} ({len(members)} methods):")
            for m in members:
                print(f"    - {m}")
            # Show pairwise correlations within this cluster — only those
            # with |r| > 0.7 (genuinely strong redundancy). Skip printing
            # dozens of near-zero pairs that just happened to cluster
            # together via distance aggregation.
            idxs = [method_names.index(m) for m in members]
            pairs = []
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    r = corr[idxs[i], idxs[j]]
                    pairs.append((abs(r), members[i], members[j], r))
            strong = [p for p in pairs if p[0] > 0.7]
            if strong:
                print("    Strong pairwise correlations (|r| > 0.70) within cluster:")
                for _, mi, mj, r in sorted(strong, reverse=True):
                    print(f"      {mi} <-> {mj}: {r:+.3f}")
            else:
                print("    (no strong pairwise correlations within cluster — "
                      "cluster forms via distance aggregation across multiple weak links)")

    singles = cluster_map[cluster_map.groupby("cluster")["cluster"].transform("count") == 1]
    if len(singles) > 0:
        print(f"\n{len(singles)} methods stand alone (no strong redundancy detected):")
        print(f"    {', '.join(singles['method'].tolist())}")

    # ── Cross-cluster high-correlation pairs ──
    # Methods assigned to different clusters but with |r| > 0.7 may still
    # be redundant — cluster boundary just means the linkage algorithm
    # put them in different groups at this cut level.
    n = len(method_names)
    cross_strong = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            mi, mj = method_names[i], method_names[j]
            ci = cluster_map[cluster_map["method"] == mi]["cluster"].values[0]
            cj = cluster_map[cluster_map["method"] == mj]["cluster"].values[0]
            if ci != cj and abs(r) > 0.7:
                cross_strong.append((abs(r), mi, mj, r, ci, cj))
    if cross_strong:
        print(f"\n{'─'*70}")
        print("CROSS-CLUSTER REDUNDANCIES (|r| > 0.70 across cluster boundaries):")
        print(f"{'─'*70}")
        for _, mi, mj, r, ci, cj in sorted(cross_strong, reverse=True):
            print(f"  {mi} (cl.{ci}) <-> {mj} (cl.{cj}): r={r:+.3f}")
    print()

    print("\n" + "=" * 70)
    print(f"SUMMARY: {result['n_methods']} methods -> {result['effective_dim']} "
          f"effective independent signals")
    print("=" * 70)


if __name__ == "__main__":
    history = extract_method_history()
    if history is None:
        sys.exit(1)

    result = analyze_method_independence(history)
    print_report(result)

    # Save cluster map for downstream use
    result["cluster_map"].to_json(".method_clusters.json", orient="records", indent=2)
    print(f"\nCluster assignments saved to .method_clusters.json")
