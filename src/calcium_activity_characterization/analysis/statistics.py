# https://www.theanalysisfactor.com/dunns-test-post-hoc-test-after-kruskal-wallis/
# Usage example
# ----------------
# import pandas as pd
# from violin_stats_kw_dunn import kw_dunn_violin_stats
#
# # Example DataFrame with columns: 'event_id' (group) and 'duration' (value)
# df = pd.DataFrame({
# 'event_id': [1,1,1,2,2,2,3,3,3,3],
# 'duration': [20,25,30,40,45,42,70,60,75,72],
# })
#
# result = kw_dunn_violin_stats(
# df=df,
# group_col='event_id',
# value_col='duration',
# p_adjust_method='holm', # or 'fdr_bh'
# n_boot_ci=1000, # bootstrap reps for HL CI (set 0 to skip)
# random_state=42,
# )
#
# # Access pieces you can paste into your thesis/figures:
# print(result['global_test']) # dict: {'test': 'Kruskal–Wallis', 'H': ..., 'df': ..., 'p_value': ...}
# print(result['group_summary'].head()) # per-group medians/IQR, n, mean, std, skew, kurtosis
# print(result['pairwise'].head()) # pairwise Dunn p_adj, Cliff's delta, HL median diff + CI
#
# # Example: filter significant pairs
# sig = result['pairwise'][result['pairwise']['p_adj'] < 0.05]
# print(sig.sort_values('p_adj'))

from __future__ import annotations
import numpy as np
from calcium_activity_characterization.logger import logger
import pandas as pd
from typing import Optional, Literal, Iterable
from scipy import stats
from statsmodels.stats.multitest import multipletests
from calcium_activity_characterization.analysis.metrics import detect_asymmetric_iqr_outliers
from itertools import combinations

try:
    import scikit_posthocs as sp
except Exception:  # pragma: no cover
    sp = None  # type: ignore


def _summary(a: np.ndarray) -> dict[str, float]:
    """Return robust per-group summaries used in tables/prints.

    Args:
        a: 1D numeric array.

    Returns:
        Dict with n, mean, std, q1, median, q3, iqr, min, max, skew, kurtosis.
    """
    a = np.asarray(a, float).ravel()
    n = a.size
    if n == 0:
        return {k: float("nan") for k in ["n","mean","std","q1","median","q3","iqr","min","max","skew","kurtosis"]} | {"n": 0.0}
    q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
    return {
        "n": float(n),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if n > 1 else float("nan"),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "skew": float(stats.skew(a, bias=False)) if n > 2 else float("nan"),
        "kurtosis": float(stats.kurtosis(a, fisher=True, bias=False)) if n > 3 else float("nan"),
    }

def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta in [-1,1] via MWU/U.

    Args:
        x: Sample 1.
        y: Sample 2.

    Returns:
        δ = 2U/(n_x n_y) − 1.
    """
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def _vd_a(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha–Delaney A = U/(n_x n_y)."""
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(u / (len(x) * len(y)))


def _median_diff_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int = 0, alpha: float = 0.05, seed: int = 7) -> tuple[float, float, float]:
    """Median difference (x−y) with optional percentile bootstrap CI (fast & robust).

    Args:
        x: Sample 1.
        y: Sample 2.
        n_boot: Bootstrap reps (0 to skip).
        alpha: 1−confidence level.
        seed: RNG seed.

    Returns:
        (estimate, ci_low, ci_high); CI is NaN if n_boot == 0.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    est = float(np.median(x) - np.median(y))
    if n_boot <= 0:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nx, ny = len(x), len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        xb = x[rng.integers(0, nx, nx)]
        yb = y[rng.integers(0, ny, ny)]
        boots[b] = np.median(xb) - np.median(yb)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return est, float(lo), float(hi)

def kw_dunn(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    p_adjust_method: Literal["holm", "fdr_bh"] = "holm",
    filter_outliers: bool = False,
    outliers_bounds: tuple[float, float] = (3.0, 3.0),
    outliers_bygroup: str | None = None,
    n_boot_ci: int = 0,
) -> pd.DataFrame:
    """Run Kruskal–Wallis and Dunn post-hoc, returning a pairwise DataFrame.

    Returns:
        DataFrame with one row per pair and columns:
        ['group1','group2','n1','n2','median1','q1_1','q3_1','median2','q1_2','q3_2',
         'dunn_p_raw','p_adj','cliffs_delta','median_diff','med_ci_low','med_ci_high'].
    """
    try:
        if sp is None:
            raise ImportError("scikit-posthocs is required for Dunn's test. Install via `pip install scikit-posthocs`. ")

        work = df[[group_col, value_col]].dropna().copy()
        if filter_outliers:
            ql, qu = outliers_bounds if outliers_bounds is not None else (3, 3)
            filtered_df, outliers, lb, ub = detect_asymmetric_iqr_outliers(work.copy(), value_col, ql, qu, outliers_bygroup)
            logger.info(
                "plot_violin: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                len(outliers), work.shape[0], value_col, lb, ub
            )
            work = filtered_df

        groups = [g for g in pd.unique(work[group_col]) if pd.notna(g)]
        if len(groups) < 2:
            raise ValueError("Need at least two groups with data to compare.")
        groups_sorted = sorted(groups)

        # Prepare per-group arrays
        arrays: dict[any, np.ndarray] = {}
        for g in groups_sorted:
            vals = work.loc[work[group_col] == g, value_col].to_numpy(float)
            arrays[g] = vals

        # (Global test is computed but not returned, to keep minimal change in surface)
        sample_list = [arrays[g] for g in groups_sorted]
        kw = stats.kruskal(*sample_list)
        logger.info("Kruskal–Wallis: H=%.4f, df=%d, p=%.3g", float(kw.statistic), len(groups_sorted)-1, float(kw.pvalue))

        # Dunn raw p-value matrix
        dunn_pmat = sp.posthoc_dunn(work, val_col=value_col, group_col=group_col, p_adjust=None)

        # Build pairwise rows
        pairs = [(g1, g2) for i, g1 in enumerate(groups_sorted) for g2 in groups_sorted[i+1:]]
        p_raw = [float(dunn_pmat.loc[g1, g2]) for (g1, g2) in pairs]
        _, p_adj, _, _ = multipletests(p_raw, method=p_adjust_method)

        rows: list[dict[str, float | int | any]] = []
        for (g1, g2), pr, pa in zip(pairs, p_raw, p_adj):
            x, y = arrays[g1], arrays[g2]
            q1x, medx, q3x = np.quantile(x, [0.25, 0.5, 0.75])
            q1y, medy, q3y = np.quantile(y, [0.25, 0.5, 0.75])
            md, lo, hi = _median_diff_ci(x, y, n_boot=n_boot_ci)
            delta = _cliffs_delta(x, y)
            rows.append({
                "group1": g1, "group2": g2,
                "n1": len(x), "n2": len(y),
                "median1": float(medx), "q1_1": float(q1x), "q3_1": float(q3x),
                "median2": float(medy), "q1_2": float(q1y), "q3_2": float(q3y),
                "dunn_p_raw": float(pr), "p_adj": float(pa),
                "cliffs_delta": float(delta),
                "median_diff": float(md), "med_ci_low": float(lo), "med_ci_high": float(hi),
            })
        return pd.DataFrame(rows).sort_values("p_adj").reset_index(drop=True)

    except Exception as e:  # pragma: no cover
        logger.exception("kw_dunn failed.")
        raise e


def brunner_pairs(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    pairs: Optional[list[tuple[any, any]] | tuple[any, any] | list[any]] = None,
    filter_outliers: bool = False,
    outliers_bounds: tuple[float, float] = (3.0, 3.0),
    outliers_bygroup: str | None = None,
    n_boot_ci: int = 0,
    alpha: float = 0.05,
    print_results: bool = True,
    p_adjust_method: Literal[
        "none","holm","bonferroni","sidak","holm-sidak","simes-hochberg",
        "hommel","fdr_bh","fdr_by","fdr_tsbh","fdr_tsbky"
    ] = "none",
) -> pd.DataFrame:
    """Run Brunner–Munzel for one/more pairs; if `pairs=None`, compute all pairs.
    Adds multiplicity control with `p_adjust_method` (Holm/FDR/etc) and returns:
    ['group1','group2','n1','n2','median1','q1_1','q3_1','median2','q1_2','q3_2',
     'bm_stat','bm_p','p_adj','reject','vd_A','cliffs_delta','median_diff','med_ci_low','med_ci_high'].
    """
    try:
        work = df[[group_col, value_col]].dropna().copy()
        if filter_outliers:
            ql, qu = outliers_bounds if outliers_bounds is not None else (3, 3)
            filtered_df, outliers, lb, ub = detect_asymmetric_iqr_outliers(work.copy(), value_col, ql, qu, outliers_bygroup)
            logger.info(
                "plot_violin: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                len(outliers), work.shape[0], value_col, lb, ub
            )
            work = filtered_df

        arrays: dict[any, np.ndarray] = {g: sub[value_col].to_numpy(float) for g, sub in work.groupby(group_col, dropna=False)}
        groups_sorted = sorted(arrays.keys())
        if len(groups_sorted) < 2:
            raise ValueError("Need at least two groups with data to compare.")

        # Normalize/expand pairs
        if pairs is None:
            norm_pairs = list(combinations(groups_sorted, 2))
        elif isinstance(pairs, tuple) and len(pairs) == 2 and not isinstance(pairs[0], tuple):
            norm_pairs = [pairs]
        elif isinstance(pairs, list) and len(pairs) == 2 and not isinstance(pairs[0], tuple):
            norm_pairs = [tuple(pairs)]
        else:
            norm_pairs = list(pairs)  # type: ignore[arg-type]

        rows: list[dict[str, float | int | any]] = []
        for g1, g2 in norm_pairs:
            if g1 not in arrays or g2 not in arrays:
                raise ValueError(f"Pair ({g1},{g2}) includes a level not found in '{group_col}'.")
            x, y = arrays[g1], arrays[g2]
            if len(x) < 2 or len(y) < 2:
                raise ValueError(f"Group sizes too small for BM: {g1} (n={len(x)}), {g2} (n={len(y)}).")

            s1, s2 = _summary(x), _summary(y)
            bm = stats.brunnermunzel(x, y, alternative="two-sided")
            vdA = _vd_a(x, y)
            delta = _cliffs_delta(x, y)
            md, lo, hi = _median_diff_ci(x, y, n_boot=n_boot_ci, alpha=alpha)

            row = {
                "group1": g1, "group2": g2,
                "n1": int(s1["n"]), "n2": int(s2["n"]),
                "median1": s1["median"], "q1_1": s1["q1"], "q3_1": s1["q3"],
                "median2": s2["median"], "q1_2": s2["q1"], "q3_2": s2["q3"],
                "bm_stat": float(bm.statistic), "bm_p": float(bm.pvalue),
                "vd_A": float(vdA), "cliffs_delta": float(delta),
                "median_diff": float(md), "med_ci_low": float(lo), "med_ci_high": float(hi),
            }
            rows.append(row)

            if print_results:
                print(
                    f"Brunner–Munzel {g1} vs {g2}: t={bm.statistic:.3f}, p={bm.pvalue:.3g}\n"
                    f"  {g1}: n={int(s1['n'])}, median={s1['median']:.3g} [Q1={s1['q1']:.3g}, Q3={s1['q3']:.3g}]\n"
                    f"  {g2}: n={int(s2['n'])}, median={s2['median']:.3g} [Q1={s2['q1']:.3g}, Q3={s2['q3']:.3g}]\n"
                    f"  Effect sizes: A={vdA:.3f} (0.5=no effect), δ={delta:.3f}; median diff={md:.3g}"
                    + (f" (95% CI {lo:.3g}..{hi:.3g})" if n_boot_ci > 0 else "")
                )

        out = pd.DataFrame(rows)

        # Multiplicity control
        if p_adjust_method != "none":
            reject, p_adj, _, _ = multipletests(out["bm_p"].to_numpy(float), alpha=alpha, method=p_adjust_method)
        else:
            reject = np.zeros(len(out), dtype=bool)
            p_adj = out["bm_p"].to_numpy(float)

        out["p_adj"] = p_adj
        out["reject"] = reject

        # Sort by adjusted p if used, else raw p
        sort_cols = ["p_adj" if p_adjust_method != "none" else "bm_p", "group1", "group2"]
        return out.sort_values(sort_cols).reset_index(drop=True)

    except Exception as e:  # pragma: no cover
        logger.exception("brunner_pairs failed.")
        raise e

def wilcoxon_pairs(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    pairs: Optional[list[tuple[any, any]] | tuple[any, any] | list[any]] = None,
    filter_outliers: bool = False,
    outliers_bounds: tuple[float, float] = (3.0, 3.0),
    outliers_bygroup: str | None = None,
    n_boot_ci: int = 0,
    alpha: float = 0.05,
    print_results: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    exact_if_small: bool = True,
    p_adjust_method: Literal[
        "none","holm","bonferroni","sidak","holm-sidak","simes-hochberg",
        "hommel","fdr_bh","fdr_by","fdr_tsbh","fdr_tsbky"
    ] = "none",
) -> pd.DataFrame:
    """Wilcoxon rank-sum (Mann–Whitney U) for one/more pairs; if `pairs=None`, test all pairs.
    Adds multiplicity control with `p_adjust_method` and returns:
    ['group1','group2','n1','n2','median1','q1_1','q3_1','median2','q1_2','q3_2',
     'mw_U','mw_p','p_adj','reject','vd_A','cliffs_delta','median_diff','med_ci_low','med_ci_high'].
    """
    try:
        work = df[[group_col, value_col]].dropna().copy()
        if filter_outliers:
            ql, qu = outliers_bounds if outliers_bounds is not None else (3.0, 3.0)
            filtered_df, outliers, lb, ub = detect_asymmetric_iqr_outliers(
                work.copy(), value_col, ql, qu, outliers_bygroup
            )
            logger.info(
                "wilcoxon_pairs: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                len(outliers), work.shape[0], value_col, lb, ub
            )
            work = filtered_df

        arrays: dict[any, np.ndarray] = {
            g: sub[value_col].to_numpy(float) for g, sub in work.groupby(group_col, dropna=False)
        }
        groups_sorted = sorted(arrays.keys())
        if len(groups_sorted) < 2:
            raise ValueError("Need at least two groups with data to compare.")

        # Normalize/expand pairs
        if pairs is None:
            norm_pairs = list(combinations(groups_sorted, 2))
        elif isinstance(pairs, tuple) and len(pairs) == 2 and not isinstance(pairs[0], tuple):
            norm_pairs = [pairs]
        elif isinstance(pairs, list) and len(pairs) == 2 and not isinstance(pairs[0], tuple):
            norm_pairs = [tuple(pairs)]
        else:
            norm_pairs = list(pairs)  # type: ignore[arg-type]

        rows: list[dict[str, float | int | any]] = []
        for g1, g2 in norm_pairs:
            if g1 not in arrays or g2 not in arrays:
                raise ValueError(f"Pair ({g1},{g2}) includes a level not found in '{group_col}'.")
            x, y = arrays[g1], arrays[g2]
            n1, n2 = len(x), len(y)
            if n1 < 1 or n2 < 1:
                raise ValueError(f"Empty group encountered: {g1} (n={n1}), {g2} (n={n2}).")

            s1, s2 = _summary(x), _summary(y)

            method = "auto"
            if exact_if_small and min(n1, n2) <= 20:
                method = "exact"  # SciPy will fallback if ties/large
            mw = stats.mannwhitneyu(x, y, alternative=alternative, method=method)

            vdA = _vd_a(x, y)
            delta = _cliffs_delta(x, y)
            md, lo, hi = _median_diff_ci(x, y, n_boot=n_boot_ci, alpha=alpha)

            row = {
                "group1": g1, "group2": g2,
                "n1": int(s1["n"]), "n2": int(s2["n"]),
                "median1": s1["median"], "q1_1": s1["q1"], "q3_1": s1["q3"],
                "median2": s2["median"], "q1_2": s2["q1"], "q3_2": s2["q3"],
                "mw_U": float(mw.statistic), "mw_p": float(mw.pvalue),
                "vd_A": float(vdA), "cliffs_delta": float(delta),
                "median_diff": float(md), "med_ci_low": float(lo), "med_ci_high": float(hi),
            }
            rows.append(row)

            if print_results:
                print(
                    f"Wilcoxon rank-sum {g1} vs {g2}: U={mw.statistic:.3g}, p={mw.pvalue:.3g} [{alternative}]\n"
                    f"  {g1}: n={int(s1['n'])}, median={s1['median']:.3g} [Q1={s1['q1']:.3g}, Q3={s1['q3']:.3g}]\n"
                    f"  {g2}: n={int(s2['n'])}, median={s2['median']:.3g} [Q1={s2['q1']:.3g}, Q3={s2['q3']:.3g}]\n"
                    f"  Effect sizes: A={vdA:.3f} (0.5=no effect), δ={delta:.3f}; median diff={md:.3g}"
                    + (f" (95% CI {lo:.3g}..{hi:.3g})" if n_boot_ci > 0 else "")
                )

        out = pd.DataFrame(rows)

        # Multiplicity control
        if p_adjust_method != "none":
            reject, p_adj, _, _ = multipletests(out["mw_p"].to_numpy(float), alpha=alpha, method=p_adjust_method)
        else:
            reject = np.zeros(len(out), dtype=bool)
            p_adj = out["mw_p"].to_numpy(float)

        out["p_adj"] = p_adj
        out["reject"] = reject

        sort_cols = ["p_adj" if p_adjust_method != "none" else "mw_p", "group1", "group2"]
        return out.sort_values(sort_cols).reset_index(drop=True)

    except Exception as e:  # pragma: no cover
        logger.exception("wilcoxon_pairs failed.")
        raise e



def _bootstrap_ci_stat(x, y, stat_fn, n_boot: int, alpha: float, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot, float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = stat_fn(x[idx], y[idx])
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def corr_nonparametric(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    methods: Iterable[Literal["spearman","kendall"]] = ("spearman","kendall"),
    alpha: float = 0.05,
    n_boot_ci: int = 0,   # 0 = skip bootstrap CIs for correlations
    exact_kendall_if_small: bool = True,
) -> pd.DataFrame:
    """
    Compute nonparametric association between x and y:
      - Spearman's rho
      - Kendall's tau-b
      - Theil–Sen robust slope (always reported once, with CI)

    Returns a DataFrame with one row per correlation method
    and shared Theil–Sen slope columns.
    """
    # 1) Clean
    work = df[[x_col, y_col]].dropna().astype(float)
    x = work[x_col].to_numpy()
    y = work[y_col].to_numpy()
    n = len(work)
    if n < 2:
        raise ValueError("Need at least 2 paired observations.")
    
    rows = []

    # 2) Theil–Sen robust slope (+ CI at 1-alpha)
    #    (SciPy's alpha parameter is the *confidence level*.)
    ts_slope, ts_intercept, ts_lo, ts_hi = stats.theilslopes(y, x, alpha=1-alpha)

    # 3) Spearman
    if "spearman" in methods:
        rho, p = stats.spearmanr(x, y, alternative="two-sided")
        if n_boot_ci > 0:
            # bootstrap on ranks to keep it rank-based
            def _rho(ix, iy):
                return stats.spearmanr(ix, iy, alternative="two-sided").correlation
            rho_lo, rho_hi = _bootstrap_ci_stat(x, y, _rho, n_boot_ci, alpha)
        else:
            rho_lo = rho_hi = float("nan")
        rows.append({
            "method": "spearman",
            "n": n,
            "estimate": float(rho),
            "p_value": float(p),
            "ci_low": float(rho_lo),
            "ci_high": float(rho_hi),
            "theilsen_slope": float(ts_slope),
            "theilsen_slope_ci_low": float(ts_lo),
            "theilsen_slope_ci_high": float(ts_hi),
            "theilsen_intercept": float(ts_intercept),
        })

    # 4) Kendall τ-b
    if "kendall" in methods:
        kt_method = "auto"
        # SciPy will use exact when possible (no ties, small n); otherwise asymptotic.
        # If you *require* exact for tiny samples without ties:
        if exact_kendall_if_small and n <= 50:
            kt_method = "auto"  # keep auto; SciPy switches to exact when valid
        kt = stats.kendalltau(x, y, alternative="two-sided", method=kt_method)
        tau, p = kt.correlation, kt.pvalue
        if n_boot_ci > 0:
            def _tau(ix, iy):
                return stats.kendalltau(ix, iy, alternative="two-sided", method="auto").correlation
            tau_lo, tau_hi = _bootstrap_ci_stat(x, y, _tau, n_boot_ci, alpha)
        else:
            tau_lo = tau_hi = float("nan")
        rows.append({
            "method": "kendall_tau_b",
            "n": n,
            "estimate": float(tau),
            "p_value": float(p),
            "ci_low": float(tau_lo),
            "ci_high": float(tau_hi),
            "theilsen_slope": float(ts_slope),
            "theilsen_slope_ci_low": float(ts_lo),
            "theilsen_slope_ci_high": float(ts_hi),
            "theilsen_intercept": float(ts_intercept),
        })

    out = pd.DataFrame(rows)
    # nice ordering
    return out.loc[:, [
        "method","n","estimate","p_value","ci_low","ci_high",
        "theilsen_slope","theilsen_slope_ci_low","theilsen_slope_ci_high","theilsen_intercept"
    ]].sort_values("method").reset_index(drop=True)



def analyze_peak_intervals(peak_times: list[int]) -> tuple[list[int], float | None, float | None]:
    """
    Analyze periodicity of global (or trace-level) peak times.

    Computes inter-peak intervals, periodicity score (based on CV),
    and average peak frequency (events per frame).

    Args:
        peak_times (list[int]): Sorted list of peak times (in frames).

    Returns:
        tuple:
            - intervals (list[int]): list of inter-peak intervals.
            - periodicity_score (Optional[float]): [0, 1] score or None if too few events.
            - average_frequency (Optional[float]): Events per frame, or None if invalid.
    """
    if not peak_times or len(peak_times) < 2:
        return [], None, None

    peak_times = sorted(peak_times)
    intervals = np.diff(peak_times).tolist()

    if len(intervals) < 2:
        return intervals, None, None

    mean_ipi = np.mean(intervals)
    std_ipi = np.std(intervals)
    cv = std_ipi / mean_ipi if mean_ipi > 0 else None
    periodicity_score = 1 / (1 + cv) if cv is not None else None

    total_duration = peak_times[-1] - peak_times[0]
    average_frequency = len(intervals) / total_duration if total_duration > 0 else None

    return intervals, periodicity_score, average_frequency


def build_neighbor_pair_stats(
    *,
    cells_df: pd.DataFrame,
    comm_df: pd.DataFrame,
    dataset_col: str = "dataset",
    cell_id_col: str = "Cell ID",
    centroid_x_col: str = "Centroid X coordinate (um)",
    centroid_y_col: str = "Centroid Y coordinate (um)",
    neighbors_col: Optional[str] = "Neighbors (labels)",
    edges_df: Optional[pd.DataFrame] = None,
    edge_cols: tuple[str, str] = ("Cell ID", "Neighbor ID"),
) -> pd.DataFrame:
    """
    Build a per-dataset table of neighbor cell pairs with their centroid distance and
    the number of communications between the two cells.

    The function uses either:
      (A) a neighbors column in `cells_df` that contains a list/JSON list of neighbor labels, or
      (B) a separate long `edges_df` with one row per (cell_id, neighbor_id) pair.

    Communications are counted regardless of direction (origin↔cause), and pairs with
    zero communications are kept with count=0.

    Args:
        cells_df: DataFrame with at least [dataset_col, cell_id_col, centroid_x_col, centroid_y_col],
                  and optionally [neighbors_col] if `edges_df` is not provided.
        comm_df: DataFrame with communications; must include [dataset_col, "Origin cell ID", "Cause cell ID"].
        dataset_col: Column name identifying the dataset each row belongs to.
        cell_id_col: Column with integer cell labels/IDs.
        centroid_x_col: Column with centroid X coordinate (ideally in microns).
        centroid_y_col: Column with centroid Y coordinate (ideally in microns).
        neighbors_col: Column in `cells_df` that contains neighbors (list of ints). Used if `edges_df` is None.
        edges_df: Optional pre-built edge list DataFrame. If provided, must have at least [dataset_col, edge_cols[0], edge_cols[1]].
        edge_cols: Names of the two columns in `edges_df` giving the endpoints (defaults to ("Cell ID","Neighbor ID")).

    Returns:
        pd.DataFrame with columns:
            - dataset_col
            - Cell A (int)
            - Cell B (int)
            - distance_um (float)
            - n_communications (int)

        Each row represents an undirected neighbor pair (Cell A < Cell B) within a dataset.

    Raises:
        KeyError: if required columns are missing.
        ValueError: if inputs are empty or inconsistent.
    """
    try:
        # ---- Validate inputs
        if cells_df is None or cells_df.empty:
            raise ValueError("`cells_df` is empty.")
        if comm_df is None:
            raise ValueError("`comm_df` is None (pass an empty DataFrame if not available).")
        req_cells = {dataset_col, cell_id_col, centroid_x_col, centroid_y_col}
        missing_cells = req_cells - set(cells_df.columns)
        if missing_cells:
            raise KeyError(f"`cells_df` missing columns: {sorted(missing_cells)}")

        if edges_df is None and neighbors_col is None:
            raise ValueError("Provide either `edges_df` or `neighbors_col` in `cells_df`.")

        if edges_df is not None:
            req_edges = {dataset_col, edge_cols[0], edge_cols[1]}
            missing_edges = req_edges - set(edges_df.columns)
            if missing_edges:
                raise KeyError(f"`edges_df` missing columns: {sorted(missing_edges)}")

        if not comm_df.empty:
            req_comm = {dataset_col, "Origin cell ID", "Cause cell ID"}
            missing_comm = req_comm - set(comm_df.columns)
            if missing_comm:
                raise KeyError(f"`comm_df` missing columns: {sorted(missing_comm)}")

        # ---- Prepare centroids table (numeric)
        centroids = cells_df[[dataset_col, cell_id_col, centroid_x_col, centroid_y_col]].copy()
        centroids[centroid_x_col] = pd.to_numeric(centroids[centroid_x_col], errors="coerce")
        centroids[centroid_y_col] = pd.to_numeric(centroids[centroid_y_col], errors="coerce")
        centroids = centroids.dropna(subset=[centroid_x_col, centroid_y_col])

        # ---- Build neighbor pairs (undirected, per dataset)
        if edges_df is not None:
            pairs = edges_df[[dataset_col, edge_cols[0], edge_cols[1]]].copy()
            pairs.rename(columns={edge_cols[0]: "cell_u", edge_cols[1]: "cell_v"}, inplace=True)
        else:
            if neighbors_col not in cells_df.columns:
                raise KeyError(f"`neighbors_col='{neighbors_col}' not found in `cells_df`.")
            # explode neighbors
            work = cells_df[[dataset_col, cell_id_col, neighbors_col]].copy()
            # neighbors may be stored as JSON strings -> ensure list
            def _to_list(x):
                if isinstance(x, str):
                    try:
                        import json
                        val = json.loads(x)
                        return val if isinstance(val, list) else []
                    except Exception:
                        return []
                return list(x) if isinstance(x, (list, tuple, set, pd.Series, np.ndarray)) else []
            work["_nbrs"] = work[neighbors_col].apply(_to_list)
            pairs = work.explode("_nbrs", ignore_index=True)
            pairs = pairs.dropna(subset=["_nbrs"])
            pairs = pairs.rename(columns={cell_id_col: "cell_u"})
            pairs["cell_v"] = pd.to_numeric(pairs["_nbrs"], errors="coerce").astype("Int64")
            pairs = pairs.dropna(subset=["cell_v"])
            pairs["cell_v"] = pairs["cell_v"].astype(int)
            pairs = pairs[[dataset_col, "cell_u", "cell_v"]]

        # normalize undirected pair (Cell A < Cell B)
        pairs["Cell A"] = pairs[["cell_u", "cell_v"]].min(axis=1)
        pairs["Cell B"] = pairs[["cell_u", "cell_v"]].max(axis=1)
        pairs = pairs[[dataset_col, "Cell A", "Cell B"]].drop_duplicates()
        # drop self-pairs just in case
        pairs = pairs.loc[pairs["Cell A"] != pairs["Cell B"]].reset_index(drop=True)

        if pairs.empty:
            logger.warning("build_neighbor_pair_stats: no neighbor pairs found after normalization.")
            return pairs.assign(distance_um=np.nan, n_communications=0)

        # ---- Attach centroids for distance computation
        pairs = pairs.merge(
            centroids.rename(columns={
                cell_id_col: "Cell A",
                centroid_x_col: "x_a",
                centroid_y_col: "y_a",
            }),
            on=[dataset_col, "Cell A"], how="left"
        ).merge(
            centroids.rename(columns={
                cell_id_col: "Cell B",
                centroid_x_col: "x_b",
                centroid_y_col: "y_b",
            }),
            on=[dataset_col, "Cell B"], how="left"
        )

        # Drop pairs missing centroids (can happen if ROI filtered)
        before = len(pairs)
        pairs = pairs.dropna(subset=["x_a", "y_a", "x_b", "y_b"])
        dropped = before - len(pairs)
        if dropped:
            logger.info("build_neighbor_pair_stats: dropped %d pairs missing centroids.", dropped)

        # Compute Euclidean distance (um)
        pairs["distance_um"] = np.sqrt((pairs["x_a"] - pairs["x_b"])**2 + (pairs["y_a"] - pairs["y_b"])**2)

        # ---- Communications: count per unordered pair per dataset
        if comm_df.empty:
            comm_counts = pd.DataFrame(columns=[dataset_col, "Cell A", "Cell B", "n_communications"])
        else:
            comm = comm_df[[dataset_col, "Origin cell ID", "Cause cell ID"]].copy()
            comm["Cell A"] = comm[["Origin cell ID", "Cause cell ID"]].min(axis=1)
            comm["Cell B"] = comm[["Origin cell ID", "Cause cell ID"]].max(axis=1)
            comm_counts = (
                comm.groupby([dataset_col, "Cell A", "Cell B"])
                    .size()
                    .reset_index(name="n_communications")
            )

        # ---- Join pairs with counts (keep zero counts)
        out = pairs.merge(
            comm_counts,
            on=[dataset_col, "Cell A", "Cell B"],
            how="left"
        )
        out["n_communications"] = out["n_communications"].fillna(0).astype(int)

        # Final tidy columns
        out = out[[dataset_col, "Cell A", "Cell B", "distance_um", "n_communications"]].sort_values(
            by=[dataset_col, "Cell A", "Cell B"], kind="mergesort"
        ).reset_index(drop=True)

        logger.info(
            "build_neighbor_pair_stats: built %d pairs across %d datasets (mean distance=%.2f um)",
            len(out), out[dataset_col].nunique(), out["distance_um"].mean() if not out.empty else float("nan")
        )
        return out

    except Exception as exc:
        logger.exception("build_neighbor_pair_stats failed: %s", exc)
        raise