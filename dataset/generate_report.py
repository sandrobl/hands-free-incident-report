"""
Results.csv Analytics Script
==============================
Computes counts, averages, medians, and additional metrics
across all records and per category.

Saves all charts to: results_charts.pdf and individual PNG files.

Columns of interest:
  - success_a, success_d      : success outcome types
  - failure_b, failure_c,
    failure_d, failure_e      : failure outcome types
  - completion_time           : time in seconds to complete the task
  - WER_score                 : Word Error Rate (0.0 = perfect, 1.0 = total mismatch)
  - category                  : grouping dimension
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# ============================================================ Config ===================================================================
CSV_PATH   = "Results.csv"
CHARTS_PDF = "results_charts.pdf"

OUTCOME_COLS = {
    "success_a": "success",
    "failure_b": "failure",
    "failure_c": "failure",
    "failure_d": "failure",
    "failure_e": "failure",
}

OUTCOME_COLORS = {
    "success_a": "#2ecc71",
    "failure_b": "#ff0000",
    "failure_c": "#9b59b6",
    "failure_d": "#f39c12",
    "failure_e": "#34495e",
}


METRIC_COLS = ["completion_time", "WER_score"]
# Replaced dashes with equal signs
SEPARATOR   = "=" * 72

# ============================================================ Colour palette ============================================================
SUCCESS_COLOR = "#2ecc71"
FAILURE_COLOR = "#e74c3c"
NEUTRAL_COLOR = "#3498db"
CATEGORY_PAL  = plt.cm.tab10.colors

WER_BAND_COLORS = {
    "Perfect (0)":          "#27ae60",
    "Good (0 to 0.2)":      "#2ecc71",
    "Acceptable (0.2 to 0.4)": "#f1c40f",
    "Poor (0.4 to 0.6)":    "#e67e22",
    "Bad (0.6 to 0.8)":     "#e74c3c",
    "Very Bad (0.8 to 1.0)": "#c0392b",
    "Over 1.0":             "#7f0000",
}

# ============================================================ Style =====================================================================
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        "#cccccc",
    "grid.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})


# ============================================================ Data loading ==============================================================
def load_data(path):
    df = pd.read_csv(path, sep=";")
    for col in OUTCOME_COLS:
        df[col] = df[col].notna() & (df[col].astype(str).str.strip() != "")
    return df


# ============================================================ Analysis helpers ==========================================================
def outcome_summary(df, label="ALL"):
    rows = []
    for col, kind in OUTCOME_COLS.items():
        rows.append({"outcome": col, "type": kind, "count": int(df[col].sum())})
    result     = pd.DataFrame(rows)
    total      = len(df)
    total_succ = int(df[[c for c, k in OUTCOME_COLS.items() if k == "success"]].any(axis=1).sum())
    total_fail = int(df[[c for c, k in OUTCOME_COLS.items() if k == "failure"]].any(axis=1).sum())
    result.loc[len(result)] = {"outcome": "TOTAL_SUCCESS", "type": "summary", "count": total_succ}
    result.loc[len(result)] = {"outcome": "TOTAL_FAILURE", "type": "summary", "count": total_fail}
    result.loc[len(result)] = {"outcome": "UNCLASSIFIED",  "type": "summary", "count": total - total_succ - total_fail}
    result.loc[len(result)] = {"outcome": "TOTAL_RECORDS", "type": "summary", "count": total}
    result.insert(0, "scope", label)
    return result


def metric_stats(df, label="ALL"):
    rows = []
    for col in METRIC_COLS:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75]).values
        rows.append({"scope": label, "metric": col, "n": len(s),
                     "mean": round(s.mean(),3), "median": round(s.median(),3),
                     "std":  round(s.std(),3),  "min": round(s.min(),3),
                     "q1":   round(q1,3),        "q3": round(q3,3),
                     "max":  round(s.max(),3)})
    return pd.DataFrame(rows)


def wer_band_counts(df, label="ALL"):
    bins   = [-0.001, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
    labels = list(WER_BAND_COLORS.keys())
    df2    = df.copy()
    df2["wer_band"] = pd.cut(df2["WER_score"], bins=bins, labels=labels)
    counts = df2.groupby("wer_band", observed=True).size().reset_index(name="count")
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    counts.insert(0, "scope", label)
    return counts


def time_band_counts(df, label="ALL"):
    bins   = [0, 60, 120, 180, 300, np.inf]
    labels = ["<=60 s", "61-120 s", "121-180 s", "181-300 s", ">300 s"]
    df2    = df[df["completion_time"].notna()].copy()
    df2["time_band"] = pd.cut(df2["completion_time"], bins=bins, labels=labels)
    counts = df2.groupby("time_band", observed=True).size().reset_index(name="count")
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    counts.insert(0, "scope", label)
    return counts


def success_rate_by_outcome(df, label="ALL"):
    total = len(df)
    rows  = []
    for col, kind in OUTCOME_COLS.items():
        n = int(df[col].sum())
        rows.append({"scope": label, "outcome": col, "type": kind,
                     "n": n, "rate_%": round(n / total * 100, 1) if total else 0})
    return pd.DataFrame(rows)


def wer_by_success_failure(df):
    sm = df[[c for c, k in OUTCOME_COLS.items() if k == "success"]].any(axis=1)
    fm = df[[c for c, k in OUTCOME_COLS.items() if k == "failure"]].any(axis=1)
    rows = []
    for label, mask in [("success", sm), ("failure", fm)]:
        s = df[mask]["WER_score"].dropna()
        if not s.empty:
            rows.append({"outcome_type": label, "n": len(s),
                         "mean_WER":    round(s.mean(),3),
                         "median_WER":  round(s.median(),3),
                         "mean_time":   round(df[mask]["completion_time"].dropna().mean(),1),
                         "median_time": round(df[mask]["completion_time"].dropna().median(),1)})
    return pd.DataFrame(rows)


def correlation_metrics(df):
    valid = df[["completion_time", "WER_score"]].dropna()
    corr  = valid["completion_time"].corr(valid["WER_score"])
    strength = ("negligible" if abs(corr) < 0.1 else "weak" if abs(corr) < 0.3
                else "moderate" if abs(corr) < 0.5 else "strong")
    print(f"  Pearson correlation (completion_time vs WER_score): {corr:.4f}")
    print(f"  Interpretation: {strength} {'positive' if corr > 0 else 'negative'} relationship")


def print_section(title):
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")

def print_df(df):
    print(df.to_string(index=False))


# ============================================================ Chart helpers =============================================================
def save_fig(fig, filename, pdf):
    """Helper to save to both PDF and individual PNG."""
    pdf.savefig(fig)
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def add_bar_labels(ax, pad=2, fmt="{:.0f}"):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.text(p.get_x() + p.get_width()/2, h + pad,
                    fmt.format(h), ha="center", va="bottom", fontsize=8, color="#333")

def add_hbar_labels(ax, pad=1, fmt="{:.0f}"):
    for p in ax.patches:
        w = p.get_width()
        if w > 0:
            ax.text(w + pad, p.get_y() + p.get_height()/2,
                    fmt.format(w), ha="left", va="center", fontsize=8, color="#333")


# ============================================================ Chart functions ===========================================================

def chart_outcome_counts_all(df, pdf):
    raw    = {col: int(df[col].sum()) for col in OUTCOME_COLS}
    colors = [SUCCESS_COLOR if OUTCOME_COLS[c] == "success" else FAILURE_COLOR for c in raw]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(raw.keys()), list(raw.values()), color=colors, edgecolor="white", height=0.55)
    add_hbar_labels(ax, pad=1)
    ax.set_xlabel("Count")
    ax.set_title("Outcome Counts: All Records (green=success, red=failure)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(raw.values()) * 1.18)
    fig.tight_layout()
    save_fig(fig, "outcome_counts_all", pdf)


def chart_outcome_counts_per_category(df, categories, pdf):
    succ_cols = [c for c, k in OUTCOME_COLS.items() if k == "success"]
    fail_cols = [c for c, k in OUTCOME_COLS.items() if k == "failure"]
    succs = [df[df["category"]==cat][succ_cols].any(axis=1).sum() for cat in categories]
    fails = [df[df["category"]==cat][fail_cols].any(axis=1).sum() for cat in categories]
    x, w  = np.arange(len(categories)), 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w/2, succs, width=w, label="Success", color=SUCCESS_COLOR, edgecolor="white")
    b2 = ax.bar(x + w/2, fails, width=w, label="Failure",  color=FAILURE_COLOR, edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                    str(int(b.get_height())), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Success vs Failure Count per Category")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "outcome_counts_per_category", pdf)

def chart_wer_by_failure_type(df, pdf):
    failure_cols = [c for c, k in OUTCOME_COLS.items() if k == "failure"]
    data = []
    for col in failure_cols:
        subset = df[df[col] == True]
        if not subset.empty:
            mean_wer = subset["WER_score"].mean()
            data.append({"failure_type": col, "mean_WER": mean_wer})
    
    wer_df = pd.DataFrame(data).sort_values("mean_WER", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(wer_df["failure_type"], wer_df["mean_WER"], color=FAILURE_COLOR, edgecolor="white", width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
                
    ax.set_ylabel("Average WER Score")
    ax.set_xlabel("Failure Type")
    ax.set_title("Average WER Score by Failure Type")
    ax.set_ylim(0, max(wer_df["mean_WER"]) * 1.2 if not wer_df.empty else 1)
    
    fig.tight_layout()
    save_fig(fig, "wer_by_failure_type", pdf)


def chart_success_rate_per_category(df, categories, pdf):
    rows = [{"category": cat,
             "success_rate": round(df[df["category"]==cat]
                [[c for c,k in OUTCOME_COLS.items() if k=="success"]].any(axis=1).mean()*100, 1)}
            for cat in categories]
    sr = pd.DataFrame(rows).sort_values("success_rate")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = [SUCCESS_COLOR if v >= 1 else FAILURE_COLOR for v in sr["success_rate"]]
    ax.barh(sr["category"], sr["success_rate"], color=colors, edgecolor="white", height=0.55)
    add_hbar_labels(ax, pad=0.5, fmt="{:.1f}")
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("Success Rate per Category (%)")
    ax.set_xlim(0, 115); ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "success_rate_per_category", pdf)


def chart_outcome_breakdown_per_category(df, categories, pdf):
    outcome_labels = list(OUTCOME_COLS.keys())
    palette = [OUTCOME_COLORS.get(col, "#7f8c8d") for col in outcome_labels]
    totals  = [len(df[df["category"]==cat]) for cat in categories]
    data    = {col: [df[df["category"]==cat][col].sum()/tot*100
                     for cat, tot in zip(categories, totals)]
               for col in outcome_labels}
    x, w   = np.arange(len(categories)), 0.55
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom  = np.zeros(len(categories))
    for col, color in zip(outcome_labels, palette):
        vals = np.array(data[col])
        ax.bar(x, vals, width=w, bottom=bottom, label=col, color=color, edgecolor="white")
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 3:
                ax.text(xi, b+v/2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel("Share (%)")
    ax.set_title("Percentage share of total records per category (%)")
    ax.legend(loc="upper right", fontsize=8); ax.set_ylim(0, 115)
    fig.tight_layout()
    save_fig(fig, "outcome_breakdown_percent", pdf)


def chart_mean_median_per_category(df, categories, pdf):
    for metric, unit in [
        ("completion_time", "Seconds"),
        ("WER_score",       "WER"),
    ]:
        means   = [df[df["category"]==cat][metric].mean()   for cat in categories]
        medians = [df[df["category"]==cat][metric].median() for cat in categories]
        x, w    = np.arange(len(categories)), 0.38
        fig, ax = plt.subplots(figsize=(12, 5))
        b1 = ax.bar(x-w/2, means,   width=w, label="Mean",   color=NEUTRAL_COLOR, edgecolor="white")
        b2 = ax.bar(x+w/2, medians, width=w, label="Median", color="#9b59b6",     edgecolor="white")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                        f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(categories, rotation=25, ha="right")
        ax.set_ylabel(unit)
        ax.set_title(f"Mean & Median {metric} per Category")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, f"mean_median_{metric}", pdf)


def chart_boxplot_per_category(df, categories, pdf):
    for metric, unit in [
        ("completion_time", "Seconds"),
        ("WER_score",       "WER Score"),
    ]:
        data = [df[df["category"]==cat][metric].dropna().values for cat in categories]
        fig, ax = plt.subplots(figsize=(12, 5))
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color":"black","linewidth":2},
                        whiskerprops={"linewidth":1.2}, capprops={"linewidth":1.2})
        for patch, color in zip(bp["boxes"], CATEGORY_PAL):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(categories)+1))
        ax.set_xticklabels(categories, rotation=25, ha="right")
        ax.set_ylabel(unit)
        ax.set_title(f"{metric} Box Plot per Category")
        fig.tight_layout()
        save_fig(fig, f"boxplot_{metric}", pdf)


def chart_wer_bands_all(df, pdf):
    bands  = list(WER_BAND_COLORS.keys())
    colors = list(WER_BAND_COLORS.values())
    bins   = [-0.001, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
    df2    = df.copy()
    df2["wer_band"] = pd.cut(df2["WER_score"], bins=bins, labels=bands)
    counts = df2.groupby("wer_band", observed=True).size()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bands, counts.values, color=colors, edgecolor="white", width=0.65)
    add_bar_labels(ax, pad=0.3)
    ax.set_xlabel("WER Quality Band"); ax.set_ylabel("Count")
    ax.set_title("WER Score Quality Distribution: All Records")
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands, rotation=20, ha="right")
    fig.tight_layout()
    save_fig(fig, "wer_bands_all", pdf)


def chart_wer_bands_heatmap(df, categories, pdf):
    bands = list(WER_BAND_COLORS.keys())
    bins  = [-0.001, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
    df2   = df.copy()
    df2["wer_band"] = pd.cut(df2["WER_score"], bins=bins, labels=bands)
    matrix = (df2.groupby(["category","wer_band"], observed=True)
                 .size().unstack(fill_value=0)
                 .reindex(columns=bands, fill_value=0))
    fig, ax = plt.subplots(figsize=(13, 5))
    data_arr = matrix.values
    im = ax.imshow(data_arr, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks(range(len(matrix.columns))); ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)));   ax.set_yticklabels(matrix.index)
    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[1]):
            ax.text(j, i, str(data_arr[i, j]), ha="center", va="center",
                    fontsize=9, color="black" if data_arr[i, j] < data_arr.max()*0.6 else "white")
    ax.set_title("WER Quality Band Counts per Category (heatmap)")
    ax.set_xlabel("WER Band"); ax.set_ylabel("Category")
    ax.grid(False)
    fig.tight_layout()
    save_fig(fig, "wer_heatmap", pdf)


def chart_time_bands_all(df, pdf):
    bins    = [0, 60, 120, 180, 300, np.inf]
    labels  = ["<=60 s", "61-120 s", "121-180 s", "181-300 s", ">300 s"]
    palette = ["#2ecc71","#3498db","#f1c40f","#e67e22","#e74c3c"]
    df2     = df[df["completion_time"].notna()].copy()
    df2["time_band"] = pd.cut(df2["completion_time"], bins=bins, labels=labels)
    counts  = df2.groupby("time_band", observed=True).size()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, counts.values, color=palette, edgecolor="white", width=0.6)
    add_bar_labels(ax, pad=0.3)
    ax.set_xlabel("Time Band"); ax.set_ylabel("Count")
    ax.set_title("Completion Time Distribution: All Records")
    fig.tight_layout()
    save_fig(fig, "time_bands_all", pdf)


def chart_time_bands_per_category(df, categories, pdf):
    bins    = [0, 60, 120, 180, 300, np.inf]
    labels  = ["<=60 s", "61-120 s", "121-180 s", "181-300 s", ">300 s"]
    palette = ["#2ecc71","#3498db","#f1c40f","#e67e22","#e74c3c"]
    df2     = df[df["completion_time"].notna()].copy()
    df2["time_band"] = pd.cut(df2["completion_time"], bins=bins, labels=labels)
    matrix  = (df2.groupby(["category","time_band"], observed=True)
                  .size().unstack(fill_value=0)
                  .reindex(columns=labels, fill_value=0))
    x, n, w = np.arange(len(categories)), len(labels), 0.14
    fig, ax  = plt.subplots(figsize=(14, 6))
    for i, (band, color) in enumerate(zip(labels, palette)):
        offset = (i - n/2 + 0.5) * w
        vals   = [matrix.loc[cat, band] if cat in matrix.index else 0 for cat in categories]
        ax.bar(x+offset, vals, width=w, label=band, color=color, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Completion Time Distribution per Category")
    ax.legend(title="Time Band", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "time_bands_per_category", pdf)


def chart_scatter_time_vs_wer(df, categories, pdf):
    valid = df.dropna(subset=["completion_time","WER_score"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat, color in zip(categories, CATEGORY_PAL):
        sub = valid[valid["category"]==cat]
        ax.scatter(sub["completion_time"], sub["WER_score"],
                   label=cat, color=color, alpha=0.7, edgecolors="white", s=55)
    m, b = np.polyfit(valid["completion_time"], valid["WER_score"], 1)
    xs   = np.linspace(valid["completion_time"].min(), valid["completion_time"].max(), 100)
    ax.plot(xs, m*xs+b, color="black", linewidth=1.5, linestyle="--", label="Trend")
    r = valid["completion_time"].corr(valid["WER_score"])
    ax.set_xlabel("Completion Time (s)"); ax.set_ylabel("WER Score")
    ax.set_title(f"Completion Time vs WER Score (r = {r:.3f})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, "scatter_time_vs_wer", pdf)


def chart_wer_time_success_vs_failure(df, pdf):
    sf  = wer_by_success_failure(df)
    metrics = [("mean_WER","Mean WER","WER"), ("median_WER","Median WER","WER"),
               ("mean_time","Mean Time","Sec"), ("median_time","Median Time","Sec")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for ax, (col, label, unit) in zip(axes, metrics):
        vals = sf[col].values
        bars = ax.bar(sf["outcome_type"], vals,
                      color=[SUCCESS_COLOR, FAILURE_COLOR], edgecolor="white", width=0.5)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label); ax.set_ylabel(unit); ax.set_ylim(0, vals.max()*1.25)
    fig.suptitle("Success vs Failure: WER & Time Comparison", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "success_vs_failure_comparison", pdf)


def chart_category_ranking_wer(df, pdf):
    cat_wer = (df.groupby("category")["WER_score"]
                 .agg(mean_WER="mean", median_WER="median")
                 .sort_values("mean_WER").reset_index())
    y, w = np.arange(len(cat_wer)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.barh(y-w/2, cat_wer["mean_WER"],   height=w, label="Mean",   color=NEUTRAL_COLOR, edgecolor="white")
    b2 = ax.barh(y+w/2, cat_wer["median_WER"], height=w, label="Median", color="#9b59b6",     edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_width()+0.003, b.get_y()+b.get_height()/2,
                    f"{b.get_width():.3f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(cat_wer["category"])
    ax.set_xlabel("WER Score (lower = better)")
    ax.set_title("Category Ranking by WER Score (best to worst)")
    ax.legend(); ax.set_xlim(0, cat_wer["mean_WER"].max()*1.2)
    fig.tight_layout()
    save_fig(fig, "category_ranking_wer", pdf)


def chart_outliers(df, pdf):
    ct       = df["completion_time"].dropna()
    q1, q3   = ct.quantile(0.25), ct.quantile(0.75)
    hi       = q3 + 1.5*(q3-q1)
    df2      = df[df["completion_time"].notna()].copy()
    df2["is_outlier"] = df2["completion_time"] > hi
    cats     = sorted(df2["category"].unique())
    fig, ax  = plt.subplots(figsize=(12, 5))
    rng      = np.random.default_rng(42)
    for i, (cat, color) in enumerate(zip(cats, CATEGORY_PAL)):
        sub    = df2[df2["category"]==cat]
        jitter = rng.uniform(-0.2, 0.2, len(sub))
        ax.scatter([i+j for j in jitter], sub["completion_time"],
                   color=color, alpha=0.6, s=40, edgecolors="white", zorder=3)
        out = sub[sub["is_outlier"]]
        if not out.empty:
            out_mask = sub["is_outlier"].values
            out_jitter = np.array(jitter)[out_mask]
            ax.scatter([i+j for j in out_jitter], out["completion_time"],
                       color="red", s=80, marker="D", edgecolors="black", zorder=4)
    ax.axhline(hi, color="red", linestyle="--", linewidth=1.5, label=f"Fence ({hi:.0f} s)")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel("Completion Time (s)")
    ax.set_title("Completion Time per Category: Outliers in Red")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "completion_time_outliers", pdf)

# ============================================================ Visualisation orchestrator ================================================

def chart_outcome_breakdown_abs_per_category(df, categories, pdf):
    outcome_labels = list(OUTCOME_COLS.keys())
    palette = [OUTCOME_COLORS.get(col, "#7f8c8d") for col in outcome_labels]
    data = {col: [df[df["category"]==cat][col].sum() for cat in categories] for col in outcome_labels}
    x, w = np.arange(len(categories)), 0.55
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(categories))
    for col, color in zip(outcome_labels, palette):
        vals = np.array(data[col])
        ax.bar(x, vals, width=w, bottom=bottom, label=col, color=color, edgecolor="white")
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(xi, b+v/2, f"{int(v)}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel("Absolute Count")
    ax.set_title("Outcome Type Breakdown per Category Absolute")
    ax.legend(loc="upper right", fontsize=8)
    max_val = np.max(bottom) if len(bottom) > 0 else 10
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 10)
    fig.tight_layout()
    save_fig(fig, "outcome_breakdown_absolute", pdf)

def chart_failure_isolation_heatmap(df, categories, pdf):
    failure_cols = ["failure_b", "failure_c", "failure_d", "failure_e"]
    failure_cols = [c for c in failure_cols if c in df.columns]
    data = []
    for cat in categories:
        sub = df[df["category"] == cat]
        data.append([sub[col].sum() for col in failure_cols])
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(data, cmap="Reds", aspect="auto")
    ax.set_xticks(np.arange(len(failure_cols)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(failure_cols)
    ax.set_yticklabels(categories)
    for i in range(len(categories)):
        for j in range(len(failure_cols)):
            val = int(data[i, j])
            color = "white" if val > np.max(data)/2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)
    ax.set_title("Failure Isolation Heatmap")
    ax.grid(False)
    fig.colorbar(cax)
    fig.tight_layout()
    save_fig(fig, "failure_isolation_heatmap", pdf)

def chart_normalized_failure_distribution(df, categories, pdf):
    failure_cols = ["failure_b", "failure_c", "failure_d", "failure_e"]
    failure_cols = [c for c in failure_cols if c in df.columns]
    palette = ["#e74c3c", "#c0392b", "#e67e22", "#f39c12"]
    data = {col: [] for col in failure_cols}
    valid_cats = []
    for cat in categories:
        sub = df[df["category"] == cat]
        tot = sub[failure_cols].sum().sum()
        if tot > 0:
            valid_cats.append(cat)
            for col in failure_cols:
                data[col].append(sub[col].sum() / tot * 100)
    x = np.arange(len(valid_cats))
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(valid_cats))
    for col, color in zip(failure_cols, palette):
        vals = np.array(data[col])
        ax.bar(x, vals, width=0.55, bottom=bottom, label=col, color=color, edgecolor="white")
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(xi, b + v/2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(valid_cats, rotation=25, ha="right")
    ax.set_ylabel("Share of Total Failures (%)")
    ax.set_title("Normalized Failure Distribution per Category")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    save_fig(fig, "normalized_failure_distribution", pdf)

def chart_systemic_pareto_analysis(df, pdf):
    failure_cols = ["failure_b", "failure_c", "failure_d", "failure_e"]
    failure_cols = [c for c in failure_cols if c in df.columns]
    counts = {col: df[col].sum() for col in failure_cols}
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    labels = list(sorted_counts.keys())
    vals = list(sorted_counts.values())
    total = sum(vals)
    cumulative = []
    current_sum = 0
    for v in vals:
        current_sum += v
        cumulative.append(current_sum / total * 100 if total > 0 else 0)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.grid(False)
    x = np.arange(len(labels))
    ax1.bar(x, vals, color="#e74c3c", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Absolute Count")
    for i, v in enumerate(vals):
        ax1.text(i, v + (max(vals)*0.02 if max(vals)>0 else 1), str(int(v)), ha="center", va="bottom", fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#34495e", marker="o", linestyle="solid", linewidth=2)
    ax2.set_ylabel("Cumulative Percentage (%)")
    ax2.set_ylim(0, 110)
    for i, c in enumerate(cumulative):
        ax2.text(i, c + 3, f"{c:.1f}%", ha="center", va="bottom", color="#34495e", fontsize=9)
    ax1.set_title("Systemic Pareto Analysis of Failure Types")
    fig.tight_layout()
    save_fig(fig, "pareto_analysis", pdf)


def generate_plots(df, categories, output_path):
    print(f"\n  Generating charts -> {output_path} and PNG files")
    with PdfPages(output_path) as pdf:
        # Cover page
        fig = plt.figure(figsize=(10, 5))
        fig.text(0.5, 0.62, "Results.csv Analytics Charts",
                 ha="center", va="center", fontsize=22, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        chart_outcome_counts_all(df, pdf)
        chart_outcome_counts_per_category(df, categories, pdf)
        chart_success_rate_per_category(df, categories, pdf)
        chart_wer_by_failure_type(df, pdf)
        chart_outcome_breakdown_per_category(df, categories, pdf)
        chart_outcome_breakdown_abs_per_category(df, categories, pdf)
        chart_failure_isolation_heatmap(df, categories, pdf)
        chart_normalized_failure_distribution(df, categories, pdf)
        chart_systemic_pareto_analysis(df, pdf)
        chart_mean_median_per_category(df, categories, pdf)     
        chart_boxplot_per_category(df, categories, pdf)          
        chart_wer_bands_all(df, pdf)
        chart_wer_bands_heatmap(df, categories, pdf)
        chart_time_bands_all(df, pdf)
        chart_time_bands_per_category(df, categories, pdf)
        chart_scatter_time_vs_wer(df, categories, pdf)
        chart_wer_time_success_vs_failure(df, pdf)
        chart_category_ranking_wer(df, pdf)
        chart_outliers(df, pdf)

# ============================================================ Main ======================================================================
def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df         = load_data(CSV_PATH)
    categories = sorted(df["category"].dropna().unique())

    print_section("OUTCOME COUNTS: ALL RECORDS")
    print_df(outcome_summary(df, "ALL"))

    print_section("OUTCOME COUNTS: PER CATEGORY")
    print_df(pd.concat([outcome_summary(df[df["category"]==c], c) for c in categories], ignore_index=True))

    print_section("COMPLETION TIME & WER SCORE STATS: ALL RECORDS")
    print_df(metric_stats(df, "ALL"))

    print_section("COMPLETION TIME & WER SCORE STATS: PER CATEGORY")
    print_df(pd.concat([metric_stats(df[df["category"]==c], c) for c in categories], ignore_index=True))

    print_section("SUCCESS / FAILURE RATES (% of total): ALL RECORDS")
    print_df(success_rate_by_outcome(df, "ALL"))

    print_section("SUCCESS / FAILURE RATES (% of total): PER CATEGORY")
    print_df(pd.concat([success_rate_by_outcome(df[df["category"]==c], c) for c in categories], ignore_index=True))

    print_section("WER QUALITY DISTRIBUTION: ALL RECORDS")
    print_df(wer_band_counts(df, "ALL"))

    print_section("WER QUALITY DISTRIBUTION: PER CATEGORY")
    print_df(pd.concat([wer_band_counts(df[df["category"]==c], c) for c in categories], ignore_index=True))

    print_section("COMPLETION TIME DISTRIBUTION: ALL RECORDS")
    print_df(time_band_counts(df, "ALL"))

    print_section("COMPLETION TIME DISTRIBUTION: PER CATEGORY")
    print_df(pd.concat([time_band_counts(df[df["category"]==c], c) for c in categories], ignore_index=True))

    print_section("ADDITIONAL METRICS")

    print("\n  Correlation: completion_time vs WER_score")
    correlation_metrics(df)

    print("\n  Mean & Median WER and time: SUCCESS vs FAILURE")
    print_df(wer_by_success_failure(df))

    print("\n  Category ranking by mean WER_score (ascending = better)")
    cat_wer = (df.groupby("category")["WER_score"]
                 .agg(n="count", mean_WER="mean", median_WER="median")
                 .sort_values("mean_WER").reset_index())
    cat_wer[["mean_WER","median_WER"]] = cat_wer[["mean_WER","median_WER"]].round(3)
    print_df(cat_wer)

    print("\n  Category ranking by success rate (descending = better)")
    rows = []
    for cat in categories:
        sub  = df[df["category"]==cat]
        succ = sub[[c for c,k in OUTCOME_COLS.items() if k=="success"]].any(axis=1).sum()
        rows.append({"category": cat, "total": len(sub), "successes": int(succ),
                     "success_rate%": round(succ/len(sub)*100, 1)})
    print_df(pd.DataFrame(rows).sort_values("success_rate%", ascending=False))

    print("\n  Completion time outliers (beyond 1.5 x IQR fence)")
    ct      = df["completion_time"].dropna()
    q1, q3  = ct.quantile(0.25), ct.quantile(0.75)
    lo, hi  = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
    outliers = df[(df["completion_time"]<lo)|(df["completion_time"]>hi)]
    print(f"  IQR fence: [{lo:.1f}, {hi:.1f}]  outlier count: {len(outliers)}")
    if not outliers.empty:
        cols = [c for c in ["dataset(id)","category","completion_time","WER_score"] if c in df.columns]
        print_df(outliers[cols].reset_index(drop=True))
    
    print("\n  Average WER score per failure type")
    failure_cols = [c for c, k in OUTCOME_COLS.items() if k == "failure"]
    f_rows = []
    for col in failure_cols:
        subset = df[df[col] == True]
        if not subset.empty:
            f_rows.append({"failure_type": col, "mean_WER": round(subset["WER_score"].mean(), 3)})
    print_df(pd.DataFrame(f_rows).sort_values("mean_WER", ascending=False))

    print(f"\n{SEPARATOR}\n  Analysis complete.\n{SEPARATOR}")

    generate_plots(df, categories, CHARTS_PDF)


if __name__ == "__main__":
    main()