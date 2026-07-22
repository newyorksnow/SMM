""" 
Model validation: cumulative bite-count model. (scenario i, w=1)
  k_i(t) = (beta_i / l_i) * (1 - exp(-l_i * t))
against sheets entry_17 – entry_20 of the dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  MODEL PARAMETERS & FORMULA
# ─────────────────────────────────────────────
BETA  = 0.08502
L     = 0.00007
SHEETS = ["entry_17", "entry_18", "entry_19", "entry_20"]
FILE   = "Downloads/CORRECT/bite_gt_cumulative_timestamps CORRECT.xlsx"


def model(t: np.ndarray, beta: float = BETA, l: float = L) -> np.ndarray:
    """Cumulative bite-count model.  Uses expm1 for numerical stability."""
    return (beta / l) * (-np.expm1(-l * t))


# ─────────────────────────────────────────────
# 2.  VALIDATION HELPERS
# ─────────────────────────────────────────────
def load_sheet(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [c.lower().strip() for c in df.columns]
    # Expected columns: timestamp, cumulative_bites
    df = df.rename(columns={df.columns[0]: "t", df.columns[1]: "y_true"})
    df = df.dropna().reset_index(drop=True)
    df["y_pred"] = model(df["t"].values)
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    t: np.ndarray | None = None) -> dict:
    """Return a comprehensive dict of regression / goodness-of-fit metrics."""
    residuals = y_true - y_pred
    n = len(y_true)

    # --- basic error statistics ---
    me   = np.mean(residuals)                          # Mean Error (bias)
    mae  = np.mean(np.abs(residuals))                  # Mean Absolute Error
    mse  = np.mean(residuals**2)                       # Mean Squared Error
    rmse = np.sqrt(mse)                                # Root MSE
    med_ae = np.median(np.abs(residuals))              # Median Absolute Error
    max_ae = np.max(np.abs(residuals))                 # Max Absolute Error
    p95_ae = np.percentile(np.abs(residuals), 95)      # 95th-pct Abs Error

    # --- relative / normalised errors ---
    # avoid dividing by zero for early timestamps where y_true == 0
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(residuals[nonzero_mask] /
                              y_true[nonzero_mask])) * 100   # %
        smape = np.mean(
            2 * np.abs(residuals[nonzero_mask]) /
            (np.abs(y_true[nonzero_mask]) + np.abs(y_pred[nonzero_mask]))
        ) * 100                                               # %
    else:
        mape = smape = np.nan

    range_true = y_true.max() - y_true.min()
    nrmse_range = (rmse / range_true * 100) if range_true != 0 else np.nan
    nrmse_mean  = (rmse / y_true.mean() * 100) if y_true.mean() != 0 else np.nan

    # --- R-squared & variants ---
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Adjusted R² (single predictor model)
    p = 1   # one free parameter drives the curve (the ratio beta/l, with l shaping)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else np.nan

    # --- correlation ---
    pearson_r, pearson_p  = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    # --- explained variance ---
    ev = 1 - np.var(residuals) / np.var(y_true) if np.var(y_true) != 0 else np.nan

    # --- final-value accuracy ---
    y_true_final = y_true[-1]
    y_pred_final = y_pred[-1]
    final_pct_err = (
        (y_pred_final - y_true_final) / y_true_final * 100
        if y_true_final != 0 else np.nan
    )

    # --- residual normality (Shapiro-Wilk on ≤5000 samples) ---
    sample = residuals if n <= 5000 else residuals[
        np.random.choice(n, 5000, replace=False)]
    _, sw_p = stats.shapiro(sample)

    # --- Durbin-Watson (autocorrelation of residuals) ---
    if n > 1:
        diff_res = np.diff(residuals)
        dw = np.sum(diff_res**2) / np.sum(residuals**2)
    else:
        dw = np.nan

    return dict(
        n                   = n,
        # ----- bias / central tendency -----
        mean_error          = me,
        # ----- absolute error -----
        mae                 = mae,
        median_abs_error    = med_ae,
        max_abs_error       = max_ae,
        p95_abs_error       = p95_ae,
        # ----- squared error -----
        mse                 = mse,
        rmse                = rmse,
        # ----- relative error -----
        mape_pct            = mape,
        smape_pct           = smape,
        nrmse_range_pct     = nrmse_range,
        nrmse_mean_pct      = nrmse_mean,
        # ----- goodness of fit -----
        r_squared           = r2,
        r_squared_adj       = r2_adj,
        explained_variance  = ev,
        # ----- correlation -----
        pearson_r           = pearson_r,
        pearson_p_value     = pearson_p,
        spearman_r          = spearman_r,
        spearman_p_value    = spearman_p,
        # ----- residual diagnostics -----
        residual_std        = np.std(residuals),
        residual_skewness   = float(stats.skew(residuals)),
        residual_kurtosis   = float(stats.kurtosis(residuals)),
        shapiro_wilk_p      = sw_p,
        durbin_watson       = dw,
        # ----- end-point accuracy -----
        final_true          = y_true_final,
        final_pred          = y_pred_final,
        final_pct_error     = final_pct_err,
    )


def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────
# 3.  RUN VALIDATION PER SHEET
# ─────────────────────────────────────────────
print("=" * 70)
print("  MODEL VALIDATION  |  beta = 0.08502  |  l = 0.00007")
print(f"  Sheets: {', '.join(SHEETS)}")
print("=" * 70)

all_y_true = []
all_y_pred = []
all_t      = []
per_sheet_metrics = {}

for sheet in SHEETS:
    df = load_sheet(FILE, sheet)
    y_true = df["y_true"].values.astype(float)
    y_pred = df["y_pred"].values
    t      = df["t"].values

    all_y_true.append(y_true)
    all_y_pred.append(y_pred)
    all_t.append(t)

    per_sheet_metrics[sheet] = compute_metrics(y_true, y_pred, t)

    print(f"\n── {sheet} ──────────────────────────────")
    m = per_sheet_metrics[sheet]
    print(f"  Observations             : {m['n']}")
    print(f"  Duration (timestamps)    : {t[0]:.1f} → {t[-1]:.1f}  (step ≈ {np.diff(t[:10]).mean():.2f})")
    print(f"  True cumulative bites    : {int(y_true[0])} → {int(y_true[-1])}")
    print(f"  Predicted cumulative     : {y_pred[0]:.3f} → {y_pred[-1]:.3f}")


# ─────────────────────────────────────────────
# 4.  POOLED (GLOBAL) METRICS
# ─────────────────────────────────────────────
cat_y_true = np.concatenate(all_y_true)
cat_y_pred = np.concatenate(all_y_pred)
cat_t      = np.concatenate(all_t)

global_metrics = compute_metrics(cat_y_true, cat_y_pred, cat_t)
per_sheet_metrics["POOLED"] = global_metrics


# ─────────────────────────────────────────────
# 5.  METRICS REPORT
# ─────────────────────────────────────────────

SECTION = "\n" + "═" * 70
SEP     = "─" * 70

print(SECTION)
print("  VALIDATION METRICS")
print(SECTION)

COLS = SHEETS + ["POOLED"]
COL_W = 13
LABEL_W = 30

def row(label, key, dec=4):
    vals = "".join(
        f"{fmt(per_sheet_metrics[c][key], dec):>{COL_W}}"
        for c in COLS
    )
    print(f"  {label:<{LABEL_W}}{vals}")

header_vals = "".join(f"{c:>{COL_W}}" for c in COLS)
print(f"\n  {'Metric':<{LABEL_W}}{header_vals}")
print(f"  {SEP}")

print(f"  {'── DATASET INFO ──':<{LABEL_W}}")
row("N (observations)",           "n",            0)
row("Final true bites",           "final_true",   0)
row("Final predicted bites",      "final_pred",   3)
row("Final % error",              "final_pct_error", 2)

print(f"\n  {'── BIAS & CENTRAL TENDENCY ──':<{LABEL_W}}")
row("Mean Error (bias)",          "mean_error",   4)

print(f"\n  {'── ABSOLUTE ERROR ──':<{LABEL_W}}")
row("MAE",                        "mae",          4)
row("Median Absolute Error",      "median_abs_error", 4)
row("Max Absolute Error",         "max_abs_error", 4)
row("95th-pct Absolute Error",    "p95_abs_error", 4)

print(f"\n  {'── SQUARED ERROR ──':<{LABEL_W}}")
row("MSE",                        "mse",          4)
row("RMSE",                       "rmse",         4)

print(f"\n  {'── RELATIVE / NORMALISED ERROR ──':<{LABEL_W}}")
row("MAPE (%)",                   "mape_pct",     3)
row("SMAPE (%)",                  "smape_pct",    3)
row("NRMSE / range (%)",          "nrmse_range_pct", 3)
row("NRMSE / mean (%)",           "nrmse_mean_pct", 3)

print(f"\n  {'── GOODNESS OF FIT ──':<{LABEL_W}}")
row("R²",                         "r_squared",    6)
row("Adjusted R²",                "r_squared_adj", 6)
row("Explained Variance",         "explained_variance", 6)

print(f"\n  {'── CORRELATION ──':<{LABEL_W}}")
row("Pearson  r",                 "pearson_r",    6)
row("Pearson  p-value",           "pearson_p_value", 6)
row("Spearman r",                 "spearman_r",   6)
row("Spearman p-value",           "spearman_p_value", 6)

print(f"\n  {'── RESIDUAL DIAGNOSTICS ──':<{LABEL_W}}")
row("Residual Std Dev",           "residual_std",  4)
row("Residual Skewness",          "residual_skewness", 4)
row("Residual Kurtosis",          "residual_kurtosis", 4)
row("Shapiro-Wilk p (normality)", "shapiro_wilk_p", 6)
row("Durbin-Watson",              "durbin_watson", 4)

print(f"\n  {SEP}")
print("""
  METRIC GLOSSARY
  ───────────────
  Mean Error (bias)   : Signed avg residual; + means model over-predicts.
  MAE                 : Average absolute deviation from truth.
  Median AE           : Robust to outliers; complement to MAE.
  Max / P95 AE        : Worst-case and near-worst-case error magnitude.
  MSE / RMSE          : Penalise large errors; RMSE is in original units.
  MAPE                : % error, computed only on non-zero actuals.
  SMAPE               : Symmetric MAPE; bounded and more stable than MAPE.
  NRMSE/range         : RMSE normalised by (max–min); scale-free comparison.
  NRMSE/mean          : RMSE as % of mean truth; another scale-free view.
  R²                  : Fraction of variance explained; 1.0 = perfect fit.
  Adjusted R²         : R² penalised for number of parameters (p=1 here).
  Explained Variance  : Like R² but ignores mean bias.
  Pearson r           : Linear correlation coefficient.
  Spearman r          : Rank-based (monotonic) correlation.
  Residual Skewness   : 0 = symmetric; + = right-skewed residuals.
  Residual Kurtosis   : 0 = normal; excess kurtosis (>0) = heavy tails.
  Shapiro-Wilk p      : p>0.05 → residuals not significantly non-normal.
  Durbin-Watson       : ~2 = no autocorrelation; <1 or >3 = concern.
""")