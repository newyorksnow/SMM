"""
i) Calibration of a SOLO eating/biting model (no interaction term). (scenario i)

Original coupled model:
    k_i'(t) = beta_i * exp(-l_i * t) * w(t)
    w(t)    = (|k_j - k_i| + 1) * a_ij

Since agent i is eating alone, there is no second agent to generate the
gap term |k_j - k_i|, so we remove w(t) entirely (set w = 1). What's left
is a simple decaying-rate growth model:

    k_i'(t) = beta_i * exp(-l_i * t),     k_i(0) = 0

This ODE has a closed-form solution (no numerical ODE solver needed):

    k_i(t) = (beta_i / l_i) * (1 - exp(-l_i * t))

Only two parameters remain (beta_i, l_i).

CHANGES IN THIS VERSION:
  - No upper bounds on beta_i or l_i -- only the lower constraint > 0 is
    enforced (via a small positive floor of 1e-6). Previously both were
    capped (beta<=50, l<=2); those caps are removed.
  - Only the FIRST 17 of the 21 sheets are used for calibration (held out
    the remaining 4 -- useful later as an out-of-sample check).
  - No files are written (no CSV, no plot). Everything is just printed.
  - Initial guesses for curve_fit are no longer a single fixed (beta, l)
    pair -- instead we use MULTI-START optimization:
        1. Generate a random initial guess (beta0, l0).
        2. Run curve_fit() starting from that guess.
        3. Record how good the fit is (RMSE).
        4. Repeat N_STARTS times.
        5. Keep the fit with the lowest RMSE.
    This guards against the (small) chance that a single fixed starting
    point lands in a bad spot, and gives some empirical evidence that the
    fit we report is a genuine global optimum, not an artifact of one
    particular guess.

Fitting strategy: fit (beta_i, l_i) to each of the 17 replicate curves
independently using scipy.optimize.curve_fit (closed-form curve, no ODE
integration required), with multi-start initial guesses. Then report the
across-replicate mean and a 95% confidence interval (t-distribution) as
the calibrated parameters.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as tdist

DATA_PATH = "Downloads/CORRECT/bite_gt_cumulative_timestamps CORRECT.xlsx"
N_CALIB_SHEETS = 17                      # only the first 17 sheets are used
BOUNDS = ([1e-6, 1e-6], [np.inf, np.inf])  # only a lower bound (>0); no upper bound
N_STARTS = 25                            # number of random initial guesses per sheet
RNG_SEED = 0                             # fixed seed -> reproducible random guesses


def k_model(t, beta, l):
    """Closed-form solution of k' = beta * exp(-l*t), k(0) = 0.
    Uses expm1 for numerical accuracy when l*t is small."""
    return (beta / l) * (-np.expm1(-l * t))


def fit_multistart(t, y, n_starts, rng):
    """Multi-start optimization: try several random initial guesses for
    (beta, l), refit from each one, and keep whichever converged to the
    lowest RMSE. Guesses are drawn log-uniformly (beta in [1e-4, 10],
    l in [1e-6, 1]) to cover several orders of magnitude."""
    best_popt, best_rmse = None, np.inf
    for _ in range(n_starts):
        beta0 = 10 ** rng.uniform(-4, 1)
        l0 = 10 ** rng.uniform(-6, 0)
        try:
            popt, _ = curve_fit(k_model, t, y, p0=[beta0, l0], bounds=BOUNDS, maxfev=20000)
        except RuntimeError:
            continue  # this particular start failed to converge; skip it
        rmse = np.sqrt(np.mean((k_model(t, *popt) - y) ** 2))
        if rmse < best_rmse:
            best_popt, best_rmse = popt, rmse
    return best_popt, best_rmse


def main():
    xl = pd.ExcelFile(DATA_PATH)
    sheets = xl.sheet_names[:N_CALIB_SHEETS]
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for sheet in sheets:
        df = xl.parse(sheet)
        t = df.timestamp.values
        y = df.cumulative_bites.values.astype(float)

        popt, rmse = fit_multistart(t, y, N_STARTS, rng)
        beta, l = popt
        rel_rmse = rmse / (y.max() if y.max() > 0 else 1)

        rows.append(dict(sheet=sheet, beta_i=beta, l_i=l, rmse=rmse, rel_rmse=rel_rmse))
        print(f"{sheet}: beta_i={beta:.5f}  l_i={l:.6f}  rmse={rmse:.2f} ({100*rel_rmse:.1f}%)")

    res_df = pd.DataFrame(rows)

    # --- population-level summary across the 17 calibration replicates ---
    n = len(res_df)
    tcrit = tdist.ppf(0.975, df=n - 1)
    print(f"\n=== Calibrated parameters (mean across {n} replicates, 95% CI) ===")
    for p in ["beta_i", "l_i"]:
        vals = res_df[p].values
        mean, sd = vals.mean(), vals.std(ddof=1)
        se = sd / np.sqrt(n)
        lo, hi = mean - tcrit * se, mean + tcrit * se
        med = res_df[p].median()
        q1, q3 = res_df[p].quantile([0.25, 0.75])
        print(f"{p:8s} = {mean:.5f}   SD={sd:.5f}   95% CI = [{max(lo,0):.5f}, {hi:.5f}]"
              f"   | median={med:.5f}  IQR=[{q1:.5f}, {q3:.5f}]")
    print(f"\nMean relative RMSE across replicates: {100*res_df.rel_rmse.mean():.2f}%")

    return res_df


if __name__ == "__main__":
    res_df = main()

    # =========================================================================
    # NEW SECTION: plot the calibrated model (using the MEAN beta_i, l_i across
    # the 17 replicates) against the raw data from entry_11, on the same axes.
    # Nothing is normalized or rescaled -- both are plotted in their original,
    # as-recorded units.
    # =========================================================================
    import matplotlib.pyplot as plt

    beta_mean = res_df["beta_i"].mean()
    l_mean = res_df["l_i"].mean()

    xl = pd.ExcelFile(DATA_PATH)
    df_11 = xl.parse("entry_11")
    t_11 = df_11.timestamp.values
    y_11 = df_11.cumulative_bites.values.astype(float)

    t_curve = np.linspace(0, t_11.max(), 500)
    k_curve = k_model(t_curve, beta_mean, l_mean)

    plt.figure(figsize=(7, 5))
    plt.plot(t_11, y_11, "o", ms=3, color="tab:gray", label="entry_11 data")
    plt.plot(t_curve, k_curve, "-", color="tab:red",
              label=f"k_i(t), mean params (beta_i={beta_mean:.5f}, l_i={l_mean:.6f})")
    plt.xlabel("t")
    plt.ylabel("cumulative bites")
    plt.title("Calibrated model (mean params) vs entry_11 data")
    plt.legend()
    plt.tight_layout()
    plt.show()