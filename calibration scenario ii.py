"""
ii) CALIBRATION AND VALIDATION ASSUMING NO OTHER AGENT (k_j = 0)


Calibration of a SOLO eating/biting model (interaction term kept, but
pointed at an absent companion).

Original coupled model:
    k_i'(t) = beta_i * exp(-l_i * t) * w(t)
    w(t)    = (|k_j - k_i| + 1) * a_ij

Since agent i is eating alone, there is no second agent -- but instead of
removing w(t), here we keep it and assume the absent companion's value is
k_j = 0. Since k_i(t) >= 0 always, |k_j - k_i| = |0 - k_i| = k_i, so:

    k_i'(t) = beta_i * exp(-l_i * t) * (k_i(t) + 1) * a_ij,   k_i(0) = 0

a_ij is NOT calibrated here -- it is fixed at a_ij = 1, exactly as in
scenario (i).

This ODE is nonlinear in k_i (the (k_i + 1) factor depends on the state
itself), but it is still separable and has a closed-form solution -- no
numerical ODE solver needed. Separating variables:

    dk / (k + 1) = beta * exp(-l*t) dt

Integrating both sides from 0 to t (using k(0) = 0):

    ln(k_i(t) + 1) = (beta_i / l_i) * (1 - exp(-l_i * t))

    k_i(t) = exp[ (beta_i / l_i) * (1 - exp(-l_i * t)) ] - 1

Only two parameters remain (beta_i, l_i) -- exactly as in scenario (i).

CHANGES IN THIS VERSION, RELATIVE TO SCENARIO (i):
  - Only the model equation, k_model(), differs (see above).
  - The except clause in fit_multistart() additionally catches ValueError.
    This model's closed form is exp() of something that can itself be
    large, so a handful of random multi-start guesses overflow before
    converging. scipy reports that specific failure as a ValueError
    ("Residuals are not finite"), not a RuntimeError, so it must also be
    caught here or the whole run stops on the first bad guess. This is
    the ONLY structural code change beyond the equation itself.
  - Everything else -- bounds, sheet selection, multi-start procedure,
    summary statistics, the final plot -- is identical to scenario (i),
    including the order in which it all runs.

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
    """Closed-form solution of k' = beta * exp(-l*t) * (k+1), k(0) = 0,
    i.e. the w(t) model with k_j fixed at 0 and a_ij fixed at 1.
    Uses expm1 for numerical accuracy when l*t is small."""
    return np.exp((beta / l) * (-np.expm1(-l * t))) - 1.0


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
        except (RuntimeError, ValueError):
            continue  # this particular start failed to converge (or overflowed); skip it
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