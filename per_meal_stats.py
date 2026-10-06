"""
Regenerates EVERY number in Table 1 of main.tex from the workbook in the repo
(https://github.com/newyorksnow/SMM), replicating the repo's own pipeline:
  - calibration scenario i/ii.py : first 17 sheets, curve_fit (TRF), bounds [1e-6, inf), maxfev=20000,
                                   25 log-uniform starts (beta 1e-4..1e1, l 1e-6..1), np.random.default_rng(0)
  - validation scenario i/ii.py  : sheets entry_17..entry_20, fixed (beta, l) = mean of the 17 fits
New (not in the repo): 95% t-CIs (as in the calibration scripts), per-meal R^2, paired t-test on per-meal R^2, and LaTeX rows.

Usage:  python per_meal_stats.py "path/to/modified bite_gt from FIC data (smm).xlsx"
Note: the repo scripts point to "Downloads/CORRECT/bite_gt_cumulative_timestamps CORRECT.xlsx"; make sure it is the same data.
Columns expected: timestamp, cumulative_bites (as in the repo); time is taken as given (seconds).
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")   # overflow warnings from discarded multi-start trials
from scipy.optimize import curve_fit
from scipy import stats

N_CALIB, N_STARTS, SEED = 17, 25, 0
BOUNDS = ([1e-6, 1e-6], [np.inf, np.inf])
HELD_OUT = ["entry_17", "entry_18", "entry_19", "entry_20"]
# Validation pairs exactly as hard-coded in the repo's "validation scenario i/ii.py" (rounded means of the calibration fits)
REPO_PARAMS = {"i": (0.08502, 0.00007), "ii": (0.01980, 0.00474)}

def k_i(t, b, l):  return (b/l)*(-np.expm1(-l*t))                  # scenario (i)
def k_ii(t, b, l): return np.exp((b/l)*(-np.expm1(-l*t))) - 1.0    # scenario (ii)
MODELS = {"i": k_i, "ii": k_ii}

def fit_multistart(f, t, y, rng):
    best, best_rmse = None, np.inf
    for _ in range(N_STARTS):
        p0 = [10**rng.uniform(-4, 1), 10**rng.uniform(-6, 0)]
        try:
            popt, _ = curve_fit(f, t, y, p0=p0, bounds=BOUNDS, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rmse = np.sqrt(np.mean((f(t, *popt)-y)**2))
        if rmse < best_rmse: best, best_rmse = popt, rmse
    return best, best_rmse

def ci95(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    h = stats.t.ppf(.975, n-1)*x.std(ddof=1)/np.sqrt(n)
    return m, m-h, m+h

def r2(y, p): return 1 - ((y-p)**2).sum()/((y-y.mean())**2).sum()

def main(path):
    xl = pd.ExcelFile(path)
    res = {}
    for scen, f in MODELS.items():
        rng = np.random.default_rng(SEED)                      # one fresh generator per scenario, as in the repo
        fits = []
        for sh in xl.sheet_names[:N_CALIB]:
            df = xl.parse(sh); popt, _ = fit_multistart(f, df.timestamp.values, df.cumulative_bites.values.astype(float), rng)
            fits.append(popt)
        fits = np.array(fits); res[scen] = fits
        print(f"\n=== scenario ({scen}): {len(fits)} calibration meals ===")
        for j, nm in enumerate(("beta", "l")):
            m, lo, hi = ci95(fits[:, j])
            print(f"  {nm:4s} mean {m:.5f}  95% CI [{lo:.5f}, {hi:.5f}]  SD {fits[:,j].std(ddof=1):.5f}  median {np.median(fits[:,j]):.5f}")
        print(f"  l at lower bound (<=1.0001e-6): {(fits[:,1] <= 1.0001e-6).sum()} of {len(fits)} meals")
        # one-sample t-tests vs 0 (only to compare with the numbers quoted in the earlier manuscript)
        for j, nm in enumerate(("beta", "l")):
            t, p = stats.ttest_1samp(fits[:, j], 0.0); print(f"  [old manuscript check] {nm}: t(16)={t:.2f}, p={p:.4f}")

    held = {sh: xl.parse(sh).dropna() for sh in HELD_OUT}
    out = {}
    for scen, f in MODELS.items():
        b, l = REPO_PARAMS[scen]                                # validation pair as hard-coded in the repo (= rounded across-meal means)
        ys = [held[s].cumulative_bites.values.astype(float) for s in HELD_OUT]
        ps = [f(held[s].timestamp.values, b, l) for s in HELD_OUT]
        r = np.concatenate([y-p for y, p in zip(ys, ps)]); Y = np.concatenate(ys)
        per = np.array([r2(y, p) for y, p in zip(ys, ps)])
        out[scen] = per
        print(f"\n--- validation ({scen}): beta={b:.5f} l={l:.6f} ---")
        print(f"  pooled MAE {np.abs(r).mean():.4f}  RMSE {np.sqrt((r**2).mean()):.4f}  R2 {1-(r**2).sum()/((Y-Y.mean())**2).sum():.4f}")
        print("  per-meal R2 (17..20): " + ", ".join(f"{v:.3f}" for v in per))
    d = out["i"] - out["ii"]; t, p = stats.ttest_rel(out["i"], out["ii"]); m, lo, hi = ci95(d)
    print(f"\nPaired per-meal dR2 (i-ii): {np.round(d,3)}  mean {m:.3f}  SD {d.std(ddof=1):.3f}  95% CI [{lo:.4f}, {hi:.3f}]  t(3)={t:.2f}  p={p:.3f}")
    print("\nLaTeX row for Table 1:\n  $R^2$ per meal (17--20) & " + ", ".join(f"{v:.2f}" for v in out['i']) + " & " + ", ".join(f"{v:.2f}" for v in out['ii']) + r" \\")

if __name__ == "__main__":
    main(sys.argv[1])
