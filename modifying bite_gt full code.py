import pickle
import numpy as np
import pandas as pd

with open("FIC.pkl", "rb") as f:
    fic = pickle.load(f)

out_sheets = []

for meal_idx in range(len(fic["bite_gt"])):
    bite_gt = np.asarray(fic["bite_gt"][meal_idx], dtype=float)  # shape (N, 2)

    # Choose the meal end from the aligned signal stream you are using.
    # Use the recording's actual end timestamp, then build 0.1 s ticks below it.
    meal_end = float(fic["signals_proc"][meal_idx][-1, 0])

    # Exact 0.1 s grid, avoiding floating-point accumulation.
    t = np.arange(0, int(np.floor(meal_end * 10)) + 1) / 10.0

    bite_starts = np.sort(bite_gt[:, 0])
    cumulative = np.searchsorted(bite_starts, t, side="right")

    df = pd.DataFrame({
        "timestamp": t,
        "cumulative_bites": cumulative.astype(int),
    })

    out_sheets.append(df)