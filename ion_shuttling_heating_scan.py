"""
Batch‑run `ion_shuttling_heating` for a range of shuttling times and
produce a plot of \bar{n} (mean phonon number) versus the total shuttling
time.  Results are cached to disk after every evaluation so an
interrupted run can be resumed simply by executing the script again.

Usage
-----
    python ion_shuttling_heating_scan.py

The script will

1.  Evaluate `ion_shuttling_heating` for every value in
    `TARGET_TIMES_US` (micro‑seconds).
2.  Append the result to `heating_scan_results.csv` **immediately after
    each successful run** to guarantee progress is saved.
3.  When all points are available it will generate `nbar_vs_T.pdf`
    containing a line plot of the first mode’s \bar{n} versus the
    requested shuttling time.

If you stop the script part‑way through simply rerun it; previously
completed points will be skipped automatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local import – assumes this file lives next to ion_shuttling_heating.py
from ion_shuttling_heating import ion_shuttling_heating

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Requested total shuttling times (in µs).
# 200 evenly‑spaced points between 50 µs and 200 µs (inclusive).
TARGET_TIMES_US: List[float] = np.linspace(50, 200, 200).tolist()

RESULT_CSV = Path("heating_scan_results_new.csv")
PLOT_PDF   = Path("nbar_vs_T.pdf")

# Internal – the original script multiplies the time argument by
# `0.63494002` before passing it to `ion_shuttling_heating`, and the
# function immediately divides by the same number.  To reproduce the
# original behaviour while allowing users to specify physical shuttling
# times directly we apply that factor here so that the *effective* time
# inside `ion_shuttling_heating` matches the requested duration.
_TIME_FACTOR = 0.63494002

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_existing_results() -> pd.DataFrame:
    """Return stored results (empty `DataFrame` if none exist)."""

    if RESULT_CSV.exists():
        df = pd.read_csv(RESULT_CSV)
    else:
        columns = ["T_us"] + [f"nbar_mode{i}" for i in range(1, 10)]
        df = pd.DataFrame(columns=columns)
    return df


def _save_results(df: pd.DataFrame) -> None:
    """Persist *df* to :pydata:`RESULT_CSV` (over‑write)."""

    df.to_csv(RESULT_CSV, index=False)


def _plot(df: pd.DataFrame) -> None:
    """Generate a PDF of \bar{n} versus T using the first mode."""

    df_sorted = df.sort_values("T_us")

    # Basic styling mirrors the one set in `ion_shuttling_heating.py`
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "axes.grid": True,
    })

    fig, ax = plt.subplots(figsize=(4.25, 3))  # half‑column width
    ax.plot(df_sorted["T_us"], df_sorted["nbar_mode1"], "o-", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"Shuttling time $T$ (µs)")
    ax.set_ylabel(r"$\bar{n}$ (mode 1)")
    ax.set_title(r"Heating versus shuttling time")
    fig.tight_layout()
    fig.savefig(PLOT_PDF, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------


def main() -> None:
    df = _load_existing_results()

    # Convert existing T column to a *set* of floats for fast lookup
    done_times = set(df["T_us"].values.astype(float))

    total_pts = len(TARGET_TIMES_US)
    for idx, T_us in enumerate(TARGET_TIMES_US, start=1):
        if T_us in done_times:
            print(f"[{idx}/{total_pts}] T = {T_us} µs already computed – skipping")
            continue

        print(f"\n[{idx}/{total_pts}] Computing heating for T = {T_us} µs …", flush=True)

        # The heavy simulation – apply factor so the *effective* time
        # inside the integrator equals the requested shuttling time.
        try:
            nbar_all_modes = ion_shuttling_heating(T_us * 1e-6 * _TIME_FACTOR)
        except Exception as exc:
            print(f"❌  Simulation failed for T = {T_us} µs → {exc}")
            continue

        # Ensure we have exactly nine modes (ion_shuttling_heating returns
        # a list; if its length changes we still record what we obtained).
        record = {"T_us": T_us}
        for i, val in enumerate(nbar_all_modes, start=1):
            record[f"nbar_mode{i}"] = val

        # Append, persist, and add to the lookup set so a mid‑run failure
        # does not repeat the calculation.
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        _save_results(df)
        done_times.add(T_us)
        print(f"   → completed (mode 1 \u0305n = {record['nbar_mode1']:.3f}) ✓")

    # All points finished – produce plot
    if not df.empty:
        _plot(df)
        print(f"\nSaved plot → {PLOT_PDF}")
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()
