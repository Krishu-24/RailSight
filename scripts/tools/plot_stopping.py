"""Report figure: how far ahead must a hazard be seen, and how big is it on the sensor then?

    python scripts/tools/plot_stopping.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from railsight.corridor import focal_from_hfov
from railsight.risk import stopping_distance

OUT = Path("docs/assets")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kmh = np.linspace(10, 160, 151)
    v = kmh / 3.6
    f = focal_from_hfov(1920, 60.0)  # full-HD camera, 60 degree horizontal FOV

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for a, label in [(0.5, "freight (0.5 m/s²)"), (1.0, "passenger (1.0 m/s²)"),
                     (1.4, "emergency, EMU (1.4 m/s²)")]:
        d = stopping_distance(v, 2.0, a)
        ax1.plot(kmh, d, label=label)
        ax2.plot(kmh, f * 1.7 / d, label=label)  # height in px of a 1.7 m person at that range
    ax1.set(xlabel="speed (km/h)", ylabel="stopping distance (m)",
            title="Distance needed to stop (2 s reaction)")
    ax2.set(xlabel="speed (km/h)", ylabel="person height at that distance (px)",
            title="Size of a person when braking must start", yscale="log")
    ax2.axhline(16, color="grey", ls="--", lw=1)
    ax2.text(12, 17.5, "≈ smallest reliably detected object", fontsize=8, color="grey")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stopping_distance.png", dpi=150)
    for s in (40, 60, 80, 100, 120):
        d = stopping_distance(s / 3.6, 2.0, 1.0)
        print(f"{s:>4} km/h: stop {d:6.0f} m, person {f * 1.7 / d:5.1f} px tall")


if __name__ == "__main__":
    main()
