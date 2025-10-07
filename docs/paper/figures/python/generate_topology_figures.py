"""Utility scripts for generating illustrative figures for the molecular topology paper.

The figures intentionally avoid synthetic performance claims and instead provide
conceptual visuals supporting the methodology section. Run this module as a
script to regenerate all assets in ``docs/paper/figures``.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Color palette (MIT branding inspired)
# -----------------------------------------------------------------------------
MIT_RED = "#A31F34"
MIT_GRAY = "#8A8B8C"
MIT_CHARCOAL = "#222222"
MIT_SKY = "#6BA5C3"
MIT_ACCENT = "#0F6F75"  # complimentary teal tone

FIGURE_DPI = 300
FIGURE_SIZE = (6.0, 4.0)


@dataclass(frozen=True)
class FigurePaths:
    """Convenience container for resolved figure output destinations."""

    base_dir: Path

    @property
    def loops(self) -> Path:
        return self.base_dir / "synthetic_conformational_loops.pdf"

    @property
    def persistence(self) -> Path:
        return self.base_dir / "synthetic_persistence_diagram.pdf"

    @property
    def pipeline(self) -> Path:
        return self.base_dir / "training_pipeline_overview.pdf"


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _get_paths() -> FigurePaths:
    """Return resolved output destinations relative to this script."""

    current = Path(__file__).resolve()
    figure_dir = current.parents[1]  # docs/paper/figures
    figure_dir.mkdir(parents=True, exist_ok=True)
    return FigurePaths(base_dir=figure_dir)


def _style_axes(ax: plt.Axes, *, spine_color: str = MIT_CHARCOAL) -> None:
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=spine_color, labelcolor=spine_color)
    ax.set_facecolor("white")


# -----------------------------------------------------------------------------
# Figure generators
# -----------------------------------------------------------------------------

def plot_conformational_loops(path: Path) -> None:
    """Render a conceptual conformational manifold with ring-flip trajectories."""

    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    radius = 1.0
    outer_loop = np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1)

    # Perturb a subset to suggest thermal fluctuations
    noise = 0.1 * np.random.RandomState(42).normal(size=outer_loop.shape)
    noisy_loop = outer_loop + noise

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.scatter(
        noisy_loop[:, 0],
        noisy_loop[:, 1],
        s=36,
        color=MIT_GRAY,
        alpha=0.45,
        label="MD conformations",
        edgecolor="none",
    )

    # Highlight a loop trajectory connecting conformations
    path_indices = np.linspace(0, len(theta) - 1, 12, dtype=int)
    ax.plot(
        outer_loop[path_indices, 0],
        outer_loop[path_indices, 1],
        color=MIT_RED,
        linewidth=2.5,
        label="Ring-flip trajectory",
    )
    ax.scatter(
        outer_loop[path_indices, 0],
        outer_loop[path_indices, 1],
        color=MIT_RED,
        s=50,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

    # Annotate start/end
    ax.annotate(
        "start",
        xy=(outer_loop[path_indices[0], 0], outer_loop[path_indices[0], 1]),
        xytext=(-1.35, 0.1),
        arrowprops=dict(arrowstyle="->", color=MIT_CHARCOAL, linewidth=1.2),
        fontsize=11,
        color=MIT_CHARCOAL,
    )
    ax.annotate(
        "end",
        xy=(outer_loop[path_indices[-1], 0], outer_loop[path_indices[-1], 1]),
        xytext=(1.2, -0.2),
        arrowprops=dict(arrowstyle="->", color=MIT_CHARCOAL, linewidth=1.2),
        fontsize=11,
        color=MIT_CHARCOAL,
    )

    circle = plt.Circle((0.0, 0.0), 1.15, color=MIT_ACCENT, fill=False, linestyle="--", linewidth=1.2)
    ax.add_patch(circle)

    ax.set_title("Cyclic peptide conformational loop", color=MIT_CHARCOAL, fontsize=14)
    ax.set_xlabel("Principal coordinate 1", color=MIT_CHARCOAL)
    ax.set_ylabel("Principal coordinate 2", color=MIT_CHARCOAL)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    _style_axes(ax)

    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_persistence_diagram(path: Path) -> None:
    """Draw a synthetic persistence diagram emphasizing dominant $H_1$ features."""

    rng = np.random.RandomState(7)
    diag_points = np.array(
        [
            [0.15, 0.95],
            [0.22, 0.88],
            [0.30, 0.80],
            [0.55, 1.10],
            [0.62, 1.18],
            [0.70, 1.35],
        ]
    )
    # Additional small persistence noise points near the diagonal
    noise_points = rng.uniform(0.05, 0.45, size=(15, 2))
    noise_points[:, 1] = noise_points[:, 0] + rng.uniform(0.01, 0.08, size=15)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.scatter(
        diag_points[:, 0],
        diag_points[:, 1],
        s=70,
        color=MIT_RED,
        edgecolor=MIT_CHARCOAL,
        linewidth=0.8,
        label="$H_1$ features",
        zorder=3,
    )
    ax.scatter(
        noise_points[:, 0],
        noise_points[:, 1],
        s=30,
        color=MIT_GRAY,
        alpha=0.5,
        label="short-lived features",
        edgecolor="none",
        zorder=2,
    )

    diag = np.linspace(0, 1.4, 200)
    ax.plot(diag, diag, linestyle="--", color=MIT_GRAY, linewidth=1.2, label="diagonal")

    # Annotate the most persistent feature
    dominant = diag_points[-1]
    ax.annotate(
        "dominant loop",
        xy=(dominant[0], dominant[1]),
        xytext=(0.85, 1.3),
        arrowprops=dict(arrowstyle="->", color=MIT_CHARCOAL, linewidth=1.1),
        fontsize=11,
        color=MIT_CHARCOAL,
    )

    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("Birth scale", color=MIT_CHARCOAL)
    ax.set_ylabel("Death scale", color=MIT_CHARCOAL)
    ax.set_title("Synthetic persistence diagram", color=MIT_CHARCOAL, fontsize=14)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    _style_axes(ax)

    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_training_pipeline(path: Path) -> None:
    """Illustrate the data \textrightarrow{} model \textrightarrow{} topology pipeline."""

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.axis("off")

    nodes = [
        ("MD Trajectories", (0.08, 0.55)),
        ("Preprocessing\n(SE(3) alignment)", (0.32, 0.55)),
        ("VAE Encoder", (0.52, 0.72)),
        ("Latent Space", (0.72, 0.72)),
        ("VAE Decoder", (0.52, 0.38)),
        ("Generated\nConformations", (0.72, 0.38)),
        ("Persistence\nDiagrams", (0.90, 0.55)),
    ]

    bbox_props = dict(
        boxstyle="round,pad=0.4",
        linewidth=1.2,
        edgecolor=MIT_CHARCOAL,
        facecolor="#F4F4F4",
    )

    for label, (x, y) in nodes:
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color=MIT_CHARCOAL, bbox=bbox_props)

    arrows: Iterable[Tuple[Tuple[float, float], Tuple[float, float], str]] = [
        ((0.14, 0.55), (0.26, 0.55), MIT_CHARCOAL),
        ((0.38, 0.62), (0.46, 0.68), MIT_ACCENT),
        ((0.58, 0.72), (0.66, 0.72), MIT_ACCENT),
        ((0.38, 0.48), (0.46, 0.42), MIT_RED),
        ((0.58, 0.40), (0.66, 0.40), MIT_RED),
        ((0.78, 0.55), (0.86, 0.55), MIT_CHARCOAL),
        ((0.74, 0.68), (0.88, 0.60), MIT_ACCENT),
        ((0.74, 0.42), (0.88, 0.50), MIT_RED),
    ]

    for (x0, y0), (x1, y1), color in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.8),
        )

    ax.set_title("Topology-aware VAE training pipeline", fontsize=15, color=MIT_CHARCOAL, pad=12)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    paths = _get_paths()
    plot_conformational_loops(paths.loops)
    plot_persistence_diagram(paths.persistence)
    plot_training_pipeline(paths.pipeline)
    print("Generated figures:")
    for p in (paths.loops, paths.persistence, paths.pipeline):
        print(f" - {p.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
