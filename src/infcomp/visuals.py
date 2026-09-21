"""Dashboard figure in the HSBC 'US Inflation Dashboard' layout (brief §3.7).

  [ composite z-score history ................ ] [ gauge: Cooler <-> Hotter ]
  [ diffusion / breadth ...................... ] [ ranked contribution bars ]
  [ small-multiple grid, ranked by |contribution|, green / grey / red ....... ]

Swap PALETTE for LCAM brand colours; nothing else hard-codes colour.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

PALETTE = {"heating": "#C0392B", "neutral": "#9AA0A6", "cooling": "#2E8B57",
           "line": "#1F3A5F", "accent": "#D4A017", "grid": "#E6E6E6", "text": "#222222"}


def _style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.6)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=PALETTE["text"])


def gauge(ax, current: float, past: dict[int, float], lim: float = 3.0):
    ax.set_aspect("equal"); ax.axis("off")
    bands = np.linspace(-lim, lim, 13)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "g", [PALETTE["cooling"], PALETTE["neutral"], PALETTE["heating"]])
    for a, b in zip(bands[:-1], bands[1:]):
        th1 = 180 * (1 - (b + lim) / (2 * lim)); th2 = 180 * (1 - (a + lim) / (2 * lim))
        ax.add_patch(Wedge((0, 0), 1.0, th1, th2, width=0.28,
                           color=cmap((a + b) / 2 / (2 * lim) + 0.5), alpha=0.9))

    def needle(v, lw, color, label=None, r=0.95):
        v = float(np.clip(v, -lim, lim)); th = math.pi * (1 - (v + lim) / (2 * lim))
        ax.plot([0, r * math.cos(th)], [0, r * math.sin(th)], color=color, lw=lw,
                solid_capstyle="round", zorder=3)
        if label:
            ax.text(1.08 * math.cos(th), 1.08 * math.sin(th), label, ha="center",
                    va="center", fontsize=7, color=color)

    for m, v in sorted(past.items(), reverse=True):
        if not np.isnan(v):
            needle(v, 1.2, PALETTE["accent"], f"{m}m", r=0.8)
    needle(current, 3.0, PALETTE["text"])
    ax.add_patch(plt.Circle((0, 0), 0.05, color=PALETTE["text"], zorder=4))
    ax.text(-1.0, -0.14, "Cooler", fontsize=8, ha="center", color=PALETTE["cooling"])
    ax.text(1.0, -0.14, "Hotter", fontsize=8, ha="center", color=PALETTE["heating"])
    ax.text(0, -0.32, f"{current:+.2f}σ", ha="center", fontsize=14, fontweight="bold")
    ax.text(0, -0.48, "gold needles = reading N months ago", ha="center", fontsize=6.5,
            color="#666")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-0.55, 1.2)
    ax.set_title("Current reading", loc="left", fontsize=10, fontweight="bold")


def dashboard(comp: pd.Series, diff: pd.DataFrame, z: pd.DataFrame, table: pd.DataFrame,
              needles_m=(3, 6, 9, 12), history_years: int = 5, label: str = "UNVALIDATED",
              watermark: str | None = None, ltm: pd.DataFrame | None = None,
              title="US Inflation Composite"):
    as_of = table.attrs.get("as_of", comp.dropna().index[-1])
    comp = comp.loc[:as_of]
    n = len(table); ncol = 6; nrow = math.ceil(n / ncol)
    fig = plt.figure(figsize=(16, 6.5 + 1.5 * nrow))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[3, 2.2, 1.35 * nrow],
                  hspace=0.32, wspace=0.18, top=0.925, bottom=0.03)
    sm = gs[2, :].subgridspec(nrow, ncol, hspace=0.7, wspace=0.25)

    # --- composite history
    ax = fig.add_subplot(gs[0, :2]); _style(ax, "Composite (avg. expanding z-score; + = hotter)")
    c = comp.dropna()
    ax.plot(c.index, c.values, color=PALETTE["line"], lw=1.4)
    ax.fill_between(c.index, 0, c.values, where=c.values > 0, color=PALETTE["heating"], alpha=0.15)
    ax.fill_between(c.index, 0, c.values, where=c.values < 0, color=PALETTE["cooling"], alpha=0.15)
    ax.axhline(0, color="#555", lw=0.7)
    if ltm is not None:
        ax2 = ax.twinx(); lt = ltm["ltm_final"].loc[:as_of].dropna()
        ax2.plot(lt.index, lt.values, color=PALETTE["accent"], lw=0.9, ls="--", alpha=0.8)
        ax2.set_ylim(0, 5); ax2.set_ylabel("0–5 L/T/M (dashed)", fontsize=7)
        ax2.tick_params(labelsize=7); ax2.spines[["top"]].set_visible(False)

    # --- gauge
    past = {m: comp.shift(m).loc[as_of] if len(comp) > m else np.nan for m in needles_m}
    gauge(fig.add_subplot(gs[0, 2]), float(comp.loc[as_of]), past)

    # --- diffusion
    ax = fig.add_subplot(gs[1, :2]); _style(ax, "Diffusion: % of indicators heating (3m Δz > 0)")
    d = diff["heating_breadth"].loc[:as_of].dropna()
    ax.plot(d.index, d.values, color=PALETTE["line"], lw=1.0)
    ax.plot(d.index, d.rolling(6).mean(), color=PALETTE["accent"], lw=1.4, label="6m avg")
    ax.axhline(50, color="#555", lw=0.7, ls=":"); ax.set_ylim(0, 100)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # --- ranked contribution bars
    ax = fig.add_subplot(gs[1, 2]); _style(ax, "Contribution to composite")
    t = table.iloc[::-1]
    ax.barh(t["indicator"].str.slice(0, 28), t["contribution"],
            color=[PALETTE[s] for s in t["signal"]])
    ax.axvline(0, color="#555", lw=0.7); ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.6); ax.grid(axis="y", visible=False)

    # --- small multiples, ranked
    start = as_of - pd.DateOffset(years=history_years)
    for i, (key, r) in enumerate(table.iterrows()):
        a = fig.add_subplot(sm[i // ncol, i % ncol]); _style(a)
        s = z[key].loc[start:as_of].dropna()
        a.plot(s.index, s.values, color=PALETTE[r["signal"]], lw=1.2)
        a.axhline(0, color="#999", lw=0.5)
        a.set_title(f"{i+1}. {r['indicator'][:30]}", fontsize=7.5, loc="left",
                    color=PALETTE[r["signal"]], fontweight="bold")
        a.text(0.98, 0.9, f"z {r['z']:+.2f}", transform=a.transAxes, ha="right",
               fontsize=7, color=PALETTE["text"])
        a.tick_params(labelsize=6); a.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%y"))

    fig.suptitle(f"{title} — as of {pd.Timestamp(as_of):%b %Y}   [{label}]",
                 x=0.06, ha="left", fontsize=15, fontweight="bold", y=0.995 - 0.004 * nrow)
    fig.text(0.06, 0.972 - 0.004 * nrow, "Grid ranked by |contribution|. Colour: red = heating, grey = "
             "neutral, green = cooling (supportive for duration).", fontsize=8, color="#555")
    if watermark:
        fig.text(0.5, 0.5, watermark, fontsize=46, color="red", alpha=0.13, ha="center",
                 va="center", rotation=25, fontweight="bold")
    return fig
