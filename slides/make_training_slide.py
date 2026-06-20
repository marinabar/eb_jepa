"""Render the 'Training objective & procedure' slide as a PNG (16:9), in the
deck's visual vibe: grey section label, bold title, centred subtitle, a loss
'chip', and three numbered cards. Pure matplotlib so it runs anywhere.
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# palette tuned to the deck
BLUE = "#2E6BE6"
BLUE_SOFT = "#EAF1FE"
INK = "#12151B"
GREY = "#7A828C"
GREY_SOFT = "#9AA1AA"
EDGE = "#E4E7EB"

plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# --- header -----------------------------------------------------------------
ax.text(0.055, 0.905, "Method", color=GREY, fontsize=13, style="italic")
ax.text(0.055, 0.845, "Training objective & procedure", color=INK,
        fontsize=33, fontweight="bold")
ax.text(0.5, 0.775,
        "One heuristics-free objective, two phases — align the views, keep the "
        "latent isotropic, then specialize on liver.",
        color=GREY, fontsize=14.5, ha="center")

# --- loss chip (hero) -------------------------------------------------------
ax.text(0.405, 0.665,
        r"$\mathcal{L}\;=\;\mathrm{invariance(views)}\;+\;\lambda\cdot\mathrm{SIGReg(proj)}$",
        color=INK, fontsize=19, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.7", fc=BLUE_SOFT, ec=BLUE, lw=1.2))
ax.text(0.70, 0.665, r"$\lambda = 0.4$", color="white", fontsize=16.5,
        ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", fc=BLUE, ec="none"))

# --- three cards ------------------------------------------------------------
cards = [
    ("1", "Objective — LeJEPA",
     "Pull the V=4 views of a cell together in latent space; SIGReg pushes the "
     "embedding distribution toward an isotropic Gaussian — anti-collapse.",
     "no teacher · no EMA · no predictor"),
    ("2", "Procedure — two phases",
     "Phase 1 — pretrain on all 100 cell lines for a general representation. "
     "Phase 2 — finetune on liver to de-collapse intra-line structure.",
     "encoder then frozen → train the perturbator"),
    ("3", "Recipe",
     "4 views · shared encoder · L = 512. Muon + AdamW, warmup→cosine LR, "
     "grad-checkpoint. 8×B200, global batch 256.",
     "label-free model selection: the loss tracks downstream accuracy"),
]

x0, gap = 0.055, 0.022
cw = (1 - 2 * x0 - 2 * gap) / 3
ybot, ytop = 0.075, 0.55
ch = ytop - ybot

for i, (num, head, body, foot) in enumerate(cards):
    x = x0 + i * (cw + gap)
    ax.add_patch(FancyBboxPatch(
        (x, ybot), cw, ch,
        boxstyle="round,pad=0.004,rounding_size=0.018",
        fc="white", ec=EDGE, lw=1.4, mutation_aspect=0.56))
    # numbered circle
    ax.text(x + 0.028, ytop - 0.062, num, color="white", fontsize=14.5,
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.42", fc=BLUE, ec="none"))
    # header
    ax.text(x + 0.058, ytop - 0.062, head, color=INK, fontsize=15.5,
            ha="left", va="center", fontweight="bold")
    # body (wrapped)
    wrapped = textwrap.fill(body, width=34)
    ax.text(x + 0.022, ytop - 0.135, wrapped, color="#3A4049", fontsize=12.3,
            ha="left", va="top", linespacing=1.5)
    # divider + footer
    ax.plot([x + 0.022, x + cw - 0.022], [ybot + 0.085, ybot + 0.085],
            color=EDGE, lw=1)
    ax.text(x + 0.022, ybot + 0.03, textwrap.fill(foot, width=40),
            color=GREY_SOFT, fontsize=11, style="italic", ha="left", va="bottom")

# left-edge micro caption, like the other slides
ax.text(0.018, 0.5, "LeJEPA · Balestriero & LeCun 2026", color=GREY_SOFT,
        fontsize=9.5, rotation=90, ha="center", va="center")

fig.savefig("slides/training_objective.png", dpi=200,
            facecolor="white", bbox_inches=None)
print("saved slides/training_objective.png")
