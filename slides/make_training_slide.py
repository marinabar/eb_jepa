"""Render the 'Training objective & procedure' slide as a PNG (16:9), in the
deck's visual vibe: grey section label, bold title, centred 2-line subtitle, a
centred loss chip, and three numbered cards whose bodies are clean bullet lines.
Pure matplotlib so it runs anywhere.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# palette tuned to the deck
BLUE = "#2E6BE6"
BLUE_SOFT = "#EAF1FE"
INK = "#12151B"
BODY = "#3A4049"
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
ax.text(0.055, 0.910, "Method", color=GREY, fontsize=13, style="italic")
ax.text(0.055, 0.848, "Training objective & procedure", color=INK,
        fontsize=33, fontweight="bold")
ax.text(0.5, 0.782, "One heuristics-free objective, two phases.",
        color=GREY, fontsize=15, ha="center")
ax.text(0.5, 0.742,
        "Align the views, keep the latent isotropic — then specialize on liver.",
        color=GREY, fontsize=15, ha="center")

# --- loss chip (hero), centred ---------------------------------------------
ax.text(0.5, 0.652,
        r"$\mathcal{L}\;=\;\mathrm{invariance(views)}\;+\;\lambda\cdot\mathrm{SIGReg(proj)}$",
        color=INK, fontsize=18, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.75", fc=BLUE_SOFT, ec=BLUE, lw=1.2))
ax.text(0.5, 0.585, r"$\lambda = 0.4$", color=BLUE, fontsize=13.5,
        ha="center", va="center", fontweight="bold")

# --- three cards ------------------------------------------------------------
cards = [
    ("1", "Objective — LeJEPA",
     ["Pull the V = 4 views of a cell together",
      "SIGReg → isotropic Gaussian latent",
      "Anti-collapse, by construction"],
     "no teacher · no EMA · no predictor"),
    ("2", "Procedure — two phases",
     ["Phase 1 — pretrain, all 100 cell lines",
      "Phase 2 — finetune on liver",
      "De-collapse intra-line structure"],
     "encoder then frozen → perturbator"),
    ("3", "Recipe",
     ["4 views · shared encoder · L = 512",
      "Muon + AdamW · warmup → cosine",
      "8 × B200 · global batch 256"],
     "label-free selection: loss tracks accuracy"),
]

x0, gap = 0.055, 0.024
cw = (1 - 2 * x0 - 2 * gap) / 3
ybot, ytop = 0.075, 0.515
ch = ytop - ybot

for i, (num, head, lines, foot) in enumerate(cards):
    x = x0 + i * (cw + gap)
    ax.add_patch(FancyBboxPatch(
        (x, ybot), cw, ch,
        boxstyle="round,pad=0.004,rounding_size=0.018",
        fc="white", ec=EDGE, lw=1.4, mutation_aspect=0.56))
    # numbered circle + header on one baseline
    ax.text(x + 0.030, ytop - 0.058, num, color="white", fontsize=14,
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.40", fc=BLUE, ec="none"))
    ax.text(x + 0.060, ytop - 0.058, head, color=INK, fontsize=15.5,
            ha="left", va="center", fontweight="bold")
    # body as evenly-spaced bullet lines
    by0, pitch = ytop - 0.135, 0.060
    for j, line in enumerate(lines):
        ly = by0 - j * pitch
        ax.text(x + 0.030, ly, "•", color=BLUE, fontsize=12.5, va="center")
        ax.text(x + 0.052, ly, line, color=BODY, fontsize=12.3, va="center")
    # divider + italic footer
    ax.plot([x + 0.030, x + cw - 0.030], [ybot + 0.080, ybot + 0.080],
            color=EDGE, lw=1)
    ax.text(x + 0.030, ybot + 0.040, foot, color=GREY_SOFT, fontsize=11,
            style="italic", ha="left", va="center")

fig.savefig("slides/training_objective.png", dpi=200, facecolor="white")
print("saved slides/training_objective.png")
