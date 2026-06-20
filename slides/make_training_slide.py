"""Render the 'Training objective & procedure' slide as a PNG (16:9), in the
deck's visual vibe. Card text auto-shrinks to stay inside the boxes; the loss
line sits in a neutral pill (no blue fill). Pure matplotlib.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE = "#2E6BE6"
INK = "#12151B"
BODY = "#3A4049"
GREY = "#7A828C"
GREY_SOFT = "#9AA1AA"
EDGE = "#E4E7EB"
PILL = "#F4F6F8"

plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

_fit = []  # (text_obj, max_width_in_axes_fraction) — shrunk after a draw pass


def fit(t, max_w):
    _fit.append((t, max_w))
    return t


# --- header -----------------------------------------------------------------
ax.text(0.055, 0.910, "Method", color=GREY, fontsize=13, style="italic")
ax.text(0.055, 0.848, "Training objective & procedure", color=INK,
        fontsize=33, fontweight="bold")
ax.text(0.5, 0.782, "One heuristics-free objective, two phases.",
        color=GREY, fontsize=15, ha="center")
ax.text(0.5, 0.742,
        "Align the views, keep the latent isotropic — then specialize on liver.",
        color=GREY, fontsize=15, ha="center")

# --- loss line in a neutral pill -------------------------------------------
ax.text(0.5, 0.652,
        r"$\mathcal{L}\;=\;\mathrm{invariance(views)}\;+\;\lambda\cdot\mathrm{SIGReg(proj)}$",
        color=INK, fontsize=18, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.8", fc=PILL, ec=EDGE, lw=1.1))
ax.text(0.5, 0.585, r"$\lambda = 0.4$", color=GREY, fontsize=13,
        ha="center", va="center")

# --- three cards ------------------------------------------------------------
cards = [
    ("1", "Objective — LeJEPA",
     ["Pull the V = 4 views together",
      "SIGReg → isotropic Gaussian",
      "Anti-collapse by construction"],
     "no teacher · no EMA · no predictor"),
    ("2", "Procedure — two phases",
     ["Phase 1 — pretrain, 100 lines",
      "Phase 2 — finetune on liver",
      "De-collapse intra-line structure"],
     "encoder then frozen → perturbator"),
    ("3", "Recipe",
     ["4 views · encoder · L = 512",
      "Muon + AdamW · cosine LR",
      "8 × B200 · global batch 256"],
     "label-free: loss tracks accuracy"),
]

x0, gap = 0.055, 0.024
cw = (1 - 2 * x0 - 2 * gap) / 3
ybot, ytop = 0.075, 0.515
ch = ytop - ybot
pad_l, pad_r = 0.028, 0.026  # inner padding

for i, (num, head, lines, foot) in enumerate(cards):
    x = x0 + i * (cw + gap)
    ax.add_patch(FancyBboxPatch(
        (x, ybot), cw, ch,
        boxstyle="round,pad=0.004,rounding_size=0.018",
        fc="white", ec=EDGE, lw=1.4, mutation_aspect=0.56))
    # numbered circle + header
    ax.text(x + pad_l + 0.002, ytop - 0.058, num, color="white", fontsize=14,
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.40", fc=BLUE, ec="none"))
    fit(ax.text(x + 0.062, ytop - 0.058, head, color=INK, fontsize=15.5,
                ha="left", va="center", fontweight="bold"), cw - 0.062 - 0.012)
    # body bullet lines
    by0, pitch, tx = ytop - 0.135, 0.060, x + pad_l + 0.020
    for j, line in enumerate(lines):
        ly = by0 - j * pitch
        ax.text(x + pad_l, ly, "•", color=BLUE, fontsize=12.5, va="center")
        fit(ax.text(tx, ly, line, color=BODY, fontsize=12.0, va="center"),
            cw - (pad_l + 0.020) - pad_r)
    # divider + footer
    ax.plot([x + pad_l, x + cw - pad_r], [ybot + 0.080, ybot + 0.080],
            color=EDGE, lw=1)
    fit(ax.text(x + pad_l, ybot + 0.040, foot, color=GREY_SOFT, fontsize=11,
                style="italic", ha="left", va="center"), cw - pad_l - pad_r)

# --- shrink any text that exceeds its box, then save ------------------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
fig_w = fig.bbox.width
for t, max_w in _fit:
    fs = t.get_fontsize()
    while t.get_window_extent(rend).width / fig_w > max_w and fs > 8:
        fs -= 0.4
        t.set_fontsize(fs)

fig.savefig("slides/training_objective.png", dpi=200, facecolor="white")
print("saved slides/training_objective.png")
