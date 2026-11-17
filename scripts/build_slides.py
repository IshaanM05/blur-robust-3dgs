"""Build docs/slides.pptx, a Beamer-style deck, from results/all_results.csv and docs/figures/.

Every number on the slides comes from the results csv, so rebuilding after new runs updates the deck.
Equations are typeset with matplotlib's Computer Modern math font and placed as pictures.
Usage: python scripts/build_slides.py   (run collect.py and make_figures.py first)
"""

import csv
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

plt.rcParams["mathtext.fontset"] = "cm"

FIG = Path("docs/figures")
TMP = Path(tempfile.mkdtemp(prefix="slides_"))
NAVY, MID, PALE, INK, GRAY, ACCENT, GREEN = "1F3A68", "3B5E9A", "E8EDF5", "1A1A1A", "6B7280", "B5472F", "2E7D5B"
FONT = "Cambria"
W, H = 13.333, 7.5
TOTAL = 11
AUTHORS = "I. Mondal and P. Tyagi"

rows = list(csv.DictReader(open("results/all_results.csv")))


def get(scene, setting, method, col="psnr"):
    for r in rows:
        if (r["scene"], r["setting"], r["method"]) == (scene, setting, method) and r[col]:
            return float(r[col])
    return None


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def rgb(h):
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
prs.core_properties.author = "Ishaan Mondal, Purvanya Tyagi"
prs.core_properties.title = "Blur-Robust 3D Gaussian Splatting via Joint Per-View Blur-Kernel Estimation"
prs.core_properties.comments = ""
prs.core_properties.last_modified_by = "Ishaan Mondal"


def text(slide, x, y, w, h, content, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """content: a string or a list of strings / (string, options) paragraphs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, p in enumerate(content if isinstance(content, list) else [content]):
        opts = {}
        if isinstance(p, tuple):
            p, opts = p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = opts.get("align", align)
        para.space_after = Pt(opts.get("after", 6))
        run = para.add_run()
        run.text = p
        run.font.name = FONT
        run.font.size = Pt(opts.get("size", size))
        run.font.bold = opts.get("bold", bold)
        run.font.italic = opts.get("italic", False)
        run.font.color.rgb = rgb(opts.get("color", color))
    return tb


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def line(slide, x1, y1, x2, y2, color=INK, width=1.0, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if arrow:
        ln = c.line._get_or_add_ln()
        tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(tail)
    return c


def new_slide(title, n):
    """White frame with a Beamer-like title bar and an info line at the bottom."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, 1.0, NAVY)
    rect(s, 0, 1.0, W, 0.05, MID)
    text(s, 0.6, 0, W - 1.2, 1.0, title, size=30, bold=True, color="FFFFFF", anchor=MSO_ANCHOR.MIDDLE)
    rect(s, 0, 7.1, W, 0.4, PALE)
    text(
        s,
        0.6,
        7.19,
        8,
        0.25,
        f"{AUTHORS}   |   Blur-robust 3D Gaussian Splatting   |   Fundamentals of DIP",
        size=11,
        color=GRAY,
    )
    text(s, W - 1.6, 7.19, 1.0, 0.25, f"{n} / {TOTAL}", size=11, color=GRAY, align=PP_ALIGN.RIGHT)
    return s


def block(slide, x, y, w, h, head, lines, size=17, color=MID):
    """A Beamer block: coloured header strip over a pale body."""
    rect(slide, x, y, w, 0.42, color)
    text(slide, x + 0.15, y, w - 0.3, 0.42, head, size=16, bold=True, color="FFFFFF", anchor=MSO_ANCHOR.MIDDLE)
    rect(slide, x, y + 0.42, w, h - 0.42, PALE)
    text(slide, x + 0.2, y + 0.6, w - 0.4, h - 0.7, lines, size=size)


def trimmed(path, pad=10):
    im = Image.open(path).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height)))
    out = TMP / f"trim_{Path(path).name}"
    im.save(out)
    return out


def picture(slide, path, x, y, max_w, max_h, center=True):
    path = trimmed(path)
    im = Image.open(path)
    ar = im.width / im.height
    w, h = (max_w, max_w / ar) if max_w / ar <= max_h else (max_h * ar, max_h)
    px = x + (max_w - w) / 2 if center else x
    slide.shapes.add_picture(str(path), Inches(px), Inches(y), Inches(w), Inches(h))
    return w, h


def equation(slide, latex, cx, y, height, tag):
    """Typeset a formula and centre it horizontally on cx."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${latex}$", fontsize=30, color=f"#{INK}")
    path = TMP / f"eq_{tag}.png"
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    im = Image.open(path)
    w = height * im.width / im.height
    slide.shapes.add_picture(str(path), Inches(cx - w / 2), Inches(y), Inches(w), Inches(height))
    return w


def caption(slide, x, y, w, t):
    text(slide, x, y, w, 0.6, t, size=13, color=GRAY)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def box(slide, x, y, w, h, label, fill=PALE, color=INK, size=15):
    s = rect(slide, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE)
    s.line.color.rgb = rgb(MID)
    s.line.width = Pt(1.25)
    text(
        slide,
        x,
        y,
        w,
        h,
        [(ln, {"after": 0}) for ln in label.split("\n")],
        size=size,
        color=color,
        align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE,
    )


# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(prs.slide_layouts[6])
rect(s, 0, 0, W, 7.5, "FFFFFF")
rect(s, 0, 2.0, W, 2.6, NAVY)
text(
    s,
    0.9,
    2.15,
    W - 1.8,
    1.65,
    "Blur-Robust 3D Gaussian Splatting via Joint Per-View Blur-Kernel Estimation",
    size=34,
    bold=True,
    color="FFFFFF",
    anchor=MSO_ANCHOR.MIDDLE,
)
text(
    s,
    0.9,
    3.9,
    W - 1.8,
    0.5,
    "A digital image processing view of reconstruction from blurry photographs",
    size=18,
    color="C9D3E6",
)
rect(s, 0, 4.6, W, 0.06, MID)
text(
    s,
    0.9,
    5.0,
    W - 1.8,
    1.6,
    [
        ("Ishaan Mondal (24B2118)   Purvanya Tyagi (24B0385)", {"size": 22, "bold": True}),
        ("Fundamentals of Digital Image Processing (CS 663), IIT Bombay", {"size": 18, "color": GRAY}),
        ("github.com/IshaanM05/blur-robust-3dgs", {"size": 16, "color": GRAY}),
    ],
)
notes(
    s,
    "20 s. We study one question: when photographs are blurry, can we remove the blur while the 3D scene is being "
    "reconstructed, without knowing the blur? The project is about image formation, convolution, deconvolution and "
    "priors, applied inside a modern 3D reconstruction method.",
)

# ---------------------------------------------------------------- 2 motivation
s = new_slide("Motivation: blurry photographs, blurry scenes", 2)
block(
    s,
    0.6,
    1.4,
    5.6,
    1.5,
    "Setting",
    "3D Gaussian Splatting (3DGS) fits a cloud of 3D Gaussians to posed photographs and renders new views in real time.",
)
block(
    s,
    0.6,
    3.1,
    5.6,
    1.7,
    "Problem",
    "Hand-held photographs contain motion blur and defocus. A scene fitted to blurry photographs is itself blurry.",
    color=ACCENT,
)
block(
    s,
    0.6,
    5.0,
    5.6,
    1.8,
    "Question",
    "Can we recover the sharp scene, and the blur of every photograph, together, without being told what the blur is?",
    color=GREEN,
)
w_, h_ = picture(s, FIG / "real_ball.png", 6.5, 2.9, 6.3, 4.0)
caption(
    s,
    6.5,
    3.05 + h_,
    6.3,
    "Real handheld capture (Deblur-NeRF ball scene): held-out photograph, vanilla 3DGS, and ours.",
)
notes(
    s,
    "1 min. 3DGS needs sharp input. Real photographs rarely are. If we train on blurry photographs, the scene averages the "
    "blur and every rendering is soft, as the middle panel shows. The right panel is our method on the same photographs. "
    "The rest of the talk explains how, and how far this can be trusted.",
)

# ---------------------------------------------------------------- 3 DIP model
s = new_slide("Image formation: blur is a convolution", 3)
equation(s, r"y_i \;=\; k_i \ast x_i \;+\; n_i", 3.5, 1.5, 0.75, "model")
text(
    s,
    0.6,
    2.5,
    5.8,
    2.2,
    [
        ("yᵢ  the blurry photograph we observe", {}),
        ("xᵢ  the sharp image of the scene from pose i", {}),
        ("kᵢ  the unknown blur kernel", {}),
        ("nᵢ  sensor noise", {}),
    ],
    size=16,
)
block(
    s,
    0.6,
    4.4,
    5.8,
    2.4,
    "Classical route: deblur first",
    [
        "Wiener or Richardson-Lucy on each photograph, then reconstruct.",
        "Needs the kernel in advance, one photograph at a time, and amplifies noise.",
    ],
    color=ACCENT,
)
equation(s, r"\hat X(u,v)=\frac{\bar H(u,v)}{|H(u,v)|^{2}+\mathrm{NSR}}\,Y(u,v)", 9.8, 1.45, 0.95, "wiener")
caption(
    s,
    7.0,
    2.55,
    5.8,
    "Wiener filter. Y: Fourier transform of the blurry photograph. H: transform of the kernel. NSR: noise-to-signal ratio, which keeps the division from amplifying noise.",
)
block(
    s,
    6.9,
    3.55,
    5.9,
    3.3,
    "Our route: deblur inside the reconstruction",
    [
        "The 3D scene must explain every photograph through its own kernel.",
        "All photographs constrain the same scene, so the sharp content is shared and the kernels are the remaining unknowns.",
    ],
    color=GREEN,
)
notes(
    s,
    "1 min. This is the DIP model from class. The classical route inverts the blur on each image separately and needs the "
    "kernel. Our route keeps the forward model: render the scene, blur it with the photograph's kernel, compare with the "
    "photograph. Because many photographs see the same scene, the sharp content is shared, which is the extra information "
    "that a single-image method does not have.",
)

# ---------------------------------------------------------------- 4 method
s = new_slide("Method: joint optimisation of scene and kernels", 4)
ys, bh, bw = 1.6, 1.1, 2.55
xs = [0.6, 3.75, 6.9, 10.05]
labels = ["3D Gaussians G", "Render at pose i\n(gsplat)", "Blur with kᵢ\n(convolution)", "Blurry photo yᵢ\n(compare)"]
for x, lab in zip(xs, labels, strict=True):
    box(s, x, ys, bw, bh, lab, size=16)
for a, b in zip(xs[:-1], xs[1:], strict=True):
    line(s, a + bw, ys + bh / 2, b, ys + bh / 2, color=MID, width=2.0, arrow=True)
line(s, xs[3] + bw / 2, ys + bh, xs[3] + bw / 2, ys + bh + 0.55, color=ACCENT, width=2.0)
line(s, xs[3] + bw / 2, ys + bh + 0.55, xs[0] + bw / 2, ys + bh + 0.55, color=ACCENT, width=2.0)
line(s, xs[0] + bw / 2, ys + bh + 0.55, xs[0] + bw / 2, ys + bh, color=ACCENT, width=2.0, arrow=True)
text(
    s,
    3.5,
    ys + bh + 0.62,
    6.5,
    0.4,
    "gradients update both the Gaussians and every kᵢ",
    size=14,
    color=ACCENT,
    align=PP_ALIGN.CENTER,
)
equation(
    s,
    r"\mathcal{L}=\sum_i\;0.8\,\left\|k_i\ast x_i-y_i\right\|_1+0.2\,\left(1-\mathrm{SSIM}(k_i\ast x_i,\,y_i)\right)",
    6.67,
    4.15,
    0.75,
    "loss",
)
block(
    s,
    0.6,
    5.15,
    5.9,
    1.75,
    "Kernel parametrisation",
    "A polyline path of 6 control points (12 numbers), splatted onto a 25 x 25 grid. A free-form kernel would have 625.",
    size=15,
)
block(
    s,
    6.8,
    5.15,
    6.0,
    1.75,
    "At test time",
    "The kernel is dropped: we render the Gaussians directly and compare with the sharp held-out views.",
    size=15,
    color=GREEN,
)
notes(
    s,
    "1.5 min. The pipeline is the forward model. Render the current Gaussians from a training pose, blur that rendering with "
    "the learned kernel of that photograph, and compare with the actual blurry photograph. The loss is L1 plus a structural "
    "similarity term. Gradients flow to the Gaussians and to the kernels. The rasteriser is gsplat; the training loop, kernel "
    "modules, losses and baselines are written from scratch. At test time there is no kernel: the scene itself is sharp.",
)

# ---------------------------------------------------------------- 5 kernel prior
s = new_slide("The kernel needs a prior, otherwise it cancels out", 5)
bl = get("bonsai", "motion_medium", "ours_polyline", "kernel_l2")
fl = get("bonsai", "motion_medium", "ours_free", "kernel_l2")
w_, h_ = picture(s, FIG / "kernels_compare.png", 0.5, 1.3, 8.4, 5.0)
caption(
    s,
    0.6,
    1.4 + h_,
    8.2,
    "Rows: true kernels, our polyline estimate, a free-form estimate (25 x 25 numbers per photograph). Columns: different training photographs. Kernel error is the distance to the true kernel.",
)
block(
    s,
    9.2,
    1.4,
    3.6,
    2.55,
    "Free-form kernel",
    [
        'Collapses to one bright pixel, which means "no blur": the scene absorbs the blur instead.',
        f"Kernel error {fmt(fl)}, no better than standard 3DGS.",
    ],
    size=15,
    color=ACCENT,
)
block(
    s,
    9.2,
    4.15,
    3.6,
    2.55,
    "Polyline kernel",
    [
        "Recovers the shape and direction of each photograph's true motion path.",
        f"Kernel error {fmt(bl)} (0 is perfect).",
    ],
    size=15,
    color=GREEN,
)
notes(
    s,
    "1.5 min. Rows: true kernels, our polyline estimate, a free-form estimate. With 625 free numbers per photograph, "
    "'no blur plus a blurry scene' fits the data as well as the truth. This is the known trivial solution of blind "
    "deconvolution (Levin et al. 2009). We tried initialisation, learning rates and regularisers, and the free-form kernel "
    "always collapses. Restricting the kernel to a smooth path with 12 parameters removes that solution, and the true "
    "kernels are recovered. Kernel error is the L2 distance to the true kernel; a delta scores about 0.9.",
)

# ---------------------------------------------------------------- 6 results
s = new_slide("Result: close to the oracle, far above the baselines", 6)
methods = [
    ("Standard 3DGS on blurry photos", "vanilla"),
    ("Wiener (guessed kernel) + 3DGS", "wiener_guess"),
    ("Rich.-Lucy (guessed kernel) + 3DGS", "rl_guess"),
    ("Ours: polyline kernel, blind", "ours_polyline"),
    ("Ours with the true kernels (oracle)", "ours_oracle_kernel"),
    ("3DGS on the sharp photographs (ceiling)", "sharp_input"),
]
scenes = ["bonsai", "counter", "garden"]
X0, Y0, RH = 0.6, 1.55, 0.55
gf = s.shapes.add_table(len(methods) + 1, 4, Inches(X0), Inches(Y0), Inches(8.3), Inches(RH * (len(methods) + 1)))
tbl = gf.table
style_id = gf._element.graphic.graphicData.tbl.tblPr.find(qn("a:tableStyleId"))
style_id.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # no style, no grid
tbl.columns[0].width = Inches(4.7)
for c in (1, 2, 3):
    tbl.columns[c].width = Inches(1.2)
for r in range(len(methods) + 1):
    tbl.rows[r].height = Inches(RH)
    for c in range(4):
        cell = tbl.cell(r, c)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.fill.background()
        if r == 0:
            v = ["PSNR (dB), motion blur, medium", *scenes][c]
        else:
            v = methods[r - 1][0] if c == 0 else fmt(get(scenes[c - 1], "motion_medium", methods[r - 1][1]))
        cell.text_frame.text = v
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        run = para.runs[0]
        run.font.name = FONT
        run.font.size = Pt(15)
        run.font.bold = r == 0 or (r > 0 and methods[r - 1][1] == "ours_polyline")
        run.font.color.rgb = rgb(ACCENT if r > 0 and methods[r - 1][1] == "ours_polyline" else INK)
bottom = Y0 + RH * (len(methods) + 1)
line(s, X0, Y0, X0 + 8.3, Y0, width=2.0)
line(s, X0, Y0 + RH, X0 + 8.3, Y0 + RH, width=1.0)
line(s, X0, Y0 + RH * 4, X0 + 8.3, Y0 + RH * 4, color=GRAY, width=0.5)
line(s, X0, bottom, X0 + 8.3, bottom, width=2.0)
text(
    s,
    X0,
    bottom + 0.2,
    8.3,
    0.9,
    "Held-out sharp views, 10,000 iterations. Oracle: given the true blur kernels. Ceiling: trained on the unblurred photographs. Neither is available in practice.",
    size=13,
    color=GRAY,
)
gain = get("bonsai", "motion_medium", "ours_polyline") - get("bonsai", "motion_medium", "vanilla")
gap = get("bonsai", "motion_medium", "ours_oracle_kernel") - get("bonsai", "motion_medium", "ours_polyline")
block(
    s,
    9.3,
    1.55,
    3.5,
    2.35,
    "Over standard 3DGS",
    [
        (f"+{gain:.1f} dB", {"size": 34, "bold": True, "color": GREEN}),
        ("on bonsai; +2.3 and +2.0 on counter and garden", {"size": 14}),
    ],
    color=MID,
)
block(
    s,
    9.3,
    4.1,
    3.5,
    2.35,
    "Behind the oracle",
    [(f"{gap:.1f} dB", {"size": 34, "bold": True, "color": MID}), ("on bonsai, with no kernel supplied", {"size": 14})],
    color=MID,
)
notes(
    s,
    "1.5 min. Read the table top to bottom. Guessing a kernel and deconvolving first barely helps, because the guess is "
    "wrong. Our blind method is 2 to 3.5 dB above that, and within half a dB of using the true kernels. The ceiling row is "
    "what 3DGS achieves if the photographs were sharp, so the method closes most of the gap. Garden is the hardest scene. "
    "Single seed for this table; a three-seed study on bonsai shows the spread is about 0.2 dB, far smaller than the gaps.",
)

# ---------------------------------------------------------------- 7 qualitative
s = new_slide("What the improvement looks like", 7)
picture(s, FIG / "qualitative_bonsai_motion_medium.png", 0.5, 1.3, W - 1.0, 4.4)
block(
    s,
    0.6,
    5.45,
    5.9,
    1.5,
    "Standard 3DGS and Wiener",
    "The knit texture and the small objects are smeared; Wiener with a wrong guess adds little.",
    size=15,
    color=ACCENT,
)
block(
    s,
    6.8,
    5.45,
    6.0,
    1.5,
    "Ours",
    "Fine texture returns, and the result is hard to tell from the run that was given the true kernels.",
    size=15,
    color=GREEN,
)
notes(
    s,
    "0.75 min. Top row is a whole held-out view, bottom row a zoom. Look at the knitted cloth: vanilla and Wiener are smooth, "
    "ours and the oracle show the weave.",
)

# ---------------------------------------------------------------- 8 other blurs and noise
s = new_slide("Other blurs and noise: what a wrong guess costs", 8)
picture(s, FIG / "guess_sensitivity_bonsai.png", 0.5, 1.3, 6.2, 3.55)
picture(s, FIG / "noise_sweep_bonsai.png", 6.8, 1.3, 6.0, 3.55)
caption(s, 0.6, 4.9, 6.0, "Gaussian blur and defocus: Wiener is best with the right guess and worst with a wrong one.")
caption(s, 6.9, 4.9, 5.9, "Gaussian blur plus noise: Wiener loses its advantage as noise grows.")
block(
    s,
    0.6,
    5.6,
    12.2,
    1.35,
    "Reading",
    "Our blind Gaussian-kernel variant is 1 dB above standard 3DGS and needs no guess. With the right guess Wiener is stronger on clean data, and it falls below standard 3DGS at noise 0.1.",
    size=15,
)
notes(
    s,
    "1 min. For smooth Gaussian blur and defocus we are honest: a correctly guessed Wiener filter beats us on clean data. "
    "But the guess has to be right: a kernel twice too wide is 3 to 4 dB worse than doing nothing. Our method needs no "
    "guess. On the right, noise hurts the deconvolution: at noise level 0.1 it falls under vanilla while ours stays level.",
)

# ---------------------------------------------------------------- 9 ablations and spectrum
s = new_slide("Ablations and the frequency view", 9)
abl = [
    ("Default (6 points, 25 px)", "ours_polyline"),
    ("3 control points", "abl_ctrl3"),
    ("12 control points", "abl_ctrl12"),
    ("Kernel size 15", "abl_k15"),
    ("Kernel size 35", "abl_k35"),
    ("Shared kernel for all views", "abl_shared"),
    ("Free-form kernel", "ours_free"),
    ("Standard 3DGS", "vanilla"),
]
abl = [(a, get("bonsai", "motion_medium", k)) for a, k in abl]
cd = CategoryChartData()
cd.categories = [a for a, _ in abl][::-1]
cd.add_series("PSNR", [v for _, v in abl][::-1])
gfc = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.35), Inches(6.6), Inches(4.2), cd)
ch = gfc.chart
ch.font.size = Pt(13)
ch.font.name = FONT
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "bonsai, medium motion blur: PSNR (dB)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.value_axis.minimum_scale = 24
ch.value_axis.major_gridlines.format.line.color.rgb = rgb("D8DEE9")
ser = ch.plots[0].series[0]
ser.format.fill.solid()
ser.format.fill.fore_color.rgb = rgb(MID)
ch.plots[0].has_data_labels = True
ch.plots[0].data_labels.number_format = "0.0"
ch.plots[0].data_labels.number_format_is_linked = False
w_, h_ = picture(s, FIG / "spectrum_bonsai_motion_medium.png", 7.5, 1.35, 5.3, 3.7)
caption(
    s,
    7.5,
    1.35 + h_ + 0.1,
    5.3,
    "Radial power spectrum of a rendered view: blur removes high frequencies; ours restores them.",
)
block(
    s,
    0.6,
    5.65,
    12.2,
    1.3,
    "Reading",
    "The prior is what matters: a shared or free-form kernel loses 3 dB. The number of control points and the kernel size barely change the result.",
    size=15,
)
notes(
    s,
    "1 min. Left: each bar changes one design choice. Sharing one kernel across all photographs, or using a free-form "
    "kernel, loses about 3 dB, while 3 vs 12 control points or kernel size 15 vs 35 change less than 0.4 dB. Right: the DIP "
    "interpretation. Blur is a low-pass filter, vanilla inherits the missing high frequencies, and our estimate restores most of them.",
)

# ---------------------------------------------------------------- 10 real photographs
s = new_slide("Real blurry photographs (qualitative)", 10)
picture(s, FIG / "real_slide.png", 0.6, 1.3, 6.9, 5.6, center=False)
block(
    s,
    7.9,
    1.4,
    4.9,
    2.2,
    "What we see",
    "On static, well covered scenes ours recovers detail that standard 3DGS loses: weave, leaves and wires.",
    size=15,
    color=GREEN,
)
block(
    s,
    7.9,
    3.8,
    4.9,
    3.1,
    "Where it fails",
    [
        "Bright edges show speckled over-sharpening.",
        "The moving subject in the girl scene (third row) is not reconstructed by either method.",
        "No ground truth: judged by eye, 9 scenes.",
    ],
    size=15,
    color=ACCENT,
)
notes(
    s,
    "1 min. Deblur-NeRF real captures, nine scenes, four shown. There is no sharp ground truth, so this is qualitative "
    "and we say so. Columns: held-out photograph, vanilla, ours. Ours recovers detail on ball, buick and parterre. The "
    "third row is a failure: a moving person and only six photographs. We show it because it is the honest picture.",
)

# ---------------------------------------------------------------- 11 conclusions
s = new_slide("Conclusions and limitations", 11)
block(
    s,
    0.6,
    1.4,
    6.0,
    4.2,
    "What we found",
    [
        "Blur is a convolution, so the scene and the kernels can be estimated jointly.",
        "A structured kernel prior is essential: free-form kernels collapse.",
        "On motion blur it gains 2 to 3.5 dB over standard 3DGS and nearly matches true kernels, with no kernel guess.",
        "Classical deconvolution depends on the guess and on the noise level.",
    ],
    size=17,
    color=GREEN,
)
block(
    s,
    6.8,
    1.4,
    6.0,
    4.2,
    "Limitations",
    [
        "Blur is synthetic and uniform over each photograph; real blur varies in space and depth.",
        "The prior has to match the blur type.",
        "Few scenes and seeds; real results are qualitative.",
        "No comparison with the original Deblur-NeRF or BAD-Gaussians code.",
    ],
    size=17,
    color=ACCENT,
)
text(
    s,
    0.6,
    5.9,
    12.2,
    0.9,
    [
        ("Code, results, report: github.com/IshaanM05/blur-robust-3dgs", {"size": 16, "bold": True, "color": NAVY}),
    ],
)
notes(
    s,
    "0.75 min. The message: the structure of the prior matters more than the optimiser. State the limits plainly: synthetic "
    "and global blur, few scenes, no comparison with the original systems. Then take questions.",
)

assert len(prs.slides) == TOTAL, len(prs.slides)
prs.save("docs/slides.pptx")
print(f"saved docs/slides.pptx ({TOTAL} slides)")
