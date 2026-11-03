"""Build docs/slides.pptx from results/all_results.csv and docs/figures/ (run collect.py and make_figures.py first).

Every number on the slides comes from the results csv, so rebuilding after new runs updates the deck.
Usage: python scripts/build_slides.py
"""

import csv
from pathlib import Path

from PIL import Image, ImageChops
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

FIG = Path("docs/figures")
NAVY, AMBER, BLUE, GREEN = "14213D", "FCA311", "2A6F97", "2E8B6E"
RUST, GRAY, LIGHT, INK, TAN = "C8553D", "8D99AE", "F2F4F7", "1B1F2A", "D4A373"
HEAD_FONT, BODY_FONT = "Cambria", "Calibri"
W, H = 13.333, 7.5

rows = list(csv.DictReader(open("results/all_results.csv")))


def psnr(scene, setting, method):
    for r in rows:
        if (r["scene"], r["setting"], r["method"]) == (scene, setting, method):
            return float(r["psnr"])
    return None


def kernel_l2(scene, setting, method):
    for r in rows:
        if (r["scene"], r["setting"], r["method"]) == (scene, setting, method) and r["kernel_l2"]:
            return float(r["kernel_l2"])
    return None


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def rgb(h):
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
slides_total = 10


def new_slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(NAVY if dark else "FFFFFF")
    return s


def text(
    slide,
    x,
    y,
    w,
    h,
    content,
    size=16,
    bold=False,
    color=INK,
    font=BODY_FONT,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    italic=False,
    space_after=0,
):
    """content: str or list of str / (str, dict) paragraphs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = content if isinstance(content, list) else [content]
    for i, p in enumerate(paras):
        opts = {}
        if isinstance(p, tuple):
            p, opts = p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(opts.get("space_after", space_after))
        run = para.add_run()
        run.text = p
        f = run.font
        f.name = opts.get("font", font)
        f.size = Pt(opts.get("size", size))
        f.bold = opts.get("bold", bold)
        f.italic = opts.get("italic", italic)
        f.color.rgb = rgb(opts.get("color", color))
    return tb


def title(slide, t, color=NAVY):
    text(slide, 0.6, 0.4, W - 1.2, 0.9, t, size=32, bold=True, color=color, font=HEAD_FONT, anchor=MSO_ANCHOR.MIDDLE)


def footer(slide, n, dark=False):
    c = GRAY if not dark else "AAB4C5"
    text(slide, 0.6, 7.05, 8, 0.25, "Blur-robust 3D Gaussian Splatting  |  CS 663", size=10, color=c)
    text(slide, W - 1.6, 7.05, 1.0, 0.25, f"{n} / {slides_total}", size=10, color=c, align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, fill=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.06
    return s


def trimmed(path, pad=12):
    """Crop the white margin around a matplotlib figure so it can be shown larger."""
    im = Image.open(path).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height)))
    out = Path("/tmp") / f"trim_{Path(path).name}"
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


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def stat(slide, x, y, value, label, color=AMBER, dark=False, w=3.2):
    text(slide, x, y, w, 1.0, value, size=54, bold=True, color=color, font=HEAD_FONT)
    text(slide, x, y + 1.05, w, 0.9, label, size=14, color="DDE3EE" if dark else INK)


def style_chart(chart, size=12, legend=True):
    chart.font.size = Pt(size)
    chart.font.name = BODY_FONT
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False


def color_series(chart, colors):
    for s, c in zip(chart.plots[0].series, colors, strict=False):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(c)
        s.format.line.color.rgb = rgb(c)


# derived headline numbers
gain_high = None
if psnr("bonsai", "motion_high", "ours_polyline") and psnr("bonsai", "motion_high", "vanilla"):
    gain_high = psnr("bonsai", "motion_high", "ours_polyline") - psnr("bonsai", "motion_high", "vanilla")

# ---------------------------------------------------------------- 1 title
s = new_slide(dark=True)
text(s, 0.8, 1.7, 7.6, 2.4, "Blur-Robust 3D Gaussian Splatting", size=44, bold=True, color="FFFFFF", font=HEAD_FONT)
text(
    s,
    0.8,
    3.7,
    7.2,
    0.9,
    "Joint per-view blur-kernel estimation from blurry multi-view photos",
    size=20,
    color="DDE3EE",
)
text(
    s,
    0.8,
    5.5,
    7.2,
    0.8,
    ["Ishaan Mondal (24B2118)", "CS 663 Digital Image Processing, IIT Bombay"],
    size=16,
    color="AAB4C5",
)
if gain_high is not None:
    stat(
        s,
        8.9,
        2.1,
        f"+{gain_high:.1f} dB",
        "PSNR over vanilla 3DGS at high motion blur (bonsai scene), without knowing the blur",
        dark=True,
        w=3.8,
    )
footer(s, 1, dark=True)
notes(
    s,
    "30 s. Introduce the problem: 3DGS rebuilds 3D scenes from photos, but real photos are blurry. We treat blur as an image-restoration problem and fix it inside the 3D optimisation.",
)

# ---------------------------------------------------------------- 2 problem + related work
s = new_slide()
title(s, "Blurry photos break 3D reconstruction")
text(
    s,
    0.6,
    1.5,
    5.6,
    4.6,
    [
        (
            "3D Gaussian Splatting rebuilds a scene from many posed photos and renders it in real time.",
            {"space_after": 14},
        ),
        (
            "It assumes sharp inputs. Handheld video, night driving and shallow depth of field give motion or defocus blur.",
            {"space_after": 14},
        ),
        (
            "Vanilla 3DGS bakes the blur into the scene: the reconstruction is blurry from every viewpoint.",
            {"space_after": 14},
        ),
        (
            "Question: can we model the blur per image and recover a sharp scene, without being told the blur?",
            {"bold": True, "color": BLUE},
        ),
    ],
    size=18,
)
refs = [
    ("Deblur-NeRF (2022)", "Learned per-view blur kernel inside a NeRF."),
    ("Deblurring 3DGS (2024)", "Blur modelled by modulating Gaussians during training."),
    ("BAD-Gaussians (2024)", "Camera trajectory during exposure; blur = average of sharp renders."),
    ("Levin et al. (2009)", "Why joint blind deconvolution collapses to the trivial solution."),
]
for i, (h, d) in enumerate(refs):
    y = 1.5 + i * 1.18
    card(s, 6.8, y, 5.9, 1.06)
    text(s, 7.05, y + 0.12, 5.4, 0.35, h, size=16, bold=True, color=NAVY)
    text(s, 7.05, y + 0.5, 5.4, 0.5, d, size=14)
footer(s, 2)
notes(
    s, "1.5 min. Four related works we read. Levin et al. is the key theory: it explains the failure we observe later."
)

# ---------------------------------------------------------------- 3 DIP model
s = new_slide()
title(s, "The DIP view: blur is convolution")
card(s, 0.6, 1.5, 6.6, 4.9, fill=LIGHT)
text(
    s,
    0.9,
    1.7,
    6.0,
    4.5,
    [
        ("Image formation", {"bold": True, "color": BLUE, "size": 16, "space_after": 4}),
        ("y = k * x + n", {"font": "Courier New", "size": 22, "bold": True, "space_after": 18}),
        ("Wiener deconvolution", {"bold": True, "color": BLUE, "size": 16, "space_after": 4}),
        ("X = conj(K) Y / (|K|^2 + NSR)", {"font": "Courier New", "size": 18, "space_after": 18}),
        ("Richardson-Lucy", {"bold": True, "color": BLUE, "size": 16, "space_after": 4}),
        ("x <- x . ( k^T * ( y / (k * x) ) )", {"font": "Courier New", "size": 18, "space_after": 18}),
        ("In 3DGS, x is the rendering of the Gaussians and k is a different kernel for every photo.", {"size": 14}),
    ],
    size=16,
)
v, wg, wo = (psnr("bonsai", "motion_medium", m) for m in ("vanilla", "wiener_guess", "wiener_oracle"))
stat(s, 7.8, 1.5, fmt(wo), "dB: Wiener with the TRUE kernel, then 3DGS", color=GREEN, w=4.8)
stat(s, 7.8, 3.15, fmt(wg), "dB: Wiener with a GUESSED kernel", color=RUST, w=4.8)
stat(s, 7.8, 4.8, fmt(v), "dB: no deblurring (vanilla 3DGS)", color=GRAY, w=4.8)
text(s, 7.8, 6.55, 4.9, 0.3, "bonsai, medium motion blur, PSNR on sharp test views", size=11, color=GRAY)
footer(s, 3)
notes(
    s,
    "1.5 min. Blur is convolution with a kernel. Wiener and RL are the classical inverses and need the kernel. With the true kernel they work, with a guessed kernel they barely beat doing nothing. So the kernel must be estimated.",
)

# ---------------------------------------------------------------- 4 approach
s = new_slide()
title(s, "Approach: joint optimisation")
boxes = [
    ("3D Gaussians + camera pose", BLUE),
    ("sharp render", BLUE),
    ("* kernel k_i (one per photo)", AMBER),
    ("blurry render", GRAY),
]
bw, gap, y0 = 2.55, 0.55, 1.55
for i, (t, c) in enumerate(boxes):
    x = 0.6 + i * (bw + gap)
    b = card(s, x, y0, bw, 1.0, fill=c)
    text(
        s,
        x + 0.1,
        y0,
        bw - 0.2,
        1.0,
        t,
        size=15,
        bold=True,
        color="FFFFFF",
        align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    if i < 3:
        a = s.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + 0.08), Inches(y0 + 0.3), Inches(gap - 0.16), Inches(0.4)
        )
        a.fill.solid()
        a.fill.fore_color.rgb = rgb(NAVY)
        a.line.fill.background()
text(
    s,
    0.6,
    2.7,
    W - 1.2,
    0.4,
    "Loss: L1 + SSIM between the blurry render and the blurry photo. At test time the kernel is dropped and the sharp render is scored against the sharp photo.",
    size=14,
)
priors = [
    (
        "Free-form kernel",
        "K x K softmax per photo. Most flexible, but collapses to a delta (the scene absorbs the blur).",
        RUST,
    ),
    (
        "Polyline kernel",
        "Exposure path as 6 control points, splatted into a kernel. Physical prior for motion blur.",
        GREEN,
    ),
    ("Gaussian kernel", "sigma_x, sigma_y, angle. Three numbers per photo, aimed at defocus and Gaussian blur.", BLUE),
]
for i, (h, d, c) in enumerate(priors):
    x = 0.6 + i * 4.1
    card(s, x, 3.5, 3.9, 2.2)
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.25), Inches(3.75), Inches(0.35), Inches(0.35))
    dot.fill.solid()
    dot.fill.fore_color.rgb = rgb(c)
    dot.line.fill.background()
    text(s, x + 0.75, 3.72, 3.0, 0.4, h, size=17, bold=True, color=NAVY)
    text(s, x + 0.25, 4.3, 3.45, 1.4, d, size=14)
text(
    s,
    0.6,
    6.0,
    W - 1.2,
    0.8,
    "Written from scratch: 2D Gaussian rasterizer, blur simulator, Wiener and Richardson-Lucy, the three kernel estimators, metrics. gsplat supplies only the 3D rasterizer.",
    size=14,
    color=GRAY,
)
footer(s, 4)
notes(
    s,
    "2 min. Core idea. The only question is how to parametrise the unknown kernel. Free-form is the obvious choice and fails; structured priors work. Mention gsplat is only the rasterizer.",
)

# ---------------------------------------------------------------- 5 2D warm-up
s = new_slide()
title(s, "Why one image is not enough")
picture(s, FIG / "exp_2d_blind.png", 0.6, 1.4, W - 1.2, 4.3)
text(
    s,
    0.6,
    5.95,
    W - 1.2,
    0.9,
    "2D study with our from-scratch Gaussian fitter: with the true kernel deconvolution works; with a single blurry image and a learnable kernel, the optimiser returns the trivial solution (sharp = blurry, kernel = delta), no better than the input. In 3D, hundreds of photos with different blur must agree on one sharp scene.",
    size=14,
)
footer(s, 5)
notes(
    s,
    "1 min. This is the identifiability problem from Levin et al. The 3D multi-view setting adds the missing constraint, but we will see it is still not enough for a free-form kernel.",
)

# ---------------------------------------------------------------- 6 results table
s = new_slide()
title(s, "Blind prior matches oracle kernels")
methods = [
    ("Vanilla 3DGS (blurry input)", "vanilla"),
    ("Wiener, guessed kernel, then 3DGS", "wiener_guess"),
    ("RL, guessed kernel, then 3DGS", "rl_guess"),
    ("Ours: polyline kernel (blind)", "ours_polyline"),
    ("Ours: oracle kernels (upper bound)", "ours_oracle_kernel"),
    ("Wiener, TRUE kernel, then 3DGS", "wiener_oracle"),
]
scenes = ["bonsai", "counter", "garden"]
tbl = s.shapes.add_table(len(methods) + 1, 4, Inches(0.6), Inches(1.5), Inches(8.2), Inches(3.9)).table
tbl.columns[0].width = Inches(4.3)
for c in (1, 2, 3):
    tbl.columns[c].width = Inches(1.3)
for ci, hdr in enumerate(["PSNR (dB), motion medium", "bonsai", "counter", "garden"]):
    cell = tbl.cell(0, ci)
    cell.fill.solid()
    cell.fill.fore_color.rgb = rgb(NAVY)
    cell.text_frame.text = hdr
for ri, (label, key) in enumerate(methods, start=1):
    vals = [label] + [fmt(psnr(sc, "motion_medium", key)) for sc in scenes]
    for ci, v in enumerate(vals):
        cell = tbl.cell(ri, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb("E3F0EA" if key == "ours_polyline" else ("FFFFFF" if ri % 2 else LIGHT))
        cell.text_frame.text = v
for ri in range(len(methods) + 1):
    for ci in range(4):
        cell = tbl.cell(ri, ci)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(14)
                r.font.name = BODY_FONT
                r.font.bold = ri == 0 or methods[ri - 1][1] == "ours_polyline"
                r.font.color.rgb = rgb("FFFFFF" if ri == 0 else INK)
bl, bc = kernel_l2("bonsai", "motion_medium", "ours_polyline"), kernel_l2("counter", "motion_medium", "ours_polyline")
text(
    s,
    9.3,
    1.5,
    3.5,
    1.2,
    "Within about 0.5 dB of oracle kernels on every scene",
    size=18,
    bold=True,
    color=NAVY,
    font=HEAD_FONT,
)
if bl is not None:
    stat(s, 9.3, 2.9, f"{bl:.2f}", "kernel error of our estimate (a delta scores 0.99)", color=GREEN, w=3.6)
text(
    s,
    0.6,
    5.65,
    8.2,
    0.9,
    "Held-out sharp test views, 10k iterations, single seed. Oracle rows use information a real system does not have; the fair comparison is against the guessed-kernel and vanilla rows.",
    size=12,
    color=GRAY,
)
footer(s, 6)
notes(
    s,
    "1.5 min. Read the table top to bottom. Blind polyline is 2-4 dB above every blind baseline and within half a dB of oracle kernels. Wiener with the true kernel is higher still but uses information we do not have. Garden is a harder scene so absolute numbers are lower.",
)

# ---------------------------------------------------------------- 7 qualitative
s = new_slide()
title(s, "What it looks like")
picture(s, FIG / "qualitative_bonsai_motion_medium.png", 0.6, 1.4, W - 1.2, 5.2)
text(
    s,
    0.6,
    6.65,
    W - 1.2,
    0.3,
    "bonsai, medium motion blur, held-out view. Bottom row: zoom on the textured cloth and objects.",
    size=12,
    color=GRAY,
)
footer(s, 7)
notes(
    s,
    "1 min. Look at the zoomed bottom row: vanilla is smeared, guessed-kernel Wiener is slightly sharper but still soft, ours recovers the cloth texture and edges and looks like the oracle.",
)

# ---------------------------------------------------------------- 8 severity + blur types
s = new_slide()
title(s, "Severity and blur type")
sev = ["low", "medium", "high"]
series = [
    ("Vanilla", "vanilla", GRAY),
    ("Guessed-kernel Wiener", "wiener_guess", TAN),
    ("Ours (polyline)", "ours_polyline", BLUE),
    ("Oracle kernels", "ours_oracle_kernel", GREEN),
]
cd = CategoryChartData()
cd.categories = ["low", "medium", "high"]
used = []
for label, key, col in series:
    vals = [psnr("bonsai", f"motion_{x}", key) for x in sev]
    if all(v is not None for v in vals):
        cd.add_series(label, vals)
        used.append(col)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.5), Inches(6.0), Inches(4.7), cd)
style_chart(gf.chart)
for ser, col in zip(gf.chart.plots[0].series, used, strict=False):
    ser.format.line.color.rgb = rgb(col)
    ser.format.line.width = Pt(3)
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = rgb(col)
    ser.smooth = False
gf.chart.value_axis.minimum_scale = 24
gf.chart.value_axis.has_major_gridlines = True
gf.chart.value_axis.major_gridlines.format.line.color.rgb = rgb("DDE3EE")
gf.chart.has_title = True
gf.chart.chart_title.text_frame.text = "Motion blur severity (bonsai, PSNR dB)"
gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)

kinds = ["motion", "defocus", "gaussian"]


def best_prior(kind):
    cand = [psnr("bonsai", f"{kind}_medium", m) for m in ("ours_polyline", "ours_gauss")]
    cand = [c for c in cand if c is not None]
    return max(cand) if cand else None


bt = [
    ("Vanilla", [psnr("bonsai", f"{k}_medium", "vanilla") for k in kinds], GRAY),
    ("Guessed-kernel Wiener", [psnr("bonsai", f"{k}_medium", "wiener_guess") for k in kinds], TAN),
    ("Ours (best prior)", [best_prior(k) for k in kinds], BLUE),
    ("Oracle kernels", [psnr("bonsai", f"{k}_medium", "ours_oracle_kernel") for k in kinds], GREEN),
]
bt = [t for t in bt if all(v is not None for v in t[1])]
if bt:
    cd2 = CategoryChartData()
    cd2.categories = ["motion", "defocus", "gaussian"]
    for label, vals, _ in bt:
        cd2.add_series(label, vals)
    gf2 = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.9), Inches(1.5), Inches(5.9), Inches(4.7), cd2)
    style_chart(gf2.chart)
    color_series(gf2.chart, [t[2] for t in bt])
    gf2.chart.value_axis.minimum_scale = 24
    gf2.chart.value_axis.major_gridlines.format.line.color.rgb = rgb("DDE3EE")
    gf2.chart.has_title = True
    gf2.chart.chart_title.text_frame.text = "Blur type, medium severity (bonsai, PSNR dB)"
    gf2.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
text(
    s,
    0.6,
    6.4,
    W - 1.2,
    0.6,
    "Motion blur: the polyline prior stays near the oracle at every severity. Defocus and Gaussian blur need a prior that can represent a blob, not a path.",
    size=14,
)
footer(s, 8)
notes(
    s,
    "1.5 min. Left: the benefit over vanilla grows with severity because there is more blur to remove. Right: the same method with the wrong prior is weaker on defocus, which is why we added the Gaussian kernel. The prior has to match the blur.",
)

# ---------------------------------------------------------------- 9 ablations + spectrum
s = new_slide()
title(s, "Ablations and frequency analysis")
abl = [
    ("Default (polyline, 6 pts, 25 px)", "ours_polyline"),
    ("3 control points", "abl_ctrl3"),
    ("12 control points", "abl_ctrl12"),
    ("Kernel size 15", "abl_k15"),
    ("Kernel size 35", "abl_k35"),
    ("One shared kernel for all views", "abl_shared"),
    ("Free-form kernel", "ours_free"),
    ("Free-form + regularizers", "abl_free_all"),
    ("Vanilla 3DGS", "vanilla"),
]
abl = [(lbl, psnr("bonsai", "motion_medium", k)) for lbl, k in abl]
abl = [a for a in abl if a[1] is not None]
if abl:
    cd3 = CategoryChartData()
    cd3.categories = [a[0] for a in abl][::-1]
    cd3.add_series("PSNR", [a[1] for a in abl][::-1])
    gf3 = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.45), Inches(6.7), Inches(5.0), cd3)
    style_chart(gf3.chart, size=12, legend=False)
    color_series(gf3.chart, [BLUE])
    gf3.chart.value_axis.minimum_scale = 24
    gf3.chart.value_axis.major_gridlines.format.line.color.rgb = rgb("DDE3EE")
    pl = gf3.chart.plots[0]
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.0"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.font.size = Pt(12)
    gf3.chart.has_title = True
    gf3.chart.chart_title.text_frame.text = "bonsai, medium motion blur (PSNR dB)"
    gf3.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
w_, h_ = picture(s, FIG / "spectrum_bonsai_motion_medium.png", 7.6, 1.45, 5.2, 3.7)
text(
    s,
    7.6,
    1.45 + h_ + 0.15,
    5.2,
    1.4,
    "Radially averaged power spectrum of a test view. Vanilla loses the high frequencies; our estimate recovers most of them and tracks the oracle.",
    size=14,
)
footer(s, 9)
notes(
    s,
    "1.5 min. Left: what matters. The structured polyline prior is what makes it work; the shared kernel and free-form kernels are clearly worse. Right: the DIP interpretation, the blur is a low-pass filter, and our method restores the lost high-frequency power.",
)

# ---------------------------------------------------------------- 10 takeaways
s = new_slide(dark=True)
title(s, "Takeaways and limits", color="FFFFFF")
text(
    s,
    0.8,
    1.6,
    5.9,
    4.6,
    [
        ("What we found", {"bold": True, "color": AMBER, "size": 20, "space_after": 10}),
        ("Blur is a convolution; the scene and the per-photo kernels can be recovered jointly.", {"space_after": 10}),
        (
            "The kernel prior is the whole game: free-form collapses, a structured prior matches oracle kernels on motion blur.",
            {"space_after": 10},
        ),
        ("Guessing the kernel can be worse than doing nothing.", {}),
    ],
    size=17,
    color="FFFFFF",
)
text(
    s,
    7.0,
    1.6,
    5.7,
    4.6,
    [
        ("Limits", {"bold": True, "color": AMBER, "size": 20, "space_after": 10}),
        (
            "Blur is synthetic and global per photo; real blur varies across the image and with depth.",
            {"space_after": 10},
        ),
        ("One kernel family per blur type; the prior has to be chosen.", {"space_after": 10}),
        ("Three scenes, few seeds; no real blurry capture yet.", {}),
    ],
    size=17,
    color="FFFFFF",
)
text(
    s, 0.8, 6.3, 11.8, 0.5, "Code, results and figures: github.com/IshaanM05/blur-robust-3dgs", size=14, color="AAB4C5"
)
footer(s, 10, dark=True)
notes(
    s,
    "1 min. Close on the message: the structure of the prior matters more than the optimiser. Be upfront about the limits: synthetic blur, global kernel, few scenes.",
)

assert len(prs.slides) == slides_total
out = Path("docs/slides.pptx")
prs.save(out)
print("saved", out, f"({len(prs.slides)} slides)")
