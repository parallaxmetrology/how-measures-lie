"""Reproduce each failure on a synthetic input and draw it. Run from the folder root:

    .venv/bin/python demos/make_figures.py

Writes figures/*.png and figures/numbers.json (the page quotes these numbers).
"""
from __future__ import annotations

import numpy as np
from common import ACCENT, BLUE, INK, INK2, MUTED, RULE, centroid, lum, minmax01, plt, record, robust01, save, show, write_numbers
from scipy import ndimage as ndi
from skimage import feature, filters, morphology, transform

N = 512


# ---------------------------------------------------------------------------------------
# 1. The dead field
# ---------------------------------------------------------------------------------------

def square_image(cx, cy, side=60, ground=0.82, ink=0.08):
    img = np.full((N, N), ground)
    h = side // 2
    img[cy - h:cy + h, cx - h:cx + h] = ink
    return img


def naive_mask_centroid(img, pct=85):
    """edge = robust01(sobel); mask = edge >= its 85th percentile; centroid of the mask."""
    edge = robust01(filters.sobel(img))
    mask = edge >= np.percentile(edge, pct)
    return edge, mask, centroid(mask.astype(float))


def fig_dead_field():
    img = square_image(380, 150)
    edge, mask, (rx, ry) = naive_mask_centroid(img)
    tx, ty = centroid((img < 0.5).astype(float))
    raw = filters.sobel(img)
    support = float((raw > 1e-6).mean())

    xs = np.arange(80, 440, 40)
    reported = [naive_mask_centroid(square_image(int(x), 256))[2][0] for x in xs]
    true = [centroid((square_image(int(x), 256) < 0.5).astype(float))[0] for x in xs]

    fig = plt.figure(figsize=(9.2, 2.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.5)
    a0, a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(4))
    show(a0, img, "The image")
    a0.plot((tx + .5) * (N - 1), (ty + .5) * (N - 1), "o", color=ACCENT, ms=7, label="true centre of the square")
    a0.plot((rx + .5) * (N - 1), (ry + .5) * (N - 1), "x", color=BLUE, ms=9, mew=2.4, label="reported")
    show(a1, edge, "Edge, stretched")
    a1.text(256, 270, "all zeros", ha="center", color="white", fontsize=10)
    show(a2, mask.astype(float), "Top 15% of edge")
    a2.text(256, 270, f"{mask.mean():.0%} of frame", ha="center", color=INK, fontsize=10)
    a3.plot(true, true, color=RULE, lw=6, solid_capstyle="round", label="correct")
    a3.plot(true, reported, "o-", color=BLUE, ms=4, label="reported")
    a3.set_xlabel("true horizontal offset"); a3.set_ylabel("reported offset")
    a3.set_title("Move the square: answer stays put")
    a3.legend(fontsize=8.5, loc="upper left")
    fig.legend(*a0.get_legend_handles_labels(), loc="lower left", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.02, -0.06))
    save(fig, "dead_field")
    record("dead", support_pct=support * 100, mask_pct=float(mask.mean()) * 100,
           true_x=tx, true_y=ty, reported_x=rx, reported_y=ry,
           p98_edge=float(np.percentile(raw, 98)))


# ---------------------------------------------------------------------------------------
# 2. Stretched dust
# ---------------------------------------------------------------------------------------

def fig_dust():
    from parallax_notan.colour import oklab_chroma
    from parallax_notan.notan import NOTAN_SCALES
    img = np.full((N, N, 3), 0.3, np.float32)
    img[:, N // 2:] = 0.7
    C_raw = oklab_chroma(img)
    C = minmax01(C_raw)
    L = minmax01(lum(img))
    parts = []
    for s in NOTAN_SCALES:
        parts.append((np.abs(ndi.gaussian_filter(L, s) - np.median(L)),
                      0.5 * np.abs(ndi.gaussian_filter(C, s) - np.median(C))))
    lv = sum(p[0] for p in parts) / 3
    cv = sum(p[1] for p in parts) / 3
    share = float(cv.sum() / (lv.sum() + cv.sum()))

    # A muted but real colour, for scale.
    muted = np.full((N, N, 3), 0.42, np.float32)
    muted[..., 0] += 0.025; muted[..., 2] -= 0.02
    muted_chroma = float(np.percentile(oklab_chroma(muted), 98))

    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.75), gridspec_kw={"width_ratios": [1, 1, 1.5], "wspace": 0.3})
    show(ax[0], img, "Two greys, no colour")
    show(ax[1], C, "Chroma, stretched to 0..1")
    vals = [float(C_raw.max()), muted_chroma, 0.1]
    labels = ["this image\n(rounding)", "a muted\ngrey-brown", "a clear\ngreen"]
    ax[2].bar(range(3), vals, color=[ACCENT, MUTED, MUTED], width=0.6)
    ax[2].set_yscale("log"); ax[2].set_xticks(range(3), labels)
    ax[2].axhline(1e-6, color=INK2, lw=0.8, ls="--")
    ax[2].text(2.35, 1.4e-6, "1e-6", fontsize=8, color=INK2, ha="right")
    ax[2].set_title("Raw chroma before stretching (log)")
    save(fig, "stretched_dust")
    record("dust", raw_max=float(C_raw.max()), share_pct=share * 100, muted=muted_chroma)


# ---------------------------------------------------------------------------------------
# 3. The starved mask
# ---------------------------------------------------------------------------------------

def fig_starved():
    from parallax_notan import notan_mass
    H, W = 420, 560
    yy, xx = np.mgrid[0:H, 0:W]
    ground = np.array([0.36, 0.10, 0.12])
    field = np.array([0.22, 0.34, 0.24])
    rect = ((xx > 90) & (xx < 470) & (yy > 70) & (yy < 330)).astype(float)
    rect = ndi.gaussian_filter(rect, 22)
    img = ground * (1 - rect[..., None]) + field * rect[..., None]
    img[34:48, 500:516] = [0.04, 0.03, 0.03]      # one small hard dark mark
    img = np.clip(img + np.random.default_rng(3).normal(0, 0.004, img.shape), 0, 1)

    edges = feature.canny(lum(img), sigma=2.0, low_threshold=0.04, high_threshold=0.10)
    cov = float(edges.mean())
    ex, ey = centroid(edges.astype(float))
    m = notan_mass(img.astype(np.float32))
    nx, ny = centroid(m)

    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.6), gridspec_kw={"wspace": 0.16})
    show(ax[0], img, "Soft field, one hard mark")
    show(ax[1], np.where(edges, 0.0, 1.0), f"Edge mask: {cov:.2%} of frame")
    show(ax[2], 1 - minmax01(m), "Value mass (Notan)")
    for a, (cx, cy), col, lab in ((ax[0], (ex, ey), BLUE, "edge centroid"), (ax[0], (nx, ny), ACCENT, "mass centroid")):
        a.plot((cx + .5) * (W - 1), (cy + .5) * (H - 1), "x" if col == BLUE else "o", color=col, ms=9, mew=2.4, label=lab)
    ax[0].legend(loc="lower left", fontsize=8, bbox_to_anchor=(0, -0.28), ncol=2)
    save(fig, "starved_mask")
    record("starved", coverage_pct=cov * 100, edge_cx=ex, edge_cy=ey, mass_cx=nx, mass_cy=ny)


# ---------------------------------------------------------------------------------------
# 4. Ceilings
# ---------------------------------------------------------------------------------------

def blobs(n, seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N))
    pts = rng.uniform(20, N - 20, (n, 2))
    yy, xx = np.ogrid[:N, :N]
    for x, y in pts:
        img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 5.0 ** 2))
    return img


def fig_ceiling():
    ns = [3, 6, 10, 15, 20, 30, 50, 80, 120, 200]
    capped, free = [], []
    for n in ns:
        img = blobs(n, n)
        capped.append(len(feature.peak_local_max(img, min_distance=3, threshold_abs=0.3, num_peaks=20)))
        free.append(len(feature.peak_local_max(img, min_distance=3, threshold_abs=0.3)))
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.9), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.3})
    show(ax[0], blobs(80, 80) / blobs(80, 80).max(), "80 blobs")
    ax[1].plot(ns, free, "o-", color=MUTED, ms=4, label="peaks found, no cap")
    ax[1].plot(ns, capped, "o-", color=BLUE, ms=4, label="reported, num_peaks=20")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("blobs in the image"); ax[1].set_ylabel("regions")
    ax[1].set_title("Every rich image reads 20")
    ax[1].legend(fontsize=8.5)
    save(fig, "ceiling")
    record("ceiling", capped_at_80=capped[ns.index(80)], free_at_80=free[ns.index(80)],
           capped_at_200=capped[-1], free_at_200=free[-1])


# ---------------------------------------------------------------------------------------
# 5. Ratios with tiny denominators
# ---------------------------------------------------------------------------------------

def fig_ratio():
    fig_l = 0.30
    grounds = np.geomspace(0.002, 0.3, 40)
    ratio = fig_l / grounds
    diff = fig_l - grounds
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.8), gridspec_kw={"wspace": 0.32})
    ax[0].plot(grounds, ratio, color=BLUE)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("ground luminance (the denominator)"); ax[0].set_ylabel("figure / ground")
    ax[0].set_title("Same figure, darker ground: a 'record'")
    ax[1].plot(grounds, diff, color=MUTED)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 0.32)
    ax[1].set_xlabel("ground luminance"); ax[1].set_ylabel("figure - ground")
    ax[1].set_title("The difference barely changes")
    save(fig, "ratio")
    record("ratio", fig_l=fig_l, ratio_at_002=float(fig_l / 0.002), ratio_at_02=float(fig_l / 0.02),
           ratio_at_02_diff=float(fig_l - 0.02), diff_at_002=float(fig_l - 0.002))


# ---------------------------------------------------------------------------------------
# 6. Resolution
# ---------------------------------------------------------------------------------------

def sierpinski(n):
    i = np.arange(n)
    return ((i[:, None] & i[None, :]) == 0)


def box_dim(mask):
    n = mask.shape[0]
    sizes = [2 ** k for k in range(1, int(np.log2(n)) - 2)]
    counts = []
    for s in sizes:
        m = mask[: n - n % s, : n - n % s]
        blocks = m.reshape(m.shape[0] // s, s, m.shape[1] // s, s).any(axis=(1, 3))
        counts.append(blocks.sum())
    slope = np.polyfit(np.log(1 / np.array(sizes)), np.log(counts), 1)[0]
    return float(slope), sizes, counts


def fig_resolution():
    exact = {n: box_dim(sierpinski(n))[0] for n in (256, 512, 1024, 2048)}
    src = sierpinski(2048).astype(float)
    sizes = [600, 700, 800, 1000, 1200, 1400]   # below ~600 the 0.5 threshold empties the mask
    thresholds = (0.1, 0.25, 0.5)
    res = {t: [box_dim(transform.resize(src, (n, n), order=1, anti_aliasing=True) > t)[0] for n in sizes]
           for t in thresholds}
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.6], "wspace": 0.3})
    show(ax[0], ~sierpinski(512), "Generated at 512 (exact)")
    show(ax[1], ~(transform.resize(src, (600, 600), order=1, anti_aliasing=True) > 0.5), "2048 resampled to 600")
    ax[2].axhline(np.log(3) / np.log(2), color=RULE, lw=6, label="exact, log 3 / log 2")
    for t, col in zip(thresholds, (MUTED, INK2, BLUE)):
        ax[2].plot(sizes, res[t], "o-", color=col, ms=4, label=f"resampled, threshold {t}")
    ax[2].set_xlabel("pixels after one resample"); ax[2].set_ylabel("measured dimension")
    ax[2].set_title("One resample moves the 'fact'")
    ax[2].legend(fontsize=8, loc="lower right")
    save(fig, "resolution")
    allv = [v for t in thresholds for v in res[t]]
    record("resolution", exact=exact[1024], lo=min(allv), hi=max(allv),
           at600_t05=res[0.5][sizes.index(600)], at600_t01=res[0.1][sizes.index(600)])


# ---------------------------------------------------------------------------------------
# 7. Minimum mark size
# ---------------------------------------------------------------------------------------

def mark_field(diam, seed=0, n=60):
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N))
    yy, xx = np.ogrid[:N, :N]
    for x, y in rng.uniform(30, N - 30, (n, 2)):
        img[(xx - x) ** 2 + (yy - y) ** 2 <= (diam / 2) ** 2] = 1.0
    return img


def detect(field, floor=41, sigma=0.0):
    f = ndi.gaussian_filter(field, sigma) if sigma > 0 else field
    thr = 0.5 * f.max() if sigma > 0 else 0.5
    m = ndi.binary_opening(f > thr, structure=morphology.disk(1))
    floor_px = floor if sigma == 0 else max(4, int(floor / 4))
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return int((np.asarray(sizes) >= floor_px).sum())


def fig_marks():
    diams = [3, 4, 5, 6, 7, 8, 10, 12, 16, 20]
    plain = [detect(mark_field(d, d)) / 60 for d in diams]
    coal = [detect(mark_field(d, d), sigma=d / 3) / 60 for d in diams]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.8), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.3})
    show(ax[0], 1 - mark_field(5, 5), "60 marks, 5 px across")
    ax[1].plot(diams, plain, "o-", color=BLUE, ms=4, label="detector with a 41 px floor")
    ax[1].plot(diams, coal, "o-", color=ACCENT, ms=4, label="smoothed to mark scale first")
    ax[1].axvline(2 * np.sqrt(41 / np.pi), color=RULE, lw=5)
    ax[1].text(2 * np.sqrt(41 / np.pi) + 0.3, 0.5, "floor", fontsize=8, color=INK2)
    ax[1].set_xlabel("mark diameter (px)"); ax[1].set_ylabel("share of marks found")
    ax[1].set_title("Below the floor: a clean, wrong zero")
    ax[1].legend(fontsize=8.5, loc="center right")
    save(fig, "mark_size")
    record("marks", found_at_5=plain[diams.index(5)], found_coal_at_5=coal[diams.index(5)])


# ---------------------------------------------------------------------------------------
# 10. Ablation fill
# ---------------------------------------------------------------------------------------

def fig_fill():
    img = np.full((N, N), 0.72)
    img[120:220, 90:200] = 0.15            # mass A, upper left
    img[300:420, 330:450] = 0.25           # mass B, lower right
    img += np.random.default_rng(2).normal(0, 0.02, img.shape)
    box = (slice(300, 420), slice(330, 450))

    def offcentre(im):
        w = np.abs(ndi.gaussian_filter(im, 6) - np.median(im))
        cx, cy = centroid(w)
        return float(np.hypot(cx, cy))

    fills = {"median": float(np.median(img)), "white": 1.0, "black": 0.0}
    vals, ims = {}, {}
    for k, v in fills.items():
        a = img.copy(); a[box] = v
        ims[k] = a; vals[k] = offcentre(a)
    fig = plt.figure(figsize=(9.2, 2.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.4], wspace=0.65)
    for i, k in enumerate(fills):
        show(fig.add_subplot(gs[0, i]), np.clip(ims[k], 0, 1), f"fill {k}")
    a = fig.add_subplot(gs[0, 3])
    a.bar(range(3), [vals[k] for k in fills], color=[BLUE, MUTED, MUTED], width=0.6)
    a.set_xticks(range(3), list(fills))
    a.set_ylabel("off-centre force"); a.set_title("Same removal, three answers")
    save(fig, "ablation_fill")
    record("fill", **{f"fill_{k}": v for k, v in vals.items()}, original=offcentre(img))


# ---------------------------------------------------------------------------------------
# 12. Single random draws
# ---------------------------------------------------------------------------------------

def phase_scramble(img, rng):
    F = np.fft.fft2(img - img.mean())
    ph = np.angle(np.fft.fft2(rng.standard_normal(img.shape)))
    return np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * (np.angle(F) + ph)))) + img.mean()


def fig_null():
    rng = np.random.default_rng(11)
    f = np.fft.fftfreq(256)
    k = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2); k[0, 0] = 1
    base = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((256, 256))) / k ** 1.1))
    base = minmax01(base)
    yy, xx = np.mgrid[0:256, 0:256]
    img = base + 0.18 * np.exp(-((xx - 175) ** 2 + (yy - 90) ** 2) / (2 * 28 ** 2))

    def score(im):
        w = np.abs(ndi.gaussian_filter(im, 4) - np.median(im))
        cx, cy = centroid(w)
        return float(np.hypot(cx, cy))

    real = score(img)
    draws = np.array([score(phase_scramble(img, np.random.default_rng(s))) for s in range(400)])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.9), gridspec_kw={"width_ratios": [1.3, 1.3], "wspace": 0.3})
    ax[0].hist(draws, bins=36, color=RULE, edgecolor=MUTED, lw=0.4)
    ax[0].axvline(real, color=ACCENT, lw=2, label="the image")
    ax[0].axvline(draws[0], color=BLUE, lw=1.4, ls="--", label="one null draw")
    ax[0].set_xlabel("score"); ax[0].set_ylabel("null draws")
    ax[0].set_title("400 scrambled copies")
    ax[0].legend(fontsize=8.5)
    ns = np.arange(1, 401)
    for order_seed, col in ((1, BLUE), (2, MUTED), (3, INK2)):
        d = np.random.default_rng(order_seed).permutation(draws)
        ge = np.cumsum(d >= real)
        ax[1].plot(ns, (ge + 1) / (ns + 1), color=col, lw=1.2)
    ax[1].set_xscale("log"); ax[1].set_xlabel("draws used"); ax[1].set_ylabel("p estimate, (ge+1)/(n+1)")
    ax[1].axhline(0.05, color=RULE, lw=3)
    ax[1].set_title("Three seed orders: p settles only late")
    save(fig, "single_draw")
    p_full = float(((draws >= real).sum() + 1) / (len(draws) + 1))
    p20 = [float(((np.random.default_rng(s).permutation(draws)[:20] >= real).sum() + 1) / 21) for s in (1, 2, 3)]
    record("null", real=real, p_400=p_full, p20_a=p20[0], p20_b=p20[1], p20_c=p20[2],
           first_draw=float(draws[0]))


# ---------------------------------------------------------------------------------------
# 13. Hand-picked foils
# ---------------------------------------------------------------------------------------

def fig_foils():
    rng = np.random.default_rng(5)
    G = 64
    ink = np.zeros((G, G), bool)
    for _ in range(9):                                  # clustered ink
        cx, cy = rng.uniform(6, G - 6, 2)
        for _ in range(60):
            x, y = rng.normal([cx, cy], 3.2)
            if 0 <= x < G and 0 <= y < G:
                ink[int(y), int(x)] = True
    for _ in range(40):                                 # stray marks
        ink[rng.integers(0, G), rng.integers(0, G)] = True
    ty, tx = 8, 56
    ink[ty, tx] = True                                  # the "target" mark
    r = 4
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2 <= r * r).astype(float)
    nb = ndi.convolve(ink.astype(float), disk, mode="constant") - ink
    iso = 1 - nb / (disk.sum() - 1)
    target = float(iso[ty, tx])
    cells = np.argwhere(ink)
    dense = cells[np.argsort(iso[ink])][:40]
    foils = dense[rng.choice(len(dense), 6, replace=False)]
    foil_vals = iso[foils[:, 0], foils[:, 1]]
    all_vals = iso[ink]
    more = int((all_vals > target).sum())

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.0), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.28})
    show(ax[0], ~ink, "Ink, target and six hand-picked foils")
    ax[0].plot(tx, ty, "o", mfc="none", mec=ACCENT, ms=11, mew=2)
    ax[0].plot(foils[:, 1], foils[:, 0], "s", mfc="none", mec=BLUE, ms=8, mew=1.6)
    ax[1].hist(all_vals, bins=24, color=RULE, edgecolor=MUTED, lw=0.4, label="every inked cell")
    for v in foil_vals:
        ax[1].axvline(v, color=BLUE, lw=1.2)
    ax[1].axvline(target, color=ACCENT, lw=2.2)
    ax[1].set_xlabel("isolation (empty share of neighbourhood)"); ax[1].set_ylabel("cells")
    ax[1].set_title(f"'Most isolated' of 7; {more} cells beat it")
    ax[1].plot([], [], color=BLUE, label="hand-picked foils"); ax[1].plot([], [], color=ACCENT, label="target")
    ax[1].legend(fontsize=8.5, loc="upper left")
    save(fig, "foils")
    record("foils", target=target, foil_max=float(foil_vals.max()), more_isolated=more, cells=int(ink.sum()))


# ---------------------------------------------------------------------------------------
# 14. Unseeded randomness
# ---------------------------------------------------------------------------------------

def fig_hough():
    rng = np.random.default_rng(4)
    img = np.zeros((300, 300))
    from skimage.draw import line
    for _ in range(40):
        r0, c0, r1, c1 = rng.integers(0, 300, 4)
        rr, cc = line(r0, c0, r1, c1)
        img[rr, cc] = 1
    edges = img > 0

    def resultant(lines):
        a = np.array([np.arctan2(seg[1][1] - seg[0][1], seg[1][0] - seg[0][0]) for seg in lines])
        return float(np.abs(np.mean(np.exp(2j * a))))

    free = [resultant(transform.probabilistic_hough_line(edges, threshold=10, line_length=30, line_gap=3)[:22])
            for _ in range(12)]
    seeded = [resultant(transform.probabilistic_hough_line(edges, threshold=10, line_length=30, line_gap=3, rng=0)[:22])
              for _ in range(12)]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.6), gridspec_kw={"width_ratios": [1, 1.7], "wspace": 0.3})
    show(ax[0], ~edges, "40 lines")
    ax[1].plot(free, np.zeros(12) + 1, "o", color=BLUE, alpha=0.75, label="no seed, 12 runs")
    ax[1].plot(seeded, np.zeros(12), "o", color=MUTED, label="rng=0, 12 runs")
    ax[1].set_yticks([0, 1], ["seeded", "unseeded"]); ax[1].set_ylim(-0.6, 1.6)
    ax[1].set_xlabel("orientation resultant of the first 22 lines")
    ax[1].set_title("Same script, same image, different numbers")
    save(fig, "unseeded")
    record("hough", free_min=min(free), free_max=max(free), seeded_unique=len(set(seeded)))


# ---------------------------------------------------------------------------------------
# 15. The default lens misses the construction mode
# ---------------------------------------------------------------------------------------

def fig_lens():
    from baseline import blend_mass
    from parallax_notan import notan_mass
    rng = np.random.default_rng(8)
    img = np.full((N, N, 3), 0.66, np.float32)
    yy, xx = np.mgrid[0:N, 0:N]
    flat = ((xx - 150) ** 2 / 110 ** 2 + (yy - 330) ** 2 / 140 ** 2) <= 1     # a flat dark mass
    img[flat] = 0.32
    busy = (xx > 330) & (xx < 470) & (yy > 60) & (yy < 200)                    # a small busy patch
    tex = 0.66 + rng.uniform(-0.3, 0.3, (N, N))
    for c in range(3):
        img[..., c][busy] = tex[busy]
    b = blend_mass(img)
    m = notan_mass(img)
    share = lambda f, r: float(f[r].sum() / f.sum())  # noqa: E731
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.9), gridspec_kw={"wspace": 0.14})
    show(ax[0], img, "Flat mass, busy patch")
    show(ax[1], 1 - b, f"Edge blend: {share(b, flat):.0%} / {share(b, busy):.0%}")
    show(ax[2], 1 - minmax01(m), f"Notan: {share(m, flat):.0%} / {share(m, busy):.0%}")
    save(fig, "default_lens")
    record("lens", blend_flat=share(b, flat), blend_busy=share(b, busy),
           notan_flat=share(m, flat), notan_busy=share(m, busy),
           flat_area=float(flat.mean()), busy_area=float(busy.mean()))


# ---------------------------------------------------------------------------------------
# 16. The substrate
# ---------------------------------------------------------------------------------------

def fig_substrate():
    from parallax_notan.colour import oklab_chroma
    rng = np.random.default_rng(6)
    H, W = 380, 520
    paper = np.array([0.86, 0.80, 0.66])
    img = np.ones((H, W, 3)) * paper
    img += rng.normal(0, 0.012, (H, W, 1)) * np.array([1.0, 0.9, 0.6])   # tinted grain
    from skimage.draw import line_aa
    for _ in range(26):
        r0, c0 = rng.integers(40, H - 40), rng.integers(40, W - 40)
        r1, c1 = np.clip(r0 + rng.integers(-60, 60), 0, H - 1), np.clip(c0 + rng.integers(-90, 90), 0, W - 1)
        for d in range(-2, 3):
            rr, cc, v = line_aa(r0 + d, c0, int(r1) + d, int(c1))
            ok = (rr >= 0) & (rr < H)
            img[rr[ok], cc[ok]] = img[rr[ok], cc[ok]] * (1 - v[ok, None]) + 0.12 * v[ok, None]
    img = np.clip(img, 0, 1)

    def colour_in_void(im):
        """Share of the image's colour that sits where there is no ink: a
        distribution-relative score of the kind used to say "colour lives in the empty space"."""
        c = oklab_chroma(im.astype(np.float32))
        if c.max() < 1e-6:                      # dust guard: no colour at all
            return 0.0
        void = lum(im) > 0.5
        return float(c[void].sum() / c.sum())

    whole = colour_in_void(img)
    grey = colour_in_void(np.repeat(lum(img)[..., None], 3, 2))
    bare = colour_in_void(img[0:36, 0:120])
    inked = float((lum(img) <= 0.5).mean())
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.8), gridspec_kw={"width_ratios": [1.4, 1.2], "wspace": 0.3})
    show(ax[0], img, "Grey ink on tinted paper")
    ax[0].add_patch(plt.Rectangle((0, 0), 120, 36, fill=False, ec=ACCENT, lw=1.6))
    ax[1].bar(range(3), [whole, grey, bare], color=[BLUE, MUTED, ACCENT], width=0.6)
    ax[1].set_xticks(range(3), ["whole\nimage", "colour\nremoved", "bare paper\n(orange box)"])
    ax[1].text(1, 0.03, "0", ha="center", color=INK2)
    ax[1].set_title("The colour score reads the paper")
    ax[1].set_ylabel("colour-in-void score")
    save(fig, "substrate")
    record("substrate", whole=whole, grey=grey, bare=bare, inked_pct=inked * 100)


if __name__ == "__main__":
    for f in (fig_dead_field, fig_dust, fig_starved, fig_ceiling, fig_ratio, fig_resolution,
              fig_marks, fig_fill, fig_null, fig_foils, fig_hough, fig_lens, fig_substrate):
        f()
    write_numbers()
