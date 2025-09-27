import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.ticker import MultipleLocator

# https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    radii
    center
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    print(f'center: {mean_x:.2f}, {mean_y:.2f}')
    print(f'radii: {scale_x:.2f}, {scale_y:.2f}')
    
    ellipse.set_transform(transf + ax.transData)

    
    return ax.add_patch(ellipse), [ell_radius_x, ell_radius_y], [mean_x, mean_y], cov


def backtransform_ILR(z1, z2):
    sqr2 = np.sqrt(2)
    sqr6 = np.sqrt(6)

    y1 =  z1 / sqr2 + z2 / sqr6
    y2 = -z1 / sqr2 + z2 / sqr6
    y3 = -2*z2 / sqr6

    sum_y = np.exp(y1)+np.exp(y2)+np.exp(y3)
    x1 = np.exp(y1)/sum_y
    x2 = np.exp(y2)/sum_y
    x3 = np.exp(y3)/sum_y

    return [x1, x2, x3]


# Vibe Coding - adapted
# ternary plots are very confusing and reading off values is difficult. Therefore we create a
# shaded band parallel to each triangle edge to help with that. Additional annotating makes the 
# plot more intuitive
# added a function to plot the inverse transform of certainty ellipses

# drawing a shaded region for axis and value 1 and value 2
def draw_constant_band(ax, axis, v1, v2, *, scale=1.0, N=2,
                       color="black", line_kwargs=None, fill_kwargs=None):
    # (same as before) — draws the two constant-component lines and shades between
    axis = axis.lower(); assert axis in {"t","l","r"}
    low, high = sorted([float(v1), float(v2)])
    lk = dict(ls="--", lw=1.5, color=color); lk.update(line_kwargs or {})
    fk = dict(facecolor=color, alpha=0.15, edgecolor="none"); fk.update(fill_kwargs or {})

    if axis == "t":
        l1 = np.linspace(0, scale-low, N); r1 = scale-low-l1; t1 = np.full(N, low)
        l2 = np.linspace(0, scale-high, N); r2 = scale-high-l2; t2 = np.full(N, high)
    elif axis == "l":
        t1 = np.linspace(0, scale-low, N); r1 = scale-low-t1; l1 = np.full(N, low)
        t2 = np.linspace(0, scale-high, N); r2 = scale-high-t2; l2 = np.full(N, high)
    else:  # axis == "r"
        t1 = np.linspace(0, scale-low, N); l1 = scale-low-t1; r1 = np.full(N, low)
        t2 = np.linspace(0, scale-high, N); l2 = scale-high-t2; r2 = np.full(N, high)

    ax.plot(t1, l1, r1, **lk); ax.plot(t2, l2, r2, **lk)
    T = np.concatenate([t1, t2[::-1]])
    L = np.concatenate([l1, l2[::-1]])
    R = np.concatenate([r1, r2[::-1]])
    ax.fill(T, L, R, **fk)

# annotate the axii to make reading of values easier for user
def annotate_axis_gap(ax, axis, v1, v2, *, scale=1.0, at=0.5,
                      color="black", lw=2.0, label=None, text_kw=None):
    """
    Draw a short double-ended segment parallel to the chosen axis between v1 and v2,
    positioned at fraction 'at' (0..1) across the band, with a label Δ.
    """
    axis = axis.lower(); assert axis in {"t","l","r"}
    low, high = sorted([float(v1), float(v2)])
    gap = abs(high - low)
    txt = label if label is not None else f"Δ{axis} = {gap:.2g}"

    if axis == "t":
        # place inside both lines at l = at*(scale - high)
        l = at*(scale - high)
        p_lo = (low,  l,  scale - low  - l)
        p_hi = (high, l,  scale - high - l)
    elif axis == "l":
        t = at*(scale - high)
        p_lo = (scale - low  - t, low, t)
        p_hi = (scale - high - t, high, t)
    else:  # 'r'
        t = at*(scale - high)
        p_lo = (t, scale - low  - t, low)
        p_hi = (t, scale - high - t, high)

    # draw the segment and end caps
    ax.plot([p_lo[0], p_hi[0]], [p_lo[1], p_hi[1]], [p_lo[2], p_hi[2]],
            color=color, lw=lw, zorder= 10, clip_on= False)
    ax.scatter([p_lo[0], p_hi[0]], [p_lo[1], p_hi[1]], [p_lo[2], p_hi[2]],
               s=20, facecolors=color, edgecolors="black", linewidths=0.8, zorder=11, clip_on=False)

    # label near the upper endpoint
    tk = dict(ha="left", va="bottom", fontsize=10, color=color); tk.update(text_kw or {})
    ax.text(p_hi[0], p_hi[1], p_hi[2], txt, **tk)

# additional function to label an annotation with the correct values
# not recommended, it adds confusion
def label_point(ax, t, l, r, text, dx=6, dy=6, units='points', color="black", **kw):
    trans = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units=units)
    return ax.text(t, l, r, text, transform=trans, color=color,
                   va="bottom", ha="left",
                   path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                   **kw)

# Plotting certainty ellipse in ternary coordinates
def ternary_ellipse_from_ilr(mu, Sigma, level=0.90, n_boundary=400, n_fill=2000,
                             psi=None, rng=None):
    """
    Back-transform an ILR ellipse (D=3) to ternary compositions only.
    Returns a dict:
      - 'boundary' : (n,3) compositions on the ellipse rim (sum to 1)
      - 'fill'         : (m,3) compositions for the interior (empty if n_fill=0)
    """
    mu = np.asarray(mu, float).reshape(2)
    Sigma = np.asarray(Sigma, float).reshape(2,2)

    # ILR basis (Helmert sub-matrix by default; columns orthonormal in CLR space)
    if psi is None:
        psi = np.array([[ 1/np.sqrt(2),  1/np.sqrt(6)],
                        [-1/np.sqrt(2),  1/np.sqrt(6)],
                        [ 0.0,          -2/np.sqrt(6)]], float)
    else:
        psi = np.asarray(psi, float).reshape(3,2)

    # Chi-square thresholds for df=2
    chi2_df2 = {0.50: 1.3862943611, 0.75: 2.7725887222,
                0.90: 4.6051701860, 0.95: 5.9914645471, 0.99: 9.21034037198}
    if level not in chi2_df2:
        raise ValueError("Use level in {0.50, 0.75, 0.90, 0.95, 0.99}.")
    k = np.sqrt(chi2_df2[level])

    # Ellipse transform in ILR
    w, V = np.linalg.eigh(Sigma)                 # Sigma = V diag(w) V^T
    A = V @ np.diag(np.sqrt(np.maximum(w, 0.0))) # guard tiny negatives

    t = np.linspace(0, 2*np.pi, int(n_boundary), endpoint=True)
    circle = np.vstack((np.cos(t), np.sin(t)))
    boundary_ilr = mu + (k * (A @ circle)).T

    if n_fill > 0:
        rng = rng or np.random.default_rng()
        u = rng.random(n_fill)
        r = k * np.sqrt(u)
        phi = 2*np.pi * rng.random(n_fill)
        fill_ilr = mu + (A @ np.vstack((r*np.cos(phi), r*np.sin(phi)))).T
    else:
        fill_ilr = np.empty((0,2))

    # Inverse ILR -> compositions (a,b,c), sum to 1
    def ilr_inverse(Y):
        clr = Y @ psi.T
        clr -= np.max(clr, axis=-1, keepdims=True)
        Z = np.exp(clr)
        return Z / np.sum(Z, axis=-1, keepdims=True)

    return {
        "boundary": ilr_inverse(boundary_ilr),
        "fill": ilr_inverse(fill_ilr),
    }


def ternary_scatter_with_ilr_ellipse(
    ax,
    df,
    tag_pattern,
    comps=("carbs_perc", "fats_perc", "proteins_perc"),  # (t,l,r) for mpltern
    ilr_cols=("e_z1", "e_z2"),
    color="#E69F00",
    s=4,
    alpha=0.3,
    level=0.90,
    n_fill=300,
    scatter=True,
    plot_kwargs=None,
    scatter_kwargs=None,
    plot_centroid=True,
    centroid_marker="D",
    centroid_mfc=None,      # marker face color, defaults to color if None
    centroid_mec="black",   # marker edge color
    centroid_mew=0.8,       # marker edge width
    # crosshair options
    draw_crosshair=False,
    crosshair_linewidth=1.25,
    crosshair_alpha=0.9,
    ellipse = True
):
    """
    Add a ternary scatter + ILR confidence ellipse + centroid (via backtransform_ILR),
    with optional 'crosshair' lines through the centroid (two endpoint segments only).
    Returns dict with: index, mu, Sigma, boundary, centroid_abc.
    """
    plot_kwargs = plot_kwargs or {}
    scatter_kwargs = scatter_kwargs or {}

    # Subset and ensure needed columns
    cols_needed = list(comps) + list(ilr_cols)
    subset = df[df["tags"].str.contains(tag_pattern, case=False, na=False)][cols_needed].dropna()

    # Components closed to 1 (defensive renorm)
    X = subset.loc[:, comps].astype(float).to_numpy()
    row_sums = X.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        X = X / row_sums[:, None]

    # ILR stats from same rows
    Y = subset.loc[:, ilr_cols].astype(float).to_numpy()
    mu = Y.mean(axis=0)
    Sigma = np.cov(Y, rowvar=False)

    # Ellipse boundary (ternary triples)
    out = ternary_ellipse_from_ilr(mu, Sigma, level=level, n_fill=n_fill)
    B = out["boundary"]

    # Centroid via backtransform
    c_abc = np.array(backtransform_ILR(mu[0], mu[1]), dtype=float)
    t0, l0, r0 = c_abc.tolist()

    # Plot layers
    if scatter:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=s, alpha=alpha, color=color, **scatter_kwargs)
    if ellipse:
        ax.plot(B[:, 0], B[:, 1], B[:, 2], lw=2, color=color, **plot_kwargs)

    if plot_centroid:
        ax.plot(t0, l0, r0,
                linestyle="none",
                marker=centroid_marker,
                markerfacecolor=(color if centroid_mfc is None else centroid_mfc),
                markeredgecolor=centroid_mec,
                markeredgewidth=centroid_mew, zorder = 30)

    # Crosshair: draw each as a 2-point segment, handle = filled circle at one endpoint
    if draw_crosshair:
        mfc = (color if centroid_mfc is None else centroid_mfc)
        mec = centroid_mec
        mew = centroid_mew

        # t-constant line: endpoints (t0, 0, 1-t0) and (t0, 1-t0, 0); handle at l=0 side
        if t0 < 1.0:
            P1 = (t0, 0.0, 1.0 - t0)
            P2 = (t0, 1.0 - t0, 0.0)
            ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]],
                    lw=crosshair_linewidth, alpha=crosshair_alpha, color='black')
            ax.plot(*P1, linestyle="none", marker="o",
                    markerfacecolor=mfc, markeredgecolor=mfc, markeredgewidth=mew, zorder = 10)

        # l-constant line: endpoints (0, l0, 1-l0) and (1-l0, l0, 0); handle at r=0 side
        if l0 < 1.0:
            P1 = (0.0, l0, 1.0 - l0)
            P2 = (1.0 - l0, l0, 0.0)
            ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]],
                    lw=crosshair_linewidth, alpha=crosshair_alpha, color='black')
            ax.plot(*P2, linestyle="none", marker="o",
                    markerfacecolor=mfc, markeredgecolor=mfc, markeredgewidth=mew, zorder = 10)

        # r-constant line: endpoints (0, 1-r0, r0) and (1-r0, 0, r0); handle at t=0 side
        if r0 < 1.0:
            P1 = (0.0, 1.0 - r0, r0)
            P2 = (1.0 - r0, 0.0, r0)
            ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]],
                    lw=crosshair_linewidth, alpha=crosshair_alpha, color='black')
            ax.plot(*P1, linestyle="none", marker="o", 
                    markerfacecolor=mfc, markeredgecolor=mfc, markeredgewidth=mew, zorder = 10)

    return {"index": subset.index, "mu": mu, "Sigma": Sigma, "boundary": B, "centroid_abc": c_abc}



def ternary_scatter_with_ilr_bands(
    ax,
    c1, c2,                              # iterables of length 3 in (t,l,r)
    band_color="#999999",
    gap_color="#000000",
    centroid_colors=("#0072B2", "#E69F00"),
    annotate=True,
    labels=None,                         # dict like {"t": "Δcarbs …", "l": "...", "r": "..."} or None
    at=0,                                # baseline along axis for annotate_axis_gap
    marker="D",
    markeredgecolor="black",
    markeredgewidth=0.8,
):
    """
    Draw constant bands for t,l,r between centroids c1 and c2, annotate gaps, and plot both centroids.
    Assumes mpltern axis order (t,l,r) and that c1/c2 are compositions summing to ax.get_ternary_sum().
    """
    # Ensure floats
    import numpy as np
    c1 = np.asarray(c1, dtype=float).reshape(3)
    c2 = np.asarray(c2, dtype=float).reshape(3)

    # Determine scale to format labels (1.0 or 100.0 typical)
    try:
        scale = float(ax.get_ternary_sum())
    except Exception:
        scale = 1.0

    # Helper to format default labels in percentage points if scale in {1,100}
    def _default_label(axis_key, v1, v2):
        if labels and axis_key in labels:
            return labels[axis_key]
        if np.isclose(scale, 1.0):
            delta_pp = (v2 - v1) * 100.0
        else:
            delta_pp = (v2 - v1)           # already in percent-sum space
        sign = "\u0394"
        return f"{sign}{axis_key}: {delta_pp:+.1f} pp"

    # Draw bands and optional annotations for each axis
    axes = ("t", "l", "r")
    vals1 = {"t": c1[0], "l": c1[1], "r": c1[2]}
    vals2 = {"t": c2[0], "l": c2[1], "r": c2[2]}
    for k in axes:
        draw_constant_band(ax, k, vals1[k], vals2[k], color=band_color)
        if annotate:
            annotate_axis_gap(ax, k, vals1[k], vals2[k],
                                   at=at, color=gap_color,
                                   label=_default_label(k, vals1[k], vals2[k]))

    # Plot centroids as diamonds
    ax.plot(c1[0], c1[1], c1[2],
            linestyle="none", marker=marker,
            markerfacecolor=centroid_colors[0],
            markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)
    ax.plot(c2[0], c2[1], c2[2],
            linestyle="none", marker=marker,
            markerfacecolor=centroid_colors[1],
            markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)

    # Return deltas for downstream use
    return {
        "delta_t": vals2["t"] - vals1["t"],
        "delta_l": vals2["l"] - vals1["l"],
        "delta_r": vals2["r"] - vals1["r"],
        "scale": scale,
    }


def setup_ternary_axis(ax, major_step, minor_step=None, draw_grid=True, draw_subgrid=True, show_labels=True):
    def _norm_step(p):
        if p is None: return None
        p = float(p)
        if p <= 0: raise ValueError("Tick step must be positive.")
        if p >= 1.0: p /= 100.0
        if not (0.0 < p < 1.0): raise ValueError("Tick step must be in (0,1) or (1,100].")
        return p

    smaj = _norm_step(major_step)
    smin = _norm_step(minor_step) if minor_step is not None else None

    try:
        total = float(ax.get_ternary_sum())
    except Exception:
        total = 1.0

    # Major ticks
    tmaj_f = np.round(np.arange(smaj, 1.0, smaj), 10)
    tmaj   = tmaj_f * total
    labels = [f"{int(round(f*100))}%" for f in tmaj_f] if show_labels else []

    for axis in (ax.taxis, ax.laxis, ax.raxis):
        axis.set_ticks(tmaj, minor=False)
        axis.set_ticklabels(labels, minor=False)

    # Minor ticks (no labels)
    sub_ok = False
    if smin is not None and draw_subgrid:
        tmin_f = np.round(np.arange(smin, 1.0, smin), 10)
        tmin   = tmin_f * total
        try:
            for axis in (ax.taxis, ax.laxis, ax.raxis):
                axis.set_ticks(tmin, minor=True)
                axis.set_ticklabels([], minor=True)
            sub_ok = True
        except Exception:
            sub_ok = False  # older mpltern: no minor ticks

    # Grid control
    if draw_grid:
        ax.grid(True, which='major', linestyle = ':', linewidth = 0.8)
        if sub_ok:
            ax.grid(True, which='minor', linestyle = ':', linewidth = 0.6)
    else:
        ax.grid(False)

    return tmaj


def ilr_scatter_with_ilr_ellipse(
    ax,
    df,
    tag_pattern,
    tag_col="tags",
    ilr_cols=("e_z1","e_z2"),
    color="#E69F00",
    s=1,
    n_std=2.146,            # 90% area for 2D Gaussian
    centroid_marker="D",
    centroid_mec="black",
):
    """
    Plot ILR scatter + confidence ellipse + centroid for ONE tag on an existing Axes.
    Returns { "index", "mu", "Sigma" } for that tag.
    """
    cols = list(ilr_cols)
    sub = df[df[tag_col].str.contains(tag_pattern, case=False, na=False)][cols].dropna()
    if sub.empty:
        return {"index": pd.Index([]), "mu": np.array([np.nan, np.nan]), "Sigma": np.full((2,2), np.nan)}

    x = sub[ilr_cols[0]].astype(float).to_numpy()
    y = sub[ilr_cols[1]].astype(float).to_numpy()

    # scatter
    ax.scatter(x, y, s=s, c=color, label=str(tag_pattern))

    # ellipse 
    confidence_ellipse(x, y, ax, edgecolor=color, n_std=n_std)

    # centroid (ILR mean)
    mu = np.array([x.mean(), y.mean()], dtype=float)
    ax.plot(mu[0], mu[1], linestyle="none",
            marker=centroid_marker, markerfacecolor=color, markeredgecolor=centroid_mec)

    # covariance
    Sigma = np.cov(np.c_[x, y], rowvar=False)

    return {"index": sub.index, "mu": mu, "Sigma": Sigma}