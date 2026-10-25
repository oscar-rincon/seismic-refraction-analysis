"""Plotting helpers: wiggle plots of shot gathers and traveltime curves."""

import matplotlib.pyplot as plt
import numpy as np


def plot_wiggle(gather, ax=None, t_max=None, scale=0.45, fill_color="#9ecae1",
                fill_alpha=0.8, line_color="#4d4d4d", line_width=0.4,
                pick_color="#08306b", pick_size=7, source_size=5,
                axis_color="#7f7f7f", label_color="k", grid_color="#e5e5e5",
                picks=None, title=None, fontsize=7, legend=True,
                legend_loc="lower right"):
    """Wiggle plot of one shot gather (trace-normalized).

    `picks` is an optional array of first-break times (s), one per receiver,
    drawn as short horizontal markers. `t_max` limits the displayed time
    window (s).

    Style knobs: `fill_color`/`fill_alpha` (positive lobes), `line_color`/
    `line_width` (wiggle trace), `pick_color`/`pick_size`/`source_size`
    (markers), `axis_color` (spines, tick marks and tick values),
    `label_color` (axis labels and title), `grid_color` (None disables the
    grid), `fontsize` in pt (title is +1, tick values -1). Set
    `legend=False` when the caller draws a shared legend instead.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    t_ms = gather.time * 1e3
    trace_spacing = np.min(np.diff(gather.receiver_x)) if gather.n_receivers > 1 else 1.0

    for i, x in enumerate(gather.receiver_x):
        trace = gather.data[:, i]
        trace = trace / (np.max(np.abs(trace)) + 1e-12) * scale * trace_spacing
        ax.plot(x + trace, t_ms, color=line_color, lw=line_width)
        ax.fill_betweenx(t_ms, x, x + trace, where=trace > 0,
                         color=fill_color, alpha=fill_alpha, lw=0)

    if picks is not None:
        ax.plot(gather.receiver_x, np.asarray(picks) * 1e3, "_",
                color=pick_color, markersize=pick_size, markeredgewidth=1.2,
                label="First-break picks")
    ax.plot(gather.source_x, 0, "v", color=pick_color, markersize=source_size,
            clip_on=False, label="Source")
    if legend:
        ax.legend(loc=legend_loc, fontsize=fontsize - 1, frameon=False)

    ax.set_xlabel("Position along line (m)", fontsize=fontsize, color=label_color)
    ax.set_ylabel("Time (ms)", fontsize=fontsize, color=label_color)
    ax.set_title(title or f"Source at {gather.source_x:+.1f} m",
                 fontsize=fontsize + 1, color=label_color)
    ax.tick_params(labelsize=fontsize - 1, length=2, width=0.5,
                   color=axis_color, labelcolor=axis_color)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(axis_color)
    if grid_color is not None:
        ax.grid(True, color=grid_color, lw=0.4)
        ax.set_axisbelow(True)
    if t_max is not None:
        ax.set_ylim(0, t_max * 1e3)
    ax.invert_yaxis()
    return ax


def plot_all_gathers(gathers, t_max=None, scale=0.45, picks_per_shot=None,
                     ncols=3, fontsize=7, width=7.0, panel_aspect=0.9,
                     legend_loc="center", **wiggle_kwargs):
    """Wiggle panels for every shot gather on an `ncols`-wide grid.

    The figure is `width` inches wide; each panel is `panel_aspect` times as
    tall as it is wide. A single legend is drawn in the first empty grid slot
    (if any, at `legend_loc` inside that slot), otherwise above the grid.
    Any extra keyword argument (e.g. `fill_color`, `line_color`, `pick_color`,
    `axis_color`, `label_color`, `grid_color`) is passed to `plot_wiggle`.
    """
    n = len(gathers)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    panel_w = width / ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width, panel_aspect * panel_w * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for i, (gather, ax) in enumerate(zip(gathers, flat)):
        picks = picks_per_shot[i] if picks_per_shot is not None else None
        plot_wiggle(gather, ax=ax, t_max=t_max, scale=scale, picks=picks,
                    fontsize=fontsize, legend=False, **wiggle_kwargs)
        if i % ncols:
            ax.set_ylabel("")
        if i < n - ncols:
            ax.set_xlabel("")
        else:  # bottom-most panel of its column: show its tick labels
            ax.tick_params(labelbottom=True)

    handles, labels = flat[0].get_legend_handles_labels()
    if n < len(flat):  # use the first empty slot for the legend
        spare = flat[n]
        spare.axis("off")
        for ax in flat[n + 1:]:
            ax.set_visible(False)
        spare.legend(handles, labels, loc=legend_loc, fontsize=fontsize,
                     frameon=False)
        fig.tight_layout()
    else:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   fontsize=fontsize, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_pick_review(gathers, original_picks, selected_picks, review, width=13.0):
    """Draw one enlarged raw-trace figure per shot with explicit review status."""
    figures = []
    for shot, gather in enumerate(gathers):
        ncols = min(3, gather.n_receivers)
        nrows = int(np.ceil(gather.n_receivers / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, 2.6 * nrows),
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f"Source {gather.source_x:g} m: first-arrival review")
        for receiver, ax in enumerate(axes.ravel()):
            if receiver >= gather.n_receivers:
                ax.set_visible(False)
                continue
            record = review[(review.shot_index == shot)
                            & (review.receiver_index == receiver)].iloc[0]
            old = original_picks[shot][receiver]
            new = selected_picks[shot][receiver]
            times = [t for t in (old, new, record.candidate_ms / 1000) if np.isfinite(t)]
            lo, hi = max(0, min(times) - 0.010), max(times) + 0.015
            mask = (gather.time >= lo) & (gather.time <= hi)
            trace = gather.data[mask, receiver]
            scale = max(np.max(np.abs(trace)), np.finfo(float).tiny)
            ax.plot(gather.time[mask] * 1000, trace / scale, color="black", lw=0.8)
            ax.axvline(old * 1000, color="0.5", ls=":", label="Original")
            if np.isfinite(new):
                color = "darkorange" if record.needs_review else "#0072b2"
                ax.axvline(new * 1000, color=color, label="Selected")
            if record.needs_review and np.isfinite(record.candidate_ms):
                ax.axvline(record.candidate_ms, color="purple", ls="--",
                           label="Unaccepted AIC")
            ax.set(title=f"({shot}, {receiver}) receiver {gather.receiver_x[receiver]:g} m",
                   xlabel="Time (ms)", ylabel="Normalized amplitude")
            status = record.review_flags if record.needs_review else record.selection_method
            ax.text(0.02, 0.97, status.replace("; ", "\n"), transform=ax.transAxes,
                    va="top", fontsize=7, color="firebrick" if record.needs_review else "#0072b2",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(alpha=0.2)
        figures.append(fig)
    return figures


def plot_traveltime_curves(gathers, picks_per_shot, ax=None, width=7.0,
                           height=3.5, fontsize=7, colors=None,
                           line_width=0.8, marker="o", marker_size=3,
                           source_marker="s", source_size=6,
                           axis_color="#7f7f7f", label_color="k",
                           grid_color="#e5e5e5", legend_loc="best",
                           title=None):
    """Traveltime (first-break) curves of every shot on a common axis.

    Sources are drawn as `source_marker` on the zero-time line in the same
    color as their curve. `colors` defaults to tab10. `axis_color` styles
    spines, tick marks and tick values; `label_color` the axis labels.
    `grid_color=None` disables the grid; `title=None` draws no title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(width, height))

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (gather, picks) in enumerate(zip(gathers, picks_per_shot)):
        c = colors[i % len(colors)]
        ax.plot(gather.receiver_x, np.asarray(picks) * 1e3, marker=marker,
                ls="-", lw=line_width, color=c, ms=marker_size,
                label=f"Source at {gather.source_x:+.1f} m")
        ax.plot(gather.source_x, 0, source_marker, color=c,
                markersize=source_size, clip_on=False)

    ax.set_xlabel("Position along line (m)", fontsize=fontsize, color=label_color)
    ax.set_ylabel("First-break traveltime (ms)", fontsize=fontsize,
                  color=label_color)
    if title:
        ax.set_title(title, fontsize=fontsize + 1, color=label_color)
    ax.tick_params(labelsize=fontsize - 1, length=2, width=0.5,
                   color=axis_color, labelcolor=axis_color)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(axis_color)
    if grid_color is not None:
        ax.grid(True, color=grid_color, lw=0.4)
        ax.set_axisbelow(True)
    ax.legend(loc=legend_loc, fontsize=fontsize - 1, frameon=False)
    return ax


def _tomogram_node_values(manager):
    """Interpolate cell velocities to mesh nodes for display only."""
    from matplotlib.tri import Triangulation

    mesh = manager.paraDomain
    velocity = np.asarray(manager.velocity)
    if any(cell.nodeCount() != 3 for cell in mesh.cells()):
        raise ValueError("Interpolated tomograms require a triangular 2D mesh.")
    if len(velocity) != mesh.cellCount() or not np.all(np.isfinite(velocity)):
        raise ValueError("Each mesh cell requires a finite velocity.")
    nodes = np.array([(node.x(), node.y()) for node in mesh.nodes()])
    triangles = np.array([[node.id() for node in cell.nodes()]
                          for cell in mesh.cells()])
    areas = np.array([cell.size() for cell in mesh.cells()])
    weighted = np.zeros(mesh.nodeCount())
    weights = np.zeros(mesh.nodeCount())
    for corner in range(3):
        np.add.at(weighted, triangles[:, corner], velocity * areas)
        np.add.at(weights, triangles[:, corner], areas)
    nodal_velocity = np.divide(weighted, weights, out=np.zeros_like(weighted),
                               where=weights > 0)
    return Triangulation(nodes[:, 0], nodes[:, 1], triangles), nodal_velocity


def plot_traveltime_fit(manager, data, ax=None, width=9.0, height=3.8):
    """Plot observed arrival points and connected model predictions by source.

    Predictions must correspond to the exact row order of the fitted data.
    Only retained observations are drawn; curves connect receiver positions,
    not a continuously sampled forward calculation.
    """
    observed = np.asarray(data["t"])
    predicted = np.asarray(manager.inv.response)
    if (observed.size == 0 or predicted.shape != observed.shape
            or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted))):
        raise ValueError("Require a finite model prediction for each observed arrival.")
    positions = np.array([p.x() for p in data.sensorPositions()])
    sources = positions[np.asarray(data["s"], dtype=int)]
    receivers = positions[np.asarray(data["g"], dtype=int)]
    if ax is None:
        _, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    for i, source in enumerate(np.unique(sources)):
        rows = np.flatnonzero(sources == source)
        rows = rows[np.argsort(receivers[rows])]
        color = plt.get_cmap("tab10")(i % 10)
        ax.plot(receivers[rows], predicted[rows] * 1000, color=color, lw=1.4,
                label=f"Source {source:+g} m")
        ax.scatter(receivers[rows], observed[rows] * 1000, color=color,
                   s=22, edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot(source, 0, "s", color=color, ms=5)
    from matplotlib.lines import Line2D
    source_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=source_handles + [
        Line2D([], [], color="0.3", lw=1.4, label="Model prediction"),
        Line2D([], [], color="0.3", marker="o", ls="", label="Observed pick"),
        Line2D([], [], color="0.3", marker="s", ls="", label="Source"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set(xlabel="Position along line (m)", ylabel="First-break traveltime (ms)",
           title="Observed picks and connected model predictions")
    ax.grid(alpha=0.2)
    return ax


def plot_velocity_contours(manager, levels, ax=None, width=9.0, height=4.0,
                           xlim=None, cmap="viridis"):
    """Display velocity intervals with labeled isolines on the actual mesh.

    Ranges are velocity display bins, not material classes or resolved layers.
    """
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import FuncFormatter

    levels = np.asarray(levels, dtype=float)
    if (levels.ndim != 1 or len(levels) < 2 or not np.all(np.isfinite(levels))
            or np.any(np.diff(levels) <= 0)):
        raise ValueError("Contour boundaries must be finite and strictly increasing.")
    triangulation, values = _tomogram_node_values(manager)
    if ax is None:
        _, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    colors = plt.get_cmap(cmap, len(levels) - 1)
    filled = ax.tricontourf(triangulation, values, levels=levels, cmap=colors,
                            norm=BoundaryNorm(levels, colors.N), extend="both")
    interior = levels[(levels > values.min()) & (levels < values.max())]
    if len(interior):
        ax.tricontour(triangulation, values, levels=interior,
                      colors="0.2", linewidths=0.65)
    colorbar = ax.figure.colorbar(filled, ax=ax, orientation="horizontal",
                                  pad=0.18, fraction=0.09, ticks=levels,
                                  spacing="proportional")
    colorbar.set_label("Vp (m/s)")
    ax.set(xlabel="Distance (m)", ylabel="Depth (m)")
    ax.set_ylim(manager.paraDomain.yMin(), 0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):g}"))
    if xlim is not None:
        ax.set_xlim(*xlim)
    return ax, colorbar


VP_CLASS_EDGES = [500.0, 1000.0, 1500.0]  # m/s
VP_CLASS_COLORS = ["#eadbb8", "#c8a06e", "#8c6a4a", "#7d8186"]
VP_CLASS_LABELS = ["< 500", "500–1000", "1000–1500", "≥ 1500"]


def plot_material_classes(manager, ax=None, width=11.0, height=5.0, xlim=None):
    """Plot indicative Vp classes, not confirmed lithology or Vs site classes.

    Boundaries use the same nodal interpolation as the velocity contour plot.
    Exactly 500, 1000 and 1500 m/s belong to the next higher class.
    Colors and thresholds stay fixed across models, including absent classes.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = VP_CLASS_COLORS
    labels = VP_CLASS_LABELS
    velocity = np.asarray(manager.velocity)
    if velocity.size == 0 or not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
        raise ValueError("Material-class display requires positive finite velocities.")
    upper = max(2000.0, 500.0 * np.ceil(velocity.max() / 500.0))
    ax, colorbar = plot_velocity_contours(
        manager, [0, 500, 1000, 1500, upper], ax=ax,
        width=width, height=height, xlim=xlim, cmap=ListedColormap(colors))
    colorbar.remove()
    ax.legend(
        handles=[Patch(facecolor=color, edgecolor="0.3", label=label)
                 for color, label in zip(colors, labels)],
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
        frameon=False, fontsize=9, title="Vp (m/s)", title_fontsize=9)
    return ax


def vp_class_layers(depth, vp):
    """Merge a Vp(z) profile into contiguous Vp-class layers.

    Returns a list of dicts with top_m, bottom_m, vp_mean_m_s and class index;
    boundaries sit midway between the last and first sample of adjacent layers.
    """
    depth, vp = np.asarray(depth, float), np.asarray(vp, float)
    classes = np.searchsorted(VP_CLASS_EDGES, vp, side="right")
    layers, start = [], 0
    for i in range(1, len(depth) + 1):
        if i == len(depth) or classes[i] != classes[start]:
            layers.append({"top_m": depth[start], "bottom_m": depth[i - 1],
                           "vp_mean_m_s": vp[start:i].mean(), "class": int(classes[start])})
            start = i
    for upper, lower in zip(layers[:-1], layers[1:]):
        upper["bottom_m"] = lower["top_m"] = (upper["bottom_m"] + lower["top_m"]) / 2
    return layers


def plot_1d_profile(vp_profile, vs_profile=None, width=6.5, height=5.0):
    """Laterally averaged Vp(z), MASW Vs(z) and the Vp-class column.

    `vp_profile` needs depth_m and vp_m_s; optional vp_p10_m_s / vp_p90_m_s
    are drawn as the lateral spread. Layers come from the averaged Vp with
    the same classes and colors as `plot_material_classes`.
    """
    z = vp_profile["depth_m"].to_numpy()
    vp = vp_profile["vp_m_s"].to_numpy()
    layers = vp_class_layers(z, vp)

    fig, (ax_v, ax_m) = plt.subplots(
        1, 2, figsize=(width, height), sharey=True,
        gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.05})
    if {"vp_p10_m_s", "vp_p90_m_s"} <= set(vp_profile.columns):
        ax_v.fill_betweenx(z, vp_profile["vp_p10_m_s"], vp_profile["vp_p90_m_s"],
                           color="#8c6a4a", alpha=0.2, lw=0, label="Vp P10–P90")
    ax_v.plot(vp, z, color="#5a3d26", lw=2, label="Vp mean")
    if vs_profile is not None:
        ax_v.plot(vs_profile["vs_m_s"], vs_profile["depth_m"], color="0.35",
                  lw=1.6, ls="--", label="Vs (MASW)")
    ax_v.set_xlabel("Velocity (m/s)")
    ax_v.set_ylabel("Depth (m)")
    ax_v.set_xlim(left=0)
    ax_v.grid(alpha=0.3)
    ax_v.legend(fontsize=8, loc="lower left", frameon=False)

    for layer in layers:
        ax_m.axhspan(layer["top_m"], layer["bottom_m"],
                     color=VP_CLASS_COLORS[layer["class"]], ec="0.3", lw=0.6)
        ax_m.text(0.5, (layer["top_m"] + layer["bottom_m"]) / 2,
                  VP_CLASS_LABELS[layer["class"]], ha="center", va="center", fontsize=8,
                  color="white" if layer["class"] >= 2 else "black")
    for layer in layers[:-1]:
        ax_v.axhline(layer["bottom_m"], color="0.4", ls=":", lw=0.9)
        ax_m.text(1.04, layer["bottom_m"], f'{layer["bottom_m"]:.1f} m',
                  transform=ax_m.get_yaxis_transform(), va="center", fontsize=8)
    ax_m.set_xticks([])
    ax_m.set_xlabel("Vp (m/s)")
    ax_v.set_ylim(z.max(), min(z.min(), 0))
    return fig, layers


def plot_tomogram(manager, ax=None, width=7.0, height=None, fontsize=7,
                  label_size=None, tick_size=None,
                  v_exag=2.0, cbar_thickness=0.12, xlim=None,
                  cmap="viridis", v_min=None, v_max=None, show_mesh=False,
                  mesh_color="k", mesh_alpha=0.3, mesh_width=0.2,
                  rays=True, ray_color="w", ray_alpha=0.3, ray_width=0.5,
                  axis_color="#7f7f7f", label_color="k", cbar_label=r"$V_p$ (m/s)",
                  xlabel="Position along line (m)", ylabel="Depth (m)",
                  title=None, smooth=False):
    """Styled pyGIMLi traveltime tomogram with a horizontal colorbar.

    `manager` is the fitted TravelTimeManager. `v_min`/`v_max` clip the
    color scale (defaults: model range rounded to 50 m/s). `height=None`
    sizes the figure from the model aspect ratio (equal axes) plus room for
    the labels and colorbar. `rays=True` overlays the ray paths;
    `xlim=(x0, x1)` crops the distance axis (e.g. to the sensor range).
    `v_exag` stretches the depth axis relative to the distance axis (1 = true
    scale) and drives the auto height; `cbar_thickness` is the colorbar
    height in inches. `label_size` (axis and colorbar labels) and `tick_size` (tick values)
    in pt default to `fontsize` and `fontsize - 1`; the title uses
    `fontsize + 1`. `show_mesh=True` draws the cell edges with `mesh_color`, `mesh_alpha`
    and `mesh_width` (pt). `smooth=True` displays area-weighted node velocities
    with Gouraud interpolation on the original triangular mesh. This changes
    only the display, not the cell model or its resolution. Returns (ax, cbar).
    """
    import pygimli as pg
    from matplotlib.ticker import MaxNLocator

    label_size = fontsize if label_size is None else label_size
    tick_size = fontsize - 1 if tick_size is None else tick_size

    if ax is None:
        if height is None:
            bb = manager.paraDomain.boundingBox()
            x_range = bb.max()[0] - bb.min()[0]
            z_range = bb.max()[1] - bb.min()[1]
            height = 0.9 * width * v_exag * z_range / x_range + 1.0
        fig, ax = plt.subplots(figsize=(width, height))

    velocity = np.asarray(manager.velocity)
    v_min = 50 * np.floor(velocity.min() / 50) if v_min is None else v_min
    v_max = 50 * np.ceil(velocity.max() / 50) if v_max is None else v_max
    if smooth:
        triangulation, nodal_velocity = _tomogram_node_values(manager)
        mappable = ax.tripcolor(
            triangulation, nodal_velocity, shading="gouraud", cmap=cmap,
            vmin=v_min, vmax=v_max, edgecolors="none", antialiased=False)
    else:
        pg.show(manager.paraDomain, velocity, ax=ax, cMap=cmap, cMin=v_min,
                cMax=v_max, logScale=False, showMesh=False, colorBar=False)
        mappable = next(c for c in ax.collections if hasattr(c, "get_array")
                        and c.get_array() is not None and len(c.get_array()) == len(velocity))
        mappable.set_edgecolor("none")
        mappable.set_antialiased(False)
    if show_mesh:
        from matplotlib.collections import PolyCollection
        mesh = manager.paraDomain
        polys = [[(n.x(), n.y()) for n in cell.nodes()] for cell in mesh.cells()]
        ax.add_collection(PolyCollection(
            polys, facecolors="none", edgecolors=mesh_color,
            linewidths=mesh_width, alpha=mesh_alpha))
    if rays:
        manager.drawRayPaths(ax=ax, color=ray_color, alpha=ray_alpha, lw=ray_width)
    mappable.set_clim(v_min, v_max)
    ax.set_aspect("auto")
    if xlim is not None:
        ax.set_xlim(*xlim)
    # fixed margins (inches): room below for x label, colorbar and its label
    fh, fw = ax.figure.get_figheight(), ax.figure.get_figwidth()
    ax.figure.subplots_adjust(left=0.6 / fw, right=1 - 0.15 / fw,
                              top=1 - (0.4 if title else 0.1) / fh, bottom=0.9 / fh)
    ax.figure.canvas.draw()  # fix the axes position before placing the colorbar
    pos = ax.get_position()
    fig_h = ax.figure.get_figheight()
    cax = ax.figure.add_axes([pos.x0 + 0.2 * pos.width,
                              pos.y0 - (label_size / 72 + 0.42) / fig_h,
                              0.6 * pos.width, cbar_thickness / fig_h])
    cbar = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))

    ax.set_xlabel(xlabel, fontsize=label_size, color=label_color)
    ax.set_ylabel(ylabel, fontsize=label_size, color=label_color)
    if title:
        ax.set_title(title, fontsize=fontsize + 1, color=label_color)
    ax.tick_params(labelsize=tick_size, length=2, width=0.5,
                   color=axis_color, labelcolor=axis_color)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(axis_color)

    cbar.set_label(cbar_label, fontsize=label_size, color=label_color)
    cbar.ax.tick_params(labelsize=tick_size, length=2, width=0.5,
                        color=axis_color, labelcolor=axis_color)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(axis_color)
    return ax, cbar
