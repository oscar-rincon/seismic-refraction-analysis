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


def plot_tomogram(manager, ax=None, width=7.0, height=None, fontsize=7,
                  label_size=None, tick_size=None,
                  v_exag=2.0, cbar_thickness=0.12, xlim=None,
                  cmap="viridis", v_min=None, v_max=None, show_mesh=False,
                  mesh_color="k", mesh_alpha=0.3, mesh_width=0.2,
                  rays=True, ray_color="w", ray_alpha=0.3, ray_width=0.5,
                  axis_color="#7f7f7f", label_color="k", cbar_label=r"$V_p$ (m/s)",
                  xlabel="Position along line (m)", ylabel="Depth (m)",
                  title=None):
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
    and `mesh_width` (pt). Returns (ax, cbar).
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
    pg.show(manager.paraDomain, velocity, ax=ax, cMap=cmap, cMin=v_min,
            cMax=v_max, logScale=False, showMesh=False, colorBar=False)
    mappable = next(c for c in ax.collections if hasattr(c, "get_array")
                    and c.get_array() is not None and len(c.get_array()) == len(velocity))
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
                              top=1 - 0.1 / fh, bottom=0.9 / fh)
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
