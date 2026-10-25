"""Refraction (RF) processing: first-break picking, layered interpretation
and (optional) 2D traveltime tomography with pyGIMLi.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# First-break picking
# ----------------------------------------------------------------------

def pick_first_breaks(gather, v_min=150.0, v_max=1500.0,
                      noise_window=0.01, threshold=5.0):
    """Offset-guided automatic first-break picks for one shot gather.

    Receivers are processed outward from the source. Traveltimes must
    increase with offset, and the move-out between neighboring receivers
    is bounded by the plausible velocity range [v_min, v_max] (m/s); the
    pick on each trace is therefore only searched inside the time window
    the previous pick allows. This keeps low-SNR far-offset traces from
    triggering on noise.

    Within the window, the first break is the earliest sample exceeding
    `threshold` times the trace noise RMS (estimated from the first
    `noise_window` seconds), refined back to the signal onset. If nothing
    exceeds the threshold, the strongest sample in the window is used.

    For a soft site (e.g. grass/soccer field) the defaults are adequate:
    direct waves of ~200-300 m/s and refractors up to ~1500 m/s.
    Review the picks over the wiggle plots and correct individual ones
    via the `overrides` of `pick_line`.
    Returns an array of pick times (s), one per receiver.
    """
    n_noise = max(int(noise_window / gather.dt), 8)
    picks = np.full(gather.n_receivers, np.nan)

    t_prev = None
    offset_prev = None
    for i in np.argsort(gather.offsets):
        trace = gather.data[:, i] - np.mean(gather.data[:n_noise, i])
        noise_rms = np.std(trace[:n_noise])
        offset = gather.offsets[i]

        if t_prev is None:
            t_lo, t_hi = offset / v_max, offset / v_min
        else:
            t_lo = t_prev + (offset - offset_prev) / v_max
            t_hi = t_prev + (offset - offset_prev) / v_min
        j_lo = int(t_lo / gather.dt)
        j_hi = min(int(t_hi / gather.dt) + 1, gather.n_samples)

        window = np.abs(trace[j_lo:j_hi])
        above = np.nonzero(window > threshold * noise_rms)[0]
        j = j_lo + (above[0] if len(above) else int(np.argmax(window)))

        # Refine: walk back to the preceding energy minimum (onset).
        energy = trace ** 2
        while j > 0 and energy[j - 1] < energy[j]:
            j -= 1

        picks[i] = j * gather.dt
        t_prev, offset_prev = picks[i], offset
    return picks


def pick_line(gathers, overrides=None, **kwargs):
    """Pick first breaks for every shot gather of the line.

    `overrides` fixes individual bad autopicks manually:
        {(shot_index, receiver_index): time_in_seconds, ...}
    Keyword arguments are forwarded to `pick_first_breaks`.
    Returns a list of pick arrays (s), one per shot.
    """
    picks_per_shot = [pick_first_breaks(g, **kwargs) for g in gathers]
    for (i_shot, i_rec), t in (overrides or {}).items():
        picks_per_shot[i_shot][i_rec] = t
    return picks_per_shot


def picks_table(gathers, picks_per_shot):
    """Picks as a DataFrame (rows: receivers, columns: shots, values: ms)."""
    table = {
        f"source {g.source_x:+.1f} m": np.round(np.asarray(p) * 1e3, 2)
        for g, p in zip(gathers, picks_per_shot)
    }
    return pd.DataFrame(table, index=[f"{x:.1f} m" for x in gathers[0].receiver_x])


# ----------------------------------------------------------------------
# Apparent velocities and two-layer interpretation
# ----------------------------------------------------------------------

def apparent_velocities(gathers, picks_per_shot):
    """Apparent velocity of the first arrivals for every shot and direction.

    For each shot, the picks on each side of the source are fit with a
    straight line t(offset); the inverse slope is the apparent velocity.
    Asymmetry between shots/directions indicates dipping layers or lateral
    changes. Returns a DataFrame with one row per shot side.
    """
    rows = []
    for gather, picks in zip(gathers, picks_per_shot):
        picks = np.asarray(picks)
        for side, mask in [("left", gather.receiver_x < gather.source_x),
                           ("right", gather.receiver_x > gather.source_x)]:
            mask = mask & ~np.isnan(picks)
            if mask.sum() < 3:
                continue
            slope, intercept = np.polyfit(gather.offsets[mask], picks[mask], 1)
            rows.append({
                "source_x_m": gather.source_x,
                "side": side,
                "n_picks": int(mask.sum()),
                "v_apparent_m_s": round(1.0 / slope, 1),
                "intercept_ms": round(intercept * 1e3, 2),
            })
    return pd.DataFrame(rows)


def two_layer_interpretation(gather, picks, n_direct=4):
    """Intercept-time interpretation of one off-end shot.

    NOTE: this requires the direct wave to be visible on the first
    `n_direct` receivers. With wide geophone spacing on soft ground
    (e.g. 5 m spacing on a soccer field) the crossover distance can be
    shorter than the nearest offset, making v1 unobservable - check that
    the returned v1 < v2 before trusting the result.

    The first `n_direct` picks (nearest receivers) are fit as the direct
    wave (slope = 1/v1); the remaining picks as the refracted head wave
    (slope = 1/v2, intercept ti). Refractor depth below the source:
        z = ti * v1 * v2 / (2 * sqrt(v2**2 - v1**2))
    Returns a dict with v1, v2, crossover distance, intercept time, depth.
    """
    picks = np.asarray(picks)
    valid = ~np.isnan(picks)
    offsets = gather.offsets[valid]
    order = np.argsort(offsets)
    x, t = offsets[order], picks[valid][order]

    slope1, intercept1 = np.polyfit(x[:n_direct], t[:n_direct], 1)
    slope2, intercept2 = np.polyfit(x[n_direct:], t[n_direct:], 1)
    v1, v2 = 1.0 / slope1, 1.0 / slope2
    x_cross = (intercept2 - intercept1) / (slope1 - slope2)
    depth = intercept2 * v1 * v2 / (2.0 * np.sqrt(v2**2 - v1**2))

    return {
        "source_x": gather.source_x,
        "v1_m_s": v1,
        "v2_m_s": v2,
        "crossover_m": x_cross,
        "intercept_s": intercept2,
        "depth_m": depth,
        # Fit lines for plotting: t(x) for direct and refracted branches
        "fit_direct": (slope1, intercept1),
        "fit_refracted": (slope2, intercept2),
    }


def drop_outlier_picks(gathers, picks_per_shot, manager, data, max_residual=0.010):
    """Discard picks that a first inversion cannot explain.

    Compares every pick with the traveltime computed from the inverted
    model (`manager.inv.response`) and sets picks whose absolute residual
    exceeds `max_residual` (s) to NaN. Use a generous threshold so only
    gross picking errors are removed, then re-run the inversion on the
    returned picks.
    """
    residuals = np.asarray(data["t"]) - np.asarray(manager.inv.response)

    cleaned = [np.array(p, dtype=float) for p in picks_per_shot]
    k = 0  # residual index; picks were added shot by shot, NaNs skipped
    for gather, picks in zip(gathers, cleaned):
        for i, t in enumerate(picks):
            if np.isnan(t) or t <= 0:
                continue
            if np.abs(residuals[k]) > max_residual:
                print(f"Dropped pick: shot {gather.source_x:+.1f} m, "
                      f"receiver {gather.receiver_x[i]:.1f} m "
                      f"(residual {residuals[k] * 1e3:+.1f} ms)")
                picks[i] = np.nan
            k += 1
    return cleaned


# ----------------------------------------------------------------------
# 2D traveltime tomography (pyGIMLi, optional dependency)
# ----------------------------------------------------------------------

def build_traveltime_data(gathers, picks_per_shot, pick_error=0.001):
    """Combine all shots/picks into a pyGIMLi traveltime DataContainer."""
    import pygimli.physics.traveltime as tt

    # Unique sensor positions: every source and receiver along the line.
    positions = sorted(
        {float(g.source_x) for g in gathers}
        | {float(x) for g in gathers for x in g.receiver_x}
    )
    index_of = {x: i for i, x in enumerate(positions)}

    data = tt.DataContainerTT()
    for x in positions:
        data.createSensor([x, 0.0])

    shots, receivers, times = [], [], []
    for gather, picks in zip(gathers, picks_per_shot):
        for x_rec, t in zip(gather.receiver_x, picks):
            if np.isnan(t) or t <= 0:
                continue
            shots.append(index_of[gather.source_x])
            receivers.append(index_of[float(x_rec)])
            times.append(t)

    n = len(times)
    data.resize(n)
    data.set("s", np.array(shots, dtype=float))
    data.set("g", np.array(receivers, dtype=float))
    data.set("t", np.array(times))
    data.set("err", np.full(n, pick_error))
    data.set("valid", np.ones(n))
    print(f"Traveltime data: {data.sensorCount()} sensors, {n} picks")
    return data


def export_refrapy_picks(data, filename: str | Path) -> Path:
    """Write 2D picks in the strict Refrapick/Refrainv .sgt layout.

    Sensor indices are one-based on disk and traveltimes are in seconds.
    The GUI only reads s/g/t, so uncertainty and validity fields are not
    exported; only valid rows are included. Existing output is overwritten.
    """
    positions = np.array([[p.x(), p.y(), p.z()] for p in data.sensorPositions()])
    if positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("Refrapy export requires finite sensor coordinates.")
    if np.any(positions[:, 2] != 0):
        raise ValueError("Refrapy export requires 2D x/elevation coordinates, not 3D geometry.")
    valid = np.asarray(data["valid"]) != 0
    shots = np.asarray(data["s"])[valid]
    receivers = np.asarray(data["g"])[valid]
    times = np.asarray(data["t"])[valid]
    if not len(times) or not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("Refrapy export requires finite, positive valid pick times.")
    for indices in (shots, receivers):
        if (not np.all(np.isfinite(indices)) or np.any(indices != np.floor(indices))
                or np.any(indices < 0) or np.any(indices >= len(positions))):
            raise ValueError("Pick sensor indices must be integers within the sensor table.")

    filename = Path(filename)
    with filename.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"{len(positions)} # shot/geophone points\n#x y\n")
        for x, elevation, _ in positions:
            output.write(f"{x:.12g} {elevation:.12g}\n")
        output.write(f"{len(times)} # measurements\n#s g t\n")
        for shot, receiver, time in zip(shots, receivers, times):
            output.write(f"{int(shot) + 1} {int(receiver) + 1} {time:.12g}\n")
    return filename


def create_mesh(data, para_dx=0.5, para_depth=12.0, quality=33.5,
                max_cell_size=None, para_boundary=2.0):
    """Build the 2D triangular inversion mesh for the traveltime data.

    para_dx       : node refinement between sensors, RELATIVE to the sensor
                    spacing (1 = nodes only at sensors, 0.5 = one extra
                    node between sensors, 0.33 = two, ...)
    para_depth    : model depth (m); ~1/4 to 1/5 of the spread length is
                    the depth refraction data can actually illuminate
    quality       : minimum triangle angle (deg) for the mesh generator;
                    higher = nicer, more uniform triangles but more cells
                    (do not exceed ~34, the generator may not converge)
    max_cell_size : maximum triangle area (m^2) in the parameter region;
                    None lets the generator decide from the node spacing
    para_boundary : lateral margin around the sensors, in sensor spacings

    Returns the mesh; inspect it with pygimli.show(mesh, markers=True).
    """
    import pygimli.physics.traveltime as tt

    # Triangle (the mesh generator) is only guaranteed to terminate for
    # minimum angles up to ~34 deg; above that it can loop forever.
    if quality > 34.0:
        print(f"quality={quality} is too high for the mesh generator; "
              "clamping to 34.0")
        quality = 34.0

    manager = tt.TravelTimeManager(verbose=False)
    kwargs = dict(paraDX=para_dx, paraDepth=para_depth, quality=quality,
                  paraBoundary=para_boundary)
    if max_cell_size is not None:
        kwargs["paraMaxCellSize"] = max_cell_size
    mesh = manager.createMesh(data=data, **kwargs)
    print(f"Mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes "
          f"(paraDX={para_dx}, depth={para_depth} m, quality={quality})")
    return mesh


def invert_tomography(data, mesh=None, para_dx=0.5, para_depth=12.0,
                      quality=33.5, max_cell_size=None, lam=30.0,
                      z_weight=0.3, v_top=300.0, v_bottom=3000.0,
                      correlation_lengths=None, para_boundary=2.0):
    """Run pyGIMLi 2D refraction tomography; returns the fitted manager.

    Pass a `mesh` from `create_mesh` to control the mesh style explicitly;
    otherwise one is built here from para_dx/para_depth/quality/
    max_cell_size. z_weight < 1 favors horizontally layered models.
    `correlation_lengths=[Lx, Lz]` (m) switches to geostatistical
    regularization: Lx >> Lz forces smooth lateral continuity, i.e. a
    horizontally layered model. `para_boundary` is the lateral mesh margin
    around the sensors in sensor spacings (only used if no `mesh` is given).
    The resulting model is in `manager.paraDomain` / `manager.velocity`;
    plot it with `manager.showResult()` and `manager.drawRayPaths(ax)`.
    """
    import pygimli.physics.traveltime as tt

    if mesh is None:
        mesh = create_mesh(data, para_dx=para_dx, para_depth=para_depth,
                           quality=quality, max_cell_size=max_cell_size,
                           para_boundary=para_boundary)

    manager = tt.TravelTimeManager(verbose=False)
    extra = {}
    if correlation_lengths is not None:
        extra["correlationLengths"] = list(correlation_lengths)
    manager.invert(data=data, mesh=mesh, secNodes=2, lam=lam,
                   zWeight=z_weight, vTop=v_top, vBottom=v_bottom,
                   verbose=False, **extra)

    velocity = np.array(manager.velocity)
    print(f"Inversion finished: chi2 = {manager.inv.chi2():.2f}, "
          f"Vp = {velocity.min():.0f}-{velocity.max():.0f} m/s "
          f"(mean {velocity.mean():.0f} m/s)")
    return manager
