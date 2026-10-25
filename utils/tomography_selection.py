"""Select isotropic traveltime models by held-out arrival prediction."""

import numpy as np
import pandas as pd
import pygimli as pg
from pygimli.physics import TravelTimeManager

from utils.refraction import create_mesh


def spatial_folds(data, n_folds=3):
    """Interleave receiver locations; keep reciprocal measurements together."""
    if n_folds < 2:
        raise ValueError("At least two validation folds are required.")
    sources = np.asarray(data["s"], dtype=int)
    receivers = np.asarray(data["g"], dtype=int)
    pairs = sorted({tuple(sorted((s, g))) for s, g in zip(sources, receivers)})
    if len(pairs) < n_folds:
        raise ValueError("Not enough independent source/receiver pairs for the folds.")
    assignment = {pair: i % n_folds for i, pair in enumerate(pairs)}
    return np.array([assignment[tuple(sorted((s, g)))]
                     for s, g in zip(sources, receivers)])


def subset(data, mask):
    """Copy selected rows without changing the sensor table or input data."""
    selected = pg.DataContainer(data)
    selected.markInvalid(np.flatnonzero(~np.asarray(mask, dtype=bool)))
    selected.removeInvalid()
    return selected


def fit_model(data, mesh, lam, max_iter=12, z_weight=1.0):
    """Fit from a homogeneous start without velocity gradient or residual rejection.

    `z_weight` scales the vertical smoothness constraint: 1 is isotropic,
    values < 1 favour laterally continuous (horizontal) structure.
    """
    if not z_weight > 0:
        raise ValueError("z_weight must be positive.")
    sources = np.asarray(data["s"], dtype=int)
    receivers = np.asarray(data["g"], dtype=int)
    positions = np.array([[p.x(), p.y()] for p in data.sensorPositions()])
    distances = np.linalg.norm(positions[sources] - positions[receivers], axis=1)
    times = np.asarray(data["t"])
    apparent = distances / times
    initial_velocity = float(np.median(apparent[distances > 0]))
    if not np.isfinite(initial_velocity) or initial_velocity <= 0:
        raise ValueError("Cannot construct a positive starting velocity from the training data.")
    manager = TravelTimeManager(verbose=False)
    manager.invert(
        data=data, mesh=pg.Mesh(mesh), secNodes=2, lam=lam, zWeight=z_weight,
        vTop=initial_velocity, vBottom=initial_velocity,
        maxIter=max_iter, stopAtChi1=False, verbose=False,
    )
    velocity = np.asarray(manager.velocity)
    if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
        raise ValueError("Inversion returned non-finite or non-positive velocities.")
    return manager


def predict(manager, scheme):
    """Use a separate forward operator so the fitted manager is unchanged."""
    forward = TravelTimeManager(verbose=False)
    response = np.asarray(forward.simulate(
        mesh=manager.paraDomain, scheme=scheme, vel=manager.velocity,
        secNodes=2, returnArray=True, noiseLevel=0.0, noiseAbs=0.0,
    ))
    if response.shape != (scheme.size(),) or not np.all(np.isfinite(response)):
        raise ValueError("Forward calculation did not predict every requested arrival.")
    return response


def select_model(data, depths=(8.0, 12.0, 18.0),
                 lambdas=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 300.0),
                 n_folds=3, max_iter=12):
    """Rank a bounded search by pooled held-out RMS, then refit the winner.

    The same folds are used for all candidates. No observations are discarded.
    This is selection CV, not an independent estimate of final-model accuracy.
    """
    observed = np.asarray(data["t"])
    if (data.size() == 0 or not np.all(np.isfinite(observed))
            or np.any(observed <= 0) or np.any(np.asarray(data["valid"]) == 0)):
        raise ValueError("Selection requires nonempty, valid, positive finite arrivals.")
    if not depths or not lambdas or min(depths) <= 0 or min(lambdas) <= 0:
        raise ValueError("Supply positive candidate depths and regularization strengths.")
    folds = spatial_folds(data, n_folds)
    rows, predictions, meshes = [], {}, {}
    for depth in depths:
        meshes[depth] = create_mesh(
            data, para_dx=0.5, para_depth=depth, max_cell_size=2.0,
            para_boundary=0.5,
        )
        for lam in lambdas:
            held_out = np.full(data.size(), np.nan)
            fold_rms = []
            for fold in range(n_folds):
                test_mask = folds == fold
                train = subset(data, ~test_mask)
                test = subset(data, test_mask)
                manager = fit_model(train, meshes[depth], lam, max_iter=max_iter)
                held_out[test_mask] = predict(manager, test)
                fold_rms.append(float(np.sqrt(np.mean(
                    (observed[test_mask] - held_out[test_mask]) ** 2)) * 1000))
            rms = float(np.sqrt(np.mean((observed - held_out) ** 2)) * 1000)
            rows.append(dict(depth_m=depth, lam=lam, cv_rms_ms=rms,
                             cv_mae_ms=float(np.mean(np.abs(observed - held_out)) * 1000),
                             fold_rms_min_ms=min(fold_rms), fold_rms_max_ms=max(fold_rms)))
            predictions[(depth, lam)] = held_out
            print(f"depth={depth:g} m, lambda={lam:g}: held-out RMS={rms:.3f} ms", flush=True)
    ranking = pd.DataFrame(rows).sort_values(
        ["cv_rms_ms", "cv_mae_ms", "depth_m", "lam"]).reset_index(drop=True)
    best = ranking.iloc[0]
    depth, lam = float(best.depth_m), float(best.lam)
    manager = fit_model(data, meshes[depth], lam, max_iter=max_iter)
    final_prediction = predict(manager, data)
    np.testing.assert_allclose(final_prediction, np.asarray(manager.inv.response),
                               rtol=1e-5, atol=1e-8)
    positions = np.array([[p.x(), p.y()] for p in data.sensorPositions()])
    fit = pd.DataFrame(dict(
        source_x_m=positions[np.asarray(data["s"], dtype=int), 0],
        receiver_x_m=positions[np.asarray(data["g"], dtype=int), 0],
        fold=folds, observed_ms=observed * 1000,
        predicted_ms=final_prediction * 1000,
        held_out_ms=predictions[(depth, lam)] * 1000,
    ))
    fit["residual_ms"] = fit.observed_ms - fit.predicted_ms
    fit["held_out_residual_ms"] = fit.observed_ms - fit.held_out_ms
    return manager, ranking, fit
