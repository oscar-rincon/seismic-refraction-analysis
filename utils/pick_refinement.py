"""Independent, local AIC onset candidates for visual first-break review."""

import numpy as np
import pandas as pd


def aic_candidate(trace, dt, reference_time, half_window=0.012, min_samples=8):
    """Locate a variance change near a reference pick, without moveout constraints."""
    trace = np.asarray(trace, dtype=float)
    if (trace.ndim != 1 or not np.all(np.isfinite(trace)) or dt <= 0
            or not np.isfinite(dt) or not np.isfinite(reference_time)
            or reference_time <= 0 or half_window <= 0 or min_samples < 2):
        raise ValueError("AIC requires a finite trace, positive times/window and valid sample count.")
    lo = max(0, int((reference_time - half_window) / dt))
    hi = min(len(trace), int((reference_time + half_window) / dt) + 1)
    if hi - lo < 2 * min_samples + 1:
        raise ValueError("AIC search window is too short or outside the recording.")
    values = trace[lo:hi]
    values = values - np.mean(values)
    scale = np.max(np.abs(values))
    if scale == 0:
        raise ValueError("No waveform variation in the AIC search window.")
    values = values / scale
    n = len(values)
    k = np.arange(min_samples, n - min_samples + 1)
    cumulative = np.cumsum(values)
    squares = np.cumsum(values ** 2)
    left = (squares[k - 1] - cumulative[k - 1] ** 2 / k) / k
    right_n = n - k
    right = (squares[-1] - squares[k - 1]
             - (cumulative[-1] - cumulative[k - 1]) ** 2 / right_n) / right_n
    floor = max(np.var(values) * 1e-12, np.finfo(float).tiny)
    aic = k * np.log(np.maximum(left, floor)) + right_n * np.log(np.maximum(right, floor))
    selected = int(np.argmin(aic))
    index = lo + int(k[selected])
    return index * dt, selected in (0, len(k) - 1)


def refine_line(gathers, original_picks):
    """Propose AIC picks with window-stability and signal-contrast review flags.

    Flags are heuristics, not calibrated uncertainties or approval. No inversion
    prediction, source position, or neighboring trace controls the candidate.
    """
    if len(gathers) != len(original_picks):
        raise ValueError("Each gather requires one array of reference picks.")
    candidates, records = [], []
    for shot_index, (gather, original) in enumerate(zip(gathers, original_picks)):
        if len(original) != gather.n_receivers:
            raise ValueError("Each receiver requires one reference pick.")
        refined = np.full(gather.n_receivers, np.nan)
        for receiver_index, reference in enumerate(original):
            trace = gather.data[:, receiver_index]
            flags = []
            try:
                times_and_edges = [
                    aic_candidate(trace, gather.dt, reference, half_window=window)
                    for window in (0.008, 0.012, 0.016)
                ]
            except ValueError as error:
                candidate, spread, ratio = np.nan, np.nan, np.nan
                flags.append(str(error))
            else:
                candidate = times_and_edges[1][0]
                spread = np.ptp([time for time, _ in times_and_edges])
                index = round(candidate / gather.dt)
                width = max(8, round(0.003 / gather.dt))
                before = trace[max(0, index - width):index]
                after = trace[index:min(len(trace), index + width)]
                before_std, after_std = np.std(before), np.std(after)
                ratio = after_std / max(before_std, np.finfo(float).eps * np.max(np.abs(trace)))
                if any(edge for _, edge in times_and_edges):
                    flags.append("search boundary")
                if spread > 0.002:
                    flags.append("window-sensitive (>2 ms)")
                if ratio < 3:
                    flags.append("weak variance contrast (<3)")
                if abs(candidate - reference) > 0.005:
                    flags.append("large change (>5 ms)")
                refined[receiver_index] = candidate
            records.append(dict(
                shot_index=shot_index, receiver_index=receiver_index,
                source_x_m=gather.source_x, receiver_x_m=gather.receiver_x[receiver_index],
                original_ms=reference * 1000, candidate_ms=candidate * 1000,
                shift_ms=(candidate - reference) * 1000,
                window_spread_ms=spread * 1000, variance_std_ratio=ratio,
                review_flags="; ".join(flags) if flags else "visual confirmation required",
            ))
        candidates.append(refined)
    return candidates, pd.DataFrame(records)


def select_refined_picks(gathers, original_picks, overrides=None):
    """Use stable, high-contrast AIC candidates; flag unresolved references.

    Original picks remain where waveform evidence is ambiguous, explicitly
    labeled for review rather than silently replaced by a later phase. Manual
    overrides take precedence; NaN explicitly discards an unreliable trace.
    """
    candidates, review = refine_line(gathers, original_picks)
    selected = [np.asarray(picks, dtype=float).copy() for picks in original_picks]
    stable = review.review_flags.eq("visual confirmation required").to_numpy()
    methods = []
    for accepted, row in zip(stable, review.itertuples()):
        if accepted:
            selected[row.shot_index][row.receiver_index] = candidates[row.shot_index][row.receiver_index]
        methods.append("stable AIC candidate" if accepted else "unresolved reference")
    review["selection_method"] = methods
    review["needs_review"] = ~stable
    for (shot, receiver), value in (overrides or {}).items():
        if not (0 <= shot < len(gathers) and 0 <= receiver < gathers[shot].n_receivers):
            raise ValueError("Manual override refers to an unknown trace.")
        if not np.isnan(value) and (
                not np.isfinite(value) or value <= 0 or value > gathers[shot].time[-1]):
            raise ValueError("Manual pick must be inside the recording or NaN to discard.")
        selected[shot][receiver] = value
        rows = (review.shot_index == shot) & (review.receiver_index == receiver)
        review.loc[rows, "selection_method"] = "manual exclusion" if np.isnan(value) else "manual pick"
        review.loc[rows, "needs_review"] = False
    review["selected_ms"] = [
        selected[row.shot_index][row.receiver_index] * 1000 for row in review.itertuples()
    ]
    review["selected_shift_ms"] = review.selected_ms - review.original_ms
    return selected, review
