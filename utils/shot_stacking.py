"""Re-stack individual hammer hits after correcting trigger jitter.

The field stacks are the plain mean of the individual hits, but the hits
are offset by a nearly constant per-hit lag (trigger jitter), which blurs
the first arrivals. Here each hit is aligned by cross-correlation before
averaging. The absolute time zero is unknown, so hits are aligned to the
median lag: the stack keeps the median trigger time of the hits.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import io
import os

import numpy as np

from .seg2_io import ShotGather, read_shot_gather


@contextlib.contextmanager
def _tolerant_seg2_dates():
    """Accept the invalid acquisition dates written in the per-hit files."""
    import obspy.io.seg2.seg2 as seg2
    from obspy import UTCDateTime

    original = seg2._parse_date_and_time

    def safe(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ValueError:
            return UTCDateTime(0)

    seg2._parse_date_and_time = safe
    try:
        yield
    finally:
        seg2._parse_date_and_time = original


def read_hits(paths: list[str]) -> np.ndarray:
    """Read per-hit SEG-2 files into an array (n_hits, n_samples, n_receivers)."""
    with _tolerant_seg2_dates(), contextlib.redirect_stdout(io.StringIO()):
        return np.stack([read_shot_gather(p).data for p in sorted(paths)])


def shift_samples(trace: np.ndarray, shift: int) -> np.ndarray:
    """Delay (shift > 0) or advance (shift < 0) along axis 0, zero padded."""
    out = np.zeros_like(trace)
    if shift > 0:
        out[shift:] = trace[:-shift]
    elif shift < 0:
        out[:shift] = trace[-shift:]
    else:
        out[:] = trace
    return out


def estimate_hit_lags(hits: np.ndarray, window: int = 1200, min_corr: float = 0.8) -> np.ndarray:
    """Lag (samples) of each hit relative to hit 0; median over well-correlated traces."""
    lags = []
    for k in range(hits.shape[0]):
        trace_lags, corrs = [], []
        for j in range(hits.shape[2]):
            a = hits[0, :window, j] - hits[0, :window, j].mean()
            b = hits[k, :window, j] - hits[k, :window, j].mean()
            xc = np.correlate(b, a, "full")
            trace_lags.append(np.argmax(xc) - (window - 1))
            corrs.append(xc.max() / np.sqrt((a @ a) * (b @ b) + 1e-30))
        trace_lags, corrs = np.array(trace_lags), np.array(corrs)
        good = corrs > min_corr
        if not good.any():
            raise ValueError(f"Hit {k} does not correlate with hit 0 (max r={corrs.max():.2f})")
        lags.append(int(np.median(trace_lags[good])))
    return np.array(lags)


def align_and_stack(hits: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Return (stack, applied_shifts) with hits aligned to their median lag."""
    lags = estimate_hit_lags(hits, **kwargs)
    shifts = -(lags - int(np.median(lags)))
    aligned = np.stack([shift_samples(h, int(s)) for h, s in zip(hits, shifts)])
    return aligned.mean(axis=0), shifts


def restack_line(gathers: list[ShotGather], folder: str, hit_patterns: dict[float, str]) -> list[ShotGather]:
    """Replace each gather's stack by an aligned re-stack of its hits.

    `hit_patterns` maps source_x -> filename substring of its per-hit files.
    Sources without readable hits keep the original field stack.
    """
    out = []
    for g in gathers:
        pattern = hit_patterns.get(g.source_x)
        paths = glob.glob(os.path.join(folder, f"*{pattern}*.dat")) if pattern else []
        try:
            hits = read_hits(paths) if paths else None
        except Exception as exc:  # e.g. the unreadable 60 m hit files
            print(f"  {g.source_x:+.1f} m: hits unreadable ({type(exc).__name__}); keeping field stack")
            hits = None
        if hits is None or hits.shape[1:] != g.data.shape:
            if paths and hits is not None:
                print(f"  {g.source_x:+.1f} m: hit shape mismatch; keeping field stack")
            elif not paths:
                print(f"  {g.source_x:+.1f} m: no hit files; keeping field stack")
            out.append(g)
            continue
        stack, shifts = align_and_stack(hits)
        shifts_ms = ", ".join(f"{s * g.dt * 1000:+.2f}" for s in shifts)
        print(f"  {g.source_x:+.1f} m: {len(hits)} hits re-stacked, shifts [{shifts_ms}] ms")
        out.append(dataclasses.replace(g, data=stack))
    return out
