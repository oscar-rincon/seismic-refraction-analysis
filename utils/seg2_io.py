"""Reading PASI Gea24 SEG-2 (.dat) shot gathers with ObsPy.

A "shot gather" is one file: all geophone traces recorded for a single
source (sledgehammer) position. Geometry (source and receiver positions)
is read directly from the SEG-2 trace headers.
"""

import glob
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import obspy

# ObsPy warns about vendor-specific SEG-2 headers; the fields we use are standard.
warnings.filterwarnings("ignore", category=UserWarning, module="obspy")


@dataclass
class ShotGather:
    """One SEG-2 record: all traces for a single source position."""

    filename: str
    source_x: float            # source position along the line (m)
    receiver_x: np.ndarray     # receiver positions along the line (m), one per trace
    data: np.ndarray           # trace samples, shape (n_samples, n_receivers)
    dt: float                  # sample interval (s)

    @property
    def n_receivers(self) -> int:
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def fs(self) -> float:
        """Sampling rate (Hz)."""
        return 1.0 / self.dt

    @property
    def time(self) -> np.ndarray:
        """Time axis in seconds, one value per sample."""
        return np.arange(self.n_samples) * self.dt

    @property
    def offsets(self) -> np.ndarray:
        """Absolute source-to-receiver distances (m)."""
        return np.abs(self.receiver_x - self.source_x)


def read_shot_gather(path: str, source_x: float | None = None) -> ShotGather:
    """Read one SEG-2 file into a ShotGather.

    Geometry is taken from the SEG-2 headers (SOURCE_LOCATION and
    RECEIVER_LOCATION). Pass `source_x` to override the header value,
    e.g. when the operator did not update it in the field.
    """
    stream = obspy.read(path, format="SEG2")

    receiver_x = np.array(
        [float(tr.stats.seg2.RECEIVER_LOCATION.split()[0]) for tr in stream]
    )
    header_source_x = float(stream[0].stats.seg2.SOURCE_LOCATION.split()[0])
    data = np.column_stack([tr.data.astype(float) for tr in stream])

    return ShotGather(
        filename=os.path.basename(path),
        source_x=source_x if source_x is not None else header_source_x,
        receiver_x=receiver_x,
        data=data,
        dt=float(stream[0].stats.delta),
    )


def read_line(folder: str, exclude: str = "_shot", source_x_overrides: dict | None = None) -> list[ShotGather]:
    """Read every stacked shot gather in a folder, sorted by source position.

    By default the per-hit files (containing "_shot" in the name) are
    skipped, keeping only the stacked record of each source position.
    `source_x_overrides` maps filename -> source position (m) for files
    whose SEG-2 header geometry is wrong.
    """
    overrides = source_x_overrides or {}
    paths = [
        p for p in sorted(glob.glob(os.path.join(folder, "*.dat")))
        if exclude not in os.path.basename(p)
    ]
    if not paths:
        raise FileNotFoundError(f"No .dat files found in {folder}")

    gathers = [
        read_shot_gather(p, source_x=overrides.get(os.path.basename(p)))
        for p in paths
    ]
    gathers.sort(key=lambda g: g.source_x)

    print(f"Read {len(gathers)} shot gathers from {folder}:")
    for g in gathers:
        print(
            f"  {g.filename}: source at {g.source_x:+.1f} m, "
            f"{g.n_receivers} receivers ({g.receiver_x[0]:.1f}-{g.receiver_x[-1]:.1f} m), "
            f"{g.n_samples} samples at {g.fs:.0f} Hz"
        )
    return gathers
