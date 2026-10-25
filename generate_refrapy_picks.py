"""Generate preliminary automatic L1 picks for review in the Refrapy GUI."""

import argparse
from pathlib import Path

from utils import refraction, seg2_io


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=root / "L1" / "refrapy_inversion" / "L1_automatic_picks.sgt",
        help="Output .sgt path (existing output is overwritten)",
    )
    args = parser.parse_args()
    gathers = seg2_io.read_line(
        str(root / "L1" / "RF"),
        source_x_overrides={"2026-10-08_RF_47.5m.dat": 47.5},
    )
    picks = refraction.pick_line(
        gathers, v_min=150.0, v_max=1500.0,
        noise_window=0.010, threshold=3.0,
    )
    data = refraction.build_traveltime_data(gathers, picks, pick_error=0.0035)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = refraction.export_refrapy_picks(data, args.output)
    print(f"Saved {data.size()} automatic picks to {path.resolve()}")
    print("PRELIMINARY: review arrivals before inversion; no residual-based rejection was applied.")
    print("Assumes flat geometry and source correction 42.5 -> 47.5 m; confirm field notes.")
    print("The GUI format does not store the assumed 3.5 ms pick uncertainty.")


if __name__ == "__main__":
    main()
