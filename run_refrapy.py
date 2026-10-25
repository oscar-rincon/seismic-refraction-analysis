"""Launch Refrapy with Linux icons and PASI along-line header compatibility."""

import argparse
import os
from pathlib import Path
import runpy
import sys
import tkinter

import obspy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", choices=("pick", "inv"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Refrapy",
        help="Upstream Refrapy checkout (default: ../Refrapy)",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    script = source / {"pick": "Refrapick.py", "inv": "Refrainv.py"}[args.app]
    icon = source / "images" / "ico_refrapy.gif"
    if not script.is_file() or not icon.is_file():
        parser.error(
            f"Refrapy scripts/images not found in {source}. Clone "
            "https://github.com/viictorjs/Refrapy.git there or use --source."
        )

    original_iconbitmap = tkinter.Wm.iconbitmap
    original_read = obspy.read

    def iconbitmap(self, bitmap=None, default=None):
        # Upstream uses Windows .ico bitmaps, which Tk on Linux cannot load.
        if sys.platform.startswith("linux") and bitmap and str(bitmap).endswith(".ico"):
            self.iconphoto(True, tkinter.PhotoImage(master=self, file=str(icon)))
            return None
        return original_iconbitmap(self, bitmap, default)

    def read_waveform(*args, **kwargs):
        stream = original_read(*args, **kwargs)
        for trace in stream:
            if trace.stats.get("_format") == "SEG2":
                # PASI stores x/y/z; Refrapy expects a scalar along-line position.
                for key in ("SOURCE_LOCATION", "RECEIVER_LOCATION"):
                    value = trace.stats.seg2[key]
                    trace.stats.seg2[key] = str(float(str(value).split()[0]))
        return stream

    cwd = Path.cwd()
    try:
        tkinter.Wm.iconbitmap = iconbitmap
        obspy.read = read_waveform
        os.chdir(source)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        tkinter.Wm.iconbitmap = original_iconbitmap
        obspy.read = original_read
        os.chdir(cwd)


if __name__ == "__main__":
    main()
