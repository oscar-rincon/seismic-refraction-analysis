# seismic-refraction-analysis

Seismic refraction (RF) and MASW processing of PASI Gea24 SEG-2 field data.

- `L1_RF_analysis.ipynb` — first-break picking, two-layer interpretation and 2D traveltime tomography (pyGIMLi) of the L1 refraction line.
- `L1_MASW_analysis.ipynb` — phase-shift dispersion imaging, fundamental-mode picking and Vs depth profile of the L1 MASW records.
- `utils/` — reusable processing modules (`seg2_io`, `refraction`, `masw`, `plotting`).
- `L1/` — raw SEG-2 (`.dat`) field records (`RF/` and `MASW/`).

## Installation

Requires [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda); pyGIMLi is distributed through its own conda channel, so pip alone is not enough.

1. Create the environment from the repository root:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate it:

   ```bash
   conda activate seismic-l1
   ```

3. Verify the installation:

   ```bash
   python -c "import pygimli, obspy; print(pygimli.__version__, obspy.__version__)"
   ```

4. Start Jupyter (or select the `seismic-l1` kernel in VS Code) and run the notebooks:

   ```bash
   jupyter notebook
   ```

To update an existing environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

## Usage

Open either notebook and adjust the parameter cell at the top (data folder, geometry overrides, picking thresholds, inversion settings) for each new measurement, then run all cells. All processing logic lives in `utils/` so the notebooks stay short.

[L1_RF_analysis.ipynb](L1_RF_analysis.ipynb) runs a single tomography (depth 8 m, λ = 1, homogeneous start) with
`Z_WEIGHT = 0.5`, which mildly favours horizontal layering (1 = isotropic). With
the same 3-fold check this changes held-out RMS only from 3.41 to 3.44 ms. Depth
and λ were chosen earlier by
cross-validated prediction of held-out picks. The parameter search itself is in
[L1_RF_optimized.ipynb](L1_RF_optimized.ipynb) and `utils/tomography_selection.py`;
re-run it if the picks change substantially. All picks are used, without
residual-based rejection. After the tomogram the notebook shows observed picks
against model predictions, velocity contours in `CONTOUR_INTERVAL` (250 m/s)
display bins, and the material-class section. Figures, per-arrival predictions
and the model are saved in `L1/RF_results/`. The joint Vp + Vs table is limited
to the model depth to avoid extrapolation.

An additional indicative material-class cross-section uses fixed colors and
Vp thresholds: <500 m/s (loose/low-velocity soil, light sand), 500-1000 m/s
(soil/fill, tan), 1000-1500 m/s (dense or wetter soil, dark brown), and >=1500 m/s
(saturated soil or weathered rock, gray). Threshold values belong to the
higher class. The figure is saved as `L1/RF_results/indicative_material_classes.png`.
These are user-selected interpretive hypotheses, not calibrated material
identifications: Vp alone cannot confirm density, saturation or lithology.
The map uses the velocity-contour display interpolation, leaves the inverted
model unchanged, and does not extrapolate the 1D MASW Vs classes laterally.

Section 6.1 adds a 1D summary (`L1/RF_results/profile_1d.png`): the Vp mean over
x = 0–55 m with its lateral P10–P90 band and a column of the same
Vp classes computed from the mean profile, labelled with boundary depths. The
average hides lateral changes (see the band), so the layer depths are indicative.

Before picking, `RESTACK_ALIGNED = True` re-stacks the individual hammer hits
(`utils/shot_stacking.py`). The field stacks are the plain mean of 3 hits that
are offset by trigger jitter of up to ~2 ms; each hit is aligned by
cross-correlation (median over traces with r > 0.8) to the median-lag hit, so
the absolute time zero stays the median trigger time. This raised stable AIC
picks from 21 to 23 of 60. The 60 m hit files are not valid SEG-2, so its field
stack is kept. Band-pass filtering was tested and rejected for picking: a
zero-phase 10–150 Hz filter reduced the number of stable AIC picks, and
filtering the legacy picker input produced early, implausible triggers.

The main RF notebook now actually uses waveform refinement, not just the
legacy reference picks: stable, high-contrast AIC candidates replace references
when the local-window checks agree. Ambiguous traces keep their reference
pick and are explicitly marked orange, not silently declared improved.
`PICK_OVERRIDES` applies after refinement; `np.nan` explicitly excludes a trace.
The overview is zoomed to 100 ms. Pick diagnostics and
`L1_waveform_refined_picks.sgt` are saved under `L1/RF_results/`; enlarged
per-trace panels are available through `plotting.plot_pick_review` and in
[L1_pick_review.ipynb](L1_pick_review.ipynb).
No inversion residual is used to reposition picks. Review unresolved picks
before treating the inversion as geological evidence.

## Refrapy desktop picking and inversion

Use a separate environment for Refrapy; the notebook environment above is
unchanged. A graphical desktop/display is required (including when using a
remote VS Code session).

```bash
conda env create -f environment-refrapy.yml
conda activate refrapy-l1
```

For an existing installation of this environment:

```bash
conda env update -f environment-refrapy.yml
```

Clone [upstream Refrapy](https://github.com/viictorjs/Refrapy) next to this
repository, unless it is already there:

```bash
git clone https://github.com/viictorjs/Refrapy.git ../Refrapy
```

From this repository, launch either application:

```bash
python run_refrapy.py pick
python run_refrapy.py inv
```

For a checkout elsewhere, add `--source /path/to/Refrapy`. The launcher sets
the working directory so upstream can find its images and replaces unsupported
Windows `.ico` icons with the supplied GIF icon on Linux. It does not modify
the upstream files or the raw field data. At read time, it also converts SEG-2
`SOURCE_LOCATION` and `RECEIVER_LOCATION` headers from PASI's `x y z` strings
to the first (along-line x) coordinate, matching `utils/seg2_io.py`. This is
a 2D line workflow, not support for arbitrary 3D acquisition geometry.

The local sibling checkout's `Refrapick.py` and `Refrainv.py` also have a
project-dialog fix: cancelling folder/name selection, selecting an invalid
project, or attempting to create an existing project preserves the previous
project state. Without this fix, cancelling a folder dialog can replace the
project path with a tuple and opening waveforms then raises
`TypeError: can only concatenate tuple (not "str") to tuple`.
This fix lives in that checkout, not in the launcher; a fresh upstream clone
does not include it. Restart the application after applying the fix.

The local `Refrainv.py` also clips the contour set directly on modern
Matplotlib, with a fallback for older versions. This fixes
`AttributeError: 'QuadContourSet' object has no attribute 'collections'`
when drawing the tomography result. This change also lives in the sibling
checkout and requires restarting Refrainv.

### Test with the L1 refraction records

1. In Refrapick, create a project in a writable results directory. Refrapy
   creates `data`, `picks`, `models`, and `gps` subdirectories.
2. Open the five stacked `.dat` files in [L1/RF](L1/RF), excluding filenames
   containing `_shot`. Navigate to that folder in the open dialog; the raw
   records do not need to be copied into the project's `data` folder.
3. Check the geometry: 12 receivers at 0, 5, ..., 55 m; source positions
   -5, 12.5, 27.5, 47.5, and 60 m. Each record has 4096 samples at 8000 Hz.
4. **Before picking**, select `2026-10-08_RF_47.5m.dat`, open **Edit
   parameters**, and use **Edit source position** to set **47.5 m**. Its raw
   header says 42.5 m; this is the same override used by
   [the RF notebook](L1_RF_analysis.ipynb). Confirm the actual field position
   before interpretation. The launcher does not silently override geometry.
5. Pick first arrivals and save the `.sgt` picks under the project's `picks`
   directory. Supply surveyed topography if available; omitting it assumes
   flat geometry.
6. In Refrainv, load the same project and the saved picks. Check the traveltime
   curves and geometry before running time-terms or tomography inversion.

Refrapy upstream documents an older Windows/Python 3.8 stack. This separate
environment uses Python 3.12 and pyGIMLi 1.5.5; the launcher addresses Linux
startup, but inspect inversion results and compare them with the RF notebook
rather than assuming numerical equivalence with the older upstream stack.
The pyGIMLi version is pinned because the tested conda build of 1.6.1 installs
NumPy below 2.3 through its SciPy constraint while declaring NumPy >=2.4 in its
Python package metadata.

### Automatic picks for the GUI

Generate first-arrival picks for the five stacked L1 records:

```bash
conda activate refrapy-l1
python generate_refrapy_picks.py
```

The output is [L1_automatic_picks.sgt](L1/refrapy_inversion/L1_automatic_picks.sgt).
Use `--output /path/to/project/picks/L1_automatic_picks.sgt` to save elsewhere.
Existing output is overwritten. This exports all automatic picks, without
the notebook's inversion-dependent residual rejection. They require visual
review and assume flat geometry and the 47.5 m source correction.

In Refrapick, load/create a project, open the five stacked records, correct the
47.5 m record's source coordinate, then use **Load pick file** to overlay the
automatic picks. Load them only once into a fresh waveform session to avoid
duplicate markers. In Refrainv, load/create a project and use **Load pick
file** to open the same file (navigate to its saved location).

This exporter uses Refrapy's required `#s g t` header, one-based source/receiver
indices, and seconds. The strict GUI format cannot store pick uncertainties;
do not assume its default uncertainty reproduces the notebook's 3.5 ms.
The notebook also exports its retained picks in this GUI-compatible layout.

### Inversion results notebook

For waveform-based pick refinement, open
[L1_pick_review.ipynb](L1_pick_review.ipynb) with the `refrapy-l1` kernel.
It proposes independent local AIC onset candidates, flags sensitivity to the
search window and weak variance changes, and shows all 60 enlarged raw traces.
Use `MANUAL_OVERRIDES` (seconds; `np.nan` to discard) after inspecting the panels.
Outputs are saved separately in `L1/pick_review/`, including
`L1_AIC_candidates.sgt` for Refrapick/Refrainv and a comparison CSV.
These are unverified candidates, not automatically accepted improvements.
Existing picks and inversion notebooks are not replaced; the local search
can still select noise or a later phase and needs human confirmation.

Open [L1_Refrapy_inversion.ipynb](L1_Refrapy_inversion.ipynb), select the
`refrapy-l1` interpreter/kernel, and run all cells from the repository root.
If the environment predates the notebook, update and register its kernel:

```bash
conda env update -f environment-refrapy.yml
conda activate refrapy-l1
python -m ipykernel install --user --name refrapy-l1 --display-name "Python (refrapy-l1)"
```

The notebook runs pyGIMLi directly on the stacked L1 records; it does not read
a saved Refrainv GUI model. It shows automatic first-arrival picks, QC
rejections, the P-wave velocity model with ray paths, observed/predicted
traveltimes, and residual statistics. It saves figures, CSV diagnostics,
the mesh, cell velocities, `.sgt` picks, and settings in
`L1/refrapy_inversion/`. Re-running overwrites those exports.
Review the automatic picks, the source-coordinate override, and the flat-line
geometry before interpreting these preliminary results.

### Quantitative inversion selection without forced horizontal layers

[L1_RF_optimized.ipynb](L1_RF_optimized.ipynb) compares 24 combinations of
model depth and isotropic regularization using three-fold held-out prediction.
It uses a homogeneous starting velocity from each training set, retains all
60 automatic picks, and does not impose the earlier 80 m horizontal
correlation. Select the `refrapy-l1` kernel and run all cells.
Per-shot plots show observations, final-model predictions and out-of-fold
predictions, with residuals. Results and settings are saved to
`L1/optimized_inversion/`; re-running overwrites those exports.
The velocity figure uses area-weighted nodal interpolation for a smooth display,
with no mesh edges or ray overlays and a labeled velocity colorbar. This is
visualization only: saved cell velocities and forward predictions are unchanged.
The original RF notebooks remain unchanged for comparison.

This selects within a bounded search, not a unique or assumption-free earth
model. Geometry and automatic picking assumptions remain; flat surface
elevation is not the same as flat subsurface layers. Selection CV measures
interpolation on the same survey and is not an independent test-set score.
