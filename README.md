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
