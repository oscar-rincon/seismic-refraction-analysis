"""MASW processing: phase-shift dispersion imaging (Park et al., 1998),
fundamental-mode curve picking and empirical Vs depth profile.

Pure NumPy implementation chosen for readability; no compiled
dependencies are required.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


# ----------------------------------------------------------------------
# Dispersion imaging
# ----------------------------------------------------------------------

def dispersion_image(gather, c_min=50.0, c_max=800.0, c_step=2.5,
                     f_min=5.0, f_max=100.0):
    """Phase-shift wavefield transformation of one shot gather.

    Each trace is Fourier-transformed and amplitude-normalized; for every
    trial phase velocity c the spectra are phase-shifted back by the
    travel distance (offset) and stacked. Coherent surface-wave energy
    stacks constructively at the true phase velocity:

        A(f, c) = | sum_over_receivers  exp(2*pi*i*f*offset/c) * U(f) / |U(f)| |

    Returns (frequencies, velocities, energy) where energy has shape
    (n_frequencies, n_velocities) and is normalized to 1 at each frequency.
    """
    spectra = np.fft.rfft(gather.data, axis=0)
    frequencies = np.fft.rfftfreq(gather.n_samples, gather.dt)

    band = (frequencies >= f_min) & (frequencies <= f_max)
    frequencies = frequencies[band]
    spectra = spectra[band, :]
    spectra = spectra / (np.abs(spectra) + 1e-30)  # keep phase only

    velocities = np.arange(c_min, c_max + c_step, c_step)
    energy = np.zeros((len(frequencies), len(velocities)))
    for j, c in enumerate(velocities):
        # Phase shift that undoes propagation over each offset at velocity c
        shift = np.exp(2j * np.pi * np.outer(frequencies / c, gather.offsets))
        energy[:, j] = np.abs(np.sum(shift * spectra, axis=1))

    energy = energy / (energy.max(axis=1, keepdims=True) + 1e-30)
    return frequencies, velocities, energy


# ----------------------------------------------------------------------
# Dispersion curve picking
# ----------------------------------------------------------------------

def pick_dispersion_curve(frequencies, velocities, energy,
                          f_min=10.0, f_max=60.0, smooth_points=5):
    """Pick the fundamental-mode ridge (energy maximum per frequency).

    Restricted to [f_min, f_max]; the picked curve is smoothed with a
    moving average of `smooth_points` samples.
    Returns a DataFrame with frequency (Hz), phase velocity c_R (m/s)
    and wavelength (m).
    """
    band = (frequencies >= f_min) & (frequencies <= f_max)
    f = frequencies[band]
    c_picked = velocities[np.argmax(energy[band, :], axis=1)]

    kernel = np.ones(smooth_points) / smooth_points
    c_smooth = np.convolve(c_picked, kernel, mode="same")
    # The moving average is biased at the edges; keep the raw picks there.
    half = smooth_points // 2
    c_smooth[:half], c_smooth[-half:] = c_picked[:half], c_picked[-half:]

    return pd.DataFrame({
        "frequency_Hz": f,
        "c_rayleigh_m_s": c_smooth,
        "wavelength_m": c_smooth / f,
    })


def plot_dispersion(frequencies, velocities, energy, curve=None, ax=None,
                    title="MASW dispersion image"):
    """Plot the dispersion energy image with the picked curve overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    mesh = ax.pcolormesh(frequencies, velocities, energy.T,
                         cmap="inferno", shading="auto")
    plt.colorbar(mesh, ax=ax, label="Normalized energy")
    if curve is not None:
        ax.plot(curve["frequency_Hz"], curve["c_rayleigh_m_s"], "c--",
                lw=2.2, label="Fundamental mode $c_R(f)$")
        ax.legend(loc="upper right")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.set_title(title)
    return ax


def average_curves(curves):
    """Average dispersion curves from several shots (e.g. forward/reverse).

    All curves must share the same frequency axis (same recording length
    and sampling rate). Returns a DataFrame in the same format.
    """
    f = curves[0]["frequency_Hz"].to_numpy()
    c_mean = np.mean([c["c_rayleigh_m_s"].to_numpy() for c in curves], axis=0)
    return pd.DataFrame({
        "frequency_Hz": f,
        "c_rayleigh_m_s": c_mean,
        "wavelength_m": c_mean / f,
    })


# ----------------------------------------------------------------------
# Vs depth profile (empirical wavelength-depth transform)
# ----------------------------------------------------------------------

def _pick_depths(curve, depth_factor, vs_factor):
    z = curve["wavelength_m"].to_numpy() / depth_factor
    vs = curve["c_rayleigh_m_s"].to_numpy() * vs_factor
    return z, vs


def vs_profile(curve, depth_factor=2.5, vs_factor=1.09, z_max=None, n_depths=100,
               min_wavelength=None, depths=None):
    """Empirical Vs(z) profile from the picked dispersion curve.

    Each (c_R, wavelength) pair is mapped to a pseudo-depth
    z = wavelength / depth_factor and Vs = vs_factor * c_R
    (vs_factor ~ 1.09 corresponds to Poisson's ratio ~ 0.35).
    Picks with wavelength < `min_wavelength` are dropped: below twice the
    receiver spacing the phase-shift image has alias peaks of similar
    energy, so those picks are ambiguous. `depths` evaluates the profile at
    given depths (NaN outside the picked range) instead of a regular grid.
    Returns a DataFrame with depth (m) and Vs (m/s).
    """
    z, vs = _pick_depths(curve, depth_factor, vs_factor)
    if min_wavelength is not None:
        keep = curve["wavelength_m"].to_numpy() >= min_wavelength
        z, vs = z[keep], vs[keep]

    order = np.argsort(z)
    z, vs = z[order], vs[order]
    z_unique, idx = np.unique(np.round(z, 2), return_index=True)
    vs_unique = vs[idx]

    vs_of_z = interpolate.PchipInterpolator(z_unique, vs_unique, extrapolate=True)
    if depths is not None:
        z_out = np.asarray(depths, float)
        vs_out = np.where((z_out >= z_unique[0]) & (z_out <= z_unique[-1]),
                          vs_of_z(z_out), np.nan)
        return pd.DataFrame({"depth_m": z_out, "vs_m_s": vs_out})
    z_fine = np.linspace(z_unique[0], z_max or z_unique[-1], n_depths)
    return pd.DataFrame({"depth_m": z_fine, "vs_m_s": vs_of_z(z_fine)})


def aliased_picks(curve, min_wavelength, depth_factor=2.5, vs_factor=1.09):
    """Picks excluded by `vs_profile(min_wavelength=...)`, as depth and Vs."""
    z, vs = _pick_depths(curve, depth_factor, vs_factor)
    flagged = curve["wavelength_m"].to_numpy() < min_wavelength
    return pd.DataFrame({"depth_m": z[flagged], "vs_m_s": vs[flagged]})


def add_shot_profiles(profile, curves, **profile_kwargs):
    """Add one vs_shot{i}_m_s column per shot curve, at the profile depths.

    Each shot is converted with the same `profile_kwargs` as the mean
    profile. The mean curve is averaged per frequency before the
    wavelength-depth mapping, so it need not lie between the shot profiles.
    """
    out = profile.copy()
    for i, c in enumerate(curves, start=1):
        out[f"vs_shot{i}_m_s"] = vs_profile(
            c, depths=profile["depth_m"], **profile_kwargs)["vs_m_s"].to_numpy()
    return out


def layer_averages(profile, boundaries):
    """Average Vs within layers defined by depth `boundaries` (m).

    Example: boundaries=[1.0, 4.0] gives three layers
    (0 to 1 m, 1 to 4 m, deeper than 4 m). Boundaries outside the
    profile depth range are ignored.
    Returns a DataFrame with one row per layer.
    """
    z = profile["depth_m"].to_numpy()
    vs = profile["vs_m_s"].to_numpy()
    inner = [b for b in boundaries if z.min() < b < z.max()]
    edges = [z.min()] + inner + [z.max()]

    rows = []
    for top, bottom in zip(edges[:-1], edges[1:]):
        inside = (z >= top) & (z <= bottom)
        rows.append({
            "top_m": round(top, 2),
            "bottom_m": round(bottom, 2),
            "vs_mean_m_s": round(float(vs[inside].mean()), 1),
        })
    return pd.DataFrame(rows)


def plot_vs_profile(profile, layers=None, ax=None):
    """Plot Vs versus depth, optionally with the stepped layer model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    ax.plot(profile["vs_m_s"], profile["depth_m"], "b-", lw=2,
            label="MASW $V_s(z)$")
    if layers is not None:
        for _, layer in layers.iterrows():
            ax.plot([layer["vs_mean_m_s"]] * 2,
                    [layer["top_m"], layer["bottom_m"]], "r--", lw=2)
        ax.plot([], [], "r--", lw=2, label="Layer averages")

    ax.set_xlabel("Shear-wave velocity $V_s$ (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Vs depth profile")
    ax.invert_yaxis()
    ax.legend()
    return ax
