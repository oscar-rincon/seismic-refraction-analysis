"""Joint interpretation of refraction (Vp) and MASW (Vs) results:
elastic properties, material classification and a summary figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Combining the two measurements
# ----------------------------------------------------------------------

def vp_profile_from_tomogram(manager, x_center, z_min=0.3, z_max=10.0, n_depths=40):
    """Extract a 1D Vp(z) profile from the 2D pyGIMLi tomogram.

    Velocities are interpolated from the inverted model along a vertical
    line at `x_center` (m, usually the middle of the spread), between
    `z_min` and `z_max` depth. Returns a DataFrame with depth_m and vp_m_s.
    """
    import pygimli as pg

    depths = np.linspace(z_min, z_max, n_depths)
    points = [[x_center, -z] for z in depths]
    vp = np.array(pg.interpolate(manager.paraDomain, manager.velocity, points))
    return pd.DataFrame({"depth_m": depths, "vp_m_s": vp})


def elastic_properties(vp_profile, vs_profile, rho=1800.0):
    """Combine Vp(z) and Vs(z) into dynamic elastic properties.

    The Vs profile is interpolated onto the Vp profile depths (only the
    depth range covered by both is kept). Vs is capped at Vp/sqrt(2),
    the physical limit for an elastic solid. `rho` is the assumed soil
    density (kg/m3). Moduli are returned in MPa.
    """
    z = vp_profile["depth_m"].to_numpy()
    vp = vp_profile["vp_m_s"].to_numpy()

    z_vs = vs_profile["depth_m"].to_numpy()
    inside = (z >= z_vs.min()) & (z <= z_vs.max())
    z, vp = z[inside], vp[inside]
    vs = np.interp(z, z_vs, vs_profile["vs_m_s"].to_numpy())
    vs = np.minimum(vs, vp / np.sqrt(2.0))

    ratio = vp / vs
    poisson = (ratio**2 - 2.0) / (2.0 * (ratio**2 - 1.0))
    G = rho * vs**2 / 1e6
    E = 2.0 * G * (1.0 + poisson)

    return pd.DataFrame({
        "depth_m": z,
        "vp_m_s": vp,
        "vs_m_s": vs,
        "vp_vs_ratio": ratio,
        "poisson_ratio": poisson,
        "shear_modulus_MPa": G,
        "youngs_modulus_MPa": E,
    })


# ----------------------------------------------------------------------
# Material classification
# ----------------------------------------------------------------------

# (vs_max in m/s, label, plot color). Vs ranges follow common geotechnical
# practice (e.g. NEHRP site classes), with labels worded for a grass
# sports-field setting. Classified in order: the first row whose vs_max
# exceeds the measured Vs applies.
VS_CLASSES = [
    (180.0, "Soft soil: turf, rootzone & loose topsoil", "#8fce8f"),
    (360.0, "Stiff soil: compacted sandy/clayey fill", "#f2c57c"),
    (760.0, "Very dense soil / gravelly subbase", "#c9a0dc"),
    (1500.0, "Weathered / soft rock", "#9aa7b8"),
]
SATURATED_NOTE = "high Vp/Vs: likely saturated"  # appended when nu > 0.45


def classify_materials(properties, poisson_saturated=0.45,
                       smooth_points=5, min_thickness=1.0):
    """Assign a material class to every depth and merge contiguous layers.

    Classification is by Vs (stiffness); Poisson's ratio from the joint
    Vp/Vs adds a saturation indicator, since water raises Vp but not Vs.
    Vs and Poisson's ratio are median-smoothed over `smooth_points`
    samples before classifying, and interpreted layers thinner than
    `min_thickness` (m) are merged into the layer above - both guard
    against jitter in the picked dispersion curve.
    Returns a DataFrame with one row per interpreted layer: top_m,
    bottom_m, mean velocities, mean Poisson's ratio, material and color.
    """
    from scipy.signal import medfilt

    df = properties
    kernel = smooth_points + 1 - smooth_points % 2  # medfilt needs odd size
    vs_smooth = medfilt(df["vs_m_s"].to_numpy(), kernel)
    nu_smooth = medfilt(df["poisson_ratio"].to_numpy(), kernel)

    labels, colors = [], []
    for vs, nu in zip(vs_smooth, nu_smooth):
        for vs_max, label, color in VS_CLASSES:
            if vs < vs_max:
                break
        if nu > poisson_saturated:
            label = f"{label} ({SATURATED_NOTE})"
        labels.append(label)
        colors.append(color)

    # Merge runs thinner than min_thickness into the layer above
    # (the topmost run merges into the layer below instead).
    z = df["depth_m"].to_numpy()
    dz = np.median(np.diff(z))
    min_run = max(int(round(min_thickness / dz)), 1)
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if i - start < min_run:
                source = start - 1 if start > 0 else i  # neighbor to copy from
                if source < len(labels):
                    labels[start:i] = [labels[source]] * (i - start)
                    colors[start:i] = [colors[source]] * (i - start)
            start = i

    # Merge consecutive depths with the same label into layers.
    layers = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segment = df.iloc[start:i]
            layers.append({
                "top_m": round(segment["depth_m"].iloc[0], 2),
                "bottom_m": round(segment["depth_m"].iloc[-1], 2),
                "vp_mean_m_s": round(segment["vp_m_s"].mean(), 0),
                "vs_mean_m_s": round(segment["vs_m_s"].mean(), 0),
                "poisson_mean": round(segment["poisson_ratio"].mean(), 2),
                "material": labels[start],
                "color": colors[start],
            })
            start = i

    # Make layers contiguous: adjacent layers meet at the midpoint
    # between the last depth of one and the first depth of the next.
    for upper, lower in zip(layers[:-1], layers[1:]):
        boundary = round((upper["bottom_m"] + lower["top_m"]) / 2, 2)
        upper["bottom_m"] = boundary
        lower["top_m"] = boundary
    return pd.DataFrame(layers)


# ----------------------------------------------------------------------
# Summary figure
# ----------------------------------------------------------------------

def plot_interpretation(properties, layers, title="Joint RF + MASW interpretation"):
    """Four panels: velocities, Poisson's ratio, stiffness moduli and the
    interpreted material column, on a shared depth axis."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True,
                             gridspec_kw={"width_ratios": [1, 0.8, 1, 1.4]})
    ax_vel, ax_nu, ax_mod, ax_col = axes
    z = properties["depth_m"]

    ax_vel.plot(properties["vp_m_s"], z, "r-o", ms=3, label="$V_p$ (refraction)")
    ax_vel.plot(properties["vs_m_s"], z, "b-s", ms=3, label="$V_s$ (MASW)")
    ax_vel.set_xlabel("Velocity (m/s)")
    ax_vel.set_ylabel("Depth (m)")
    ax_vel.set_title("Wave velocities")
    ax_vel.legend(loc="lower right", fontsize=9)

    ax_nu.plot(properties["poisson_ratio"], z, color="darkviolet", lw=2)
    ax_nu.set_xlabel(r"Poisson's ratio $\nu$")
    ax_nu.set_title("Poisson's ratio")
    ax_nu.set_xlim(0.1, 0.5)

    ax_mod.plot(properties["shear_modulus_MPa"], z, "g-^", ms=3, label="$G$")
    ax_mod.plot(properties["youngs_modulus_MPa"], z, "-", color="navy", label="$E$")
    ax_mod.set_xlabel("Modulus (MPa)")
    ax_mod.set_title("Dynamic stiffness")
    ax_mod.legend(loc="lower right", fontsize=9)

    # Interpreted material column with labels.
    for _, layer in layers.iterrows():
        ax_col.axhspan(layer["top_m"], layer["bottom_m"],
                       color=layer["color"], alpha=0.8, ec="black", lw=0.8)
        ax_col.text(0.5, (layer["top_m"] + layer["bottom_m"]) / 2,
                    f'{layer["material"]}\n'
                    f'$V_p\\approx${layer["vp_mean_m_s"]:.0f} m/s, '
                    f'$V_s\\approx${layer["vs_mean_m_s"]:.0f} m/s, '
                    f'$\\nu\\approx${layer["poisson_mean"]:.2f}',
                    ha="center", va="center", fontsize=8, wrap=True,
                    transform=ax_col.get_yaxis_transform())
    ax_col.set_title("Interpreted materials")
    ax_col.set_xticks([])

    # Shade the layer boundaries across the property panels too.
    for ax in (ax_vel, ax_nu, ax_mod):
        for boundary in layers["bottom_m"].iloc[:-1]:
            ax.axhline(boundary, color="gray", ls="--", lw=0.8, alpha=0.7)

    ax_vel.set_ylim(z.max(), z.min())
    fig.suptitle(title)
    return fig
