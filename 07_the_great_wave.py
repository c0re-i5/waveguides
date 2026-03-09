"""
=============================================================================
  THE GREAT WAVE — Capstone Project
  One earthquake, three media, one equation
=============================================================================

  This is the grand finale: a multi-medium simulation that
  demonstrates the central thesis of this entire repo.

  We simulate three scenarios with ONE universal FDTD solver:

    1. TSUNAMI (fluid): earthquake → ocean wave → coastal amplification
    2. ACOUSTIC BOOM (sound): explosion → sound through a duct → resonance
    3. LIGHT PULSE (optical): laser → fiber optic → signal detection

  All three use the SAME wave equation, the SAME solver code,
  and the SAME numerical methods from scripts 01-06.

  The only difference: the speed array c(x).

  "The Great Wave off Kanagawa" by Hokusai (c. 1831) depicts
  a towering wave — the same physics that governs tsunamis,
  sound in pipes, and light in fibers.

  ═══════════════════════════════════════════════════════════════
  This is the 4th repo in the series:
    1. fast-inverse-square-root  — the number
    2. bit-tricks-and-wave-functions — the transform
    3. navigational-pathfinding — the path
    4. waveguides — the wave
  ═══════════════════════════════════════════════════════════════
"""

import math
import time
import numpy as np

SEPARATOR = "=" * 72
SUBSEP = "─" * 72


# ── Visualization helpers ────────────────────────────────────────────────────

def ascii_plot(values, width=64, height=13, label="", show_zero=True):
    """Render a 1D signal as ASCII art."""
    n = len(values)
    if n == 0:
        return

    min_val = min(values)
    max_val = max(values)
    if abs(max_val - min_val) < 1e-12:
        max_val = min_val + 1.0

    indices = [int(i * (n - 1) / (width - 1)) for i in range(width)]
    sampled = [values[idx] for idx in indices]

    grid = [[" " for _ in range(width)] for _ in range(height)]

    if show_zero and min_val < 0 < max_val:
        zero_row = int((max_val - 0) / (max_val - min_val) * (height - 1))
        zero_row = max(0, min(height - 1, zero_row))
        for c in range(width):
            grid[zero_row][c] = "·"

    for c in range(width):
        val = sampled[c]
        row = int((max_val - val) / (max_val - min_val) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][c] = "█"

    for c in range(1, width):
        row_prev = int((max_val - sampled[c - 1]) / (max_val - min_val) * (height - 1))
        row_curr = int((max_val - sampled[c]) / (max_val - min_val) * (height - 1))
        row_prev = max(0, min(height - 1, row_prev))
        row_curr = max(0, min(height - 1, row_curr))
        for r in range(min(row_prev, row_curr), max(row_prev, row_curr) + 1):
            if grid[r][c] == " ":
                grid[r][c] = "│"

    for r in range(height):
        if r == 0:
            axis = f"{max_val:>8.3f}"
        elif r == height - 1:
            axis = f"{min_val:>8.3f}"
        elif r == height // 2:
            axis = f"{(max_val + min_val) / 2:>8.3f}"
        else:
            axis = "        "
        print(f"    {axis} ┤{''.join(grid[r])}")
    print(f"    {'':>8} └{'─' * width}")
    if label:
        padding = max(0, (width - len(label)) // 2)
        print(f"    {'':>8}  {' ' * padding}{label}")


def ascii_2d(grid_data, width=60, height=25, label=""):
    """Render 2D scalar field with block brightness."""
    arr = np.array(grid_data)
    if arr.size == 0:
        return

    ny, nx = arr.shape
    chars = " ░▒▓█"

    max_abs = np.max(np.abs(arr))
    if max_abs < 1e-15:
        max_abs = 1.0

    row_indices = [int(r * (ny - 1) / (height - 1)) for r in range(height)]
    col_indices = [int(c * (nx - 1) / (width - 1)) for c in range(width)]

    print(f"    ┌{'─' * width}┐")
    for ri in row_indices:
        row_str = ""
        for ci in col_indices:
            val = arr[ri, ci]
            norm = abs(val) / max_abs
            idx = min(int(norm * (len(chars) - 1)), len(chars) - 1)
            row_str += chars[idx]
        print(f"    │{row_str}│")
    print(f"    └{'─' * width}┘")
    if label:
        padding = max(0, (width - len(label)) // 2 + 5)
        print(f"{' ' * padding}{label}")


# ── Universal FDTD Solver ────────────────────────────────────────────────────

def fdtd_1d(c_array, dx, dt, n_steps, initial_u, initial_v=None,
            bc_left="absorbing", bc_right="absorbing",
            snapshot_interval=None):
    """
    Universal 1D FDTD solver — the SAME function for all three media.

    This is the central thesis made code: one solver, three worlds.
    """
    N = len(c_array)
    u_curr = np.array(initial_u, dtype=float)

    if initial_v is not None:
        u_prev = u_curr - dt * np.array(initial_v, dtype=float)
    else:
        u_prev = u_curr.copy()

    if snapshot_interval is None:
        snapshot_interval = max(1, n_steps // 6)

    snapshots = [(u_curr.copy(), 0)]
    max_amplitudes = [np.max(np.abs(u_curr))]

    for step in range(1, n_steps + 1):
        u_next = np.zeros(N)

        for i in range(1, N - 1):
            r = c_array[i] * dt / dx
            u_next[i] = 2 * u_curr[i] - u_prev[i] + r**2 * (
                u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
            )

        # Boundary conditions
        if bc_left == "fixed":
            u_next[0] = 0
        elif bc_left == "free":
            u_next[0] = u_next[1]
        elif bc_left == "absorbing":
            r0 = c_array[0] * dt / dx
            u_next[0] = u_curr[1] + (r0 - 1)/(r0 + 1) * (u_next[1] - u_curr[0])

        if bc_right == "fixed":
            u_next[N - 1] = 0
        elif bc_right == "free":
            u_next[N - 1] = u_next[N - 2]
        elif bc_right == "absorbing":
            rn = c_array[N - 1] * dt / dx
            u_next[N - 1] = u_curr[N - 2] + (rn - 1)/(rn + 1) * (u_next[N - 2] - u_curr[N - 1])

        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        max_amplitudes.append(np.max(np.abs(u_curr)))

        if step % snapshot_interval == 0 or step == n_steps:
            snapshots.append((u_curr.copy(), step))

    return snapshots, max_amplitudes


def fdtd_2d(c_field, dx, dt, n_steps, initial_u,
            bc="absorbing", snapshot_interval=None):
    """
    Universal 2D FDTD solver.

    c_field: 2D array of wave speeds c(x,y)
    """
    Ny, Nx = c_field.shape
    u_curr = np.array(initial_u, dtype=float)
    u_prev = u_curr.copy()

    if snapshot_interval is None:
        snapshot_interval = max(1, n_steps // 4)

    snapshots = [(u_curr.copy(), 0)]

    for step in range(1, n_steps + 1):
        u_next = np.zeros((Ny, Nx))

        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                r_sq = (c_field[i, j] * dt / dx) ** 2
                u_next[i, j] = (
                    2 * u_curr[i, j] - u_prev[i, j]
                    + r_sq * (
                        u_curr[i+1, j] + u_curr[i-1, j]
                        + u_curr[i, j+1] + u_curr[i, j-1]
                        - 4 * u_curr[i, j]
                    )
                )

        # Boundary conditions
        if bc == "fixed":
            u_next[0, :] = 0
            u_next[-1, :] = 0
            u_next[:, 0] = 0
            u_next[:, -1] = 0
        elif bc == "absorbing":
            # Simple first-order absorbing on all edges
            r_edge = c_field[0, :] * dt / dx
            u_next[0, 1:-1] = u_curr[1, 1:-1] + (r_edge[1:-1] - 1)/(r_edge[1:-1] + 1) * (u_next[1, 1:-1] - u_curr[0, 1:-1])
            r_edge = c_field[-1, :] * dt / dx
            u_next[-1, 1:-1] = u_curr[-2, 1:-1] + (r_edge[1:-1] - 1)/(r_edge[1:-1] + 1) * (u_next[-2, 1:-1] - u_curr[-1, 1:-1])
            r_edge = c_field[:, 0] * dt / dx
            u_next[1:-1, 0] = u_curr[1:-1, 1] + (r_edge[1:-1] - 1)/(r_edge[1:-1] + 1) * (u_next[1:-1, 1] - u_curr[1:-1, 0])
            r_edge = c_field[:, -1] * dt / dx
            u_next[1:-1, -1] = u_curr[1:-1, -2] + (r_edge[1:-1] - 1)/(r_edge[1:-1] + 1) * (u_next[1:-1, -2] - u_curr[1:-1, -1])

        u_prev = u_curr
        u_curr = u_next.copy()

        if step % snapshot_interval == 0 or step == n_steps:
            snapshots.append((u_curr.copy(), step))

    return snapshots


# =============================================================================
# SIMULATION 1: THE TSUNAMI
# =============================================================================

def simulation_tsunami():
    print(f"\n{SEPARATOR}")
    print("  ☸  SIMULATION 1: TSUNAMI")
    print("  Earthquake → ocean crossing → coastal amplification")
    print(SEPARATOR)

    print("""
    A magnitude 9.0 earthquake displaces the ocean floor near a
    subduction zone. The resulting wave crosses 4000 km of deep
    ocean, hits a continental shelf, slows down, and amplifies.

    Physical parameters (real scale, normalized):
      • Deep ocean: depth = 4000 m → c = √(9.81·4000) = 198 m/s
      • Continental shelf: depth = 200 m → c = 44 m/s
      • Coastal shallows: depth = 10 m → c = 9.9 m/s

    Normalized: c_deep = 1.0, c_shelf = 0.22, c_coast = 0.05

    Key physics: c ∝ √(depth), so as depth decreases:
      • Wave SLOWS DOWN
      • Wavelength COMPRESSES
      • Amplitude INCREASES (Green's law: A ∝ h^{-1/4})
    """)

    # ── 1D Tsunami Simulation ──
    print(f"\n{SUBSEP}")
    print("    1D Cross-Section: Deep Ocean → Shelf → Coast")
    print(SUBSEP)

    N = 800
    x = np.linspace(0, 1, N)
    dx = 1.0 / N

    # Bathymetric profile: deep → shelf → coast
    depth = np.ones(N)
    for i in range(N):
        if x[i] < 0.65:
            depth[i] = 1.0       # deep ocean
        elif x[i] < 0.78:
            # continental slope
            t = (x[i] - 0.65) / 0.13
            depth[i] = 1.0 - 0.95 * t
        elif x[i] < 0.92:
            depth[i] = 0.05      # continental shelf
        else:
            # coastal shallows
            t = (x[i] - 0.92) / 0.08
            depth[i] = 0.05 - 0.045 * t
            depth[i] = max(depth[i], 0.003)  # minimum depth

    c_tsunami = np.sqrt(depth)  # c = √(g·h), normalized

    print("\n    Depth profile (deep ocean → continental shelf → coast):")
    ascii_plot(depth.tolist(), width=60, height=7,
              label="Normalized depth h(x)")

    print("\n    Wave speed profile (c = √(g·h)):")
    ascii_plot(c_tsunami.tolist(), width=60, height=7,
              label="c(x) — wave speed")

    dt = 0.35 * dx / np.max(c_tsunami)

    # Earthquake source: broad uplift
    sigma = 0.04
    x_eq = 0.15  # earthquake near the left edge
    initial = np.exp(-((x - x_eq) ** 2) / (2 * sigma ** 2))

    snaps, amplitudes = fdtd_1d(
        c_tsunami, dx, dt, n_steps=1200,
        initial_u=initial,
        bc_left="absorbing", bc_right="fixed",  # coastline = rigid wall
        snapshot_interval=200,
    )

    for u, step in snaps:
        label = f"step {step}  |  max|u| = {np.max(np.abs(u)):.3f}"
        ascii_plot(u.tolist(), width=60, height=9, label=label)
        print()

    # Show amplification
    print("    Wave amplification as it approaches the coast:")
    print(f"    Initial amplitude: {amplitudes[0]:.3f}")
    print(f"    Maximum amplitude: {max(amplitudes):.3f}")
    print(f"    Amplification factor: {max(amplitudes)/amplitudes[0]:.1f}×")

    print("""
    The tsunami is barely noticeable in the deep ocean (< 1 m tall
    in real life), but Green's law amplifies it dramatically as it
    enters shallow water. A 0.5 m deep-ocean wave can become a
    10 m coastal wave — the physics of the 2004 and 2011 disasters.
    """)

    # ── 2D Tsunami Simulation ──
    print(f"\n{SUBSEP}")
    print("    2D Tsunami: Ocean with Island and Submarine Ridge")
    print(SUBSEP)

    Nx, Ny = 80, 80
    dx2 = 1.0 / Nx
    x2 = np.linspace(0, 1, Nx)
    y2 = np.linspace(0, 1, Ny)
    xx, yy = np.meshgrid(x2, y2)

    # Create bathymetry: deep ocean with a submarine ridge and an island
    depth_2d = np.ones((Ny, Nx)) * 1.0  # deep ocean

    # Continental shelf on right edge
    for j in range(Nx):
        if x2[j] > 0.75:
            t = (x2[j] - 0.75) / 0.25
            depth_2d[:, j] = 1.0 - 0.9 * t
            depth_2d[:, j] = np.maximum(depth_2d[:, j], 0.05)

    # Submarine ridge (diagonal) — acts as a waveguide
    ridge_mask = np.abs(yy - 0.6 * xx - 0.2) < 0.05
    depth_2d[ridge_mask] = 0.3  # shallower along ridge

    # Island (circular shallow spot)
    island_mask = (xx - 0.55) ** 2 + (yy - 0.5) ** 2 < 0.02
    depth_2d[island_mask] = 0.02  # very shallow

    c_2d = np.sqrt(depth_2d)
    dt2 = 0.3 * dx2 / np.max(c_2d)

    # Earthquake source
    source = np.exp(-((xx - 0.1)**2 + (yy - 0.5)**2) / (2 * 0.04**2))

    snaps_2d = fdtd_2d(c_2d, dx2, dt2, 200, source,
                       bc="absorbing", snapshot_interval=40)

    print("\n    Bathymetry (depth profile):")
    ascii_2d(depth_2d, width=50, height=18, label="Depth — dark = deep, bright = shallow")

    print("\n    Tsunami propagation snapshots:")
    for u2d, step in snaps_2d[1:]:
        ascii_2d(u2d, width=50, height=18,
                label=f"step {step} — wave amplitude")
        print()

    print("""
    The 2D simulation shows:
      • Waves slow down and refract over the submarine ridge
      • The island creates a shadow zone behind it (diffraction)
      • Waves amplify as they approach the continental shelf
      • The ridge acts as a WAVEGUIDE, focusing wave energy

    Same equation: ∂²η/∂t² = c(x,y)² · ∇²η with c = √(g·h(x,y))
    """)


# =============================================================================
# SIMULATION 2: ACOUSTIC BOOM
# =============================================================================

def simulation_acoustic():
    print(f"\n{SEPARATOR}")
    print("  ♪  SIMULATION 2: ACOUSTIC BOOM")
    print("  Explosion → duct propagation → resonant chamber")
    print(SEPARATOR)

    print("""
    An impulsive pressure wave (from an explosion or loud clap)
    enters a duct system. The duct narrows, causing the wave to
    partially reflect and partially transmit. At the end, a
    resonant chamber creates standing waves.

    Physical parameters:
      • Wide duct:   c = 343 m/s (air)
      • Narrow duct: c = 343 m/s (same speed, but impedance changes)
      • Helium fill: c = 1007 m/s (lighter gas, faster sound)

    For this simulation, we model impedance changes through
    the effective speed profile of the waveguide.
    """)

    N = 600
    x = np.linspace(0, 1, N)
    dx = 1.0 / N

    # Speed profile: uniform duct → impedance step → resonant cavity
    c_acoustic = np.ones(N)
    for i in range(N):
        if x[i] < 0.35:
            c_acoustic[i] = 1.0       # wide duct (air)
        elif x[i] < 0.45:
            c_acoustic[i] = 1.5       # helium-filled section (faster)
        elif x[i] < 0.65:
            c_acoustic[i] = 1.0       # air again
        elif x[i] < 0.70:
            c_acoustic[i] = 0.6       # narrow constriction (effective slower speed)
        else:
            c_acoustic[i] = 1.0       # resonant chamber

    dt = 0.4 * dx / np.max(c_acoustic)

    print("    Speed profile along the duct system:")
    ascii_plot(c_acoustic.tolist(), width=60, height=7,
              label="c(x) — air | helium | air | constriction | chamber")

    # Impulsive source (sharp pressure spike)
    sigma = 0.015
    initial = np.exp(-((x - 0.1) ** 2) / (2 * sigma ** 2))

    snaps, amplitudes = fdtd_1d(
        c_acoustic, dx, dt, n_steps=800,
        initial_u=initial,
        bc_left="absorbing", bc_right="fixed",  # chamber end = rigid wall
        snapshot_interval=100,
    )

    for u, step in snaps:
        label = f"step {step}"
        ascii_plot(u.tolist(), width=60, height=9, label=label)
        print()

    # Frequency analysis at a detector point
    print(f"\n{SUBSEP}")
    print("    Frequency Analysis at Chamber (Fourier — repo 2)")
    print(SUBSEP)

    # Re-run to collect time series at a detector point
    detector_pos = int(0.85 * N)
    u_curr = initial.copy()
    u_prev = initial.copy()
    time_series = []

    for step in range(1500):
        u_next = np.zeros(N)
        for i in range(1, N - 1):
            r = c_acoustic[i] * dt / dx
            u_next[i] = 2 * u_curr[i] - u_prev[i] + r**2 * (
                u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
            )
        u_next[0] = u_curr[1] + (c_acoustic[0]*dt/dx - 1)/(c_acoustic[0]*dt/dx + 1) * (u_next[1] - u_curr[0])
        u_next[N - 1] = 0  # rigid end

        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        time_series.append(u_curr[detector_pos])

    # FFT
    spectrum = np.abs(np.fft.rfft(time_series))
    freqs = np.fft.rfftfreq(len(time_series), d=dt)
    # Show up to reasonable frequency
    max_freq_idx = min(len(spectrum), int(len(spectrum) * 0.3))
    spec_plot = spectrum[:max_freq_idx].tolist()

    print("\n    Pressure time series at detector (x = 0.85):")
    ascii_plot(time_series[:500], width=60, height=7,
              label="Pressure vs time")

    print("\n    Frequency spectrum (FFT):")
    ascii_plot(spec_plot, width=60, height=9,
              label="Frequency spectrum — peaks are resonant modes")

    # Find peaks
    if len(spec_plot) > 2:
        peaks = []
        for i in range(1, len(spec_plot) - 1):
            if spec_plot[i] > spec_plot[i-1] and spec_plot[i] > spec_plot[i+1]:
                if spec_plot[i] > max(spec_plot) * 0.1:
                    peaks.append((freqs[i], spec_plot[i]))

        if peaks:
            print(f"\n    Detected resonant frequencies:")
            for f, amp in peaks[:6]:
                print(f"      f = {f:.3f}  (amplitude: {amp:.1f})")

    print("""
    The acoustic system shows:
      • Partial reflections at each impedance change
      • Resonant frequencies in the chamber (standing waves)
      • The helium section transmits the wave faster
      • The constriction acts as a filter (impedance mismatch)

    Same solver: fdtd_1d(c_acoustic, dx, dt, ...)
    Same equation: ∂²P/∂t² = c(x)² · ∂²P/∂x²
    """)


# =============================================================================
# SIMULATION 3: LIGHT PULSE THROUGH FIBER
# =============================================================================

def simulation_optical():
    print(f"\n{SEPARATOR}")
    print("  ◉  SIMULATION 3: LIGHT PULSE")
    print("  Laser → fiber optic → detector")
    print(SEPARATOR)

    print("""
    A short laser pulse is coupled into a multimode fiber optic.
    The fiber has a core (n = 1.48) and cladding (n = 1.46).
    As the pulse travels, different modes propagate at slightly
    different speeds, causing modal dispersion: the pulse
    broadens over distance.

    Physical parameters (normalized):
      • Core:     n = 1.48 → c = c₀/1.48 ≈ 0.676·c₀
      • Cladding: n = 1.46 → c = c₀/1.46 ≈ 0.685·c₀
    """)

    N = 800
    x = np.linspace(0, 1, N)
    dx = 1.0 / N

    # Step-index fiber profile
    core_width = 0.08
    c_fiber = np.ones(N) * (1.0 / 1.46)  # cladding speed
    for i in range(N):
        if abs(x[i] - 0.5) < core_width:
            c_fiber[i] = 1.0 / 1.48  # core speed (slower = higher n)

    dt = 0.4 * dx / np.max(c_fiber)

    print("    Refractive index profile (fiber cross-section along x):")
    n_profile = [1.0 / c for c in c_fiber]
    ascii_plot(n_profile, width=60, height=7,
              label="n(x) — fiber core has higher refractive index")

    # Narrow pulse (simulates short laser pulse in the core)
    sigma = 0.012
    initial = np.exp(-((x - 0.25) ** 2) / (2 * sigma ** 2))
    # Only excite the core region
    for i in range(N):
        if abs(x[i] - 0.5) >= core_width * 1.5:
            initial[i] *= 0.1  # mostly in core

    # Actually, for 1D fiber sim, let's do longitudinal propagation
    # along the fiber axis with a waveguide speed model
    print(f"\n{SUBSEP}")
    print("    Longitudinal Propagation Along Fiber Axis")
    print(SUBSEP)

    # Model the fiber as a longitudinal 1D domain where
    # the effective propagation speed depends on modal structure
    N_long = 800
    x_long = np.linspace(0, 1, N_long)
    dx_long = 1.0 / N_long

    # Effective speed profile along fiber length
    # Include some bends/splices with impedance mismatches
    c_long = np.ones(N_long) * 0.676  # core effective speed

    # Splice point at x = 0.4 (small impedance bump)
    for i in range(N_long):
        if 0.39 < x_long[i] < 0.41:
            c_long[i] = 0.67  # slight mismatch at splice

    # Bend loss region at x = 0.65
    for i in range(N_long):
        if 0.64 < x_long[i] < 0.68:
            c_long[i] = 0.680  # slight speed change at bend

    dt_long = 0.4 * dx_long / np.max(c_long)

    # Short laser pulse
    sigma_pulse = 0.015
    pulse = np.exp(-((x_long - 0.1) ** 2) / (2 * sigma_pulse ** 2))

    snaps_fiber, amps_fiber = fdtd_1d(
        c_long, dx_long, dt_long, n_steps=1000,
        initial_u=pulse,
        bc_left="absorbing", bc_right="absorbing",
        snapshot_interval=150,
    )

    for u, step in snaps_fiber:
        label = f"step {step}"
        ascii_plot(u.tolist(), width=60, height=9, label=label)
        print()

    # Show pulse broadening
    initial_width = sigma_pulse
    final_snap = snaps_fiber[-1][0]
    final_nonzero = np.where(np.abs(final_snap) > 0.01 * np.max(np.abs(final_snap)))[0]
    if len(final_nonzero) > 1:
        final_width = (final_nonzero[-1] - final_nonzero[0]) * dx_long
    else:
        final_width = initial_width

    print(f"    Initial pulse width: {initial_width:.4f} (normalized)")
    print(f"    Final pulse width:   {final_width:.4f} (normalized)")
    print(f"    Broadening factor:   {final_width/initial_width:.1f}×")

    # WDM demonstration: two wavelengths at different speeds
    print(f"\n{SUBSEP}")
    print("    WDM Demo: Two Wavelengths, Slightly Different Speeds")
    print(SUBSEP)

    print("""
    Wavelength Division Multiplexing sends multiple signals at
    different wavelengths. Chromatic dispersion means each
    wavelength sees a slightly different refractive index.
    """)

    # Channel 1: slightly faster (shorter wavelength)
    c_ch1 = np.ones(N_long) * 0.676
    # Channel 2: slightly slower (longer wavelength)
    c_ch2 = np.ones(N_long) * 0.670

    pulse1 = np.exp(-((x_long - 0.10) ** 2) / (2 * 0.02 ** 2))
    pulse2 = np.exp(-((x_long - 0.10) ** 2) / (2 * 0.02 ** 2))

    snaps1, _ = fdtd_1d(c_ch1, dx_long, dt_long, 600, pulse1,
                        bc_left="absorbing", bc_right="absorbing",
                        snapshot_interval=600)
    snaps2, _ = fdtd_1d(c_ch2, dx_long, dt_long, 600, pulse2,
                        bc_left="absorbing", bc_right="absorbing",
                        snapshot_interval=600)

    print("\n    After propagation:")
    combined = snaps1[-1][0] + snaps2[-1][0]
    ascii_plot(snaps1[-1][0].tolist(), width=60, height=7,
              label="λ₁ (1550 nm) — faster")
    print()
    ascii_plot(snaps2[-1][0].tolist(), width=60, height=7,
              label="λ₂ (1310 nm) — slower")
    print()
    ascii_plot(combined.tolist(), width=60, height=7,
              label="Combined WDM signal")

    print("""
    The two wavelengths separate in time due to chromatic dispersion.
    At the receiver, they're demultiplexed — repo 2's Fourier
    analysis applied to light.

    Same solver: fdtd_1d(c_fiber, dx, dt, ...)
    Same equation: ∂²E/∂t² = c(x)² · ∂²E/∂x² with c = c₀/n(x)
    """)


# =============================================================================
# GRAND UNIFIED SUMMARY
# =============================================================================

def grand_summary():
    print(f"\n{SEPARATOR}")
    print("  THE GRAND UNIFICATION")
    print("  Three simulations, one equation")
    print(SEPARATOR)

    print("""
    All three simulations used the EXACT SAME function:

        fdtd_1d(c_array, dx, dt, n_steps, initial_u, bc_left, bc_right)

    The ONLY difference was the c_array parameter:

    ┌───────────────────────────────────────────────────────────────────┐
    │  Simulation    │  c(x) source              │  Key phenomenon     │
    ├───────────────────────────────────────────────────────────────────┤
    │  TSUNAMI       │  c = √(g·h(x))            │  Shoaling,          │
    │                │  depth → speed             │  amplification      │
    │                │                             │                     │
    │  ACOUSTIC      │  c = √(γP/ρ)              │  Impedance          │
    │                │  gas properties → speed     │  mismatch,          │
    │                │                             │  resonance          │
    │                │                             │                     │
    │  OPTICAL       │  c = c₀/n(x)               │  Modal dispersion,  │
    │                │  refractive index → speed   │  pulse broadening   │
    └───────────────────────────────────────────────────────────────────┘

    The solver doesn't know which universe it's in.
    It just propagates waves at the speed you specify.

    This is the deepest insight of mathematical physics:
    the STRUCTURE of the equation determines the behavior,
    not the interpretation of its variables.
    """)

    # Final connections box
    print(f"\n{SUBSEP}")
    print("    THE FOUR REPOS — ONE STORY")
    print(SUBSEP)

    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  REPO 1: The Number                                            │
    │  ─────────────────                                             │
    │  A real number stored as bits. The fast inverse square root.   │
    │  Lesson: representation matters. Finite precision is real.     │
    │                                                                 │
    │  REPO 2: The Transform                                         │
    │  ────────────────────                                          │
    │  Fourier analysis decomposes any signal into frequencies.      │
    │  Lesson: every wave is a sum of simpler waves. The FFT is     │
    │  the most important algorithm in signal processing.            │
    │                                                                 │
    │  REPO 3: The Path                                              │
    │  ──────────────                                                │
    │  Dijkstra, A*, and the Eikonal equation find shortest paths.  │
    │  Lesson: wave propagation IS pathfinding. Wavefronts and      │
    │  travel-time isosurfaces are the same thing.                   │
    │                                                                 │
    │  REPO 4: The Wave  ← YOU ARE HERE                              │
    │  ──────────────────                                            │
    │  The wave equation unifies sound, light, and water.            │
    │  Lesson: one equation, one solver, three worlds.               │
    │  The same c(x) array governs everything.                       │
    │                                                                 │
    │  TOGETHER:                                                      │
    │  A number (repo 1) is sampled and transformed (repo 2),       │
    │  propagated through space (repo 3), and simulated as           │
    │  a physical wave (repo 4).                                     │
    │                                                                 │
    │  ∂²u/∂t² = c² · ∇²u                                           │
    │                                                                 │
    │  That's it. That's the whole thing.                             │
    └─────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{SEPARATOR}")
    print("  THE GREAT WAVE — Capstone Project")
    print("  One earthquake, three media, one equation")
    print(SEPARATOR)

    print("""
    This capstone simulates waves in three different media
    using a SINGLE universal FDTD solver. Each simulation
    follows the recipe from Script 05, with different c(x).

    Medium 1: Water (tsunami)   — c = √(g·h)
    Medium 2: Air   (acoustic)  — c = √(γP/ρ)
    Medium 3: Glass (optical)   — c = c₀/n
    """)

    t_start = time.perf_counter()

    simulation_tsunami()
    simulation_acoustic()
    simulation_optical()
    grand_summary()

    elapsed = time.perf_counter() - t_start

    print(f"\n{SEPARATOR}")
    print(f"  Capstone complete. Total time: {elapsed:.1f}s")
    print(f"  Three media. One equation. One solver.")
    print(SEPARATOR)
    print()
