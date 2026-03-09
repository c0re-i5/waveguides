"""
=============================================================================
  UNIFIED WAVE PHYSICS — One Equation, Three Worlds
  Sound, light, and water obey the SAME mathematics
=============================================================================

  Every script so far has used the same equation:

      ∂²u/∂t² = c(x)² · ∇²u

  The ONLY difference between acoustic, optical, and fluid
  simulations is what c(x) means:

      Sound: c = √(γP/ρ)    — gas pressure and density
      Light: c = c₀/n(x)     — speed of light / refractive index
      Water: c = √(g·h(x))   — gravity × depth

  This script proves that a single solver handles all three.
  Different physics, identical mathematics.

  Connection to repo 1: the same bit pattern can represent
  completely different values depending on interpretation.
  The same wave equation describes completely different physics
  depending on which c(x) you provide.
"""

import math
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


# ── Universal FDTD solver ────────────────────────────────────────────────────

def fdtd_1d(c_array, dx, dt, n_steps, initial_u, initial_v=None,
            bc_left="fixed", bc_right="fixed"):
    """
    Universal 1D FDTD solver.

    Parameters:
        c_array:   speed at each grid point (this is the ONLY physics input)
        dx:        spatial step size
        dt:        time step size
        n_steps:   number of time steps
        initial_u: initial displacement
        initial_v: initial velocity (optional)
        bc_left:   'fixed', 'free', or 'absorbing'
        bc_right:  'fixed', 'free', or 'absorbing'

    Returns:
        list of snapshot tuples (u_array, step_number)
    """
    N = len(c_array)
    u_curr = np.array(initial_u, dtype=float)

    if initial_v is not None:
        u_prev = u_curr - dt * np.array(initial_v, dtype=float)
    else:
        u_prev = u_curr.copy()

    snapshots = [(u_curr.copy(), 0)]
    snapshot_interval = max(1, n_steps // 5)

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

        if step % snapshot_interval == 0 or step == n_steps:
            snapshots.append((u_curr.copy(), step))

    return snapshots


# =============================================================================
# PART 1: THE UNIVERSAL EQUATION
# =============================================================================

def part1_universal_equation():
    print(f"\n{SEPARATOR}")
    print("  PART 1: ONE EQUATION, THREE WORLDS")
    print("  The universal wave equation")
    print(SEPARATOR)

    print("""
    The scalar wave equation in 1D:

        ∂²u/∂t² = c(x)² · ∂²u/∂x²

    What u and c mean in each domain:

    ┌─────────────┬──────────────────┬───────────────────────────┐
    │  Domain     │  u (field)       │  c (wave speed)           │
    ├─────────────┼──────────────────┼───────────────────────────┤
    │  Acoustic   │  pressure (Pa)   │  √(γP/ρ)  ≈ 343 m/s     │
    │  Optical    │  E-field (V/m)   │  c₀/n(x)  ≈ 2×10⁸ m/s   │
    │  Fluid      │  surface η (m)   │  √(g·h)   ≈ 1-200 m/s   │
    └─────────────┴──────────────────┴───────────────────────────┘

    The solver doesn't know which domain it's in.
    It just sees: c(x) array → same FDTD formula → answer.

    This is the power of mathematical abstraction: completely
    different physical systems, identical computational structure.
    """)


# =============================================================================
# PART 2: DIMENSIONAL ANALYSIS
# =============================================================================

def part2_dimensional_analysis():
    print(f"\n{SEPARATOR}")
    print("  PART 2: DIMENSIONAL ANALYSIS")
    print("  Normalizing away the physics")
    print(SEPARATOR)

    print("""
    The three domains span enormous ranges:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Domain     Speed (m/s)   Wavelength      Frequency            │
    ├─────────────────────────────────────────────────────────────────┤
    │  Acoustic   343           0.02-17 m       20 Hz - 20 kHz      │
    │  Optical    2×10⁸         400-700 nm      430-750 THz          │
    │  Fluid      1-200         100-1000 km     0.001-0.01 Hz       │
    └─────────────────────────────────────────────────────────────────┘

    Speeds differ by 10⁸, wavelengths by 10¹², frequencies by 10¹⁶!

    But we can NORMALIZE to dimensionless units:
      x' = x/L,  t' = t·c₀/L,  c'(x) = c(x)/c₀

    where L = characteristic length, c₀ = reference speed.
    """)

    # Show that all three look identical when normalized
    N = 400
    x = np.linspace(0, 1, N)
    dx = 1.0 / N

    # Acoustic: speed step from c₁ to c₂ (air to helium)
    c_acoustic = np.where(x < 0.5, 1.0, 2.67)   # normalized: air=1, helium=2.67

    # Optical: refractive index step (glass to air)
    c_optical = np.where(x < 0.5, 1.0, 1.5)      # normalized: glass=1, air=1.5

    # Fluid: depth step (deep to shelf)
    c_fluid = np.where(x < 0.5, 1.0, 0.316)      # normalized: c ∝ √depth

    print("    Normalized speed profiles (all just a step in c(x)):\n")

    for name, c_arr in [("Acoustic (air→helium)", c_acoustic),
                         ("Optical (glass→air)", c_optical),
                         ("Fluid (deep→shelf)", c_fluid)]:
        ascii_plot(c_arr.tolist(), width=56, height=5, label=f"c'(x) — {name}")
        print()

    # Simulate all three with the SAME solver
    dt = 0.4 * dx / max(c_acoustic.max(), c_optical.max(), c_fluid.max())

    # Same Gaussian pulse for all three
    sigma = 0.03
    x0 = 0.25
    initial = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    print(f"\n{SUBSEP}")
    print("    Same pulse, same solver, different c(x):")
    print(f"{SUBSEP}\n")

    for name, c_arr in [("ACOUSTIC", c_acoustic),
                         ("OPTICAL", c_optical),
                         ("FLUID", c_fluid)]:
        snapshots = fdtd_1d(c_arr, dx, dt, 250, initial,
                           bc_left="absorbing", bc_right="absorbing")

        print(f"    ── {name} ──\n")
        for u, step in snapshots[1:4]:
            label = f"step {step}"
            ascii_plot(u.tolist(), width=56, height=7, label=label)
            print()

    print("""
    All three show the SAME behavior: the pulse hits the interface
    and splits into reflected + transmitted. The ratios differ
    because c₁/c₂ differs — but the mechanism is IDENTICAL.

    This is Snell's law in all three domains:
      Reflected fraction R = (c₂ − c₁)² / (c₂ + c₁)²
    """)


# =============================================================================
# PART 3: IMPEDANCE MISMATCH — UNIVERSAL REFLECTION
# =============================================================================

def part3_impedance():
    print(f"\n{SEPARATOR}")
    print("  PART 3: UNIVERSAL IMPEDANCE MISMATCH")
    print("  One formula for all reflections")
    print(SEPARATOR)

    print("""
    When a wave crosses from medium 1 to medium 2:

    ┌─────────────────────────────────────────────────────────────┐
    │  Reflection coefficient:  R = (Z₂ − Z₁) / (Z₂ + Z₁)     │
    │  Transmission coefficient: T = 2Z₂ / (Z₂ + Z₁)           │
    │                                                             │
    │  where Z = impedance = ρ·c                                 │
    │                                                             │
    │  Energy: R² + (Z₁/Z₂)·T² = 1  (energy conservation)      │
    └─────────────────────────────────────────────────────────────┘

    Different domains, same formula:
    """)

    cases = [
        ("Air → Water",           "Acoustic", 413,     1.48e6,  "ρ·c"),
        ("Air → Steel",           "Acoustic", 413,     4.56e7,  "ρ·c"),
        ("Glass → Air",           "Optical",  1.5,     1.0,     "n"),
        ("Diamond → Air",         "Optical",  2.42,    1.0,     "n"),
        ("Deep ocean → Shelf",    "Fluid",    200,     31.6,    "√(g·h)·ρ"),
        ("Ocean → Coast (10m)",   "Fluid",    200,     9.9,     "√(g·h)·ρ"),
    ]

    print(f"    {'Transition':>22} {'Domain':>10} {'Z₁':>10} {'Z₂':>10} {'|R|':>8} {'|T|':>8} {'R²(%)':>8}")
    print(f"    {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 8}")

    for name, domain, z1, z2, unit in cases:
        R = (z2 - z1) / (z2 + z1)
        T = 2 * z2 / (z2 + z1)
        R_pct = R**2 * 100
        print(f"    {name:>22} {domain:>10} {z1:>10.1f} {z2:>10.1f} {abs(R):>8.4f} {abs(T):>8.4f} {R_pct:>7.1f}%")

    print("""
    Sound hitting water? 99.9% reflected.
    Light leaving glass? 4% reflected (that's why windows glare).
    Tsunami hitting shelf? Partially reflected, partially amplified.

    Same equation, same formula, wildly different scales.
    """)


# =============================================================================
# PART 4: WAVEGUIDE CONFINEMENT — SAME MECHANISM
# =============================================================================

def part4_confinement():
    print(f"\n{SEPARATOR}")
    print("  PART 4: WAVEGUIDE CONFINEMENT")
    print("  Three ways to trap a wave — all the same trick")
    print(SEPARATOR)

    print("""
    A waveguide confines waves along a path. In all three media,
    the mechanism is identical: a SLOW region surrounded by a FAST
    region (or vice versa with reflective boundaries).

    ┌─────────────────────────────────────────────────────────────┐
    │  Sound pipe:  walls reflect → wave trapped between walls   │
    │  Fiber optic: n_core > n_clad → total internal reflection  │
    │  Ocean ridge: deeper channel → waves focus along ridge     │
    └─────────────────────────────────────────────────────────────┘

    Let's simulate all three using the SAME solver:
    """)

    N = 500
    x = np.linspace(0, 1, N)
    dx = 1.0 / N

    # Create three waveguide speed profiles
    # (higher c outside the guide, lower c inside)
    guide_width = 0.15

    def make_guide(c_inside, c_outside):
        c = np.full(N, c_outside)
        center = 0.5
        mask = (x > center - guide_width) & (x < center + guide_width)
        c[mask] = c_inside
        return c

    # Acoustic: sound channel (low speed in center — like SOFAR channel)
    c_sound = make_guide(0.8, 1.0)

    # Optical: fiber (slow in core — higher refractive index)
    c_light = make_guide(0.67, 1.0)  # n_core=1.5 → c = c₀/1.5 = 0.67·c₀

    # Fluid: submarine channel (faster in deep trench)
    c_water = make_guide(1.0, 0.5)  # deep spot → higher c

    dt = 0.3 * dx / 1.0  # conservative CFL

    # Launch pulse inside the guide
    sigma = 0.02
    initial = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
    velocity = np.zeros(N)

    for name, c_arr in [("ACOUSTIC CHANNEL (SOFAR)", c_sound),
                        ("OPTICAL FIBER (Graded Index)", c_light),
                        ("SUBMARINE CHANNEL", c_water)]:
        print(f"\n    ── {name} ──")
        print(f"    Speed profile:")
        ascii_plot(c_arr.tolist(), width=56, height=5,
                  label="c(x) — guide region is the dip/peak")

        snaps = fdtd_1d(c_arr, dx, dt, 600, initial,
                       bc_left="absorbing", bc_right="absorbing")

        for u, step in snaps[-3:]:
            label = f"step {step}"
            ascii_plot(u.tolist(), width=56, height=7, label=label)
        print()

    print("""
    In each case, the wave tendency is governed by the speed profile.
    The solver didn't change — only c(x) did.

    This is why waveguide physics is UNIVERSAL: the guiding mechanism
    is an intrinsic property of the wave equation, not of any
    particular medium.
    """)


# =============================================================================
# PART 5: PARAMETER MAPPING TABLE
# =============================================================================

def part5_parameter_mapping():
    print(f"\n{SEPARATOR}")
    print("  PART 5: THE ROSETTA STONE")
    print("  Translating between sound, light, and water")
    print(SEPARATOR)

    print("""
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                   UNIFIED WAVE PARAMETER MAP                                │
    ├──────────────┬────────────────────┬────────────────────┬────────────────────┤
    │  Concept     │  Acoustic          │  Optical           │  Fluid             │
    ├──────────────┼────────────────────┼────────────────────┼────────────────────┤
    │  Field u     │  Pressure P        │  E-field           │  Surface η         │
    │  Speed c     │  √(γP₀/ρ)         │  c₀/n(x)           │  √(g·h(x))        │
    │  Impedance Z │  ρ·c              │  μ₀·c₀/n           │  ρ·c              │
    │  Waveguide   │  Pipe/duct         │  Fiber/slab        │  Channel/ridge     │
    │  Confinement │  Rigid walls       │  Total internal    │  Depth variation   │
    │              │                     │  reflection        │                    │
    │  Dispersion  │  Pipe cutoff       │  Chromatic/modal   │  Shallow water     │
    │  Source      │  Speaker/vibration │  Laser/LED         │  Earthquake/wind   │
    │  Detector    │  Microphone        │  Photodiode        │  Tide gauge        │
    │  Frequency   │  20 Hz - 20 kHz   │  430 - 750 THz     │  mHz - Hz          │
    │  Wavelength  │  mm - m            │  nm                │  m - km            │
    │  Typical c   │  343 m/s (air)     │  2×10⁸ m/s (glass) │  200 m/s (deep)    │
    │  Attenuation │  Viscous loss      │  Absorption/       │  Bottom friction   │
    │              │                     │  scattering        │                    │
    │  Nonlinear   │  Shock waves       │  Kerr effect       │  Breaking waves    │
    │  Resonance   │  Standing waves    │  Fabry-Pérot       │  Harbor modes      │
    │              │  in pipes           │  cavities          │                    │
    │  Fourier     │  Harmonics         │  WDM channels      │  Tidal components  │
    │  (repo 2)    │                     │                    │                    │
    │  Eikonal     │  Ray acoustics     │  Geometric optics  │  Tsunami travel    │
    │  (repo 3)    │                     │                    │  times             │
    └──────────────┴────────────────────┴────────────────────┴────────────────────┘

    Every ROW is the same physics with different names.
    Every COLUMN is a different physical system with the same math.
    """)


# =============================================================================
# PART 6: SAME SOLVER DEMO — ALL THREE AT ONCE
# =============================================================================

def part6_same_solver():
    print(f"\n{SEPARATOR}")
    print("  PART 6: ONE SOLVER, THREE SIMULATIONS")
    print("  Identical code, different parameters")
    print(SEPARATOR)

    print("""
    Let's prove it: the EXACT same function call, three times,
    with different c(x) arrays. That's the ONLY difference.
    """)

    N = 400
    x = np.linspace(0, 1, N)
    dx = 1.0 / N

    simulations = {
        "SOUND IN A PIPE (resonance)": {
            "c": np.ones(N) * 1.0,  # uniform speed
            "bc_left": "fixed",
            "bc_right": "fixed",
            "desc": "Fixed ends → standing waves (like script 02)",
        },
        "LIGHT IN A FIBER (guided pulse)": {
            "c": np.where(np.abs(x - 0.5) < 0.15, 0.67, 1.0),  # step-index
            "bc_left": "absorbing",
            "bc_right": "absorbing",
            "desc": "Slow core, fast cladding → guided mode (like script 03)",
        },
        "TSUNAMI (shoaling)": {
            "c": np.where(x < 0.6, 1.0, 1.0 - 0.8 * (x - 0.6) / 0.4),
            "bc_left": "absorbing",
            "bc_right": "fixed",  # coastline
            "desc": "Speed decreases → wave amplifies (like script 04)",
        },
    }

    sigma = 0.03
    x0 = 0.2
    initial = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    for name, params in simulations.items():
        c_arr = params["c"]
        dt = 0.4 * dx / np.max(c_arr)

        print(f"\n    ── {name} ──")
        print(f"    {params['desc']}")
        print(f"    BC: left={params['bc_left']}, right={params['bc_right']}\n")

        # **Same function call for all three**
        snaps = fdtd_1d(
            c_array=c_arr,
            dx=dx,
            dt=dt,
            n_steps=400,
            initial_u=initial,
            bc_left=params["bc_left"],
            bc_right=params["bc_right"],
        )

        for u, step in snaps[-3:]:
            label = f"step {step}"
            ascii_plot(u.tolist(), width=56, height=7, label=label)
        print()

    print("""
    Three completely different physical systems.
    One function: fdtd_1d(c_array, dx, dt, n_steps, initial, bc).

    The solver is PHYSICS-AGNOSTIC. It doesn't know if it's
    simulating sound, light, or water. It just propagates waves
    at whatever speed you tell it.

    This is the deep insight: waveguide physics is NOT three
    separate fields. It's ONE field with three vocabularies.
    """)


# =============================================================================
# PART 7: ENERGY CONSERVATION — THE ULTIMATE VALIDATOR
# =============================================================================

def part7_energy():
    print(f"\n{SEPARATOR}")
    print("  PART 7: ENERGY CONSERVATION")
    print("  The test every simulation must pass")
    print(SEPARATOR)

    print("""
    If a simulation conserves energy, it's probably right.
    If it doesn't, something is wrong.

    Energy in the wave equation:
      E = ½∫(∂u/∂t)² dx + ½∫c²(∂u/∂x)² dx
        = kinetic energy + potential energy
    """)

    N = 300
    x = np.linspace(0, 1, N)
    dx = 1.0 / N
    c = np.ones(N)
    dt = 0.5 * dx

    # Gaussian pulse
    initial = np.exp(-((x - 0.5) ** 2) / (2 * 0.03 ** 2))
    u_prev = initial.copy()
    u_curr = initial.copy()

    energies = []

    for step in range(500):
        u_next = np.zeros(N)
        for i in range(1, N - 1):
            r = c[i] * dt / dx
            u_next[i] = 2 * u_curr[i] - u_prev[i] + r**2 * (
                u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
            )
        # Fixed boundaries (energy-conserving)
        u_next[0] = 0
        u_next[N - 1] = 0

        # Compute energy
        dudt = (u_next - u_prev) / (2 * dt)
        dudx = np.gradient(u_curr, dx)
        KE = 0.5 * np.sum(dudt ** 2) * dx
        PE = 0.5 * np.sum((c * dudx) ** 2) * dx
        E = KE + PE
        energies.append(E)

        u_prev = u_curr.copy()
        u_curr = u_next.copy()

    E0 = energies[0]
    deviations = [(e - E0) / E0 * 100 for e in energies]

    ascii_plot(energies, width=56, height=9, label="Total energy over 500 steps")
    print()

    print(f"    Initial energy:  {energies[0]:.6f}")
    print(f"    Final energy:    {energies[-1]:.6f}")
    print(f"    Max deviation:   {max(abs(d) for d in deviations):.6f}%")

    print("""
    For fixed boundary conditions, the FDTD scheme conserves
    energy to machine precision. This is a consequence of the
    scheme being time-reversible (symplectic).

    Absorbing boundary conditions intentionally DRAIN energy
    (that's how they absorb). The energy loss equals
    the energy of the absorbed waves — by design.

    Energy conservation is the universal validation criterion.
    If your acoustic, optical, or fluid simulation conserves
    energy (with reflective boundaries), the physics is right.
    """)


# =============================================================================
# PART 8: CROSS-REPO CONNECTIONS
# =============================================================================

def part8_connections():
    print(f"\n{SEPARATOR}")
    print("  PART 8: THE CONNECTIONS")
    print("  How all four repos form one story")
    print(SEPARATOR)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  REPO 1: Bit Tricks and the Fast Inverse Square Root       │
    │                                                             │
    │  → 1/√x appears in normalization everywhere                │
    │  → IEEE 754 precision limits our finite differences         │
    │  → The bit-level trick is a LOOKUP TABLE — like how we     │
    │    use precomputed c(x) arrays instead of computing speed  │
    │    from equations of state every time step                  │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  REPO 2: Bit Tricks and Wave Functions                     │
    │                                                             │
    │  → Fourier transforms decompose waves into frequencies     │
    │  → Sampling theorem sets the grid resolution (≥2 pts/λ)   │
    │  → Standing waves = superposition of Fourier modes         │
    │  → WDM in fibers = frequency multiplexing = FFT's domain  │
    │  → Pipe resonances = the harmonic series (Fourier basis)   │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  REPO 3: Navigational Pathfinding                          │
    │                                                             │
    │  → The Eikonal equation (|∇T| = 1/c) computes travel      │
    │    times — exactly what Dijkstra's algorithm does on a     │
    │    graph                                                    │
    │  → Wavefronts are isosurfaces of the travel time field     │
    │  → Fermat's principle (fastest path) = shortest-path       │
    │    algorithm on a speed-weighted domain                    │
    │  → Tsunami warning systems compute wave arrival times      │
    │    using the same Dijkstra-like wave travel time method    │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  REPO 4: Waveguides (this repo)                            │
    │                                                             │
    │  → The wave equation unifies all three previous repos      │
    │  → Same equation governs sound, light, and water           │
    │  → FDTD turns the continuous equation into discrete code   │
    │  → Stability conditions from numerical analysis            │
    │  → One solver handles any medium with the right c(x)       │
    └─────────────────────────────────────────────────────────────┘

    Together: a number (repo 1) sampled and transformed (repo 2),
    propagated through space (repo 3), and simulated as a physical
    wave (repo 4). The same math keeps reappearing because
    it's the same universe.
    """)


# =============================================================================
# PART 9: SUMMARY
# =============================================================================

def part9_summary():
    print(f"\n{SEPARATOR}")
    print("  SUMMARY: UNIFIED WAVE PHYSICS")
    print(SEPARATOR)

    print("""
    This script demonstrated the core thesis of the repo:

      ✓ One wave equation: ∂²u/∂t² = c(x)²·∇²u
      ✓ Three physical domains: sound, light, water
      ✓ Same solver handles all three: just change c(x)
      ✓ Universal impedance mismatch: R = (Z₂−Z₁)/(Z₂+Z₁)
      ✓ Universal confinement: slow region guides waves
      ✓ Universal validation: energy conservation
      ✓ Parameter mapping: every concept translates across domains

    ┌─────────────────────────────────────────────────────────────┐
    │  NEXT: Script 07 — THE GREAT WAVE                          │
    │  The capstone: a full multi-medium simulation.             │
    │  An earthquake generates a tsunami, sound reverberates     │
    │  through a channel, and light pulses through a fiber.      │
    │  Three worlds, one equation, one solver.                   │
    └─────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{SEPARATOR}")
    print("  UNIFIED WAVE PHYSICS — One Equation, Three Worlds")
    print("  Sound, light, and water: the same mathematics")
    print(SEPARATOR)

    part1_universal_equation()
    part2_dimensional_analysis()
    part3_impedance()
    part4_confinement()
    part5_parameter_mapping()
    part6_same_solver()
    part7_energy()
    part8_connections()
    part9_summary()

    print(f"\n{SEPARATOR}")
    print("  Done. Run 07_the_great_wave.py for the capstone.")
    print(SEPARATOR)
