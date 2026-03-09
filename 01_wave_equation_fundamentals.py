"""
=============================================================================
  WAVE EQUATION FUNDAMENTALS — One Equation to Rule Them All
  The wave equation derived, visualized, and understood
=============================================================================

  The wave equation:

      ∂²u/∂t² = c² · ∂²u/∂x²

  governs sound, light, water waves, earthquakes, and vibrating strings.
  This script derives it from first principles, shows its solutions,
  and builds intuition for superposition, reflection, and refraction —
  the foundations everything else in this repo builds on.

  Connection to repo 2: the signals in signals_and_sampling.py are
  solutions to this equation. Fourier decomposition works because
  the wave equation is LINEAR — solutions add.

  Connection to repo 3: wave propagation in 03_wave_propagation.py
  is the discrete version of this equation on a grid.
"""

import math
import numpy as np

SEPARATOR = "=" * 72
SUBSEP = "─" * 72


# ── Visualization helpers ────────────────────────────────────────────────────

def ascii_plot(values, width=64, height=15, label="", show_zero=True):
    """
    Render a 1D signal as ASCII art.

    Uses block characters for waveform display. For each column,
    maps the signal value to a row and places a marker.
    """
    n = len(values)
    if n == 0:
        return

    min_val = min(values)
    max_val = max(values)
    if abs(max_val - min_val) < 1e-12:
        max_val = min_val + 1.0

    # Resample signal to fit width
    indices = [int(i * (n - 1) / (width - 1)) for i in range(width)]
    sampled = [values[idx] for idx in indices]

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Zero line
    if show_zero and min_val < 0 < max_val:
        zero_row = int((max_val - 0) / (max_val - min_val) * (height - 1))
        zero_row = max(0, min(height - 1, zero_row))
        for c in range(width):
            grid[zero_row][c] = "·"

    # Plot signal
    for c in range(width):
        val = sampled[c]
        row = int((max_val - val) / (max_val - min_val) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][c] = "█"

    # Connect vertically for continuity
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


def ascii_2d(field, width=60, height=25, label=""):
    """
    Render a 2D scalar field as ASCII art using density characters.
    Maps values to a grayscale character palette.
    """
    ny, nx = field.shape
    palette = " ·:;+*#@"

    min_val = field.min()
    max_val = field.max()
    if abs(max_val - min_val) < 1e-12:
        max_val = min_val + 1.0

    # Resample to display size
    display = np.zeros((height, width))
    for r in range(height):
        for c in range(width):
            src_r = int(r * (ny - 1) / (height - 1))
            src_c = int(c * (nx - 1) / (width - 1))
            display[r, c] = field[src_r, src_c]

    print(f"    ┌{'─' * width}┐")
    for r in range(height):
        row_chars = []
        for c in range(width):
            normalized = (display[r, c] - min_val) / (max_val - min_val)
            idx = int(normalized * (len(palette) - 1))
            idx = max(0, min(len(palette) - 1, idx))
            row_chars.append(palette[idx])
        print(f"    │{''.join(row_chars)}│")
    print(f"    └{'─' * width}┘")
    if label:
        padding = max(0, (width - len(label)) // 2)
        print(f"     {' ' * padding}{label}")


# =============================================================================
# PART 1: THE WAVE EQUATION — FROM STRINGS TO PHYSICS
# =============================================================================

def part1_derivation():
    print(f"\n{SEPARATOR}")
    print("  PART 1: THE WAVE EQUATION")
    print("  From a vibrating string to universal physics")
    print(SEPARATOR)

    print("""
    Imagine a guitar string. Pluck it, and a wave travels along it.
    What determines the wave's behavior?

    TWO things:
      • TENSION (T): how tightly the string is pulled
      • DENSITY (μ): how heavy the string is per unit length

    Newton's second law applied to a tiny element of the string gives:

        (μ · dx) · ∂²u/∂t² = T · (∂²u/∂x²) · dx

    Cancel dx and define c² = T/μ:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   ∂²u/∂t² = c² · ∂²u/∂x²                                  │
    │                                                             │
    │   c = √(T/μ) = wave speed                                  │
    │                                                             │
    │   u(x, t) = displacement at position x, time t             │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    This is the wave equation. ONE equation for ALL waves:

        Sound:   c = √(γP/ρ) ≈ 343 m/s in air
        Light:   c = c₀/n ≈ 2×10⁸ m/s in glass
        Water:   c = √(g·h) — depends on depth!
        Seismic: c ≈ 5000 m/s in rock

    The medium sets the speed. The equation does the rest.
    """)


# =============================================================================
# PART 2: D'ALEMBERT'S SOLUTION — TRAVELING WAVES
# =============================================================================

def part2_traveling_waves():
    print(f"\n{SEPARATOR}")
    print("  PART 2: D'ALEMBERT'S SOLUTION")
    print("  Any wave = sum of left-moving + right-moving")
    print(SEPARATOR)

    print("""
    In 1747, d'Alembert showed the general solution is:

        u(x, t) = f(x − ct) + g(x + ct)

    Two waves: one moving RIGHT at speed c, one LEFT at speed c.
    f and g can be ANY shape — the wave equation preserves them.

    Let's watch a Gaussian pulse travel:
    """)

    # Simulate a rightward-traveling Gaussian pulse
    L = 10.0  # domain length
    N = 500   # spatial points
    x = np.linspace(0, L, N)
    c = 2.0   # wave speed

    def gaussian_pulse(x, x0, width=0.3):
        return np.exp(-((x - x0) ** 2) / (2 * width ** 2))

    print("    Rightward-traveling pulse at three instants:\n")

    for i, t in enumerate([0.0, 1.0, 2.0]):
        # f(x - ct): pulse moves right
        u = gaussian_pulse(x, 2.0 - c * (-t))  # starts at x=2
        u_shifted = gaussian_pulse(x, 2.0 + c * t)
        label = f"t = {t:.1f}s  (pulse center at x = {2.0 + c * t:.1f})"
        ascii_plot(u_shifted.tolist(), width=60, height=9, label=label)
        print()

    print("""
    The pulse slides rightward without changing shape.
    Speed c = 2.0 m/s, so in 1 second it moves 2 meters.

    d'Alembert's solution is also why repo 3's BIDIRECTIONAL search
    works: two wavefronts (one from start, one from goal) propagating
    in opposite directions until they meet.
    """)


# =============================================================================
# PART 3: SUPERPOSITION — WAVES ADD
# =============================================================================

def part3_superposition():
    print(f"\n{SEPARATOR}")
    print("  PART 3: SUPERPOSITION")
    print("  The most powerful property of the wave equation")
    print(SEPARATOR)

    print("""
    Because the wave equation is LINEAR, if u₁(x,t) and u₂(x,t) are
    both solutions, then u₁ + u₂ is ALSO a solution.

    This is why Fourier decomposition (repo 2) works:
    any wave = sum of sine waves, and each sine wave independently
    satisfies the wave equation.

    Let's see two pulses collide and pass through each other:
    """)

    L = 10.0
    N = 500
    x = np.linspace(0, L, N)

    def gaussian(x, x0, w=0.3):
        return np.exp(-((x - x0) ** 2) / (2 * w ** 2))

    print("    Two pulses approaching each other:\n")

    times = [0.0, 0.8, 1.25, 1.7, 2.5]
    c = 2.0

    for t in times:
        # Right-moving pulse starts at x=2
        right = gaussian(x, 2.0 + c * t)
        # Left-moving pulse starts at x=8
        left = gaussian(x, 8.0 - c * t)
        total = right + left

        label = f"t = {t:.2f}s"
        ascii_plot(total.tolist(), width=60, height=9, label=label)
        print()

    print("""
    At t ≈ 1.25s the pulses overlap perfectly — CONSTRUCTIVE interference
    doubles the amplitude. Then they pass through each other and continue
    unaffected. Waves don't collide; they superpose.

    This is the same interference that repo 3's quantum-inspired
    pathfinder exploits: waves add where they agree (correct path)
    and cancel where they disagree (wrong paths).
    """)


# =============================================================================
# PART 4: REFLECTION AT BOUNDARIES
# =============================================================================

def part4_reflection():
    print(f"\n{SEPARATOR}")
    print("  PART 4: REFLECTION AT BOUNDARIES")
    print("  Where waveguides begin")
    print(SEPARATOR)

    print("""
    So far we've had infinite domains. But real waves hit BOUNDARIES —
    walls, pipe ends, coastlines, fiber edges.

    What happens depends on the boundary condition:

    ┌────────────────────────────────────────────────────────────┐
    │  FIXED END (Dirichlet): u = 0 at boundary                │
    │    → Reflected wave is INVERTED (phase flip of π)         │
    │    → String clamped, closed pipe end, rigid wall          │
    │                                                           │
    │  FREE END (Neumann): ∂u/∂x = 0 at boundary               │
    │    → Reflected wave is UPRIGHT (no phase flip)            │
    │    → String on a ring, open pipe end                      │
    └────────────────────────────────────────────────────────────┘

    Let's simulate both:
    """)

    # -- Simulate fixed-end reflection using FDTD --
    N = 300
    L = 1.0
    dx = L / N
    c = 1.0
    dt = 0.5 * dx / c  # CFL condition

    # Initialize: rightward Gaussian pulse
    x = np.linspace(0, L, N)

    def simulate_reflection(boundary_type, steps_per_snapshot=120):
        """Run a 1D FDTD simulation with the specified boundary."""
        u_prev = np.exp(-((x - 0.3) ** 2) / (2 * 0.02 ** 2))
        # Initialize with rightward motion
        u_curr = np.exp(-((x - (0.3 + c * dt)) ** 2) / (2 * 0.02 ** 2))

        snapshots = [(u_curr.copy(), 0)]

        step = 0
        for snap in range(1, 5):
            for _ in range(steps_per_snapshot):
                u_next = np.zeros(N)
                # Interior: standard FDTD
                r = (c * dt / dx) ** 2
                for i in range(1, N - 1):
                    u_next[i] = 2 * u_curr[i] - u_prev[i] + r * (
                        u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
                    )

                # Boundary at x=0: absorbing (let pulse leave cleanly)
                u_next[0] = u_curr[1] + (c * dt - dx) / (c * dt + dx) * (u_next[1] - u_curr[0])

                # Boundary at x=L (right end)
                if boundary_type == "fixed":
                    u_next[N - 1] = 0.0
                else:  # free
                    u_next[N - 1] = u_next[N - 2]

                u_prev = u_curr.copy()
                u_curr = u_next.copy()
                step += 1

            snapshots.append((u_curr.copy(), step))

        return snapshots

    print("    ── Fixed End (Dirichlet): u = 0 ──\n")
    fixed_snaps = simulate_reflection("fixed")
    for u_snap, step in fixed_snaps:
        label = f"step {step}"
        ascii_plot(u_snap.tolist(), width=60, height=7, label=label)
        print()

    print("""    The pulse hits the right wall and returns INVERTED.
    u = 0 at the boundary means the reflected wave has opposite sign.
    This is how a string clamped at both ends creates standing waves.
    """)

    print("    ── Free End (Neumann): ∂u/∂x = 0 ──\n")
    free_snaps = simulate_reflection("free")
    for u_snap, step in free_snaps:
        label = f"step {step}"
        ascii_plot(u_snap.tolist(), width=60, height=7, label=label)
        print()

    print("""    The pulse hits the right wall and returns UPRIGHT.
    The free boundary reflects without inverting — the wave bounces back
    with the same sign. Open pipe ends and free string ends do this.
    """)


# =============================================================================
# PART 5: REFRACTION — WAVES CHANGE SPEED
# =============================================================================

def part5_refraction():
    print(f"\n{SEPARATOR}")
    print("  PART 5: REFRACTION")
    print("  When waves cross from one medium to another")
    print(SEPARATOR)

    print("""
    When a wave enters a medium with a DIFFERENT wave speed,
    two things happen:

      1. Part of the wave REFLECTS back
      2. Part of the wave TRANSMITS forward (at a new speed)

    The amounts are governed by the IMPEDANCE MISMATCH:

        Reflection coefficient:  R = (Z₂ − Z₁) / (Z₂ + Z₁)
        Transmission coefficient: T = 2·Z₂ / (Z₂ + Z₁)

    where Z = ρ·c is the impedance (density × wave speed).

    ┌─────────────────────────────────────────────────────────────┐
    │  SAME impedance (Z₁ = Z₂):  R = 0, T = 1 → no reflection │
    │  HUGE mismatch (Z₂ >> Z₁):  R ≈ 1      → total reflection │
    │  This is why:                                               │
    │    • Sound echoes off walls (air→solid = huge mismatch)    │
    │    • Light reflects off glass (partial mismatch: n changes) │
    │    • Tsunamis amplify at shore (depth drops → c drops)     │
    └─────────────────────────────────────────────────────────────┘
    """)

    # Simulate pulse hitting a medium interface
    N = 400
    L = 2.0
    dx = L / N
    x = np.linspace(0, L, N)

    # Two media: c1=1.0 for x<1.0, c2=0.5 for x>=1.0
    c_field = np.where(x < 1.0, 1.0, 0.5)
    dt = 0.4 * dx / np.max(c_field)

    u_prev = np.exp(-((x - 0.3) ** 2) / (2 * 0.03 ** 2))
    u_curr = np.exp(-((x - (0.3 + 1.0 * dt)) ** 2) / (2 * 0.03 ** 2))

    print("    Pulse hitting a medium interface at x = 1.0")
    print("    Left medium: c = 1.0,  Right medium: c = 0.5\n")

    snapshots = []
    snap_steps = [0, 150, 250, 350, 500]
    step = 0
    snap_idx = 0

    if snap_idx < len(snap_steps) and step == snap_steps[snap_idx]:
        snapshots.append((u_curr.copy(), step))
        snap_idx += 1

    for _ in range(max(snap_steps)):
        u_next = np.zeros(N)
        for i in range(1, N - 1):
            r = (c_field[i] * dt / dx) ** 2
            u_next[i] = 2 * u_curr[i] - u_prev[i] + r * (
                u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
            )
        # Absorbing boundaries (Mur first-order)
        u_next[0] = u_curr[1] + (c_field[0] * dt - dx) / (c_field[0] * dt + dx) * (u_next[1] - u_curr[0])
        u_next[N - 1] = u_curr[N - 2] + (c_field[-1] * dt - dx) / (c_field[-1] * dt + dx) * (u_next[N - 2] - u_curr[N - 1])

        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        step += 1

        if snap_idx < len(snap_steps) and step == snap_steps[snap_idx]:
            snapshots.append((u_curr.copy(), step))
            snap_idx += 1

    for u_snap, s in snapshots:
        label = f"step {s}" + (" — interface at x=1.0 marked by |" if s == 0 else "")
        ascii_plot(u_snap.tolist(), width=60, height=7, label=label)
        print()

    # Calculate reflection coefficient
    Z1, Z2 = 1.0, 0.5  # impedances (simplified: ρ=1 for both)
    R = (Z2 - Z1) / (Z2 + Z1)
    T = 2 * Z2 / (Z2 + Z1)

    print(f"""
    The pulse splits at the interface:
      • Reflected pulse moves leftward (R = {R:.3f} → {abs(R)*100:.0f}% amplitude)
      • Transmitted pulse moves rightward at SLOWER speed
      • Transmitted pulse is NARROWER (compressed wavelength)

    Reflection coefficient: R = (Z₂ − Z₁)/(Z₂ + Z₁) = {R:.3f}
    Transmission coefficient: T = 2Z₂/(Z₂ + Z₁) = {T:.3f}

    This is the foundation of:
      • Optical fibers (light bouncing inside glass)
      • Acoustic pipes (sound between rigid walls)
      • Tsunami amplification (wave hitting shallow coast)
    """)


# =============================================================================
# PART 6: STANDING WAVES — WHEN BOUNDARIES CREATE RESONANCE
# =============================================================================

def part6_standing_waves():
    print(f"\n{SEPARATOR}")
    print("  PART 6: STANDING WAVES")
    print("  Reflection + superposition = resonance modes")
    print(SEPARATOR)

    print("""
    When a wave bounces back and forth between two boundaries,
    it interferes with itself. At SPECIFIC frequencies, the
    forward and backward waves create a STANDING WAVE:

        u(x, t) = sin(nπx/L) · cos(nπct/L)

    The wave oscillates in time but the SHAPE stays fixed.
    These are the MODES or HARMONICS of the waveguide.

    Mode frequencies:

        fₙ = n · c / (2L)      for n = 1, 2, 3, ...

    This is why a guitar string has a fundamental note (n=1)
    plus overtones (n=2,3,...). Why a flute has a lowest pitch.
    Why a fiber optic cable has discrete propagation modes.
    """)

    L = 1.0
    N = 500
    x = np.linspace(0, L, N)

    print("    The first four standing wave modes:\n")

    for n in range(1, 5):
        mode = np.sin(n * np.pi * x / L)
        freq = n / (2 * L)  # c = 1
        label = f"Mode n={n}  |  f = {freq:.2f} · c/L  |  {n} half-wavelength{'s' if n > 1 else ''}"
        ascii_plot(mode.tolist(), width=60, height=7, label=label)
        print()

    print("""
    Notice:
      • Mode 1 (fundamental): one half-wavelength fits in L
      • Mode 2 (first overtone): two half-wavelengths
      • Mode n: n half-wavelengths

    The NODES (zero crossings) are fixed — the wave doesn't move there.
    The ANTINODES (peaks) oscillate at maximum amplitude.

    This is the waveguide "filtering" effect: only waves whose
    wavelengths fit the guide geometry can persist. Everything
    else destructively interferes with itself and dies out.

    ┌─────────────────────────────────────────────────────────────┐
    │  A waveguide is a FILTER in physical space.                 │
    │  Only certain frequencies (modes) can propagate.           │
    │  This connects directly to Fourier analysis (repo 2):     │
    │  the waveguide decomposes the wave into allowed modes.     │
    └─────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# PART 7: THE 2D WAVE EQUATION
# =============================================================================

def part7_2d_waves():
    print(f"\n{SEPARATOR}")
    print("  PART 7: THE 2D WAVE EQUATION")
    print("  Waves on a surface — from drumheads to tsunamis")
    print(SEPARATOR)

    print("""
    In two dimensions:

        ∂²u/∂t² = c² · (∂²u/∂x² + ∂²u/∂y²) = c² · ∇²u

    A point source creates circular wavefronts (Huygens' principle).
    This is the equation we'll solve for tsunami simulations.

    Let's simulate a 2D wave from a point source:
    """)

    # 2D FDTD simulation
    Nx, Ny = 80, 80
    L = 1.0
    dx = L / Nx
    dy = L / Ny
    c = 1.0
    dt = 0.3 * min(dx, dy) / c

    u_prev = np.zeros((Ny, Nx))
    u_curr = np.zeros((Ny, Nx))

    # Point source: Gaussian blob at center
    cx_idx, cy_idx = Nx // 2, Ny // 2
    for i in range(Ny):
        for j in range(Nx):
            r2 = ((j - cx_idx) * dx) ** 2 + ((i - cy_idx) * dy) ** 2
            u_curr[i, j] = np.exp(-r2 / (2 * 0.03 ** 2))
    u_prev = u_curr.copy()

    print("    2D wave from a point source — four snapshots:\n")

    r = (c * dt / dx) ** 2
    snap_steps = [0, 30, 60, 100]
    snap_idx = 0
    step = 0

    if snap_idx < len(snap_steps) and step == snap_steps[snap_idx]:
        print(f"    Step {step}:")
        ascii_2d(u_curr, width=50, height=20, label=f"t = {step * dt:.3f}s")
        print()
        snap_idx += 1

    for _ in range(max(snap_steps)):
        u_next = np.zeros((Ny, Nx))
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                laplacian = (
                    u_curr[i + 1, j] + u_curr[i - 1, j]
                    + u_curr[i, j + 1] + u_curr[i, j - 1]
                    - 4 * u_curr[i, j]
                ) / (dx ** 2)
                u_next[i, j] = 2 * u_curr[i, j] - u_prev[i, j] + (c * dt) ** 2 * laplacian

        # Fixed boundaries (u = 0 at edges)
        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        step += 1

        if snap_idx < len(snap_steps) and step == snap_steps[snap_idx]:
            print(f"    Step {step}:")
            ascii_2d(u_curr, width=50, height=20, label=f"t = {step * dt:.3f}s")
            print()
            snap_idx += 1

    print("""
    The initial blob spreads as a circular wavefront — Huygens' principle
    in action. When it hits the fixed boundaries, it reflects back.

    This is the same physics as:
      • A stone dropped in a pond
      • An earthquake epicenter radiating seismic waves
      • A speaker cone pushing air outward
      • A flashlight beam in a 2D waveguide

    In the capstone (script 07), we'll use this with VARIABLE wave speed
    c(x,y) = √(g·h(x,y)) to simulate tsunamis over real bathymetry.
    """)


# =============================================================================
# PART 8: WAVE SPEED IN DIFFERENT MEDIA
# =============================================================================

def part8_media_comparison():
    print(f"\n{SEPARATOR}")
    print("  PART 8: WAVE SPEED ACROSS MEDIA")
    print("  One equation, wildly different speeds")
    print(SEPARATOR)

    media = [
        ("Sound in air (20°C)", 343, "c = √(γP/ρ)", "γ=1.4, P=101kPa"),
        ("Sound in water", 1480, "c = √(K/ρ)", "K = bulk modulus"),
        ("Sound in steel", 5960, "c = √(E/ρ)", "E = Young's modulus"),
        ("Tsunami (deep, h=4km)", 198, "c = √(g·h)", "g=9.81, h=4000m"),
        ("Tsunami (coast, h=10m)", 10, "c = √(g·h)", "g=9.81, h=10m"),
        ("Light in vacuum", 299_792_458, "c = 1/√(ε₀μ₀)", "fundamental"),
        ("Light in glass (n=1.5)", 199_861_639, "c = c₀/n", "n=1.5"),
        ("Light in fiber (n=1.47)", 203_940_788, "c = c₀/n", "n=1.47"),
        ("Seismic P-wave (granite)", 5800, "c = √((K+4G/3)/ρ)", "K,G = moduli"),
    ]

    print(f"\n    {'Medium':<28} {'Speed (m/s)':>14}  {'Formula':<18} {'Parameters'}")
    print(f"    {'─' * 28} {'─' * 14}  {'─' * 18} {'─' * 20}")

    for name, speed, formula, params in media:
        print(f"    {name:<28} {speed:>14,}  {formula:<18} {params}")

    print("""
    Key observations:

    • Light is ~1 MILLION times faster than sound
    • Sound in water is ~4× faster than in air (water is stiffer)
    • A deep-ocean tsunami travels at JET SPEED (713 km/h)
    • A coastal tsunami moves at WALKING SPEED — but amplifies enormously
    • The speed ratio between deep and coastal water is 20:1
      That 20× slowdown is what creates the deadly amplification

    ALL of these are governed by ∂²u/∂t² = c² · ∇²u.
    The medium sets c. The boundaries shape the wave.
    The math is identical.
    """)


# =============================================================================
# PART 9: SUMMARY — THE FOUNDATION
# =============================================================================

def part9_summary():
    print(f"\n{SEPARATOR}")
    print("  SUMMARY: THE FOUNDATION")
    print(SEPARATOR)

    print("""
    This script established the building blocks:

      ✓ The wave equation: ∂²u/∂t² = c² · ∇²u
      ✓ d'Alembert's solution: any wave = left + right traveling
      ✓ Superposition: waves add linearly (enables Fourier, repo 2)
      ✓ Reflection: fixed end inverts, free end preserves
      ✓ Refraction: impedance mismatch → partial reflection
      ✓ Standing waves: modes, harmonics, resonance
      ✓ 2D waves: circular wavefronts, Huygens' principle
      ✓ Media comparison: same equation, vastly different speeds

    ┌─────────────────────────────────────────────────────────────┐
    │  NEXT: What happens when you PUT BOUNDARIES AROUND waves?  │
    │                                                             │
    │  Script 02 → Sound in pipes (acoustic waveguides)          │
    │  Script 03 → Light in fibers (optical waveguides)          │
    │  Script 04 → Water over varying depth (fluid waveguides)   │
    │                                                             │
    │  Same equation. Different boundaries. Different physics.    │
    └─────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{SEPARATOR}")
    print("  WAVE EQUATION FUNDAMENTALS")
    print("  One Equation to Rule Them All")
    print(SEPARATOR)
    print("  ∂²u/∂t² = c² · ∇²u")
    print(f"  {'— The wave equation. The starting point for everything.'}")

    part1_derivation()
    part2_traveling_waves()
    part3_superposition()
    part4_reflection()
    part5_refraction()
    part6_standing_waves()
    part7_2d_waves()
    part8_media_comparison()
    part9_summary()

    print(f"\n{SEPARATOR}")
    print("  Done. Run 02_acoustic_waveguides.py next.")
    print(SEPARATOR)
