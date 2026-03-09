"""
=============================================================================
  FLUID CHANNELS AND SHALLOW WATER — Water Finds the Way
  Tsunamis, tides, and the ocean as a waveguide
=============================================================================

  Scripts 02-03 showed sound in pipes and light in fibers.
  Now: water waves, where the DEPTH of the ocean controls
  the wave speed — and the ocean floor becomes a waveguide.

  The shallow water equations:

      c = √(g·h)

  Deep water → fast waves. Shallow water → slow waves, tall waves.
  That speed variation means the ocean REFRACTS waves just like
  glass refracts light (Snell's law, same formula).

  Connection to repo 3: tsunami arrival time = shortest path
  through a weighted domain where cost = 1/√(g·h).
  The Eikonal equation from repo 3 IS the high-frequency
  limit of the wave equation.
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


def ascii_2d(field, width=60, height=25, label=""):
    """Render a 2D scalar field as ASCII art."""
    ny, nx = field.shape
    palette = " ·:;+*#@"

    min_val = field.min()
    max_val = field.max()
    if abs(max_val - min_val) < 1e-12:
        max_val = min_val + 1.0

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
# PART 1: THE SHALLOW WATER EQUATIONS
# =============================================================================

def part1_shallow_water():
    print(f"\n{SEPARATOR}")
    print("  PART 1: THE SHALLOW WATER EQUATIONS")
    print("  Why depth controls everything")
    print(SEPARATOR)

    print("""
    For water waves much LONGER than the water is DEEP:

        wavelength >> depth     (λ/h > ~20)

    The full Navier-Stokes equations simplify to:

        MOMENTUM:    ∂u/∂t = −g · ∂η/∂x
                     ∂v/∂t = −g · ∂η/∂y

        CONTINUITY:  ∂η/∂t = −∂(h·u)/∂x − ∂(h·v)/∂y

    For small waves (η << h), these combine into a WAVE EQUATION:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   ∂²η/∂t² = c² · ∇²η     where c = √(g·h)                │
    │                                                             │
    │   g = 9.81 m/s²                                            │
    │   h = water depth                                           │
    │   η = surface elevation above mean                          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    THE SAME wave equation from scripts 01-03. Only now:
      • c depends on h (depth) instead of n (index) or T/μ (tension)
      • The "waveguide" is the ocean floor topography
    """)

    # Wave speed vs depth
    g = 9.81
    depths = np.logspace(0, 4, 200)  # 1m to 10000m
    speeds = np.sqrt(g * depths)
    speeds_kmh = speeds * 3.6

    print("    Wave speed c = √(g·h) at various depths:\n")
    depth_table = [1, 10, 50, 200, 1000, 4000, 10000]
    print(f"    {'Depth':>10} {'c (m/s)':>10} {'c (km/h)':>10} {'Comparable to'}")
    print(f"    {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 25}")
    for h in depth_table:
        c = math.sqrt(g * h)
        ckm = c * 3.6
        if ckm < 20:
            comp = "walking"
        elif ckm < 60:
            comp = "car in city"
        elif ckm < 200:
            comp = "highway driving"
        elif ckm < 400:
            comp = "high-speed rail"
        elif ckm < 800:
            comp = "commercial aircraft"
        else:
            comp = "supersonic jet"
        print(f"    {h:>8} m {c:>8.1f} {ckm:>8.0f} {comp}")

    print("""
    A tsunami in the deep ocean (h = 4000m) moves at 713 km/h —
    as fast as a commercial jet. But in shallow coastal water
    (h = 10m), it crawls at 36 km/h.

    Key insight: the tsunami doesn't "slow down" because something
    is in the way. It slows down because the PHYSICS changes —
    the wave speed is determined by the depth. The ocean floor
    IS the waveguide, shaping the wave as it propagates.
    """)


# =============================================================================
# PART 2: SHOALING — WHY TSUNAMIS GROW NEAR SHORE
# =============================================================================

def part2_shoaling():
    print(f"\n{SEPARATOR}")
    print("  PART 2: SHOALING")
    print("  Why a barely-noticeable deep-ocean wave becomes a wall of water")
    print(SEPARATOR)

    print("""
    Conservation of energy flux:

        E · c_g = constant along a wave ray

    where E ∝ A² (energy scales as amplitude squared).
    As the wave enters shallower water, c_g decreases,
    so A must INCREASE.

    The amplification factor (Green's law):

        A_shallow / A_deep ≈ (h_deep / h_shallow)^(1/4)
    """)

    g = 9.81
    h_deep = 4000.0  # meters
    A_deep = 0.5  # meters (typical tsunami in deep ocean)

    # Calculate amplification as wave approaches shore
    depths_shore = np.logspace(np.log10(1), np.log10(4000), 200)[::-1]
    amplitudes = A_deep * (h_deep / depths_shore) ** 0.25

    print(f"    Tsunami approaching shore (A_deep = {A_deep} m at h = {h_deep:.0f} m):\n")

    shore_table = [4000, 1000, 200, 50, 10, 5, 2, 1]
    print(f"    {'Depth':>10} {'c (m/s)':>10} {'Amplitude (m)':>15} {'Amplification':>15}")
    print(f"    {'─' * 10} {'─' * 10} {'─' * 15} {'─' * 15}")
    for h in shore_table:
        c = math.sqrt(g * h)
        A = A_deep * (h_deep / h) ** 0.25
        amp_factor = (h_deep / h) ** 0.25
        danger = " ← DANGER" if A > 3 else ""
        print(f"    {h:>8} m {c:>8.1f} {A:>13.2f} {amp_factor:>13.1f}×{danger}")

    print(f"""
    At 10m depth: the 0.5m wave has grown to {A_deep * (h_deep/10)**0.25:.1f}m
    At 1m depth:  {A_deep * (h_deep/1)**0.25:.1f}m — and this is GREEN'S LAW, the MINIMUM

    Real amplification can be much worse due to:
      • Bathymetric focusing (underwater ridges concentrate energy)
      • Harbor resonance (trapped waves amplify)
      • Wave breaking and runup (adds more height)

    The 2011 Tōhoku tsunami reached 40.5m at maximum runup.
    """)

    # Plot amplitude vs depth
    print("    Amplitude amplification as depth decreases:\n")
    ascii_plot(amplitudes.tolist(), width=56, height=9,
              label="Wave amplitude (m) vs. decreasing depth (4000m → 1m)")
    print()

    print("""    The curve is gentle at first — barely noticeable in deep water.
    Then it ROCKETS up as the depth drops below ~50m.
    Ships in the deep ocean don't even notice a tsunami passing
    beneath them. But at the coast, it's catastrophic.

    This is the same physics as impedance mismatch (script 01,03):
    the energy is CONSERVED, but squeezed into a smaller "channel"
    (shallower water), so the amplitude must increase.
    """)


# =============================================================================
# PART 3: BATHYMETRIC REFRACTION — THE OCEAN AS A LENS
# =============================================================================

def part3_refraction():
    print(f"\n{SEPARATOR}")
    print("  PART 3: BATHYMETRIC REFRACTION")
    print("  The ocean floor bends waves like a glass lens")
    print(SEPARATOR)

    print("""
    Since c = √(g·h), waves traveling over varying depth REFRACT.
    Snell's law applies:

        sin(θ₁)/c₁ = sin(θ₂)/c₂  =  sin(θ₁)/√(g·h₁) = sin(θ₂)/√(g·h₂)

    This is the EXACT same Snell's law from optical fibers (script 03)!
    Replace refractive index n with 1/√(h).

    ┌──────────────────────────────────────────────────────────────────┐
    │  Optics:   n₁·sin(θ₁) = n₂·sin(θ₂)       n = c₀/c            │
    │  Water:    sin(θ₁)/c₁ = sin(θ₂)/c₂        c = √(g·h)         │
    │                                                                 │
    │  The ocean floor is to water waves what glass is to light.     │
    │  Depth variations = refractive index variations.               │
    └──────────────────────────────────────────────────────────────────┘
    """)

    # Demonstrate refraction: wave approaching a sloping beach
    g = 9.81

    print("    ── Why waves arrive parallel to shore ──\n")
    print("""    Imagine waves approaching a straight beach at an angle.
    The part of the wavefront in DEEPER water travels FASTER.
    The part in SHALLOWER water lags behind.
    The wavefront ROTATES to become parallel to the shore.

    Depth contours:        Wavefront bending:
    ═══════════════        ════════════════════
    │ 100m deep  │         │    /← fast       │
    │  50m deep  │    →    │   /  (deep)      │
    │  20m deep  │         │  / ← slowing     │
    │  10m deep  │         │ /   (shallow)    │
    │   5m deep  │         │/← nearly parallel│
    │▓▓ BEACH ▓▓│         │▓▓▓ BEACH ▓▓▓▓▓▓▓│
    """)

    # Quantify: angle vs depth for wave approaching at 45°
    θ_deep = 45.0  # degrees, in deep water
    h_deep = 100.0
    c_deep = math.sqrt(g * h_deep)

    print(f"    Wave approaching at {θ_deep}° in {h_deep}m water:\n")
    print(f"    {'Depth':>10} {'c (m/s)':>10} {'Wave angle':>12} {'Turns toward shore'}")
    print(f"    {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 20}")

    for h in [100, 50, 20, 10, 5, 2]:
        c = math.sqrt(g * h)
        sin_θ = math.sin(math.radians(θ_deep)) * c / c_deep
        if sin_θ > 1:
            sin_θ = 1.0  # total refraction
        θ = math.degrees(math.asin(sin_θ))
        turn = θ_deep - θ
        print(f"    {h:>8} m {c:>8.1f} {θ:>10.1f}° {turn:>18.1f}°")

    print("""
    By the time the wave reaches 2m depth, it has rotated from 45°
    to less than 10° — nearly parallel to shore. This is why
    waves ALWAYS seem to arrive roughly perpendicular to the beach,
    regardless of the deep-water wave direction.

    SAME physics as light bending through a gradient-index lens.
    The ocean floor is nature's optical element.
    """)


# =============================================================================
# PART 4: COASTAL WAVEGUIDES — TRAPPED WAVES
# =============================================================================

def part4_coastal_waveguides():
    print(f"\n{SEPARATOR}")
    print("  PART 4: COASTAL WAVEGUIDES")
    print("  How coastline shape traps and guides waves")
    print(SEPARATOR)

    print("""
    The concept: just as a fiber optic cable traps light between
    core (high n → slow) and cladding (low n → fast), the coast
    can trap water waves between:

      • Shore (shallow → slow waves) = "core"
      • Deep ocean (deep → fast waves) = "cladding"

    Waves that approach at shallow angles to the coastline get
    REFRACTED back toward shore by the depth gradient — total
    internal reflection, ocean-style.

    These are EDGE WAVES: they propagate ALONG the coast, trapped
    between the beach and the deep ocean.

    ┌─────────────────────────────────────────────────────────────┐
    │  COASTAL TRAPPING (water) ↔ FIBER MODE (light)             │
    │                                                             │
    │  Shore (slow) → traps wave  ↔  Core (slow) → traps light  │
    │  Deep water (fast) → barrier ↔  Clad (fast) → barrier      │
    │  Depth gradient → refraction ↔  Index step → TIR           │
    └─────────────────────────────────────────────────────────────┘
    """)

    # Simulate a 1D model of edge wave trapping
    print("    ── Edge wave mode profile (amplitude vs. distance from shore) ──\n")

    N = 500
    x = np.linspace(0, 10.0, N)  # distance from shore (km)

    # Depth profile: linear slope then flat
    h_profile = np.minimum(x * 40, 400)  # 40m/km slope, max 400m

    # Edge wave decays exponentially offshore
    for mode_n in range(3):
        κ = (mode_n + 1) * 0.5  # decay rate increases with mode number
        edge_wave = np.exp(-κ * x) * np.cos(2 * np.pi * (mode_n + 1) * x / 8)
        label = f"Edge wave mode {mode_n}: trapped near shore"
        ascii_plot(edge_wave.tolist(), width=56, height=7, label=label)
        print()

    print("""    The wave amplitude is concentrated near the shore and decays
    exponentially offshore — exactly like the evanescent field
    in a fiber's cladding (script 03).

    Edge waves are responsible for:
      • Rip current spacing along beaches
      • Periodic flooding patterns
      • Harbor oscillations (seiches)

    The coastline SHAPE determines which edge wave modes can exist,
    just as the fiber core diameter determines optical modes.
    """)


# =============================================================================
# PART 5: 1D TSUNAMI PROPAGATION
# =============================================================================

def part5_tsunami_1d():
    print(f"\n{SEPARATOR}")
    print("  PART 5: 1D TSUNAMI PROPAGATION")
    print("  A wave crossing from deep to shallow water")
    print(SEPARATOR)

    print("""
    Let's simulate a tsunami crossing a 1D ocean profile:
    deep ocean → continental shelf → coast.
    """)

    g = 9.81
    L = 500.0  # 500 km domain
    N = 2000
    dx = L * 1000 / N  # meters
    x_km = np.linspace(0, L, N)
    x = x_km * 1000  # meters

    # Depth profile: deep ocean → shelf → coast
    h_profile = np.zeros(N)
    for i in range(N):
        d = x_km[i]
        if d < 300:
            h_profile[i] = 4000  # deep ocean
        elif d < 400:
            # Continental slope: 4000m → 200m over 100km
            frac = (d - 300) / 100
            h_profile[i] = 4000 - 3800 * frac
        elif d < 480:
            h_profile[i] = 200  # continental shelf
        else:
            # Final approach to coast: 200m → 5m over 20km
            frac = (d - 480) / 20
            h_profile[i] = max(5, 200 - 195 * frac)

    c_field = np.sqrt(g * h_profile)
    c_max = np.max(c_field)
    dt = 0.5 * dx / c_max

    print("    Ocean depth profile:\n")
    ascii_plot(h_profile.tolist(), width=56, height=9,
              label="Depth (m) over 500 km — deep ocean → shelf → coast")
    print()

    print("    Wave speed profile:\n")
    ascii_plot(c_field.tolist(), width=56, height=7,
              label="c = √(g·h) in m/s — fast in deep water, slow near shore")
    print()

    # Initial condition: Gaussian tsunami source at x = 50km
    σ = 20000  # 20 km width
    x_source = 50 * 1000
    η_prev = 0.5 * np.exp(-((x - x_source) ** 2) / (2 * σ ** 2))
    η_curr = η_prev.copy()

    print("    Tsunami propagation (source at 50 km):\n")

    snap_times = [0, 500, 1200, 2000, 3000, 4000]  # seconds
    snap_idx = 0
    step = 0
    t_elapsed = 0.0

    if snap_idx < len(snap_times) and t_elapsed >= snap_times[snap_idx]:
        t_min = t_elapsed / 60
        label = f"t = {t_min:.0f} min — source"
        ascii_plot(η_curr.tolist(), width=56, height=7, label=label)
        print()
        snap_idx += 1

    max_steps = int(max(snap_times) / dt) + 100
    for _ in range(max_steps):
        η_next = np.zeros(N)
        for i in range(1, N - 1):
            # Variable-speed wave equation
            r = (c_field[i] * dt / dx) ** 2
            η_next[i] = 2 * η_curr[i] - η_prev[i] + r * (
                η_curr[i + 1] - 2 * η_curr[i] + η_curr[i - 1]
            )

        # Absorbing boundary (left: open ocean)
        η_next[0] = η_curr[1] + (c_field[0] * dt - dx) / (c_field[0] * dt + dx) * (η_next[1] - η_curr[0])
        # Reflecting boundary (right: coastline)
        η_next[N - 1] = η_next[N - 2]

        η_prev = η_curr.copy()
        η_curr = η_next.copy()
        t_elapsed += dt
        step += 1

        if snap_idx < len(snap_times) and t_elapsed >= snap_times[snap_idx]:
            t_min = t_elapsed / 60
            label = f"t = {t_min:.0f} min"
            if 300 < t_elapsed < 2000:
                label += " — crossing deep ocean at ~200 m/s"
            elif 2000 < t_elapsed < 3500:
                label += " — slowing on continental slope"
            else:
                label += " — amplifying near coast!"
            ascii_plot(η_curr.tolist(), width=56, height=7, label=label)
            print()
            snap_idx += 1

    print("""
    Watch the wave:
      1. Starts as a 0.5m pulse in the deep ocean
      2. Crosses 300 km of deep ocean at ~200 m/s (jet speed)
      3. Hits the continental slope — SLOWS dramatically
      4. Wavelength COMPRESSES (wave bunches up)
      5. Amplitude GROWS (energy squeezed into less water)
      6. Reflects off the coastline

    This is the full shoaling + refraction process from a single
    simulation. The same FDTD method from script 01, with
    c(x) = √(g·h(x)) instead of constant c.
    """)


# =============================================================================
# PART 6: THE EIKONAL CONNECTION — WAVE = PATHFINDING
# =============================================================================

def part6_eikonal():
    print(f"\n{SEPARATOR}")
    print("  PART 6: THE EIKONAL EQUATION")
    print("  Wave propagation IS pathfinding (repo 3 connection)")
    print(SEPARATOR)

    print("""
    In the high-frequency limit, the wave equation becomes:

        |∇T(x,y)| = 1/c(x,y)

    This is the EIKONAL EQUATION. T(x,y) is the ARRIVAL TIME of the
    wavefront at each point.

    This is EXACTLY the same as:
      • Repo 3's weighted wave propagation
      • Dijkstra's algorithm with edge cost = distance/c
      • The Fast Marching Method

    ┌─────────────────────────────────────────────────────────────┐
    │  tsunami arrival time at (x,y) = shortest travel time      │
    │  from earthquake source, where speed = √(g·h(x,y))        │
    │                                                             │
    │  This IS pathfinding. The "cost" is 1/c at each point.     │
    │                                                             │
    │  Repo 3's Eikonal script was solving tsunami propagation!  │
    └─────────────────────────────────────────────────────────────┘
    """)

    # Compute arrival times via Eikonal (1D version: integral of 1/c)
    g = 9.81
    L = 500.0
    N = 1000
    x_km = np.linspace(0, L, N)
    dx_km = L / N

    # Same depth profile as Part 5
    h_profile = np.zeros(N)
    for i in range(N):
        d = x_km[i]
        if d < 300:
            h_profile[i] = 4000
        elif d < 400:
            frac = (d - 300) / 100
            h_profile[i] = 4000 - 3800 * frac
        elif d < 480:
            h_profile[i] = 200
        else:
            frac = (d - 480) / 20
            h_profile[i] = max(5, 200 - 195 * frac)

    c_profile = np.sqrt(g * h_profile)

    # Arrival time: integral over 1/c from source to each point
    arrival_time = np.zeros(N)
    source_idx = int(50 / L * N)  # 50 km source

    for i in range(source_idx + 1, N):
        arrival_time[i] = arrival_time[i - 1] + (dx_km * 1000) / c_profile[i]
    for i in range(source_idx - 1, -1, -1):
        arrival_time[i] = arrival_time[i + 1] + (dx_km * 1000) / c_profile[i]

    arrival_min = arrival_time / 60  # convert to minutes

    print("    Tsunami arrival time (Eikonal solution):\n")
    ascii_plot(arrival_min[source_idx:].tolist(), width=56, height=9,
              label="Arrival time (minutes) from source at 50 km to coast")
    print()

    # Key arrival times
    landmarks = [(50, "Source"), (200, "Mid-ocean"), (300, "Shelf edge"),
                 (400, "Inner shelf"), (490, "Coast")]
    print(f"    {'Location':>15} {'Distance':>10} {'Depth':>8} {'Arrival':>10}")
    print(f"    {'─' * 15} {'─' * 10} {'─' * 8} {'─' * 10}")
    for km, name in landmarks:
        idx = min(int(km / L * N), N - 1)
        t = arrival_min[idx]
        h = h_profile[idx]
        print(f"    {name:>15} {km:>8} km {h:>6.0f} m {t:>8.1f} min")

    print("""
    The arrival time curve is NONLINEAR:
      • Fast across the deep ocean (steep = rapid progress)
      • Flattens on the shelf (wave slows down)
      • Very flat at the coast (the wave crawls the last few km)

    If you draw arrival time CONTOURS on a 2D ocean map,
    you get exactly the wavefronts. And computing those contours
    is EXACTLY Dijkstra's algorithm (repo 3), where:
      • Each point is a "node"
      • Edge cost = distance / c(x,y) = transit time
      • The "shortest path" IS the earliest arrival
    """)


# =============================================================================
# PART 7: 2D SHALLOW WATER SIMULATION
# =============================================================================

def part7_2d_shallow_water():
    print(f"\n{SEPARATOR}")
    print("  PART 7: 2D SHALLOW WATER WAVES")
    print("  Waves over varying bathymetry")
    print(SEPARATOR)

    print("""
    Now the full 2D case: waves propagating over a sea floor
    with varying depth. The wave equation:

        ∂²η/∂t² = ∇·(c²(x,y) · ∇η)    where c = √(g·h)

    With c varying spatially, the wave refracts and focuses.
    """)

    g = 9.81
    Nx, Ny = 80, 80
    Lx, Ly = 200.0, 200.0  # km
    dx = Lx * 1000 / Nx  # meters
    dy = Ly * 1000 / Ny

    # Create a bathymetry: deep ocean on left, shelf approaching right
    # with a submarine ridge in the center
    h_bathy = np.zeros((Ny, Nx))
    x_grid = np.linspace(0, Lx, Nx)
    y_grid = np.linspace(0, Ly, Ny)

    for i in range(Ny):
        for j in range(Nx):
            # Base: depth decreases from left (deep) to right (shallow)
            base_depth = 4000 - 3500 * (x_grid[j] / Lx) ** 0.7
            # Add a submarine ridge in the center
            ridge_center = Ly / 2
            ridge_width = 30  # km
            ridge_height = min(base_depth * 0.6, 1500)  # max 1500m ridge
            ridge = ridge_height * np.exp(-((y_grid[i] - ridge_center) ** 2) / (2 * (ridge_width / 2) ** 2))
            h_bathy[i, j] = max(5, base_depth - ridge)

    c_bathy = np.sqrt(g * h_bathy)
    c_max = np.max(c_bathy)
    dt = 0.35 * min(dx, dy) / c_max

    print("    Bathymetry (depth in m — darker = deeper):\n")
    ascii_2d(h_bathy, width=50, height=18,
             label="Depth: deep (left/dark) → shallow (right/light), ridge in center")
    print()

    # Initialize: tsunami source on the left
    η_prev = np.zeros((Ny, Nx))
    η_curr = np.zeros((Ny, Nx))

    src_x, src_y = 20, Ly / 2  # km
    σ = 15  # km
    for i in range(Ny):
        for j in range(Nx):
            r2 = ((x_grid[j] - src_x) ** 2 + (y_grid[i] - src_y) ** 2)
            η_curr[i, j] = 1.0 * np.exp(-r2 / (2 * σ ** 2))
    η_prev = η_curr.copy()

    print("    Simulating 2D tsunami over variable bathymetry...\n")

    snap_steps = [0, 40, 80, 130]
    snapshots = [(η_curr.copy(), 0)]
    step = 0
    snap_idx = 1

    for _ in range(max(snap_steps)):
        η_next = np.zeros((Ny, Nx))
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                c_local = c_bathy[i, j]
                r_sq = (c_local * dt) ** 2
                laplacian = (
                    (η_curr[i+1, j] - 2*η_curr[i, j] + η_curr[i-1, j]) / dy**2 +
                    (η_curr[i, j+1] - 2*η_curr[i, j] + η_curr[i, j-1]) / dx**2
                )
                η_next[i, j] = 2 * η_curr[i, j] - η_prev[i, j] + r_sq * laplacian

        # Absorbing boundaries
        η_next[0, :] = η_next[1, :]
        η_next[Ny-1, :] = η_next[Ny-2, :]
        η_next[:, 0] = η_next[:, 1]
        # Right boundary: reflecting (coast)
        η_next[:, Nx-1] = η_next[:, Nx-2]

        η_prev = η_curr.copy()
        η_curr = η_next.copy()
        step += 1

        if snap_idx < len(snap_steps) and step == snap_steps[snap_idx]:
            snapshots.append((η_curr.copy(), step))
            snap_idx += 1

    for η_snap, s in snapshots:
        t_min = s * dt / 60
        label = f"Step {s} (t ≈ {t_min:.0f} min)"
        # Show absolute value for better visualization
        ascii_2d(np.abs(η_snap), width=50, height=18, label=label)
        print()

    print("""
    Watch the wavefront:
      • Starts circular from the source (Huygens' principle)
      • Moves faster in deep water (left), slower in shallow (right)
      • The submarine ridge FOCUSES the wave (acts like a lens)
      • The wavefront curves and distorts over the varying depth

    This is bathymetric refraction in action — the same physics as
    light through a gradient-index medium. The ridge acts like a
    convex lens, focusing energy toward specific coastal areas.

    Some coastlines receive MORE energy because of this focusing.
    That's why tsunami damage isn't uniform along a coast — the
    underwater topography determines who gets hit hardest.
    """)


# =============================================================================
# PART 8: SUMMARY
# =============================================================================

def part8_summary():
    print(f"\n{SEPARATOR}")
    print("  SUMMARY: FLUID WAVEGUIDES")
    print(SEPARATOR)

    print("""
    This script explored water waves shaped by depth:

      ✓ Shallow water equations: c = √(g·h) — depth controls speed
      ✓ Shoaling: waves grow as depth decreases (energy conservation)
      ✓ Bathymetric refraction: Snell's law for ocean waves
      ✓ Coastal waveguides: shoreline traps edge waves
      ✓ 1D tsunami simulation: deep → shelf → coast amplification
      ✓ Eikonal equation: tsunami arrival time = shortest path (repo 3)
      ✓ 2D simulation: wavefronts bending over bathymetry

    ┌─────────────────────────────────────────────────────────────┐
    │  THREE media, ONE equation:                                 │
    │                                                             │
    │  Sound: c = √(γP/ρ)     → pipe walls confine              │
    │  Light: c = c₀/n         → index step confines             │
    │  Water: c = √(g·h)       → depth profile confines          │
    │                                                             │
    │  NEXT: Script 05 — the NUMERICAL METHODS that let us       │
    │  simulate all of these on a computer.                       │
    └─────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{SEPARATOR}")
    print("  FLUID CHANNELS AND SHALLOW WATER")
    print("  Tsunamis, tides, and the ocean as a waveguide")
    print(SEPARATOR)

    part1_shallow_water()
    part2_shoaling()
    part3_refraction()
    part4_coastal_waveguides()
    part5_tsunami_1d()
    part6_eikonal()
    part7_2d_shallow_water()
    part8_summary()

    print(f"\n{SEPARATOR}")
    print("  Done. Run 05_numerical_methods.py next.")
    print(SEPARATOR)
