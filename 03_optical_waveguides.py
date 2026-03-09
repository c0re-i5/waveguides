"""
=============================================================================
  OPTICAL WAVEGUIDES — Light in Fibers
  Total internal reflection, modes, and why your internet is photons
=============================================================================

  Script 02 showed sound confined by rigid walls.
  Now we confine LIGHT using a subtler trick: refractive index.

  A fiber optic cable is a thin glass cylinder (core) surrounded
  by a slightly different glass (cladding). When light hits the
  core-cladding boundary at a shallow enough angle, it TOTALLY
  reflects — trapped inside the core forever.

  This is total internal reflection, and it's the reason billions
  of bits per second flow through glass threads thinner than hair.

  Connection to repo 2: the fiber's modes are Fourier components.
  Connection to repo 1: precision matters — IEEE 754 rounding
  limits how accurately we compute critical angles.
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


def ascii_fiber_cross_section(core_radius, clad_radius, width=40, height=20):
    """Render a fiber cross-section as ASCII art."""
    print(f"    ┌{'─' * width}┐")
    cy, cx = height // 2, width // 2
    for r in range(height):
        row_chars = []
        for c in range(width):
            # Scale to physical coordinates
            y = (r - cy) / cy * clad_radius * 1.2
            x_pos = (c - cx) / cx * clad_radius * 1.2
            dist = math.sqrt(x_pos ** 2 + y ** 2)
            if dist <= core_radius:
                row_chars.append("█")
            elif dist <= clad_radius:
                row_chars.append("░")
            else:
                row_chars.append(" ")
        print(f"    │{''.join(row_chars)}│")
    print(f"    └{'─' * width}┘")
    print(f"    {'':>4} █ = core (n₁)   ░ = cladding (n₂)   n₁ > n₂")


# =============================================================================
# PART 1: SNELL'S LAW — THE FOUNDATION OF OPTICS
# =============================================================================

def part1_snells_law():
    print(f"\n{SEPARATOR}")
    print("  PART 1: SNELL'S LAW")
    print("  Why light bends at interfaces")
    print(SEPARATOR)

    print("""
    When light crosses from one medium to another, it changes
    direction. This is REFRACTION, governed by Snell's law:

        n₁ · sin(θ₁) = n₂ · sin(θ₂)

    where:
      n₁, n₂ = refractive indices (how much the medium slows light)
      θ₁ = angle of incidence (from the normal)
      θ₂ = angle of refraction

    The refractive index n = c₀/c where c₀ is the speed of light
    in vacuum. Higher n → slower light → more bending.
    """)

    # Table of refractive indices
    materials = [
        ("Vacuum", 1.000),
        ("Air", 1.0003),
        ("Water", 1.333),
        ("Fused silica (fiber)", 1.458),
        ("Crown glass", 1.520),
        ("Flint glass", 1.620),
        ("Diamond", 2.417),
        ("Silicon", 3.450),
    ]

    c0 = 299_792_458  # m/s

    print(f"    {'Material':<24} {'n':>6} {'Light speed (m/s)':>20} {'% of c₀':>10}")
    print(f"    {'─' * 24} {'─' * 6} {'─' * 20} {'─' * 10}")
    for name, n in materials:
        v = c0 / n
        pct = 100 / n
        print(f"    {name:<24} {n:>6.3f} {v:>17,} {pct:>8.1f}%")

    print("""
    Light in a fiber (n = 1.458) travels at 68.6% of vacuum speed.
    That's still 205 million m/s — but slower than vacuum.

    This speed difference is what makes optical waveguides possible:
    faster medium (low n) and slower medium (high n) create a
    boundary that can TRAP light.
    """)

    # Demonstrate refraction
    print("    ── Refraction at a glass-air interface ──\n")
    n1 = 1.50  # glass
    n2 = 1.00  # air

    print(f"    Glass (n={n1}) → Air (n={n2}):\n")
    print(f"    {'θ₁ (glass)':>12}  {'θ₂ (air)':>12}  {'Bends':>10}")
    print(f"    {'─' * 12}  {'─' * 12}  {'─' * 10}")

    for θ1_deg in [10, 20, 30, 40, 41.8]:
        θ1 = math.radians(θ1_deg)
        sin_θ2 = n1 / n2 * math.sin(θ1)
        if sin_θ2 <= 1.0:
            θ2 = math.degrees(math.asin(sin_θ2))
            print(f"    {θ1_deg:>10.1f}°  {θ2:>10.1f}°  {'away from normal':>10}")
        else:
            print(f"    {θ1_deg:>10.1f}°  {'TOTAL REFL':>12}  {'← trapped!':>10}")

    θ_critical = math.degrees(math.asin(n2 / n1))
    print(f"\n    Critical angle: θ_c = arcsin(n₂/n₁) = arcsin({n2}/{n1}) = {θ_critical:.1f}°")

    print("""
    Above the critical angle, Snell's law gives sin(θ₂) > 1 — impossible.
    The light CAN'T escape. It reflects TOTALLY back into the glass.

    This is TOTAL INTERNAL REFLECTION (TIR), and it's how fiber
    optics work: keep the light hitting the boundary above θ_c,
    and it bounces forever inside the core.
    """)


# =============================================================================
# PART 2: TOTAL INTERNAL REFLECTION — THE TRAPPING MECHANISM
# =============================================================================

def part2_total_internal_reflection():
    print(f"\n{SEPARATOR}")
    print("  PART 2: TOTAL INTERNAL REFLECTION")
    print("  Light trapped inside glass — forever")
    print(SEPARATOR)

    print("""
    For TIR to work, we need:
      1. Light going from HIGH n to LOW n (core → cladding)
      2. Angle of incidence > critical angle

    In a fiber: core has n₁ (higher), cladding has n₂ (lower).

        θ_critical = arcsin(n₂/n₁)

    Any ray hitting the core-cladding interface at an angle
    greater than θ_c bounces back — perfectly, with zero loss
    from the reflection itself.
    """)

    # Typical fiber parameters
    fibers = [
        ("Multi-mode (OM1)", 62.5, 125.0, 1.490, 1.475),
        ("Multi-mode (OM3)", 50.0, 125.0, 1.482, 1.475),
        ("Single-mode (OS2)", 9.0, 125.0, 1.468, 1.458),
        ("Telecom SM", 8.2, 125.0, 1.4681, 1.4629),
    ]

    print(f"    {'Fiber type':<22} {'Core ∅':>8} {'Clad ∅':>8} {'n₁':>7} {'n₂':>7} {'θ_c':>7} {'NA':>6}")
    print(f"    {'─' * 22} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 6}")

    for name, core_d, clad_d, n1, n2 in fibers:
        θ_c = math.degrees(math.asin(n2 / n1))
        NA = math.sqrt(n1 ** 2 - n2 ** 2)
        print(f"    {name:<22} {core_d:>6.1f}μm {clad_d:>6.1f}μm {n1:>7.4f} {n2:>7.4f} {θ_c:>5.1f}° {NA:>5.3f}")

    print("""
    Notice:
      • The refractive index DIFFERENCE is tiny (< 1%)
      • But that's enough for total internal reflection
      • Single-mode fiber has a ~9 μm core — about 6× the wavelength
      • Multi-mode has 50-62.5 μm — many wavelengths fit across

    The NUMERICAL APERTURE (NA) measures how much light the fiber
    can capture:

        NA = √(n₁² − n₂²) = n₁ · sin(θ_acceptance)

    Higher NA → wider acceptance cone → easier to couple light in.
    But also → more modes → more dispersion (blurring).
    """)

    print("\n    ── Fiber cross-section (multi-mode, 62.5/125 μm) ──\n")
    ascii_fiber_cross_section(core_radius=5.0, clad_radius=10.0, width=36, height=18)


# =============================================================================
# PART 3: FIBER MODES — HOW MANY WAYS CAN LIGHT BOUNCE?
# =============================================================================

def part3_modes():
    print(f"\n{SEPARATOR}")
    print("  PART 3: FIBER MODES")
    print("  How many bouncing patterns fit in the core")
    print(SEPARATOR)

    print("""
    Just like a pipe has acoustic modes (script 02), a fiber
    has OPTICAL modes — specific electromagnetic field patterns
    that can propagate.

    The key parameter is the V-number (normalized frequency):

        V = (π · d / λ) · NA = (π · d / λ) · √(n₁² − n₂²)

    where d = core diameter, λ = wavelength.

    The NUMBER of modes ≈ V²/2 (for large V).

    ┌─────────────────────────────────────────────────────────────┐
    │  V < 2.405  →  SINGLE-MODE: only one pattern propagates   │
    │  V > 2.405  →  MULTI-MODE: many patterns, each at a       │
    │                 slightly different speed → dispersion       │
    └─────────────────────────────────────────────────────────────┘

    The magic number 2.405 is the first zero of the Bessel
    function J₀ — it falls out of solving Maxwell's equations
    in cylindrical coordinates with the fiber boundary conditions.
    """)

    # Compute V-number for various configurations
    λ = 1.55e-6  # 1550 nm (telecom wavelength)

    configs = [
        ("Single-mode telecom", 8.2e-6, 1.4681, 1.4629),
        ("Single-mode 1310nm", 9.0e-6, 1.468, 1.458),
        ("Multi-mode OM3", 50.0e-6, 1.482, 1.475),
        ("Multi-mode OM1", 62.5e-6, 1.490, 1.475),
        ("Large core (200μm)", 200.0e-6, 1.490, 1.475),
    ]

    print(f"    At λ = {λ*1e9:.0f} nm (telecom C-band):\n")
    print(f"    {'Fiber':<24} {'Core':>8} {'V-number':>10} {'≈ Modes':>10} {'Type':>12}")
    print(f"    {'─' * 24} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 12}")

    for name, d, n1, n2 in configs:
        NA = math.sqrt(n1 ** 2 - n2 ** 2)
        V = math.pi * d / λ * NA
        modes = max(1, int(V ** 2 / 2))
        fiber_type = "single-mode" if V < 2.405 else "multi-mode"
        print(f"    {name:<24} {d*1e6:>6.1f}μm {V:>10.2f} {modes:>10,} {fiber_type:>12}")

    print("""
    The single-mode fiber has V ≈ 2.0 — just under the 2.405 cutoff.
    Only ONE mode propagates. This means:
      • No modal dispersion (all light takes the same path)
      • Signals can travel hundreds of km without distortion
      • Used for all long-distance telecom

    The multi-mode fiber (V ≈ 30-50) has thousands of modes:
      • Each mode has a slightly different propagation speed
      • After some distance, the modes spread apart → pulse blurs
      • Used for short distances (< 1 km, data centers)
    """)

    # Show mode profiles
    print("    ── Transverse mode profiles (simplified 1D slab model) ──\n")
    N = 500
    a = 1.0  # normalized core half-width
    x = np.linspace(-3 * a, 3 * a, N)

    for m in range(4):
        mode = np.zeros(N)
        for i, xi in enumerate(x):
            if abs(xi) <= a:
                # Core: oscillatory (cosine for even modes, sine for odd)
                if m % 2 == 0:
                    mode[i] = math.cos((m + 1) * math.pi * xi / (2 * a))
                else:
                    mode[i] = math.sin((m + 1) * math.pi * xi / (2 * a))
            else:
                # Cladding: exponential decay
                κ = (m + 1) * 0.8  # decay rate
                if m % 2 == 0:
                    edge_val = math.cos((m + 1) * math.pi / 2)
                else:
                    edge_val = math.sin((m + 1) * math.pi / 2) * (1 if xi > 0 else -1)
                mode[i] = edge_val * math.exp(-κ * (abs(xi) - a))

        label = f"Mode {m} — {'even' if m % 2 == 0 else 'odd'} ({'core' if m == 0 else 'higher order'})"
        ascii_plot(mode.tolist(), width=56, height=7, label=label)
        print()

    print("""
    SAME physics as acoustic modes (script 02):
      • Core = pipe interior (oscillatory)
      • Cladding = beyond the pipe walls (evanescent/decaying)

    The difference: in acoustics, the wave is ZERO at the wall.
    In optics, the field PENETRATES into the cladding — an
    evanescent tail that decays exponentially. This is why
    single-mode fiber still works even with a 9μm core: the
    mode extends slightly beyond the physical glass boundary.
    """)


# =============================================================================
# PART 4: MODAL DISPERSION — WHY PULSES SPREAD
# =============================================================================

def part4_dispersion():
    print(f"\n{SEPARATOR}")
    print("  PART 4: MODAL DISPERSION")
    print("  Different modes travel at different speeds → pulse broadening")
    print(SEPARATOR)

    print("""
    In a multi-mode fiber, each mode travels at a slightly different
    group velocity. After some distance, a sharp input pulse SPREADS:

    ┌─────────────────────────────────────────────────────────────┐
    │  Input:   ▃▆█▆▃           (sharp pulse)                    │
    │  After L: ▁▂▃▅▆█▆▅▃▂▁    (broadened pulse)                │
    │  After 2L: ▁▁▂▃▄▅▅▆▆▆▅▅▄▃▂▁▁  (even broader)            │
    └─────────────────────────────────────────────────────────────┘

    The time spread for a step-index fiber is approximately:

        Δτ/L ≈ (n₁ − n₂) / c₀ × n₁/n₂

    This limits the BIT RATE × DISTANCE product:

        B × L ≤ 1 / (4 × Δτ/L)
    """)

    c0 = 299_792_458

    fibers_disp = [
        ("Step-index MM", 62.5e-6, 1.490, 1.475, 1000),
        ("Graded-index MM", 50.0e-6, 1.482, 1.475, 1000),
        ("Single-mode", 9.0e-6, 1.468, 1.458, 100000),
    ]

    print(f"    Modal dispersion comparison (per km):\n")
    print(f"    {'Fiber':<20} {'Δτ (ns/km)':>12} {'Max bit rate':>16} {'Max distance':>14}")
    print(f"    {'─' * 20} {'─' * 12} {'─' * 16} {'─' * 14}")

    for name, d, n1, n2, L_km in fibers_disp:
        if "Single" in name:
            # Single mode: chromatic dispersion dominates (~17 ps/nm/km)
            delta_tau_per_km = 0.05  # ns/km (chromatic only)
        elif "Graded" in name:
            # Graded index reduces modal dispersion by ~100×
            delta_tau_per_km = (n1 - n2) / c0 * n1 / n2 * 1e12 / 100
        else:
            delta_tau_per_km = (n1 - n2) / c0 * n1 / n2 * 1e12  # ns/km

        max_rate = 1 / (4 * delta_tau_per_km * 1e-9) / 1e9  # Gbps
        max_dist = 1 / (4 * delta_tau_per_km * 1e-9 * 1e9)  # km at 1 Gbps

        if max_rate > 1000:
            rate_str = f"{max_rate/1000:.0f} Tbps·km"
        else:
            rate_str = f"{max_rate:.1f} Gbps·km"

        print(f"    {name:<20} {delta_tau_per_km:>10.2f}  {rate_str:>16} {max_dist:>12.1f} km")

    print("""
    KEY INSIGHT: single-mode fiber has essentially ZERO modal dispersion.
    Only chromatic dispersion (different wavelengths travel at slightly
    different speeds) limits the bandwidth.

    This is why all long-distance telecom uses single-mode fiber:
    submarine cables, backbone networks, connections between cities.

    Multi-mode is used for short links (< 300m): server-to-switch
    connections in data centers, where the distance is too short
    for dispersion to matter.
    """)

    # Simulate pulse broadening
    print("    ── Pulse broadening simulation ──\n")

    N = 500
    t = np.linspace(-5, 5, N)

    # Original sharp pulse
    pulse_0 = np.exp(-t ** 2 / (2 * 0.1 ** 2))
    ascii_plot(pulse_0.tolist(), width=56, height=7, label="Input pulse (sharp)")
    print()

    # After propagation: convolve with Gaussian broadening
    for distance, sigma in [(1, 0.3), (5, 0.8), (20, 1.5)]:
        broadened = np.exp(-t ** 2 / (2 * (0.1 ** 2 + sigma ** 2)))
        peak = max(broadened)
        broadened = broadened / peak  # normalize height
        label = f"After {distance} km (σ_broadening = {sigma} ns)"
        ascii_plot(broadened.tolist(), width=56, height=7, label=label)
        print()

    print("""    As the pulse travels further, it spreads wider and shorter.
    Eventually adjacent pulses OVERLAP and the receiver can't
    distinguish 0 from 1. That's the bandwidth limit.

    Solutions:
      • Use single-mode fiber (eliminate modal dispersion)
      • Use graded-index fiber (equalize mode speeds)
      • Use optical amplifiers + regenerators
      • Use wavelength-division multiplexing (WDM) — many colors
        of light in the same fiber, each carrying a separate signal
    """)


# =============================================================================
# PART 5: WAVELENGTH DIVISION MULTIPLEXING — MANY SIGNALS, ONE FIBER
# =============================================================================

def part5_wdm():
    print(f"\n{SEPARATOR}")
    print("  PART 5: WAVELENGTH DIVISION MULTIPLEXING")
    print("  The fiber as a multi-channel waveguide")
    print(SEPARATOR)

    print("""
    A single-mode fiber guides ONE spatial mode. But light comes
    in MANY wavelengths (colors). Each wavelength propagates
    independently — meaning we can send multiple signals
    simultaneously on different wavelengths.

    This is WDM: Wavelength Division Multiplexing.

    ┌─────────────────────────────────────────────────────────────┐
    │  λ₁ = 1530 nm ───→ ┌──────────┐ ───→ λ₁ → Receiver 1     │
    │  λ₂ = 1535 nm ───→ │  Single  │ ───→ λ₂ → Receiver 2     │
    │  λ₃ = 1540 nm ───→ │  Fiber   │ ───→ λ₃ → Receiver 3     │
    │  ...            ───→ │          │ ───→ ...                  │
    │  λ_80 = 1565nm ───→ └──────────┘ ───→ λ_80 → Receiver 80  │
    └─────────────────────────────────────────────────────────────┘

    Dense WDM (DWDM) uses 80+ channels spaced 0.4 nm apart
    in the C-band (1530-1565 nm).
    """)

    # WDM channel plan
    channels = [
        (1, 1530.33, 100),
        (10, 1537.40, 100),
        (20, 1545.32, 100),
        (40, 1561.42, 100),
        (80, 1561.42 + 40 * 0.8, 100),
    ]

    total_capacity = 80 * 100  # Gbps

    print(f"    DWDM channel plan (sample):\n")
    print(f"    {'Channel':>10} {'Wavelength':>12} {'Rate':>10}")
    print(f"    {'─' * 10} {'─' * 12} {'─' * 10}")
    for ch, wl, rate in channels:
        print(f"    {'Ch ' + str(ch):>10} {wl:>10.2f} nm {rate:>8} Gbps")

    print(f"""
    80 channels × 100 Gbps = {total_capacity / 1000:.0f} Tbps per fiber.

    Modern systems push this further:
      • 400 Gbps per channel (coherent detection)
      • C+L band (160+ channels)
      • Up to 20+ Tbps per fiber pair

    This is a PHYSICAL Fourier decomposition: each wavelength is
    a frequency component, and the fiber carries all of them
    simultaneously — the same principle as repo 2's FFT, but
    implemented in glass rather than code.
    """)


# =============================================================================
# PART 6: FIBER SIMULATION — LIGHT BOUNCING IN A CORE
# =============================================================================

def part6_fiber_simulation():
    print(f"\n{SEPARATOR}")
    print("  PART 6: FIBER SIMULATION")
    print("  1D slab waveguide: light trapped between boundaries")
    print(SEPARATOR)

    print("""
    Let's simulate a light pulse in a 1D slab waveguide —
    the 2D analog of a fiber. Two cladding regions surround
    a higher-index core.

    The wave equation with spatially varying refractive index:

        ∂²E/∂t² = (c₀/n(x))² · ∂²E/∂x²

    n(x) is higher in the core, lower in the cladding.
    """)

    # 1D FDTD with refractive index profile
    N = 500
    L = 10.0  # μm (total domain)
    dx = L / N
    x = np.linspace(0, L, N)

    # Refractive index profile: core in the center
    core_half_width = 2.0  # μm
    core_center = L / 2
    n1 = 1.468  # core
    n2 = 1.458  # cladding

    n_profile = np.full(N, n2)
    for i in range(N):
        if abs(x[i] - core_center) < core_half_width:
            n_profile[i] = n1

    c0 = 1.0  # normalized
    c_field = c0 / n_profile  # wave speed at each point
    dt = 0.4 * dx / max(c_field)

    # Initial pulse: Gaussian in the core
    u_prev = np.exp(-((x - core_center) ** 2) / (2 * 0.5 ** 2))
    u_curr = u_prev.copy()

    print(f"    Slab waveguide: core width = {2*core_half_width:.0f} μm")
    print(f"    n_core = {n1:.3f}, n_clad = {n2:.3f}")
    print(f"    Δn = {n1-n2:.3f} ({(n1-n2)/n1*100:.1f}%)\n")

    print("    Refractive index profile:")
    ascii_plot(n_profile.tolist(), width=56, height=5,
              label="n(x) — higher in core, lower in cladding")
    print()

    snap_steps = [0, 60, 150, 300, 500]
    snapshots = [(u_curr.copy(), 0)]
    step = 0
    snap_idx = 1

    for _ in range(max(snap_steps)):
        u_next = np.zeros(N)
        for i in range(1, N - 1):
            r = (c_field[i] * dt / dx) ** 2
            u_next[i] = 2 * u_curr[i] - u_prev[i] + r * (
                u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
            )
        # Absorbing boundaries
        u_next[0] = u_curr[1] + (c_field[0] * dt - dx) / (c_field[0] * dt + dx) * (u_next[1] - u_curr[0])
        u_next[N - 1] = u_curr[N - 2] + (c_field[-1] * dt - dx) / (c_field[-1] * dt + dx) * (u_next[N - 2] - u_curr[N - 1])

        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        step += 1

        if snap_idx < len(snap_steps) and step == snap_steps[snap_idx]:
            snapshots.append((u_curr.copy(), step))
            snap_idx += 1

    for u_snap, s in snapshots:
        label = f"step {s} — pulse in core region"
        ascii_plot(u_snap.tolist(), width=56, height=7, label=label)
        print()

    print("""
    Notice how the pulse partially stays trapped in the core (guided mode)
    while some energy leaks into the cladding (radiation modes).
    The guided part bounces back and forth — this is the fiber
    carrying a signal over distance.

    In a real fiber, the guided modes propagate indefinitely (aside
    from absorption loss of ~0.2 dB/km in silica at 1550nm).
    The radiation modes leak away within millimeters.
    """)


# =============================================================================
# PART 7: LOSS AND AMPLIFICATION
# =============================================================================

def part7_loss():
    print(f"\n{SEPARATOR}")
    print("  PART 7: FIBER LOSS AND AMPLIFICATION")
    print("  Why signals fade and how we fix it")
    print(SEPARATOR)

    print("""
    Real fibers aren't perfect. Light gradually fades due to:

      • Rayleigh scattering (molecules in the glass scatter light)
      • Material absorption (OH ions, UV/IR absorption)
      • Bending loss (tight curves let light escape)

    Loss is measured in dB/km:

        Loss(dB) = −10 · log₁₀(P_out / P_in)
    """)

    # Fiber loss at different wavelengths
    wavelengths = [
        (850, 2.5, "Multi-mode data center"),
        (1310, 0.35, "Zero-dispersion window"),
        (1550, 0.20, "Minimum loss (telecom C-band)"),
        (1625, 0.25, "L-band"),
    ]

    print(f"    {'Wavelength':>12} {'Loss (dB/km)':>14} {'Application'}")
    print(f"    {'─' * 12} {'─' * 14} {'─' * 30}")
    for wl, loss, app in wavelengths:
        print(f"    {wl:>10} nm {loss:>12.2f} {app}")

    # Power vs distance
    print(f"\n    Signal power vs. distance (at 1550 nm, 0.2 dB/km):\n")
    distances = np.linspace(0, 100, 200)
    power_dBm = 0 - 0.2 * distances  # starting at 0 dBm
    power_linear = 10 ** (power_dBm / 10)

    ascii_plot(power_linear.tolist(), width=56, height=9,
              label="Optical power (mW) vs distance (0-100 km)")
    print()

    # With amplifiers
    print("    With erbium-doped fiber amplifiers (EDFA) every 80 km:\n")
    distances_amp = np.linspace(0, 400, 500)
    power_amp = []
    for d in distances_amp:
        spans = int(d / 80)
        remaining = d - spans * 80
        # Each span: lose 16 dB, amp restores it
        p = 0 - 0.2 * remaining  # current span loss
        power_amp.append(10 ** (p / 10))

    ascii_plot(power_amp, width=56, height=9,
              label="Power with EDFAs every 80 km (0-400 km)")

    print("""
    EDFAs (Erbium-Doped Fiber Amplifiers) boost the signal optically
    — no need to convert to electrical and back. They amplify ALL
    wavelengths in the C-band simultaneously, which is why WDM works
    so well: one amplifier boosts 80+ channels at once.

    Submarine cables spanning oceans use EDFAs every ~80 km:

        London → New York: ~5,500 km = ~69 amplifiers
        Total capacity: 200+ Tbps per cable

    ALL of this is the wave equation with glass boundaries.
    """)


# =============================================================================
# PART 8: COMPARISON — ACOUSTIC vs OPTICAL WAVEGUIDES
# =============================================================================

def part8_comparison():
    print(f"\n{SEPARATOR}")
    print("  PART 8: ACOUSTIC vs. OPTICAL — SAME PHYSICS")
    print(SEPARATOR)

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Concept           │  Acoustic pipe         │  Optical fiber       │
    ├─────────────────────┼────────────────────────┼──────────────────────┤
    │  Wave equation     │  ∂²p/∂t² = c²∇²p      │  ∂²E/∂t² = c²∇²E   │
    │  Wave speed        │  c = √(γP/ρ)           │  c = c₀/n           │
    │  Speed varies with │  Temperature, gas type  │  Refractive index   │
    │  Boundary type     │  Rigid walls            │  Index step          │
    │  Confinement       │  Physical walls         │  Total int. refl.   │
    │  Modes             │  sin(nπx/L)             │  Bessel functions    │
    │  Mode count        │  Depends on L vs λ      │  Depends on V       │
    │  Cutoff            │  f_c = mc/(2W)          │  V < 2.405          │
    │  Dispersion        │  v_g × v_p = c²         │  v_g × v_p = c²     │
    │  Signal            │  Pressure variations    │  EM field            │
    │  Typical speed     │  343 m/s (air)          │  2×10⁸ m/s (glass)  │
    │  Loss mechanism    │  Friction, radiation    │  Scattering, abs.   │
    │  Amplification     │  Electronic amplifier   │  EDFA (optical)      │
    └─────────────────────┴────────────────────────┴──────────────────────┘

    The math is IDENTICAL. Only the parameters change.
    This is the theme of this repo: one equation, many media.

    Script 04 will show the SAME physics applies to water waves,
    where the wave speed c = √(g·h) depends on depth rather than
    index or pressure.
    """)


# =============================================================================
# PART 9: SUMMARY
# =============================================================================

def part9_summary():
    print(f"\n{SEPARATOR}")
    print("  SUMMARY: OPTICAL WAVEGUIDES")
    print(SEPARATOR)

    print("""
    This script explored light confined by refractive index:

      ✓ Snell's law: n₁·sin(θ₁) = n₂·sin(θ₂)
      ✓ Total internal reflection: the trapping mechanism
      ✓ Fiber modes: V-number determines how many
      ✓ V < 2.405 = single-mode (cleanest signal)
      ✓ Modal dispersion: different modes → different speeds → blur
      ✓ WDM: many wavelengths, one fiber, massive bandwidth
      ✓ FDTD simulation: light bouncing in a slab waveguide
      ✓ Loss and amplification: EDFAs boost all channels at once
      ✓ Acoustic ↔ optical: same equation, different parameters

    ┌─────────────────────────────────────────────────────────────┐
    │  NEXT: Script 04 — WATER waves.                            │
    │  c = √(g·h). Depth controls speed.                         │
    │  The ocean floor is a waveguide.                            │
    │  The coastline is a boundary.                               │
    │  And a tsunami is the wave.                                 │
    └─────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{SEPARATOR}")
    print("  OPTICAL WAVEGUIDES — Light in Fibers")
    print("  Total internal reflection and the backbone of the internet")
    print(SEPARATOR)

    part1_snells_law()
    part2_total_internal_reflection()
    part3_modes()
    part4_dispersion()
    part5_wdm()
    part6_fiber_simulation()
    part7_loss()
    part8_comparison()
    part9_summary()

    print(f"\n{SEPARATOR}")
    print("  Done. Run 04_fluid_channels_and_shallow_water.py next.")
    print(SEPARATOR)
