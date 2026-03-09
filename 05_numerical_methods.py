"""
=============================================================================
  NUMERICAL METHODS — Simulating Waves on a Computer
  FDTD, stability, and the art of turning PDEs into code
=============================================================================

  Every simulation in scripts 01-04 used the same method under
  the hood: the Finite-Difference Time-Domain (FDTD) method.

  This script peels back the curtain: how do you turn a continuous
  PDE (the wave equation) into discrete code that a computer
  can execute? What can go wrong? How do you prevent it?

  The key ideas:
    1. Replace derivatives with finite differences
    2. Obey the CFL stability condition (or everything explodes)
    3. Handle boundaries properly (absorbing, reflecting, periodic)
    4. Understand numerical dispersion (the grid distorts waves)

  Connection to repo 1: discretizing a continuous equation is
  the same problem as discretizing a real number into IEEE 754.
  Finite precision means finite accuracy.

  Connection to repo 2: the Nyquist theorem sets the minimum
  grid resolution — you need at least 2 points per wavelength.
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


# =============================================================================
# PART 1: THE FINITE DIFFERENCE IDEA
# =============================================================================

def part1_finite_differences():
    print(f"\n{SEPARATOR}")
    print("  PART 1: FINITE DIFFERENCES")
    print("  Replacing calculus with arithmetic")
    print(SEPARATOR)

    print("""
    A derivative is a limit:

        du/dx = lim(Δx→0) [u(x+Δx) − u(x)] / Δx

    On a computer, we can't take Δx → 0. We pick a small but
    FINITE Δx and approximate:

        du/dx ≈ [u(x+Δx) − u(x)] / Δx          (forward difference)
        du/dx ≈ [u(x) − u(x−Δx)] / Δx          (backward difference)
        du/dx ≈ [u(x+Δx) − u(x−Δx)] / (2Δx)   (central difference) ← better!

    The SECOND derivative (what we need for the wave equation):

        d²u/dx² ≈ [u(x+Δx) − 2u(x) + u(x−Δx)] / Δx²

    This is the CENTRAL DIFFERENCE formula. It's at the heart
    of every FDTD simulation.
    """)

    # Demonstrate accuracy
    print("    ── Accuracy test: d(sin(x))/dx = cos(x) at x = 1.0 ──\n")

    x0 = 1.0
    exact = math.cos(x0)

    print(f"    {'Δx':>12} {'Forward':>14} {'Central':>14} {'Forward err':>14} {'Central err':>14}")
    print(f"    {'─' * 12} {'─' * 14} {'─' * 14} {'─' * 14} {'─' * 14}")

    for exp in range(1, 9):
        dx = 10 ** (-exp)
        forward = (math.sin(x0 + dx) - math.sin(x0)) / dx
        central = (math.sin(x0 + dx) - math.sin(x0 - dx)) / (2 * dx)
        f_err = abs(forward - exact)
        c_err = abs(central - exact)
        print(f"    {dx:>12.0e} {forward:>14.10f} {central:>14.10f} {f_err:>14.2e} {c_err:>14.2e}")

    print(f"\n    Exact: cos(1.0) = {exact:.10f}")

    print("""
    Central difference converges as O(Δx²) — 100× smaller Δx → 10,000× less error.
    Forward difference converges as O(Δx) — only 100× less error.

    This is why we ALWAYS use central differences for the wave equation.

    Note: at very small Δx (below ~10⁻⁸), floating-point roundoff
    takes over and error INCREASES. This is the IEEE 754 precision
    limit from repo 1 — the same bit-level constraint that gave
    us the fast inverse square root.
    """)


# =============================================================================
# PART 2: THE FDTD SCHEME FOR THE WAVE EQUATION
# =============================================================================

def part2_fdtd_scheme():
    print(f"\n{SEPARATOR}")
    print("  PART 2: THE FDTD SCHEME")
    print("  Discretizing the wave equation")
    print(SEPARATOR)

    print("""
    The wave equation: ∂²u/∂t² = c² · ∂²u/∂x²

    Replace BOTH derivatives with central differences:

    TIME:
        ∂²u/∂t² ≈ [u(x, t+Δt) − 2u(x, t) + u(x, t−Δt)] / Δt²

    SPACE:
        ∂²u/∂x² ≈ [u(x+Δx, t) − 2u(x, t) + u(x−Δx, t)] / Δx²

    Plug into the wave equation and solve for u(x, t+Δt):

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │  u[i, n+1] = 2·u[i, n] − u[i, n−1]                          │
    │              + r² · (u[i+1, n] − 2·u[i, n] + u[i−1, n])     │
    │                                                                │
    │  where r = c·Δt/Δx  (the Courant number)                     │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    This is a LEAPFROG scheme: to compute the FUTURE (n+1),
    we need the PRESENT (n) and the PAST (n−1).

    It's explicit (no matrix inversion needed), second-order
    accurate in both space and time, and remarkably simple.
    """)

    # Show the stencil visually
    print("""
    The computational stencil:

        Time n+1:              [u[i, n+1]]     ← what we compute
                                    │
        Time n:     [u[i-1, n]] ── [u[i, n]] ── [u[i+1, n]]
                                    │
        Time n-1:              [u[i, n-1]]

    Each new value depends on 3 spatial neighbors + 1 previous time.
    This pattern tiles across the whole domain at every time step.
    """)


# =============================================================================
# PART 3: THE CFL CONDITION — THE STABILITY GATE
# =============================================================================

def part3_cfl_condition():
    print(f"\n{SEPARATOR}")
    print("  PART 3: THE CFL CONDITION")
    print("  The rule that keeps simulations from exploding")
    print(SEPARATOR)

    print("""
    The Courant-Friedrichs-Lewy (CFL) condition:

    ┌──────────────────────────────────────────────────────┐
    │  1D:  r = c·Δt/Δx ≤ 1                              │
    │  2D:  r = c·Δt/Δx ≤ 1/√2                           │
    │                                                      │
    │  Violate this → simulation EXPLODES exponentially    │
    └──────────────────────────────────────────────────────┘

    Physically: the numerical "information speed" (Δx/Δt) must be
    at least as fast as the physical wave speed c. If the wave
    travels farther than one grid cell per time step, the scheme
    can't "see" it and becomes unstable.

    Let's demonstrate:
    """)

    N = 300
    L = 1.0
    dx = L / N
    c = 1.0
    x = np.linspace(0, L, N)

    def run_fdtd_1d(courant_number, n_steps):
        """Run 1D FDTD with given Courant number and return snapshots."""
        dt = courant_number * dx / c
        r = c * dt / dx

        u_prev = np.exp(-((x - 0.3) ** 2) / (2 * 0.02 ** 2))
        u_curr = np.exp(-((x - (0.3 + c * dt)) ** 2) / (2 * 0.02 ** 2))

        snapshots = [(u_curr.copy(), 0)]
        max_vals = [np.max(np.abs(u_curr))]

        for step in range(1, n_steps + 1):
            u_next = np.zeros(N)
            for i in range(1, N - 1):
                u_next[i] = 2 * u_curr[i] - u_prev[i] + r**2 * (
                    u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
                )
            u_next[0] = 0
            u_next[N - 1] = 0

            u_prev = u_curr.copy()
            u_curr = u_next.copy()
            max_vals.append(np.max(np.abs(u_curr)))

            if step in [50, 100, 200]:
                snapshots.append((u_curr.copy(), step))

            # Abort if exploding
            if np.max(np.abs(u_curr)) > 1e6:
                snapshots.append((u_curr.copy(), step))
                break

        return snapshots, max_vals

    # Stable simulation (r = 0.8)
    print("    ── STABLE: Courant number r = 0.8 (< 1.0) ──\n")
    stable_snaps, stable_maxes = run_fdtd_1d(0.8, 200)
    for u_snap, s in stable_snaps[:3]:
        label = f"step {s}  |  r = 0.8  |  max|u| = {np.max(np.abs(u_snap)):.4f}"
        ascii_plot(u_snap.tolist(), width=56, height=7, label=label)
        print()

    # Unstable simulation (r = 1.1)
    print("    ── UNSTABLE: Courant number r = 1.1 (> 1.0) ──\n")
    unstable_snaps, unstable_maxes = run_fdtd_1d(1.1, 200)
    for u_snap, s in unstable_snaps[:3]:
        max_u = np.max(np.abs(u_snap))
        label = f"step {s}  |  r = 1.1  |  max|u| = {min(max_u, 9999):.1f}"
        # Clip for display
        clipped = np.clip(u_snap, -5, 5)
        ascii_plot(clipped.tolist(), width=56, height=7, label=label)
        print()
        if max_u > 1e6:
            print(f"    ⚠️  SIMULATION EXPLODED at step {s}! max|u| = {max_u:.2e}\n")
            break

    # Show max amplitude over time
    print("    Max amplitude over time:\n")
    print(f"    {'Step':>8} {'Stable (r=0.8)':>16} {'Unstable (r=1.1)':>18}")
    print(f"    {'─' * 8} {'─' * 16} {'─' * 18}")
    for step in [0, 10, 20, 50, 100]:
        s_val = stable_maxes[step] if step < len(stable_maxes) else float('inf')
        u_val = unstable_maxes[step] if step < len(unstable_maxes) else float('inf')
        u_str = f"{u_val:.2e}" if u_val > 100 else f"{u_val:.6f}"
        print(f"    {step:>8} {s_val:>16.6f} {u_str:>18}")

    print("""
    The unstable simulation grows EXPONENTIALLY — each time step
    amplifies the error. Within 100 steps, values go from ~1 to
    astronomically large.

    The CFL condition is non-negotiable. Every simulation in
    this repo checks it before running.

    In practice: c·Δt/Δx = 0.5 to 0.9 is common (safe margin).
    """)


# =============================================================================
# PART 4: BOUNDARY CONDITIONS
# =============================================================================

def part4_boundary_conditions():
    print(f"\n{SEPARATOR}")
    print("  PART 4: BOUNDARY CONDITIONS")
    print("  What happens at the edges of your simulation domain")
    print(SEPARATOR)

    print("""
    Every simulation has a finite domain. What happens at the edges?

    ┌─────────────────────────────────────────────────────────────┐
    │  1. FIXED (Dirichlet): u = 0                               │
    │     → Perfect reflection with phase flip                   │
    │     → Used for: rigid walls, clamped strings               │
    │                                                             │
    │  2. FREE (Neumann): ∂u/∂x = 0 → u[end] = u[end-1]        │
    │     → Perfect reflection without phase flip                │
    │     → Used for: open pipe ends, free string ends           │
    │                                                             │
    │  3. PERIODIC: u[0] = u[N-1], u[N] = u[1]                  │
    │     → Waves wrap around (infinite domain illusion)         │
    │     → Used for: ring waveguides, periodic structures       │
    │                                                             │
    │  4. ABSORBING (Mur 1st order):                             │
    │     u[0,n+1] = u[1,n] + (cΔt−Δx)/(cΔt+Δx)·(u[1,n+1]−u[0,n]) │
    │     → Waves leave the domain without reflecting            │
    │     → Used for: open space (ocean, atmosphere)             │
    └─────────────────────────────────────────────────────────────┘
    """)

    N = 300
    L = 1.0
    dx = L / N
    c = 1.0
    x = np.linspace(0, L, N)
    dt = 0.5 * dx / c
    r = c * dt / dx

    def simulate_bc(bc_type, n_steps=300):
        """Simulate with different boundary conditions."""
        u_prev = np.exp(-((x - 0.3) ** 2) / (2 * 0.02 ** 2))
        u_curr = np.exp(-((x - (0.3 + c * dt)) ** 2) / (2 * 0.02 ** 2))

        snapshots = [(u_curr.copy(), 0)]

        for step in range(1, n_steps + 1):
            u_next = np.zeros(N)
            for i in range(1, N - 1):
                u_next[i] = 2 * u_curr[i] - u_prev[i] + r**2 * (
                    u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
                )

            if bc_type == "fixed":
                u_next[0] = 0
                u_next[N - 1] = 0
            elif bc_type == "free":
                u_next[0] = u_next[1]
                u_next[N - 1] = u_next[N - 2]
            elif bc_type == "periodic":
                u_next[0] = 2 * u_curr[0] - u_prev[0] + r**2 * (
                    u_curr[1] - 2 * u_curr[0] + u_curr[N - 2]
                )
                u_next[N - 1] = u_next[0]
            elif bc_type == "absorbing":
                # Mur first-order ABC
                u_next[0] = u_curr[1] + (c*dt - dx)/(c*dt + dx) * (u_next[1] - u_curr[0])
                u_next[N-1] = u_curr[N-2] + (c*dt - dx)/(c*dt + dx) * (u_next[N-2] - u_curr[N-1])

            u_prev = u_curr.copy()
            u_curr = u_next.copy()

            if step in [80, 160, 250]:
                snapshots.append((u_curr.copy(), step))

        return snapshots

    for bc_name in ["fixed", "absorbing"]:
        full_name = {"fixed": "FIXED (Dirichlet): u = 0 at boundaries",
                     "absorbing": "ABSORBING (Mur): wave exits cleanly"}[bc_name]
        print(f"    ── {full_name} ──\n")
        snaps = simulate_bc(bc_name)
        for u_snap, s in snaps:
            label = f"step {s}"
            ascii_plot(u_snap.tolist(), width=56, height=7, label=label)
            print()

    print("""
    The absorbing boundary condition is crucial for simulating
    open domains (ocean, atmosphere, free space). Without it,
    reflections from the domain edges corrupt the simulation.

    For the tsunami capstone (script 07), we use:
      • Absorbing boundaries on the open ocean edges
      • Reflecting boundaries at the coastline
    """)


# =============================================================================
# PART 5: NUMERICAL DISPERSION
# =============================================================================

def part5_numerical_dispersion():
    print(f"\n{SEPARATOR}")
    print("  PART 5: NUMERICAL DISPERSION")
    print("  The grid distorts wave speeds")
    print(SEPARATOR)

    print("""
    A continuous wave at frequency ω propagates at speed c (exactly).
    On a discrete grid, the NUMERICAL wave speed depends on the
    wave's direction and frequency:

        c_numerical = c · (Δx / (c·Δt)) · arcsin(r · sin(k·Δx/2)) / (k·Δx/2)

    where k = 2π/λ is the wavenumber.

    At LOW frequencies (λ >> Δx): c_numerical ≈ c (good!)
    At HIGH frequencies (λ ≈ 2Δx): c_numerical < c (too slow!)

    Rule of thumb: need at least 10-20 grid points per wavelength
    to keep numerical dispersion below 1%.

    This is the spatial analog of the Nyquist theorem (repo 2):
      • Nyquist: need ≥ 2 samples per wave period in TIME
      • FDTD:   need ≥ 2 grid points per wavelength in SPACE
      • Both: need MORE points for ACCURACY (10-20 is practical)
    """)

    # Compute numerical dispersion relation
    c = 1.0
    dx = 0.01
    courant_numbers = [0.5, 0.8, 1.0]

    print("    Ratio c_numerical/c at various wavelengths:\n")
    print(f"    {'λ/Δx':>8} {'Points/λ':>10}", end="")
    for r in courant_numbers:
        print(f" {'r=' + str(r):>10}", end="")
    print()
    print(f"    {'─' * 8} {'─' * 10}", end="")
    for _ in courant_numbers:
        print(f" {'─' * 10}", end="")
    print()

    for ppw in [4, 6, 8, 10, 15, 20, 30, 50, 100]:
        lam = ppw * dx
        k = 2 * math.pi / lam

        print(f"    {lam/dx:>8.0f} {ppw:>10}", end="")
        for r in courant_numbers:
            dt = r * dx / c
            arg = r * math.sin(k * dx / 2)
            if abs(arg) <= 1:
                c_num = dx / dt * math.asin(arg) / (k * dx / 2)
                ratio = c_num / c
            else:
                ratio = float('nan')
            print(f" {ratio:>10.6f}", end="")
        print()

    print("""
    Key observations:
      • At r = 1.0 (CFL limit): numerical dispersion is ZERO!
        The discrete scheme is EXACT. This is called the "magic" time step.
      • At r < 1.0: high-frequency waves travel slightly too slow
      • Need ~20 points/wavelength for < 0.1% error at r = 0.5

    In practice, r = 1.0 is risky (right at the stability limit),
    so we use r = 0.5–0.9 and ensure enough grid resolution.
    """)


# =============================================================================
# PART 6: CONVERGENCE ANALYSIS
# =============================================================================

def part6_convergence():
    print(f"\n{SEPARATOR}")
    print("  PART 6: CONVERGENCE ANALYSIS")
    print("  Proving the simulation gets more accurate with finer grids")
    print(SEPARATOR)

    print("""
    The FDTD scheme should converge to the true solution as
    Δx, Δt → 0 (while maintaining CFL). Let's verify:

    We simulate a standing wave sin(πx/L)·cos(πct/L) and
    measure the error at different grid resolutions.
    """)

    L = 1.0
    c = 1.0
    T_final = 1.0  # simulate for 1 second

    resolutions = [20, 40, 80, 160, 320, 640]
    errors = []

    for N in resolutions:
        dx = L / N
        dt = 0.5 * dx / c  # CFL = 0.5
        n_steps = int(T_final / dt)

        x_grid = np.linspace(0, L, N)

        # Exact solution: standing wave
        def exact(x_arr, t):
            return np.sin(np.pi * x_arr / L) * np.cos(np.pi * c * t / L)

        # Initialize
        u_prev = exact(x_grid, 0)
        u_curr = exact(x_grid, dt)

        r = c * dt / dx

        for step in range(2, n_steps + 1):
            u_next = np.zeros(N)
            for i in range(1, N - 1):
                u_next[i] = 2 * u_curr[i] - u_prev[i] + r**2 * (
                    u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
                )
            u_next[0] = 0
            u_next[N - 1] = 0

            u_prev = u_curr.copy()
            u_curr = u_next.copy()

        # Compute L2 error
        u_exact = exact(x_grid, n_steps * dt)
        error = np.sqrt(np.mean((u_curr - u_exact) ** 2))
        errors.append(error)

    print(f"    {'Grid points':>12} {'Δx':>12} {'L2 Error':>14} {'Ratio':>8} {'Expected':>10}")
    print(f"    {'─' * 12} {'─' * 12} {'─' * 14} {'─' * 8} {'─' * 10}")

    for i, (N, err) in enumerate(zip(resolutions, errors)):
        dx = L / N
        if i == 0:
            ratio_str = "—"
        else:
            ratio = errors[i - 1] / err if err > 0 else float('inf')
            ratio_str = f"{ratio:.2f}"
        print(f"    {N:>12} {dx:>12.6f} {err:>14.2e} {ratio_str:>8} {'~4':>10}")

    print("""
    The error ratio should converge to ~4 as we double resolution
    (4× because the scheme is second-order: error ∝ Δx²).

    ┌─────────────────────────────────────────────────────────────┐
    │  Second-order convergence means:                            │
    │    • Double the grid points → 4× less error                │
    │    • 10× the grid points → 100× less error                 │
    │    • Accuracy improves QUADRATICALLY with resolution        │
    └─────────────────────────────────────────────────────────────┘

    This guarantees the simulation converges to the true physics.
    More grid points → more accuracy. Always.
    """)


# =============================================================================
# PART 7: 2D FDTD — THE GENERAL ENGINE
# =============================================================================

def part7_2d_fdtd():
    print(f"\n{SEPARATOR}")
    print("  PART 7: THE 2D FDTD ENGINE")
    print("  The universal wave simulator")
    print(SEPARATOR)

    print("""
    In 2D, the FDTD stencil becomes:

    u[i,j,n+1] = 2·u[i,j,n] − u[i,j,n−1]
                + rx² · (u[i+1,j,n] − 2·u[i,j,n] + u[i−1,j,n])
                + ry² · (u[i,j+1,n] − 2·u[i,j,n] + u[i,j−1,n])

    where rx = c·Δt/Δx, ry = c·Δt/Δy.

    CFL condition: c·Δt · √(1/Δx² + 1/Δy²) ≤ 1
    For Δx = Δy:  c·Δt/Δx ≤ 1/√2 ≈ 0.707

    Let's measure the performance:
    """)

    # Performance benchmark
    sizes = [40, 60, 80, 100]
    times_list = []

    for Nx in sizes:
        Ny = Nx
        dx = 1.0 / Nx
        c = 1.0
        dt = 0.35 * dx / c

        u_prev = np.zeros((Ny, Nx))
        u_curr = np.zeros((Ny, Nx))

        # Point source
        u_curr[Ny // 2, Nx // 2] = 1.0
        u_prev = u_curr.copy()

        r_sq = (c * dt / dx) ** 2

        n_steps = 50
        t0 = time.perf_counter()

        for _ in range(n_steps):
            u_next = np.zeros((Ny, Nx))
            for i in range(1, Ny - 1):
                for j in range(1, Nx - 1):
                    u_next[i, j] = (
                        2 * u_curr[i, j] - u_prev[i, j]
                        + r_sq * (
                            u_curr[i+1, j] + u_curr[i-1, j]
                            + u_curr[i, j+1] + u_curr[i, j-1]
                            - 4 * u_curr[i, j]
                        )
                    )
            u_prev = u_curr
            u_curr = u_next

        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)

        cells_per_sec = Nx * Ny * n_steps / elapsed

    print(f"    {'Grid size':>12} {'Cells':>10} {'Time (s)':>10} {'Cells/sec':>14} {'Steps':>8}")
    print(f"    {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 14} {'─' * 8}")
    for Nx, elapsed in zip(sizes, times_list):
        cells = Nx * Nx
        cps = cells * 50 / elapsed
        print(f"    {Nx:>4}×{Nx:<5} {cells:>10,} {elapsed:>10.3f} {cps:>12,.0f} {50:>8}")

    print("""
    The pure Python double loop is slow — production codes use:
      • NumPy vectorization (10-100× faster)
      • C/Fortran extensions (1000× faster)
      • GPU computing (10,000× faster)

    For this educational repo, Python clarity > raw speed.
    The algorithm is identical regardless of implementation language.

    This 2D FDTD engine is what script 07's tsunami simulation uses.
    Same stencil, but with c(x,y) = √(g·h(x,y)) varying spatially.
    """)


# =============================================================================
# PART 8: PUTTING IT ALL TOGETHER — RECIPE FOR A WAVE SIMULATION
# =============================================================================

def part8_recipe():
    print(f"\n{SEPARATOR}")
    print("  PART 8: THE COMPLETE RECIPE")
    print("  Step-by-step guide to simulating ANY wave")
    print(SEPARATOR)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  1. IDENTIFY the wave equation and medium parameters       │
    │     • Sound: c = √(γP/ρ), fixed/free BC at walls          │
    │     • Light: c = c₀/n(x), index profile defines guide     │
    │     • Water: c = √(g·h(x,y)), bathymetry defines guide    │
    │                                                             │
    │  2. DISCRETIZE the domain                                   │
    │     • Choose Δx: need ≥ 10 points per wavelength           │
    │     • Choose Δt: CFL condition c·Δt/Δx ≤ 1/√(dim)        │
    │                                                             │
    │  3. SET initial conditions                                  │
    │     • u(x, 0): initial displacement                         │
    │     • ∂u/∂t(x, 0): initial velocity                        │
    │     • Or: drive with a source term                          │
    │                                                             │
    │  4. SET boundary conditions                                 │
    │     • Reflecting: u = 0 (fixed) or ∂u/∂n = 0 (free)       │
    │     • Absorbing: Mur ABC or damping layer                   │
    │     • Periodic: wrap-around                                 │
    │                                                             │
    │  5. TIME-STEP using the FDTD formula                       │
    │     • u_new = 2·u − u_old + r²·Laplacian(u)               │
    │     • Repeat for desired duration                           │
    │                                                             │
    │  6. ANALYZE results                                         │
    │     • Visualize snapshots                                   │
    │     • Compute arrival times                                 │
    │     • FFT for spectral analysis (repo 2)                   │
    │     • Measure amplification, attenuation                    │
    └─────────────────────────────────────────────────────────────┘

    This recipe is UNIVERSAL. Scripts 02-04 each followed it
    with different parameters. Script 06 will prove they're
    the same. Script 07 applies it at full scale.
    """)


# =============================================================================
# PART 9: SUMMARY
# =============================================================================

def part9_summary():
    print(f"\n{SEPARATOR}")
    print("  SUMMARY: NUMERICAL METHODS")
    print(SEPARATOR)

    print("""
    This script covered the computational engine:

      ✓ Finite differences: replace derivatives with arithmetic
      ✓ FDTD scheme: leapfrog time-stepping for the wave equation
      ✓ CFL condition: c·Δt/Δx ≤ 1 (1D) or 1/√2 (2D)
      ✓ Boundary conditions: fixed, free, periodic, absorbing
      ✓ Numerical dispersion: grid distorts wave speeds (≥10 pts/λ)
      ✓ Convergence: second-order → 2× resolution, 4× accuracy
      ✓ 2D FDTD: the universal wave simulation engine
      ✓ The complete recipe for ANY wave simulation

    ┌─────────────────────────────────────────────────────────────┐
    │  NEXT: Script 06 — the UNIFICATION.                        │
    │  One solver, three media, same equation.                    │
    │  Then Script 07 — the CAPSTONE: all three in one sim.      │
    └─────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{SEPARATOR}")
    print("  NUMERICAL METHODS — Simulating Waves on a Computer")
    print("  FDTD, stability, and turning PDEs into code")
    print(SEPARATOR)

    part1_finite_differences()
    part2_fdtd_scheme()
    part3_cfl_condition()
    part4_boundary_conditions()
    part5_numerical_dispersion()
    part6_convergence()
    part7_2d_fdtd()
    part8_recipe()
    part9_summary()

    print(f"\n{SEPARATOR}")
    print("  Done. Run 06_unified_wave_physics.py next.")
    print(SEPARATOR)
