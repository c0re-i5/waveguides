"""
=============================================================================
  ACOUSTIC WAVEGUIDES — Sound in Pipes
  Why flutes play notes, hallways echo, and organ pipes have lengths
=============================================================================

  Script 01 showed the wave equation and reflection at boundaries.
  Now we confine sound BETWEEN two boundaries: a pipe.

  When sound bounces back and forth inside a tube, only certain
  frequencies survive — the RESONANT MODES. Everything else
  destructively interferes with itself.

  This is why a flute can play a note: the pipe length
  selects the frequency. Change the length (open a hole)
  and you change the note.

  Connection to repo 2: the resonant modes ARE the Fourier
  components of the sound inside the pipe. The pipe performs
  a physical Fourier decomposition.
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


def ascii_spectrum(magnitudes, freq_labels, width=60, height=10, label=""):
    """Render a frequency spectrum as ASCII bars."""
    if not magnitudes:
        return
    max_mag = max(magnitudes) if max(magnitudes) > 0 else 1.0

    print(f"    {'Magnitude':>12}  Frequency")
    print(f"    {'─' * 12}  {'─' * 40}")
    for mag, freq in zip(magnitudes, freq_labels):
        bar_len = int(mag / max_mag * width)
        bar = "█" * bar_len
        print(f"    {mag:>10.2f}  │{bar} {freq}")
    if label:
        print(f"\n    {label}")


# =============================================================================
# PART 1: CLOSED-CLOSED PIPE — BOTH ENDS FIXED
# =============================================================================

def part1_closed_closed():
    print(f"\n{SEPARATOR}")
    print("  PART 1: CLOSED-CLOSED PIPE")
    print("  Both ends rigid — the simplest acoustic waveguide")
    print(SEPARATOR)

    print("""
    A pipe with BOTH ends closed acts like a string fixed at both ends.
    Sound pressure has a maximum at the walls (pressure antinode),
    and the displacement has a zero (displacement node).

    For displacement modes:

        uₙ(x) = sin(nπx/L)      n = 1, 2, 3, ...
        fₙ = n · c / (2L)

    This is identical to the string modes from script 01.
    The pipe LENGTH determines the fundamental frequency.
    """)

    c_sound = 343.0  # m/s in air
    L = 0.5  # 50 cm pipe

    print(f"    Pipe length: L = {L} m")
    print(f"    Sound speed: c = {c_sound} m/s")
    print(f"    Fundamental: f₁ = c/(2L) = {c_sound/(2*L):.1f} Hz\n")

    N = 500
    x = np.linspace(0, L, N)

    print("    Displacement standing wave modes:\n")
    for n in range(1, 5):
        mode = np.sin(n * np.pi * x / L)
        freq = n * c_sound / (2 * L)
        note = ""
        if 240 < freq < 280:
            note = " (≈ middle C region)"
        elif 430 < freq < 450:
            note = " (≈ A4 = concert pitch)"
        label = f"Mode n={n}  |  f = {freq:.1f} Hz{note}"
        ascii_plot(mode.tolist(), width=56, height=7, label=label)
        print()

    print("""
    Each mode has n half-wavelengths that fit exactly in the pipe.
    The frequencies are HARMONIC: f₂ = 2f₁, f₃ = 3f₁, etc.

    This harmonic series is why musical instruments sound "musical" —
    the overtones are integer multiples of the fundamental, creating
    a pleasing sound. A metal rod or a bell has NON-harmonic overtones,
    which is why they sound more like "clang" than "tone".
    """)


# =============================================================================
# PART 2: OPEN-OPEN PIPE
# =============================================================================

def part2_open_open():
    print(f"\n{SEPARATOR}")
    print("  PART 2: OPEN-OPEN PIPE")
    print("  Both ends free — pressure nodes at the openings")
    print(SEPARATOR)

    print("""
    An open pipe end is a PRESSURE NODE (the pressure equals
    atmospheric — the wave can't push against anything).

    For displacement, open ends are ANTINODES (max displacement).

    The mode shapes are COSINES:

        uₙ(x) = cos(nπx/L)     n = 0, 1, 2, ...
        fₙ = n · c / (2L)       (same frequencies as closed-closed!)

    Wait — the n=0 mode? That's just a uniform displacement.
    It represents a DC offset (air flowing through the pipe)
    and doesn't contribute to sound.

    The SAME harmonic series as a closed-closed pipe, but with
    different mode SHAPES. Cosines instead of sines.
    """)

    L = 0.5
    c_sound = 343.0
    N = 500
    x = np.linspace(0, L, N)

    print(f"    Open-open pipe, L = {L} m:\n")
    for n in range(1, 5):
        mode = np.cos(n * np.pi * x / L)
        freq = n * c_sound / (2 * L)
        label = f"Mode n={n}  |  f = {freq:.1f} Hz"
        ascii_plot(mode.tolist(), width=56, height=7, label=label)
        print()

    print("""
    Notice the key difference from closed-closed:
      • Closed-closed: zero displacement at BOTH ends (sines)
      • Open-open: maximum displacement at BOTH ends (cosines)

    Both produce the FULL harmonic series (all integer multiples).
    A flute approximates an open-open pipe.
    """)


# =============================================================================
# PART 3: OPEN-CLOSED PIPE — THE ODD HARMONICS
# =============================================================================

def part3_open_closed():
    print(f"\n{SEPARATOR}")
    print("  PART 3: OPEN-CLOSED PIPE")
    print("  One end open, one end closed — only odd harmonics")
    print(SEPARATOR)

    print("""
    A pipe closed at one end and open at the other is more interesting.

    The closed end forces a displacement NODE (u = 0).
    The open end forces a displacement ANTINODE (maximum).

    Only ODD quarter-wavelengths fit:

        uₙ(x) = sin((2n−1)πx/(2L))    n = 1, 2, 3, ...
        fₙ = (2n−1) · c / (4L)

    The fundamental is f₁ = c/(4L) — HALF the frequency of an
    open-open or closed-closed pipe of the same length.

    And only ODD harmonics exist: f₁, 3f₁, 5f₁, 7f₁, ...

    This gives a distinctive "hollow" sound (like a clarinet).
    """)

    L = 0.5
    c_sound = 343.0
    N = 500
    x = np.linspace(0, L, N)

    f1 = c_sound / (4 * L)
    print(f"    Open-closed pipe, L = {L} m")
    print(f"    Fundamental: f₁ = c/(4L) = {f1:.1f} Hz\n")

    for n in range(1, 5):
        mode_number = 2 * n - 1
        mode = np.sin(mode_number * np.pi * x / (2 * L))
        freq = mode_number * c_sound / (4 * L)
        label = f"Mode n={n}  |  harmonic {mode_number}  |  f = {freq:.1f} Hz"
        ascii_plot(mode.tolist(), width=56, height=7, label=label)
        print()

    print("""
    Compare the spectra:

    ┌────────────────────────────────────────────────────────────┐
    │  Closed-closed or Open-open:                              │
    │    Harmonics: 1, 2, 3, 4, 5, 6, 7, 8, ...               │
    │    → Rich, bright sound (flute, open organ pipe)          │
    │                                                           │
    │  Open-closed:                                             │
    │    Harmonics: 1, 3, 5, 7, 9, ...                         │
    │    → Hollow, reedy sound (clarinet, stopped organ pipe)   │
    └────────────────────────────────────────────────────────────┘

    The pipe's boundary conditions FILTER the wave — only allowing
    modes that satisfy the constraints. This is a physical bandpass
    filter, performing in space what repo 2's FFT filters do in code.
    """)


# =============================================================================
# PART 4: CUTOFF FREQUENCY — THE WAVEGUIDE GATE
# =============================================================================

def part4_cutoff():
    print(f"\n{SEPARATOR}")
    print("  PART 4: CUTOFF FREQUENCY")
    print("  Below this frequency, the wave can't propagate")
    print(SEPARATOR)

    print("""
    In a RECTANGULAR waveguide (a duct with width W and height H),
    transverse modes have a minimum frequency below which they
    CAN'T propagate — they decay exponentially instead.

    For a 2D duct of width W:

        f_cutoff(m) = m · c / (2W)     m = 1, 2, 3, ...

    Below f_cutoff(1) = c/(2W), ONLY the plane wave (m=0) propagates.
    The waveguide acts as a HIGH-PASS FILTER for higher modes.

    This is why:
      • Narrow pipes transmit fewer modes than wide pipes
      • Fiber optics can be "single mode" (thin enough to cut off all
        modes except the fundamental)
      • HVAC ducts transmit low-frequency rumble but not high-frequency
        speech (the duct width sets the cutoff)
    """)

    c_sound = 343.0
    widths = [0.05, 0.10, 0.20, 0.50, 1.00]  # meters

    print(f"    Cutoff frequencies for rectangular ducts (first 3 modes):\n")
    print(f"    {'Duct width':>12}  {'Mode 1':>10}  {'Mode 2':>10}  {'Mode 3':>10}")
    print(f"    {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

    for W in widths:
        f1 = c_sound / (2 * W)
        f2 = 2 * c_sound / (2 * W)
        f3 = 3 * c_sound / (2 * W)
        print(f"    {W:>10.2f} m  {f1:>8.0f} Hz  {f2:>8.0f} Hz  {f3:>8.0f} Hz")

    print("""
    A 5 cm wide duct cuts off everything above the plane wave at 3430 Hz.
    Human speech (300–3400 Hz) would barely propagate in a 5 cm duct!

    A 1 m wide duct has cutoff at 172 Hz — most sound passes through.

    This filter effect is IDENTICAL in principle to optical fibers:
      • Wider core → more modes → more dispersion
      • Narrower core → fewer modes → cleaner signal
    We'll see the optical version in script 03.
    """)


# =============================================================================
# PART 5: DISPERSION — SPEED DEPENDS ON FREQUENCY
# =============================================================================

def part5_dispersion():
    print(f"\n{SEPARATOR}")
    print("  PART 5: WAVEGUIDE DISPERSION")
    print("  Group velocity vs. phase velocity")
    print(SEPARATOR)

    print("""
    In free space, all sound frequencies travel at c = 343 m/s.
    Inside a waveguide, HIGHER MODES travel SLOWER. This is
    dispersion: different frequencies have different speeds.

    For mode m in a duct of width W, the propagation speed is:

        v_phase(f) = c / √(1 − (f_c/f)²)      phase velocity
        v_group(f) = c · √(1 − (f_c/f)²)       group velocity

    where f_c = m·c/(2W) is the cutoff frequency.

    ┌─────────────────────────────────────────────────────────────┐
    │  Phase velocity: speed of the wave CRESTS                  │
    │    → Can exceed c! (but no information travels this fast)  │
    │                                                            │
    │  Group velocity: speed of the ENVELOPE (the information)   │
    │    → Always ≤ c                                            │
    │    → Goes to ZERO at the cutoff frequency                  │
    │                                                            │
    │  v_phase × v_group = c²  (always!)                         │
    └─────────────────────────────────────────────────────────────┘
    """)

    c_sound = 343.0
    W = 0.10  # 10 cm duct
    f_cutoff = c_sound / (2 * W)

    print(f"    Duct width: W = {W} m")
    print(f"    Mode 1 cutoff: f_c = {f_cutoff:.0f} Hz\n")

    freqs = np.linspace(f_cutoff * 1.01, f_cutoff * 5.0, 200)
    v_group = c_sound * np.sqrt(1 - (f_cutoff / freqs) ** 2)
    v_phase = c_sound / np.sqrt(1 - (f_cutoff / freqs) ** 2)

    print("    Group velocity (information speed) vs frequency:\n")
    ascii_plot(v_group.tolist(), width=56, height=9,
              label=f"Group velocity (m/s) — approaches c={c_sound} m/s at high f")
    print()

    print("    Phase velocity (crest speed) vs frequency:\n")
    # Clip phase velocity for display
    v_phase_clipped = np.clip(v_phase, 0, 2000)
    ascii_plot(v_phase_clipped.tolist(), width=56, height=9,
              label="Phase velocity (m/s) — diverges near cutoff, approaches c at high f")
    print()

    # Show relationship
    print("    Verification: v_phase × v_group = c² at selected frequencies:\n")
    test_freqs = [f_cutoff * 1.5, f_cutoff * 2.0, f_cutoff * 3.0, f_cutoff * 5.0]
    print(f"    {'Frequency':>12}  {'v_group':>10}  {'v_phase':>10}  {'v_g × v_p':>12}  {'c²':>12}")
    print(f"    {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 12}  {'─' * 12}")
    for f in test_freqs:
        vg = c_sound * math.sqrt(1 - (f_cutoff / f) ** 2)
        vp = c_sound / math.sqrt(1 - (f_cutoff / f) ** 2)
        print(f"    {f:>10.0f} Hz  {vg:>8.1f} m/s  {vp:>8.1f} m/s  {vg*vp:>10.0f}  {c_sound**2:>10.0f}")

    print("""
    The product v_group × v_phase = c² exactly. Always.

    Near cutoff: v_group → 0, v_phase → ∞ (but no energy moves)
    Far above cutoff: both approach c (free-space behavior)

    This dispersion is why a sharp pulse SPREADS OUT as it
    travels through a waveguide — different frequency components
    travel at different speeds. Repo 2's wavelets are the tool
    for analyzing this time-frequency behavior.
    """)


# =============================================================================
# PART 6: FDTD SIMULATION — SOUND IN A PIPE
# =============================================================================

def part6_pipe_simulation():
    print(f"\n{SEPARATOR}")
    print("  PART 6: PIPE SIMULATION")
    print("  Watching sound bounce inside a tube")
    print(SEPARATOR)

    print("""
    Let's simulate a sound pulse traveling through a closed pipe
    and watch it reflect back and forth:
    """)

    # 1D FDTD in a closed pipe
    L = 1.0  # 1 meter pipe
    c = 343.0  # m/s
    N = 500
    dx = L / N
    dt = 0.8 * dx / c  # CFL condition: r = c·dt/dx < 1

    x = np.linspace(0, L, N)

    # Initial condition: Gaussian pulse at 1/4 of pipe
    u_prev = np.exp(-((x - 0.25) ** 2) / (2 * 0.02 ** 2))
    u_curr = np.exp(-((x - (0.25 + c * dt)) ** 2) / (2 * 0.02 ** 2))

    r = (c * dt / dx) ** 2

    print(f"    Pipe length: {L} m")
    print(f"    Grid points: {N}")
    print(f"    CFL number: r = c·dt/dx = {math.sqrt(r):.3f}")
    print(f"    Boundary: CLOSED at both ends (fixed: u = 0)\n")

    snapshots = []
    snap_steps = [0, 80, 160, 240, 320, 400]
    snap_idx = 0
    step = 0

    if step == snap_steps[snap_idx]:
        snapshots.append((u_curr.copy(), step))
        snap_idx += 1

    for _ in range(max(snap_steps)):
        u_next = np.zeros(N)
        for i in range(1, N - 1):
            u_next[i] = 2 * u_curr[i] - u_prev[i] + r * (
                u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
            )
        # Fixed boundaries (closed pipe: u = 0 at both ends)
        u_next[0] = 0.0
        u_next[N - 1] = 0.0

        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        step += 1

        if snap_idx < len(snap_steps) and step == snap_steps[snap_idx]:
            snapshots.append((u_curr.copy(), step))
            snap_idx += 1

    for u_snap, s in snapshots:
        t_ms = s * dt * 1000
        label = f"step {s}  |  t = {t_ms:.2f} ms"
        ascii_plot(u_snap.tolist(), width=56, height=7, label=label)
        print()

    print("""
    The pulse bounces between the closed ends:
      • Each reflection at a closed end INVERTS the pulse
      • After two reflections, the pulse is upright again
      • The period is T = 2L/c (round trip)

    For L = 1.0 m: T = 2(1.0)/343 = 5.83 ms → f = 171.5 Hz
    This matches the fundamental frequency: f₁ = c/(2L) = 171.5 Hz ✓

    The pipe has SELECTED this frequency through its geometry.
    It's a physical Fourier filter.
    """)


# =============================================================================
# PART 7: RESONANCE — FREQUENCY RESPONSE OF A PIPE
# =============================================================================

def part7_resonance():
    print(f"\n{SEPARATOR}")
    print("  PART 7: RESONANCE")
    print("  The pipe amplifies its natural frequencies")
    print(SEPARATOR)

    print("""
    If you drive a pipe with a frequency matching one of its modes,
    the wave builds up — RESONANCE. The amplitude grows until
    damping balances the input.

    Let's compute the frequency response: drive the pipe at
    different frequencies and measure the steady-state amplitude.
    """)

    L = 0.5  # 50 cm pipe
    c_sound = 343.0
    N = 200
    dx = L / N
    x = np.linspace(0, L, N)

    # Resonant frequencies
    f_resonant = [n * c_sound / (2 * L) for n in range(1, 6)]

    # Test a range of frequencies
    test_freqs = np.linspace(100, 3000, 50)
    amplitudes = []

    for f_drive in test_freqs:
        dt = min(0.8 * dx / c_sound, 0.2 / f_drive)  # CFL + resolve the wave
        r = (c_sound * dt / dx) ** 2
        ω = 2 * math.pi * f_drive

        u_prev = np.zeros(N)
        u_curr = np.zeros(N)

        # Drive for enough cycles to approach steady state
        n_cycles = 8
        total_steps = int(n_cycles / (f_drive * dt))
        total_steps = min(total_steps, 2000)  # cap for speed

        max_amp = 0.0
        for step in range(total_steps):
            t = step * dt
            u_next = np.zeros(N)
            for i in range(1, N - 1):
                u_next[i] = 2 * u_curr[i] - u_prev[i] + r * (
                    u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]
                )
            # Closed-closed: u = 0 at both ends
            u_next[0] = 0.0
            u_next[N - 1] = 0.0

            # Drive source at x = dx (just inside left end)
            u_next[1] = math.sin(ω * t) * 0.1

            u_prev = u_curr.copy()
            u_curr = u_next.copy()

            # Measure amplitude at pipe center in the last few cycles
            if step > total_steps * 0.7:
                max_amp = max(max_amp, abs(u_curr[N // 2]))

        amplitudes.append(max_amp)

    # Plot frequency response
    print(f"\n    Frequency response of a {L} m closed pipe:\n")
    ascii_plot(amplitudes, width=56, height=11,
              label=f"Amplitude at pipe center vs. driving frequency ({test_freqs[0]:.0f}–{test_freqs[-1]:.0f} Hz)")

    print(f"\n    Predicted resonant frequencies:")
    for n, f in enumerate(f_resonant, 1):
        print(f"      Mode {n}: f = {f:.1f} Hz")

    # Find actual peaks
    peaks = []
    for i in range(1, len(amplitudes) - 1):
        if amplitudes[i] > amplitudes[i-1] and amplitudes[i] > amplitudes[i+1]:
            if amplitudes[i] > max(amplitudes) * 0.1:
                peaks.append(test_freqs[i])

    if peaks:
        print(f"\n    Measured peaks in simulation:")
        for p in peaks[:5]:
            # Find nearest theoretical mode
            nearest = min(f_resonant, key=lambda f: abs(f - p))
            error = abs(p - nearest) / nearest * 100
            print(f"      f ≈ {p:.0f} Hz  (nearest mode: {nearest:.0f} Hz, error: {error:.1f}%)")

    print("""
    The peaks align with the predicted modes — the pipe RESONATES
    at frequencies where standing waves fit its length.

    This is the acoustic equivalent of a band-pass filter:
    the pipe amplifies its resonant frequencies and attenuates
    everything else. Musical instruments exploit this to select
    the notes they produce.
    """)


# =============================================================================
# PART 8: REAL INSTRUMENTS — PUTTING IT ALL TOGETHER
# =============================================================================

def part8_instruments():
    print(f"\n{SEPARATOR}")
    print("  PART 8: REAL INSTRUMENTS")
    print("  Pipes, flutes, and organ stops")
    print(SEPARATOR)

    print("""
    Real instruments are acoustic waveguides with specific
    boundary conditions and geometries:
    """)

    instruments = [
        ("Flute (C5)", "open-open", 0.325, "All harmonics"),
        ("Clarinet (Bb3)", "open-closed", 0.660, "Odd harmonics only"),
        ("Organ pipe 8' (C2)", "open-open", 2.640, "All harmonics"),
        ("Organ pipe 8' stopped", "open-closed", 1.320, "Odd harmonics only"),
        ("Piccolo (C6)", "open-open", 0.163, "All harmonics"),
        ("Didgeridoo (Bb1)", "open-closed", 1.470, "Odd harmonics only"),
    ]

    c_sound = 343.0

    print(f"    {'Instrument':<26} {'Boundary':<14} {'Length':>8} {'f₁':>10} {'Overtone series'}")
    print(f"    {'─' * 26} {'─' * 14} {'─' * 8} {'─' * 10} {'─' * 20}")

    for name, btype, length, series in instruments:
        if "open-closed" in btype:
            f1 = c_sound / (4 * length)
        else:
            f1 = c_sound / (2 * length)
        print(f"    {name:<26} {btype:<14} {length:>6.3f} m {f1:>8.1f} Hz {series}")

    print("""
    Notice:
      • An open-closed pipe is HALF the length to produce the same
        fundamental as an open-open pipe. The "stopped" organ pipe
        saves space by closing one end.

      • The clarinet and didgeridoo sound "hollow" because they only
        produce odd harmonics (1st, 3rd, 5th, ...). The even harmonics
        are missing — the boundary conditions forbid them.

      • The flute sounds "bright" because it has ALL harmonics.

    ┌─────────────────────────────────────────────────────────────┐
    │  The instrument's GEOMETRY determines what you HEAR.        │
    │  The waveguide is the filter. The air is the medium.       │
    │  The musician excites the modes. Physics does the rest.    │
    └─────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# PART 9: SUMMARY
# =============================================================================

def part9_summary():
    print(f"\n{SEPARATOR}")
    print("  SUMMARY: ACOUSTIC WAVEGUIDES")
    print(SEPARATOR)

    print("""
    This script explored sound confined by boundaries:

      ✓ Closed-closed pipes: sin modes, all harmonics (f = nc/(2L))
      ✓ Open-open pipes: cos modes, all harmonics (same frequencies)
      ✓ Open-closed pipes: only ODD harmonics (f = (2n-1)c/(4L))
      ✓ Cutoff frequency: higher modes can't propagate below f_c
      ✓ Dispersion: v_group × v_phase = c² (always)
      ✓ FDTD simulation: pulse bouncing in a pipe
      ✓ Resonance: pipe amplifies its natural frequencies
      ✓ Real instruments: geometry → sound character

    ┌─────────────────────────────────────────────────────────────┐
    │  NEXT: Script 03 applies the SAME physics to LIGHT.        │
    │  Instead of rigid walls → refractive index boundaries.     │
    │  Instead of pressure waves → electromagnetic waves.         │
    │  Instead of pipe length → fiber core diameter.              │
    │  Same equation. Different medium. Same math.               │
    └─────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{SEPARATOR}")
    print("  ACOUSTIC WAVEGUIDES — Sound in Pipes")
    print("  Why flutes play notes and hallways echo")
    print(SEPARATOR)

    part1_closed_closed()
    part2_open_open()
    part3_open_closed()
    part4_cutoff()
    part5_dispersion()
    part6_pipe_simulation()
    part7_resonance()
    part8_instruments()
    part9_summary()

    print(f"\n{SEPARATOR}")
    print("  Done. Run 03_optical_waveguides.py next.")
    print(SEPARATOR)
