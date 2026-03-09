# Waveguides: Sound, Light, and Tsunamis

**How the same wave equation governs sound in a pipe, light in a fiber, and tsunamis across an ocean — and what boundaries do to waves.**

```
∂²u/∂t² = c² · ∇²u     — "One equation. Three domains.
                           The medium sets the speed.
                           The boundaries shape the wave."
```

This repo picks up where [Navigational Pathfinding](https://github.com/c0re-i5/navigational-pathfinding), [Bit Tricks and Wave Functions](https://github.com/c0re-i5/bit-tricks-and-wave-functions), and [Exploring: From FISR to Conway and Beyond](https://github.com/c0re-i5/exploring-from-fisr-to-conway-and-beyond) left off. Those repos showed:

> **Repo 1:** *The representation of data has structure. That structure is a tool.*
>
> **Repo 2:** *Every signal is a sum of waves. Decompose it right, and the impossible becomes trivial.*
>
> **Repo 3:** *Propagate waves through space. Let interference find the path.*

This repo adds a fourth law:

> **Repo 4:** *Boundaries don't just contain waves — they shape them. The same physics, different media, one equation.*

The culmination is **The Great Wave** — a multi-medium simulation that propagates a tsunami across an ocean floor, sound through a channel, and light through a fiber, demonstrating that one numerical solver handles all three because the underlying equation is identical.

Each Python script is self-contained and meant to be read as much as run. `numpy` is the only dependency.

## The Scripts

Run them in order for the best experience, or jump to whatever catches your eye.

### 1. [`01_wave_equation_fundamentals.py`](01_wave_equation_fundamentals.py) — One Equation to Rule Them All

The wave equation derived, visualized, and understood. Start with a vibrating string and build outward to 2D.

**Topics:** the 1D wave equation · superposition principle · reflection at fixed and free boundaries · refraction at medium interfaces · standing waves · wave speed and medium properties · the 2D wave equation

### 2. [`02_acoustic_waveguides.py`](02_acoustic_waveguides.py) — Sound in Pipes

Why a flute plays notes, why your voice echoes in a hallway, and why organ pipes have specific lengths.

**Topics:** acoustic wave equation · standing waves in tubes · resonance modes and harmonics · open vs. closed pipe boundary conditions · cutoff frequencies · waveguide dispersion · group velocity vs. phase velocity

### 3. [`03_optical_waveguides.py`](03_optical_waveguides.py) — Light in Fibers

Total internal reflection, fiber optic modes, and why your internet is a beam of light bouncing down a glass tube.

**Topics:** Snell's law · critical angle · total internal reflection · step-index fiber modes · numerical aperture · V-number and mode count · modal dispersion · single-mode vs. multi-mode · pulse broadening

### 4. [`04_fluid_channels_and_shallow_water.py`](04_fluid_channels_and_shallow_water.py) — Water Finds the Way

The shallow water equations that govern tsunamis, tides, and storm surges. How ocean depth steers waves like a lens.

**Topics:** shallow water equations · wave speed = √(g·h) · bathymetric refraction · shoaling and amplification · coastal waveguides · Snell's law for ocean waves · the connection to Repo 3's Eikonal equation

### 5. [`05_numerical_methods.py`](05_numerical_methods.py) — Simulating Waves on a Computer

The engine behind every wave simulation: finite differences, stability conditions, and absorbing boundaries.

**Topics:** FDTD (finite-difference time-domain) · CFL stability condition · spatial and temporal discretization · absorbing boundary conditions · Mur boundaries · numerical dispersion · convergence analysis

### 6. [`06_unified_wave_physics.py`](06_unified_wave_physics.py) — One Solver, Three Domains

The payoff: scripts 02–04 are the *same equation* with different parameters. One FDTD solver, three physical domains, identical results.

| Medium | Wave speed | Governing parameter | Boundary type |
|--------|-----------|-------------------|--------------|
| Sound in air | 343 m/s | Air pressure / density | Pipe walls (rigid) |
| Light in glass | 2×10⁸ m/s | Refractive index | Core/cladding interface |
| Water (ocean) | √(g·h) m/s | Depth h | Coastline / bathymetry |

**Topics:** parameter mapping · dimensional analysis · normalized wave equation · acoustic ↔ optical ↔ fluid correspondence · impedance mismatch · universal reflection coefficient

### 7. [`07_the_great_wave.py`](07_the_great_wave.py) — The Capstone

Chains every technique into one multi-medium simulation: an earthquake generates a tsunami that propagates across a realistic ocean floor, sound reverberates through a channel, and light pulses through a fiber — all using the same numerical engine.

```
Earthquake source → Shallow water propagation → Coastal amplification
Sound pulse        → Acoustic pipe resonance   → Standing wave pattern
Light pulse        → Fiber optic transmission   → Modal dispersion

Three media. One wave equation. One solver. Same physics.
```

Every stage pulls from a different chapter of this repo, demonstrating that waveguide physics is universal — the medium sets the speed, the boundaries shape the wave, but the math is always the same.

### Interactive: [`wave_sandbox.html`](wave_sandbox.html) — The Wave Sandbox

A browser-based 2D wave simulator. Paint different media on a canvas, place wave sources, and watch real-time FDTD propagation. No server, no build step — just open the file in your browser.

**Features:**
- Paint media (deep ocean, glass, air, etc.) with a brush and watch waves interact
- Place continuous oscillators or drop single pulses
- Shape tools: line, rect, circle, ellipse, diamond, star (outline or filled), with adjustable thickness
- Presets: Double Slit, Tsunami, Fiber Optic, Sound Pipe, Lens, Prism
- Switch between Fluid / Acoustic / Optical modes (changes labels, same physics)
- Undo / Redo (Ctrl+Z / Ctrl+Shift+Z) for all drawing operations
- Adjustable speed, frequency, gain, and brush size with live value labels
- Real-time FDTD at 60 fps on a 300×180 grid with performance-optimised rendering

Same equation, same solver — you just paint a different c(x,y).

### Reference

[`wave_equation.txt`](wave_equation.txt) — The wave equation from first principles: Newton's second law to ∂²u/∂t² = c²∂²u/∂x².

[`shallow_water.txt`](shallow_water.txt) — The shallow water equations and why depth controls everything.

Pre-generated output is in the [`output/`](output/) folder if you just want to read the results.

## Running

Python 3.10+ and `numpy` required. Add `matplotlib` for optional visualizations.

```bash
pip install numpy
python 01_wave_equation_fundamentals.py
python 02_acoustic_waveguides.py
python 03_optical_waveguides.py
python 04_fluid_channels_and_shallow_water.py
python 05_numerical_methods.py
python 06_unified_wave_physics.py
python 07_the_great_wave.py
```

For the interactive sandbox, open [`wave_sandbox.html`](wave_sandbox.html) in any modern browser. No dependencies, no server.

Run script 07 for the full multi-medium simulation.

Pre-generated output is in the [`output/`](output/) folder if you just want to read the results.

## The Connection Map

Every topic in this repo traces back to the same starting point:

```
The Wave Equation: ∂²u/∂t² = c² · ∇²u
    │
    ├── Acoustic waveguides (sound)
    │     ├── Standing waves in pipes
    │     ├── Resonance modes & harmonics
    │     ├── Cutoff frequencies
    │     └── Group velocity vs. phase velocity
    │
    ├── Optical waveguides (light)
    │     ├── Snell's law & total internal reflection
    │     ├── Fiber modes (V-number)
    │     ├── Modal dispersion & pulse broadening
    │     └── Single-mode vs. multi-mode fibers
    │
    ├── Fluid waveguides (water)
    │     ├── Shallow water equations
    │     ├── Bathymetric refraction (depth steers waves)
    │     ├── Tsunami propagation & coastal amplification
    │     └── Eikonal equation — arrival times  [← repo 3]
    │
    ├── Numerical methods (the engine)
    │     ├── FDTD discretization
    │     ├── CFL stability condition
    │     ├── Absorbing boundaries (Mur, damping)
    │     └── Convergence & numerical dispersion
    │
    ├── The unification
    │     ├── One solver, three domains
    │     ├── Impedance mismatch = reflection  [← repo 2 FFT analysis]
    │     └── Normalized equation (c=1, boundaries differ)
    │
    └── Bridge to previous repos
          ├── FFT for spectral mode analysis     [← repo 2]
          ├── Wave propagation = pathfinding      [← repo 3]
          ├── Eikonal equation (arrival times)    [← repo 3]
          ├── Nyquist: grid resolution ≥ 2× wavelength [← repo 2]
          └── Bit-level float precision matters   [← repo 1]
```

## License

[MIT](LICENSE)
