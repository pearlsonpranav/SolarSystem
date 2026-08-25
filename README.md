# SolarSystem

A numerical simulation of the Solar System written in Python, developed to explore computational physics, numerical integration and orbital mechanics.

The project models the gravitational interaction between the Sun and eight planets and uses numerical integrators to simulate their motion in three dimensions. It also includes tools for analysing orbital behaviour, testing physical conservation laws and comparing the accuracy and stability of different numerical integration methods.

## Performance & Validation Profile

The engine programmatically benchmarks numerical drift and geometric compliance across extended trajectory windows:
* **Algorithmic Floating-Point Floor (RK4):** Restricts total system mechanical energy and angular momentum numerical variation to within $10^{-11}\%$ and $10^{-12}\%$ respectively over a 1.4M+ step execution matrix.
* **Structural Reference Frame Stability:** Limits system barycentre coordinate displacement to less than $0.0001 \text{ meters}$ across a full 1000-day orbital sweep.
* **Geometric Path Alignment:** Reuses vectorised parameter states to perform trajectory-wide coordinate mapping against analytical conic section parameters, verifying Kepler's First Law with a maximum spatial fit error of $0.000000\%$.

## Features

- 3D N-body gravitational simulation of the Sun and eight planets
- Euler, Verlet and fourth-order Runge–Kutta (RK4) integrators
- Three-dimensional planetary positions and velocities
- Orbital parameter calculations
- Kepler's three laws
- Vis-viva equation
- Orbital eccentricity and inclination
- Perihelion and aphelion detection
- Escape velocity and orbit classification
- Total kinetic, potential and mechanical energy calculations
- Centre-of-mass and barycentre calculations
- Angular momentum calculations
- Energy and angular momentum conservation checks
- Numerical integrator error and stability analysis
- 3D Solar System visualisation
- Animated planetary orbits with optional orbit trails

## Numerical Methods

The simulation currently supports three numerical integration methods:

### Euler

A first-order integration method used primarily as a baseline for comparison.

### Verlet

A second-order integration method with improved long-term behaviour for orbital systems.

### RK4

A fourth-order Runge–Kutta method providing substantially higher local accuracy and serving as a high-accuracy reference for numerical comparisons.

The `compare_integrators.py` program evaluates the methods using different time steps and compares their position errors, velocity errors, energy variation and angular momentum variation.

## Project Structure

```text
SolarSystem/
│
├── initial_conditions.py
├── mechanics.py
├── orbits.py
├── integrators.py
├── analysis.py
├── animation.py
├── compare_integrators.py
├── requirements.txt
├── .gitignore
└── README.md
```

### `initial_conditions.py`

Contains the physical constants, masses, radii and initial positions and velocities used by the simulation.

### `mechanics.py`

Contains the core mechanics calculations, including gravitational quantities, kinetic and potential energy, total energy, angular momentum, escape velocity and centre-of-mass calculations.

### `orbits.py`

Contains orbital-mechanics calculations, including orbital parameters, eccentricity, inclination, Kepler's laws, perihelion detection, orbital periods, vis-viva calculations and orbit classification.

### `integrators.py`

Contains the numerical integration methods used to evolve the Solar System:

- Euler
- Verlet
- RK4

### `analysis.py`

Runs a configurable Solar System simulation for a selected planet and analyses its resulting orbit and physical properties.

The analysis includes:

- orbital parameters
- inclination
- eccentricity
- barycentre behaviour
- Kepler's laws
- perihelion and aphelion velocities
- vis-viva behaviour
- energy conservation
- escape velocity
- orbit classification
- angular momentum conservation

### `animation.py`

Provides a real-time 3D visualisation of the Solar System with selectable numerical integrators, time steps, simulation durations and optional orbital trails.

### `compare_integrators.py`

Compares the numerical behaviour of Euler, Verlet and RK4 across several time steps.

It evaluates:

- position error
- velocity error
- total-energy variation
- angular-momentum variation

The RK4 solution at a small time step is used as the numerical reference solution for the error analysis.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd SolarSystem
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

### Orbital Analysis

```bash
python analysis.py
```

The analyser allows you to select:

- target planet
- numerical integrator
- time step

### 3D Animation

```bash
python animation.py
```

The animation allows you to select:

- numerical integrator
- time step
- orbit-trail display
- simulation duration

### Integrator Comparison

```bash
python compare_integrators.py
```

This runs a numerical comparison between the available integration methods and produces error and stability plots.

## Units

The internal simulation uses SI units for the physical calculations:

- Distance: metres
- Time: seconds
- Mass: kilograms
- Velocity: metres per second
- Energy: joules

Distances are converted to astronomical units (AU) for visualisation where appropriate.

## Project Development

The project is being developed progressively, with the emphasis shifting from implementing basic gravitational dynamics towards numerical accuracy, orbital mechanics and higher-dimensional Solar System modelling.

Current development includes:

- multi-body gravitational dynamics
- multiple numerical integration methods
- numerical accuracy testing
- conservation-law analysis
- three-dimensional orbital modelling
- 3D visualisation and animation

Future development will focus on improving numerical accuracy and stability, expanding the physical model to incorporate General Relativity (GR) adjustments for relativistic corrections like Mercury's Schwarzschild precession, and developing a more sophisticated interactive visualisation.

## Technologies

- Python
- NumPy
- Matplotlib

## Purpose

This project is primarily an exploration of computational physics and numerical methods.

Rather than treating the Solar System as a purely visual simulation, the project uses the model to investigate how numerical algorithms behave when applied to a physical dynamical system, including questions of accuracy, stability and conservation of physical quantities.

## Author

Pranav M
