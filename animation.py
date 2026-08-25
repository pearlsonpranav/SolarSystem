# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from initial_conditions import (AU, SUN_MASS, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS, SUN_RADIUS, MERCURY_RADIUS, VENUS_RADIUS, EARTH_RADIUS, MARS_RADIUS, JUPITER_RADIUS, SATURN_RADIUS, URANUS_RADIUS, NEPTUNE_RADIUS, SUN_POSITION, MERCURY_INITIAL_POSITION, VENUS_INITIAL_POSITION, EARTH_INITIAL_POSITION, MARS_INITIAL_POSITION, JUPITER_INITIAL_POSITION, SATURN_INITIAL_POSITION, URANUS_INITIAL_POSITION, NEPTUNE_INITIAL_POSITION, SUN_INITIAL_VELOCITY, MERCURY_INITIAL_VELOCITY, VENUS_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY, MARS_INITIAL_VELOCITY, JUPITER_INITIAL_VELOCITY, SATURN_INITIAL_VELOCITY, URANUS_INITIAL_VELOCITY, NEPTUNE_INITIAL_VELOCITY)
from integrators import (simulate_euler, simulate_verlet, simulate_rk4)

# Solar System Data

BODY_NAMES = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

MASSES = np.array([SUN_MASS, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS])

BODY_RADII = np.array([SUN_RADIUS, MERCURY_RADIUS, VENUS_RADIUS, EARTH_RADIUS, MARS_RADIUS, JUPITER_RADIUS, SATURN_RADIUS, URANUS_RADIUS, NEPTUNE_RADIUS])

INITIAL_POSITIONS = np.array([SUN_POSITION, MERCURY_INITIAL_POSITION, VENUS_INITIAL_POSITION, EARTH_INITIAL_POSITION, MARS_INITIAL_POSITION, JUPITER_INITIAL_POSITION, SATURN_INITIAL_POSITION, URANUS_INITIAL_POSITION, NEPTUNE_INITIAL_POSITION])

INITIAL_VELOCITIES = np.array([SUN_INITIAL_VELOCITY, MERCURY_INITIAL_VELOCITY, VENUS_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY, MARS_INITIAL_VELOCITY, JUPITER_INITIAL_VELOCITY, SATURN_INITIAL_VELOCITY, URANUS_INITIAL_VELOCITY, NEPTUNE_INITIAL_VELOCITY])

# Display Colours (Vertically displayed for readability)

BODY_COLOURS = [
    "gold",       # Sun
    "dimgray",    # Mercury
    "khaki",      # Venus
    "royalblue",  # Earth
    "orangered",  # Mars
    "orange",     # Jupiter
    "wheat",      # Saturn
    "cyan",       # Uranus
    "blue"        # Neptune
    ]

# Integrators

INTEGRATORS = {"euler": simulate_euler, "verlet": simulate_verlet, "rk4": simulate_rk4}

# Orbital Periods

ORBITAL_PERIODS = {"Mercury": 100, "Venus": 230, "Earth": 370, "Mars": 700, "Jupiter": 4400, "Saturn": 11000, "Uranus": 31000, "Neptune": 61000}

# User Input

def choose_integrator():

    print("\nAvailable integrators:")
    print("- Euler")
    print("- Verlet")
    print("- RK4")

    while True:
        choice = input("\nChoose integrator: ").strip().lower()

        if choice in INTEGRATORS:
            return choice

        print("Invalid integrator. Please enter Euler, Verlet or RK4.")

def choose_dt():

    print("\nRecommended time-step range for animation:")
    print("- 1–100 s: Very high resolution, but slow simulation progress")
    print("- 1,000–10,000 s: Practical range for animation")
    print("- 100,000+ s: Faster simulation, but lower orbital resolution")

    while True:

        try:
            dt = float(input("\nChoose time step (seconds): "))

            if dt > 0:
                return dt

            print("Time step must be greater than zero.")

        except ValueError:
            print("Please enter a valid number.")

def choose_orbit_display():

    while True:

        choice = input("\nDisplay orbit trails? (yes/no): ").strip().lower()

        if choice in {"yes", "y"}:
            return True

        if choice in {"no", "n"}:
            return False

        print("Please enter yes or no.")

def choose_simulation_mode():

    print("\nSimulation duration:")
    print("1 - Fixed duration")
    print("2 - Run until window is closed")

    while True:

        choice = input("\nChoose duration mode (1 or 2): ").strip()

        if choice == "1":

            print("\nAvailable orbital periods:")

            for planet, period in ORBITAL_PERIODS.items():
                print(f"- {planet}: {period:,} days")

            print("- Custom duration")

            while True:

                duration_choice = input("\nChoose planet or custom duration: ").strip().lower()

                for planet, period in ORBITAL_PERIODS.items():

                    if duration_choice == planet.lower():
                        return period

                if duration_choice in {"custom", "c"}:

                    while True:

                        try:
                            duration = float(input("\nEnter simulation duration (days): "))

                            if duration > 0:
                                return duration

                            print("Duration must be greater than zero.")

                        except ValueError:
                            print("Please enter a valid number.")

                print("Invalid choice. Please choose a planet or Custom.")

        if choice == "2":
            return None

        print("Please enter 1 or 2.")

# Animation

def animate_solar_system(dt, integrator, display_orbits=True, simulation_duration_days=None):

    integrator_function = INTEGRATORS[integrator]

    positions = INITIAL_POSITIONS.copy()
    velocities = INITIAL_VELOCITIES.copy()

    figure = plt.figure(figsize=(12, 10))
    ax = figure.add_subplot(111, projection="3d")

    ax.set_xlabel("x position (AU)")
    ax.set_ylabel("y position (AU)")
    ax.set_zlabel("z position (AU)")
    ax.set_title("3D Solar System Animation")

    # Display range

    display_limit = 3.0

    ax.set_xlim(-display_limit, display_limit)
    ax.set_ylim(-display_limit, display_limit)
    ax.set_zlim(-display_limit, display_limit)

    # Force equal scaling on all axes.

    ax.set_box_aspect((1, 1, 1))

    # Set consistent AU tick spacing.

    axis_ticks = np.arange(-3, 4, 1)

    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ax.set_zticks(axis_ticks)

    # Physical radii converted to AU.

    body_radii_au = BODY_RADII / AU

    # Visual radius scaling.

    radius_scale = 1200

    visual_radii = body_radii_au * radius_scale

    minimum_visual_radius = 2.5

    visual_radii[1:] = np.maximum(visual_radii[1:], minimum_visual_radius)

    visual_radii[0] = max(visual_radii[0], minimum_visual_radius)

    # Body markers

    bodies = []

    for i, body_name in enumerate(BODY_NAMES):

        position = positions[i] / AU

        body, = ax.plot([position[0]], [position[1]], [position[2]], marker="o", linestyle="", markersize=visual_radii[i], color=BODY_COLOURS[i], label=body_name)

        bodies.append(body)

    # Orbit trails

    trails = []

    if display_orbits:

        for i in range(len(BODY_NAMES)):

            trail, = ax.plot([positions[i, 0] / AU], [positions[i, 1] / AU], [positions[i, 2] / AU], color=BODY_COLOURS[i], linewidth=1)

            trails.append(trail)

    # Store trail history

    position_history = [[positions[i].copy()] for i in range(len(BODY_NAMES))]

    # Time display

    time_text = ax.text2D(0.02, 0.95, "Simulation time: 0.0 days", transform=ax.transAxes)

    simulation_time = 0.0

    # Update function

    def update(frame):

        nonlocal positions, velocities, simulation_time

        # Stop fixed-duration simulations.

        if simulation_duration_days is not None:

            if simulation_time / (24 * 60 * 60) >= simulation_duration_days:
                animation.event_source.stop()
                return bodies + trails + [time_text]

        # Advance simulation by one timestep.

        positions_history, velocities_history = integrator_function(MASSES, positions, velocities, dt, 1)

        positions = positions_history[-1]
        velocities = velocities_history[-1]

        simulation_time += dt

        current_positions = positions / AU

        # Update body positions.

        for i, body in enumerate(bodies):

            body.set_data([current_positions[i, 0]], [current_positions[i, 1]])

            body.set_3d_properties([current_positions[i, 2]])

        # Update orbit trails.

        if display_orbits:

            for i, trail in enumerate(trails):

                position_history[i].append(positions[i].copy())

                trail_positions = np.array(position_history[i]) / AU

                trail.set_data(trail_positions[:, 0], trail_positions[:, 1])

                trail.set_3d_properties(trail_positions[:, 2])

        elapsed_time_days = simulation_time / (24 * 60 * 60)

        time_text.set_text(f"Simulation time: {elapsed_time_days:.1f} days")

        return bodies + trails + [time_text]

    # Animation

    animation = FuncAnimation(figure, update, interval=20, blit=False, cache_frame_data=False)

    ax.legend(loc="upper right")

    plt.show()

    return animation

# Main

def main():

    print("Solar System Animation")

    integrator = choose_integrator()
    dt = choose_dt()
    display_orbits = choose_orbit_display()
    simulation_duration_days = choose_simulation_mode()

    print("\nStarting animation...")
    print("Integrator:", integrator.upper())
    print("Time step:", dt, "s")
    print("Orbit trails:", "ON" if display_orbits else "OFF")

    if simulation_duration_days is None:
        print("Duration: Until animation window is closed")
    else:
        print("Duration:", simulation_duration_days, "days")

    print("\nClose the animation window to stop the simulation.")

    animate_solar_system(dt, integrator, display_orbits, simulation_duration_days)

if __name__ == "__main__":
    main()