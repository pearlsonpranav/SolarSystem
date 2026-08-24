# Import Libraries

import numpy as np
import matplotlib.pyplot as plt

from initial_conditions import (G, AU, SUN_MASS, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS, SUN_POSITION, MERCURY_INITIAL_POSITION, VENUS_INITIAL_POSITION, EARTH_INITIAL_POSITION, MARS_INITIAL_POSITION, JUPITER_INITIAL_POSITION, SATURN_INITIAL_POSITION, URANUS_INITIAL_POSITION, NEPTUNE_INITIAL_POSITION, SUN_INITIAL_VELOCITY, MERCURY_INITIAL_VELOCITY, VENUS_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY, MARS_INITIAL_VELOCITY, JUPITER_INITIAL_VELOCITY, SATURN_INITIAL_VELOCITY, URANUS_INITIAL_VELOCITY, NEPTUNE_INITIAL_VELOCITY)
from mechanics import (calculate_distances, calculate_total_kinetic_energy, calculate_total_potential_energy, calculate_total_energy, calculate_escape_velocity, calculate_total_angular_momentum, calculate_centre_of_mass)
from integrators import (simulate_rk4)
from kepler import (calculate_orbital_parameters, calculate_ellipse_centre, calculate_eccentricity_vectors, calculate_eccentricity_from_vector, check_keplers_first_law, calculate_areal_velocities, check_keplers_second_law, find_perihelion_indices, calculate_orbital_period, check_keplers_third_law, calculate_vis_viva_velocity, check_vis_viva, calculate_specific_orbital_energy, classify_orbit, calculate_inclination)

# Planet Data (Vertically displayed the data for readability)

PLANETS = {
    "Mercury": {"index": 1, "mass": MERCURY_MASS, "position": MERCURY_INITIAL_POSITION, "velocity": MERCURY_INITIAL_VELOCITY, "simulation_time_days": 100},
    "Venus": {"index": 2, "mass": VENUS_MASS, "position": VENUS_INITIAL_POSITION, "velocity": VENUS_INITIAL_VELOCITY, "simulation_time_days": 230},
    "Earth": {"index": 3, "mass": EARTH_MASS, "position": EARTH_INITIAL_POSITION, "velocity": EARTH_INITIAL_VELOCITY, "simulation_time_days": 370},
    "Mars": {"index": 4, "mass": MARS_MASS, "position": MARS_INITIAL_POSITION, "velocity": MARS_INITIAL_VELOCITY, "simulation_time_days": 700},
    "Jupiter": {"index": 5, "mass": JUPITER_MASS, "position": JUPITER_INITIAL_POSITION, "velocity": JUPITER_INITIAL_VELOCITY, "simulation_time_days": 4400},
    "Saturn": {"index": 6, "mass": SATURN_MASS, "position": SATURN_INITIAL_POSITION, "velocity": SATURN_INITIAL_VELOCITY, "simulation_time_days": 11000},
    "Uranus": {"index": 7, "mass": URANUS_MASS, "position": URANUS_INITIAL_POSITION, "velocity": URANUS_INITIAL_VELOCITY, "simulation_time_days": 31000},
    "Neptune": {"index": 8, "mass": NEPTUNE_MASS, "position": NEPTUNE_INITIAL_POSITION, "velocity": NEPTUNE_INITIAL_VELOCITY, "simulation_time_days": 61000}
    }

# Functions

def plot_energy(kinetic_energies, potential_energies, total_energies, dt, planet_name):
    time = np.arange(len(kinetic_energies)) * dt / (60 * 60 * 24)

    plt.figure(figsize=(10, 6))

    plt.plot(time, kinetic_energies, label="Kinetic Energy")
    plt.plot(time, potential_energies, label="Potential Energy")
    plt.plot(time, total_energies, label="Total Energy")

    plt.xlabel("Time (days)")
    plt.ylabel("Energy (J)")
    plt.title(f"Solar System Energy Variation — {planet_name} Analysis")
    plt.legend(loc="upper right")
    plt.grid()

    plt.show()


def plot_solar_system(position_history, plot_every=1000):
    body_names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

    position_history = position_history[::plot_every]

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")

    for i, body_name in enumerate(body_names):
        x = position_history[:, i, 0] / AU
        y = position_history[:, i, 1] / AU
        z = position_history[:, i, 2] / AU

        ax.plot(x, y, z, label=body_name)

    ax.set_xlabel("x position (AU)")
    ax.set_ylabel("y position (AU)")
    ax.set_zlabel("z position (AU)")
    ax.set_title("3D Solar System — Non-Coplanar Orbits")
    ax.legend(loc="upper right")

    plt.show()

# Core

def main():

    # Select Planet

    target_planet = "Earth"

    if target_planet not in PLANETS:
        raise ValueError(f"Unknown planet: {target_planet}. Choose from {list(PLANETS.keys())}")

    planet = PLANETS[target_planet]

    planet_index = planet["index"]
    planet_mass = planet["mass"]

    print("Solar System Simulation")
    print("Target planet:", target_planet)

    # Simulation

    dt = 10.0
    simulation_time = planet["simulation_time_days"] * 24 * 60 * 60
    number_of_steps = round(simulation_time / dt)

    masses = np.array([SUN_MASS, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS])

    positions = np.array([SUN_POSITION, MERCURY_INITIAL_POSITION, VENUS_INITIAL_POSITION, EARTH_INITIAL_POSITION, MARS_INITIAL_POSITION, JUPITER_INITIAL_POSITION, SATURN_INITIAL_POSITION, URANUS_INITIAL_POSITION, NEPTUNE_INITIAL_POSITION])

    velocities = np.array([SUN_INITIAL_VELOCITY, MERCURY_INITIAL_VELOCITY, VENUS_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY, MARS_INITIAL_VELOCITY, JUPITER_INITIAL_VELOCITY, SATURN_INITIAL_VELOCITY, URANUS_INITIAL_VELOCITY, NEPTUNE_INITIAL_VELOCITY])

    rk4_positions, rk4_velocities = simulate_rk4(masses, positions, velocities, dt, number_of_steps)

    # Barycentres

    barycentres = []

    for i in range(len(rk4_positions)):
        barycentre = calculate_centre_of_mass(masses, rk4_positions[i])
        barycentres.append(barycentre)

    barycentres = np.array(barycentres)

    initial_barycentre = barycentres[0]
    final_barycentre = barycentres[-1]

    barycentre_displacement = np.linalg.norm(final_barycentre - initial_barycentre)

    print("Initial barycentre:", initial_barycentre)
    print("Final barycentre:", final_barycentre)
    print("Barycentre displacement:", barycentre_displacement, "m")

    if np.isclose(barycentre_displacement, 0.0, atol=1e-3):
        print("Barycentre is stationary.")
    else:
        print("Barycentre is not stationary.")

    initial_barycentre_velocity = calculate_centre_of_mass(masses, velocities)
    final_barycentre_velocity = calculate_centre_of_mass(masses, rk4_velocities[-1])

    print("Initial barycentre velocity:", initial_barycentre_velocity, "m/s")
    print("Final barycentre velocity:", final_barycentre_velocity, "m/s")

    # Target Planet Trajectory Relative to Sun

    sun_index = 0

    planet_positions = rk4_positions[:, planet_index, :] - rk4_positions[:, sun_index, :]
    planet_velocities = rk4_velocities[:, planet_index, :] - rk4_velocities[:, sun_index, :]

    # Orbital Distances

    distances = calculate_distances(planet_positions)

    perihelion_distance, aphelion_distance, semi_major_axis, _ = calculate_orbital_parameters(distances)

    print("Calculated semi-major axis:", semi_major_axis / AU, "AU")

    # Orbital Inclination

    REFERENCE_NORMAL = np.array([0.0, 0.0, 1.0])

    initial_inclination = calculate_inclination(planet_positions[0], planet_velocities[0], REFERENCE_NORMAL)
    final_inclination = calculate_inclination(planet_positions[-1], planet_velocities[-1], REFERENCE_NORMAL)

    print("Initial orbital inclination:", np.rad2deg(initial_inclination), "degrees")
    print("Final orbital inclination:", np.rad2deg(final_inclination), "degrees")

    # Kepler's First Law

    eccentricity_vectors = calculate_eccentricity_vectors(planet_positions, planet_velocities, SUN_MASS, planet_mass)

    eccentricities = np.linalg.norm(eccentricity_vectors, axis=1)

    eccentricity_vector = eccentricity_vectors[0]

    eccentricity_from_vector = calculate_eccentricity_from_vector(eccentricity_vector)

    mean_eccentricity = np.mean(eccentricities) # The full N-body orbit is perturbed, so eccentricity varies slightly over time; therefore, we would not expect the textbook value for e.

    print("Initial eccentricity:", eccentricity_from_vector)
    print("Mean eccentricity:", mean_eccentricity)

    centre = calculate_ellipse_centre(eccentricity_vector, semi_major_axis)

    if check_keplers_first_law(centre, semi_major_axis, eccentricity_from_vector):
        print("Kepler's First Law is satisfied.")
    else:
        print("Kepler's First Law is not satisfied.")

    # Kepler's Second Law

    areal_velocities = calculate_areal_velocities(planet_positions, planet_velocities)

    percentage_difference_areal_velocity = (np.max(areal_velocities) - np.min(areal_velocities)) / np.mean(areal_velocities) * 100

    print("Percentage difference between maximum and minimum areal velocity:", percentage_difference_areal_velocity, "%")

    if check_keplers_second_law(areal_velocities):
        print("Kepler's Second Law is satisfied.")
    else:
        print("Kepler's Second Law is not satisfied.")

    # Kepler's Third Law

    perihelion_indices = find_perihelion_indices(planet_positions, planet_velocities)

    planet_orbital_period = calculate_orbital_period(perihelion_indices, dt)

    if planet_orbital_period is not None:

        print("Measured orbital period:", planet_orbital_period / (60 * 60 * 24), "days")

        if check_keplers_third_law(planet_orbital_period, semi_major_axis, SUN_MASS, planet_mass):
            print("Kepler's Third Law is satisfied.")
        else:
            print("Kepler's Third Law is not satisfied.")

    else:
        print("Not enough perihelion passages to measure the orbital period.")

    # Perihelion and Aphelion Velocities

    perihelion_index = np.argmin(distances)
    aphelion_index = np.argmax(distances)

    perihelion_velocity = np.linalg.norm(planet_velocities[perihelion_index])
    aphelion_velocity = np.linalg.norm(planet_velocities[aphelion_index])

    print("Perihelion velocity:", perihelion_velocity, "m/s")
    print("Aphelion velocity:", aphelion_velocity, "m/s")

    # Vis-viva Equation

    theoretical_perihelion_velocity = calculate_vis_viva_velocity(SUN_MASS, planet_mass, perihelion_distance, semi_major_axis)
    theoretical_aphelion_velocity = calculate_vis_viva_velocity(SUN_MASS, planet_mass, aphelion_distance, semi_major_axis)

    print("Theoretical Perihelion velocity:", theoretical_perihelion_velocity, "m/s")
    print("Theoretical Aphelion velocity:", theoretical_aphelion_velocity, "m/s")

    print("Perihelion velocity difference:", perihelion_velocity - theoretical_perihelion_velocity, "m/s")
    print("Aphelion velocity difference:", aphelion_velocity - theoretical_aphelion_velocity, "m/s")

    if check_vis_viva(perihelion_velocity, aphelion_velocity, theoretical_perihelion_velocity, theoretical_aphelion_velocity):
        print("Calculated perihelion and aphelion velocities agree with the vis-viva equation.")
    else:
        print("Calculated perihelion and aphelion velocities do not agree with the vis-viva equation.")

    # Energy

    kinetic_energies = []
    potential_energies = []
    total_energies = []

    for i in range(len(rk4_positions)):
        kinetic_energy = calculate_total_kinetic_energy(masses, rk4_velocities[i])
        potential_energy = calculate_total_potential_energy(masses, rk4_positions[i])
        total_energy = calculate_total_energy(kinetic_energy, potential_energy)

        kinetic_energies.append(kinetic_energy)
        potential_energies.append(potential_energy)
        total_energies.append(total_energy)

    kinetic_energies = np.array(kinetic_energies)
    potential_energies = np.array(potential_energies)
    total_energies = np.array(total_energies)

    print("Initial kinetic energy:", kinetic_energies[0], "J")
    print("Final kinetic energy:", kinetic_energies[-1], "J")

    print("Initial gravitational potential energy:", potential_energies[0], "J")
    print("Final gravitational potential energy:", potential_energies[-1], "J")

    percentage_difference_total_energy = (np.max(total_energies) - np.min(total_energies)) / abs(np.mean(total_energies)) * 100

    print("Percentage difference between maximum and minimum total energy:", percentage_difference_total_energy, "%")

    if np.isclose(np.max(total_energies), np.min(total_energies), rtol=1e-10):
        print("Total Solar System energy is conserved.")
    else:
        print("Total Solar System energy is not conserved.")

    # Escape Velocity

    escape_velocities = calculate_escape_velocity(SUN_MASS, distances)
    orbital_speeds = np.linalg.norm(planet_velocities, axis=1)

    print("Initial orbital speed:", orbital_speeds[0], "m/s")
    print("Initial escape velocity:", escape_velocities[0], "m/s")

    if np.all(orbital_speeds < escape_velocities):
        print(f"{target_planet} remains below escape velocity.")
    else:
        print(f"{target_planet} exceeds escape velocity.")

    # Orbit Classification

    specific_orbital_energies = calculate_specific_orbital_energy(planet_positions, planet_velocities, SUN_MASS, planet_mass)

    orbit_type = classify_orbit(specific_orbital_energies)

    print("Orbit classification:", orbit_type)

    # Angular Momentum

    angular_momenta = []

    for i in range(len(rk4_positions)):
        angular_momentum = calculate_total_angular_momentum(masses, rk4_positions[i], rk4_velocities[i])
        angular_momenta.append(angular_momentum)

    angular_momenta = np.array(angular_momenta)

    angular_momentum_magnitudes = np.linalg.norm(angular_momenta, axis=1)

    percentage_difference_angular_momentum = (np.max(angular_momentum_magnitudes) - np.min(angular_momentum_magnitudes)) / np.mean(angular_momentum_magnitudes) * 100

    print("Percentage difference between maximum and minimum angular momentum:", percentage_difference_angular_momentum, "%")

    if np.isclose(np.max(angular_momentum_magnitudes), np.min(angular_momentum_magnitudes), rtol=1e-10):
        print("Angular momentum is conserved.")
    else:
        print("Angular momentum is not conserved.")

    # Energy Plot

    plot_energy(kinetic_energies, potential_energies, total_energies, dt, target_planet)

    # 3D Solar System Plot

    plot_solar_system(rk4_positions, plot_every=1000)

if __name__ == "__main__":
    main()