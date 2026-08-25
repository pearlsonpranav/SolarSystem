# Import Libraries

import numpy as np
import matplotlib.pyplot as plt

from initial_conditions import (AU, SUN_MASS, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS, SUN_POSITION, MERCURY_INITIAL_POSITION, VENUS_INITIAL_POSITION, EARTH_INITIAL_POSITION, MARS_INITIAL_POSITION, JUPITER_INITIAL_POSITION, SATURN_INITIAL_POSITION, URANUS_INITIAL_POSITION, NEPTUNE_INITIAL_POSITION, SUN_INITIAL_VELOCITY, MERCURY_INITIAL_VELOCITY, VENUS_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY, MARS_INITIAL_VELOCITY, JUPITER_INITIAL_VELOCITY, SATURN_INITIAL_VELOCITY, URANUS_INITIAL_VELOCITY, NEPTUNE_INITIAL_VELOCITY)
from mechanics import (calculate_distances, calculate_total_kinetic_energy, calculate_total_potential_energy, calculate_total_energy, calculate_escape_velocity, calculate_total_angular_momentum, calculate_centre_of_mass)
from orbits import (calculate_orbital_parameters, calculate_eccentricity_vectors, calculate_areal_velocities, check_keplers_second_law, find_perihelion_indices, calculate_orbital_period, check_keplers_third_law, calculate_vis_viva_velocity, check_vis_viva, calculate_specific_orbital_energy, classify_orbit, calculate_inclination, check_keplers_first_law_trajectory)
from integrators import (simulate_euler, simulate_verlet, simulate_rk4)

# Planet Data (Vertically displayed for readability)

PLANETS = {
    "Mercury": {"index": 1, "mass": MERCURY_MASS, "simulation_time_days": 100, "recommended_dt": 10},
    "Venus": {"index": 2, "mass": VENUS_MASS, "simulation_time_days": 230, "recommended_dt": 10},
    "Earth": {"index": 3, "mass": EARTH_MASS, "simulation_time_days": 370, "recommended_dt": 10},
    "Mars": {"index": 4, "mass": MARS_MASS, "simulation_time_days": 700, "recommended_dt": 100},
    "Jupiter": {"index": 5, "mass": JUPITER_MASS, "simulation_time_days": 4400, "recommended_dt": 1000},
    "Saturn": {"index": 6, "mass": SATURN_MASS, "simulation_time_days": 11000, "recommended_dt": 1000},
    "Uranus": {"index": 7, "mass": URANUS_MASS, "simulation_time_days": 31000, "recommended_dt": 1000},
    "Neptune": {"index": 8, "mass": NEPTUNE_MASS, "simulation_time_days": 61000, "recommended_dt": 1000}
    }

# Solar System Data

MASSES = np.array([SUN_MASS, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS])

INITIAL_POSITIONS = np.array([SUN_POSITION, MERCURY_INITIAL_POSITION, VENUS_INITIAL_POSITION, EARTH_INITIAL_POSITION, MARS_INITIAL_POSITION, JUPITER_INITIAL_POSITION, SATURN_INITIAL_POSITION, URANUS_INITIAL_POSITION, NEPTUNE_INITIAL_POSITION])

INITIAL_VELOCITIES = np.array([SUN_INITIAL_VELOCITY, MERCURY_INITIAL_VELOCITY, VENUS_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY, MARS_INITIAL_VELOCITY, JUPITER_INITIAL_VELOCITY, SATURN_INITIAL_VELOCITY, URANUS_INITIAL_VELOCITY, NEPTUNE_INITIAL_VELOCITY])

# Integrators

INTEGRATORS = {"euler": simulate_euler, "verlet": simulate_verlet, "rk4": simulate_rk4}

# Simulation

def simulate_solar_system(target_planet, dt, integrator):

    if target_planet not in PLANETS:
        raise ValueError(f"Unknown planet: {target_planet}. Choose from {list(PLANETS.keys())}")

    if integrator not in INTEGRATORS:
        raise ValueError(f"Unknown integrator: {integrator}. Choose from {list(INTEGRATORS.keys())}")

    if dt <= 0:
        raise ValueError("Time step must be greater than zero.")

    planet = PLANETS[target_planet]
    simulation_time_days = planet["simulation_time_days"]

    simulation_time = simulation_time_days * 24 * 60 * 60
    number_of_steps = round(simulation_time / dt)

    integrator_function = INTEGRATORS[integrator]

    positions, velocities = integrator_function(MASSES, INITIAL_POSITIONS, INITIAL_VELOCITIES, dt, number_of_steps)

    return positions, velocities, planet["index"], planet["mass"], simulation_time_days

# User Input

def choose_planet():

    print("\nAvailable planets:")

    for planet in PLANETS:
        print("-", planet)

    while True:
        target_planet = input("\nChoose planet for analysis: ").strip().title()

        if target_planet in PLANETS:
            return target_planet

        print("Invalid planet. Please choose from the list above.")

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

def choose_dt(target_planet):

    recommended_dt = PLANETS[target_planet]["recommended_dt"]

    if target_planet in {"Mercury", "Venus", "Earth"}:
        print("\nRecommended time step for inner planets: approximately 10 seconds.")
    elif target_planet == "Mars":
        print("\nRecommended time step for Mars: approximately 100 seconds.")
    else:
        print("\nRecommended time step for outer planets: approximately 1,000 seconds.")

    print("Smaller time steps improve numerical resolution but increase computation time.")
    print("Larger time steps reduce computation time but may reduce numerical accuracy.")

    while True:

        try:
            dt = float(input(f"\nChoose time step (seconds) [recommended: {recommended_dt}]: "))

            if dt > 0:
                return dt

            print("Time step must be greater than zero.")

        except ValueError:
            print("Please enter a valid number.")

# Energy Plot

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

# Core

def main():

    print("Solar System Analysis")

    target_planet = choose_planet()
    integrator = choose_integrator()
    dt = choose_dt(target_planet)

    print("\nRunning analysis simulation...")
    print("Target planet:", target_planet)
    print("Integrator:", integrator.upper())
    print("Time step:", dt, "s")
    print("Simulation time:", PLANETS[target_planet]["simulation_time_days"], "days")

    positions, velocities, planet_index, planet_mass, simulation_time_days = simulate_solar_system(target_planet, dt, integrator)

    print("\nSimulation complete.")
    print("Number of steps:", len(positions) - 1)
    print("Bodies simulated:", len(MASSES))

    # Barycentre

    barycentres = []

    for position in positions:
        barycentres.append(calculate_centre_of_mass(MASSES, position))

    barycentres = np.array(barycentres)

    initial_barycentre = barycentres[0]
    final_barycentre = barycentres[-1]

    barycentre_displacement = np.linalg.norm(final_barycentre - initial_barycentre)

    print("\nInitial barycentre:", initial_barycentre)
    print("Final barycentre:", final_barycentre)
    print("Barycentre displacement:", barycentre_displacement, "m")

    if np.isclose(barycentre_displacement, 0.0, atol=1e-3):
        print("Barycentre is stationary.")
    else:
        print("Barycentre is not stationary.")

    initial_barycentre_velocity = calculate_centre_of_mass(MASSES, velocities[0])
    final_barycentre_velocity = calculate_centre_of_mass(MASSES, velocities[-1])

    print("Initial barycentre velocity:", initial_barycentre_velocity, "m/s")
    print("Final barycentre velocity:", final_barycentre_velocity, "m/s")

    # Target Planet Trajectory Relative to Sun

    planet_positions = positions[:, planet_index, :] - positions[:, 0, :]
    planet_velocities = velocities[:, planet_index, :] - velocities[:, 0, :]

    # Orbital Distances

    distances = calculate_distances(planet_positions)

    perihelion_distance, aphelion_distance, semi_major_axis, _ = calculate_orbital_parameters(distances)

    print("Calculated semi-major axis:", semi_major_axis / AU, "AU")

    # Orbital Inclination

    reference_normal = np.array([0.0, 0.0, 1.0])

    initial_inclination = calculate_inclination(planet_positions[0], planet_velocities[0], reference_normal)
    final_inclination = calculate_inclination(planet_positions[-1], planet_velocities[-1], reference_normal)

    print("Initial orbital inclination:", np.rad2deg(initial_inclination), "degrees")
    print("Final orbital inclination:", np.rad2deg(final_inclination), "degrees")

    # Kepler's First Law

    eccentricity_vectors = calculate_eccentricity_vectors(planet_positions, planet_velocities, SUN_MASS, planet_mass)
    eccentricities_over_time = np.linalg.norm(eccentricity_vectors, axis=1)
    
    initial_eccentricity = eccentricities_over_time[0]
    mean_eccentricity = np.mean(eccentricities_over_time)
    min_eccentricity = np.min(eccentricities_over_time)
    max_eccentricity = np.max(eccentricities_over_time)
    
    print(f"Initial orbital eccentricity: {initial_eccentricity:.8f}")
    print(f"Mean orbital eccentricity:    {mean_eccentricity:.8f} [Min: {min_eccentricity:.11f}, Max: {max_eccentricity:.11f}]")

    first_law_satisfied, max_path_error = check_keplers_first_law_trajectory(planet_positions, planet_velocities, SUN_MASS, planet_mass)
    print(f"Kepler's First Law Trajectory Fit Max Error: {max_path_error:.6f} %")
    
    if first_law_satisfied:
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

    for position, velocity in zip(positions, velocities):
        kinetic_energy = calculate_total_kinetic_energy(MASSES, velocity)
        potential_energy = calculate_total_potential_energy(MASSES, position)
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

    for position, velocity in zip(positions, velocities):
        angular_momentum = calculate_total_angular_momentum(MASSES, position, velocity)
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

if __name__ == "__main__":
    main()