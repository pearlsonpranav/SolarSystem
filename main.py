# Import Libraries

import numpy as np
import matplotlib.pyplot as plt

from initial_conditions import (G, AU, SUN_MASS, EARTH_MASS, SUN_POSITION, EARTH_INITIAL_POSITION, SUN_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY)
from mechanics import (calculate_distances_2D, calculate_kinetic_energy, calculate_total_kinetic_energy, calculate_potential_energy, calculate_total_potential_energy, calculate_total_energy, calculate_escape_velocity, calculate_angular_momentum_2D, calculate_total_angular_momentum_2D, calculate_centre_of_mass)
from integrators import simulate_euler, simulate_verlet, simulate_rk4
from kepler import (calculate_orbital_parameters_2D, calculate_ellipse_centre_2D, check_keplers_first_law_2D, calculate_areal_velocities_2D, check_keplers_second_law, find_perihelion_indices_2D, calculate_orbital_period, check_keplers_third_law, calculate_vis_viva_velocity, check_vis_viva, classify_orbit)

# Functions

def plot_energy(kinetic_energies, potential_energies, total_energies, dt):
    time = np.arange(len(kinetic_energies)) * dt / (60 * 60 * 24)

    plt.figure(figsize=(10, 6))

    plt.plot(time, kinetic_energies, label="Kinetic Energy")
    plt.plot(time, potential_energies, label="Potential Energy")
    plt.plot(time, total_energies, label="Total Energy")

    plt.xlabel("Time (days)")
    plt.ylabel("Energy (J)")
    plt.title("Energy Variation During Earth's Orbit")
    plt.legend(loc="upper right")
    plt.grid()

    plt.show()

# Core

def main():
    print("Solar System Simulation")

    # Simulation

    dt = 10.0
    simulation_time = 370 * 24 * 60 * 60
    number_of_steps = round(simulation_time / dt)

    masses = np.array([SUN_MASS, EARTH_MASS])
    positions = np.array([SUN_POSITION, EARTH_INITIAL_POSITION])
    velocities = np.array([SUN_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY])

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
    if np.isclose(barycentre_displacement, 0.0, atol = 1e-5):
        print("Barycentre is stationary.")
    else:
        print("Barycentre is not stationary.")

    initial_barycentre_velocity = calculate_centre_of_mass(masses, velocities)
    final_barycentre_velocity = calculate_centre_of_mass(masses, rk4_velocities[-1])

    print("Initial barycentre velocity:", initial_barycentre_velocity, "m/s")
    print("Final barycentre velocity:", final_barycentre_velocity, "m/s")

    # Earth Trajectory
    
    earth_positions = rk4_positions[:, 1, :] - rk4_positions[:, 0, :]
    earth_velocities = rk4_velocities[:, 1, :] - rk4_velocities[:, 0, :]

    rk4_x = earth_positions[:, 0]
    rk4_y = earth_positions[:, 1]
    rk4_vx = earth_velocities[:, 0]
    rk4_vy = earth_velocities[:, 1]

    # Orbital Distances

    distances = calculate_distances_2D(rk4_x, rk4_y)

    # Kepler's First Law

    perihelion_distance, aphelion_distance, semi_major_axis, eccentricity = calculate_orbital_parameters_2D(distances)

    print("Calculated semi-major axis:", semi_major_axis / AU, "AU")
    print("Calculated eccentricity:", eccentricity)

    centre = calculate_ellipse_centre_2D(rk4_x, rk4_y)

    if check_keplers_first_law_2D(centre, semi_major_axis, eccentricity):
        print("Kepler's First Law is satisfied.")
    else:
        print("Kepler's First Law is not satisfied.")

    # Kepler's Second Law

    areal_velocities = calculate_areal_velocities_2D(rk4_x, rk4_y, rk4_vx, rk4_vy)

    percentage_difference_areal_velocity = (np.max(areal_velocities) - np.min(areal_velocities)) / np.mean(areal_velocities) * 100

    print("Percentage difference between maximum and minimum areal velocity:", percentage_difference_areal_velocity, "%")

    if check_keplers_second_law(areal_velocities):
        print("Kepler's Second Law is satisfied.")
    else:
        print("Kepler's Second Law is not satisfied.")

    # Kepler's Third Law

    perihelion_indices = find_perihelion_indices_2D(rk4_x, rk4_y, rk4_vx, rk4_vy)

    earth_orbital_period = calculate_orbital_period(perihelion_indices, dt)

    print("Measured orbital period:", earth_orbital_period / (60 * 60 * 24), "days")

    if check_keplers_third_law(earth_orbital_period, semi_major_axis, SUN_MASS, EARTH_MASS):
        print("Kepler's Third Law is satisfied.")
    else:
        print("Kepler's Third Law is not satisfied.")

    # Perihelion and Aphelion Velocities

    perihelion_index = np.argmin(distances)
    aphelion_index = np.argmax(distances)

    perihelion_velocity = np.sqrt(rk4_vx[perihelion_index] ** 2 + rk4_vy[perihelion_index] ** 2)
    aphelion_velocity = np.sqrt(rk4_vx[aphelion_index] ** 2 + rk4_vy[aphelion_index] ** 2)

    print("Perihelion velocity:", perihelion_velocity, "m/s")
    print("Aphelion velocity:", aphelion_velocity, "m/s")

    # Vis-viva Equation

    theoretical_perihelion_velocity = calculate_vis_viva_velocity(SUN_MASS, EARTH_MASS, perihelion_distance, semi_major_axis)
    theoretical_aphelion_velocity = calculate_vis_viva_velocity(SUN_MASS, EARTH_MASS, aphelion_distance, semi_major_axis)

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

    for i in range(len(rk4_x)):
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
        print("Total orbital energy is conserved.")
    else:
        print("Total orbital energy is not conserved.")

    # Escape Velocity

    escape_velocities = calculate_escape_velocity(SUN_MASS, distances)
    orbital_speeds = np.sqrt(rk4_vx ** 2 + rk4_vy ** 2)

    print("Initial orbital speed:", orbital_speeds[0], "m/s")
    print("Initial escape velocity:", escape_velocities[0], "m/s")

    if np.all(orbital_speeds < escape_velocities):
        print("Earth remains below escape velocity.")
    else:
        print("Earth exceeds escape velocity.")

    # Orbit Classification

    orbit_type = classify_orbit(total_energies)

    print("Orbit classification:", orbit_type)

    # Angular Momentum

    angular_momenta = []

    for i in range(len(rk4_x)):
        angular_momentum = calculate_total_angular_momentum_2D(masses, rk4_positions[i], rk4_velocities[i])
        angular_momenta.append(angular_momentum)

    angular_momenta = np.array(angular_momenta)

    percentage_difference_angular_momentum = ((np.max(angular_momenta) - np.min(angular_momenta)) / abs(np.mean(angular_momenta))) * 100

    print("Percentage difference between maximum and minimum angular momentum:", percentage_difference_angular_momentum, "%")

    if np.isclose(np.max(angular_momenta), np.min(angular_momenta), rtol=1e-10):
        print("Angular momentum is conserved.")
    else:
        print("Angular momentum is not conserved.")

    # Energy Plot

    plot_energy(kinetic_energies, potential_energies, total_energies, dt)

    # Orbit Plot

    rk4_x_AU = rk4_x / AU
    rk4_y_AU = rk4_y / AU

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(rk4_x_AU, rk4_y_AU, color="blue", label="Earth")
    ax.scatter([0], [0], s=200, color="orange", label="Sun")

    ax.set_xlabel("x position (AU)")
    ax.set_ylabel("y position (AU)")
    ax.set_title("Earth's Orbit")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    plt.show()

if __name__ == "__main__":
    main()