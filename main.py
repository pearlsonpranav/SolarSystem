# Import Libraries

import numpy as np
import matplotlib.pyplot as plt

from dynamics import (G, calculate_distance, calculate_gravitational_force, calculate_gravitational_force_vector, calculate_acceleration)
from integrators import simulate_verlet

# Constants

AU = 1.496e11
sun_mass = 1.989e30
earth_mass = 5.972e24

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

    # Initial Orbit

    earth_eccentricity = 0.0167
    semi_major_axis = AU

    perihelion_distance = semi_major_axis * (1 - earth_eccentricity)

    sun_position = np.array([0.0, 0.0])
    earth_position = np.array([perihelion_distance, 0.0])

    earth_orbital_speed = np.sqrt(G * sun_mass * (2 / perihelion_distance - 1 / semi_major_axis))

    earth_orbital_velocity = np.array([0.0, earth_orbital_speed])
    
    # Simulation

    dt = 10.0 # Time step in seconds - the smaller the time step, the more accurate the simulation, but it will take longer to run.
    simulation_time = 1000 * 24 * 60 * 60 # Total simulation time in seconds (1000 days) - the longer the simulation time, the more accurate the simulation, but it will take longer to run.
    number_of_steps = round(simulation_time / dt)

    verlet_x, verlet_y, verlet_vx, verlet_vy = simulate_verlet(sun_position, earth_position, sun_mass, earth_mass, earth_orbital_velocity, dt, number_of_steps)

    # Kepler's First Law

    distances = np.sqrt(verlet_x ** 2 + verlet_y ** 2)

    perihelion_distance = np.min(distances)
    aphelion_distance = np.max(distances)

    calculated_semi_major_axis = (perihelion_distance + aphelion_distance) / 2
    calculated_eccentricity = (aphelion_distance - perihelion_distance) / (aphelion_distance + perihelion_distance)

    print("Calculated semi-major axis:", calculated_semi_major_axis / AU, "AU")
    print("Calculated eccentricity:", calculated_eccentricity)

    Centre_of_ellipse = np.array([(np.max(verlet_x) + np.min(verlet_x)) / 2, (np.max(verlet_y) + np.min(verlet_y)) / 2])

    if np.isclose(calculate_distance(Centre_of_ellipse, sun_position), calculated_semi_major_axis * calculated_eccentricity, rtol=1e-5) and np.isclose(Centre_of_ellipse[1], 0.0, atol = 1e3):
        print("Kepler's First Law is satisfied.")
    else:
        print("Kepler's First Law is not satisfied.")

    # Kepler's Second Law

    def area_swept(verlet_x, verlet_y):
        areas = []

        for n in range(len(verlet_x) - 1):

            area = 0.5 * np.abs(verlet_x[n] * verlet_y[n + 1] - verlet_y[n] * verlet_x[n + 1])
            areas.append(area)

        return np.array(areas)
    
    areas = area_swept(verlet_x, verlet_y)

    percentage_difference_areas = (np.max(areas) - np.min(areas)) / np.mean(areas) * 100
    print("Percentage difference between maximum and minimum areas:", percentage_difference_areas, "%")

    if np.isclose(np.min(areas), np.max(areas), rtol=1e-5):
        print("Kepler's Second Law is satisfied.")
    else:
        print("Kepler's Second Law is not satisfied.")

    # Kepler's Third Law

    perihelion_indices = []

    for n in range(1, len(verlet_x)):
        previous_position = np.array([verlet_x[n - 1], verlet_y[n - 1]])
        previous_velocity = np.array([verlet_vx[n - 1], verlet_vy[n - 1]])

        current_position = np.array([verlet_x[n], verlet_y[n]])
        current_velocity = np.array([verlet_vx[n], verlet_vy[n]])

        previous_radial_motion = np.dot(previous_position, previous_velocity)
        current_radial_motion = np.dot(current_position, current_velocity)

        if previous_radial_motion < 0 and current_radial_motion > 0:
            perihelion_indices.append(n)

    earth_orbital_period = (perihelion_indices[1] - perihelion_indices[0]) * dt

    print("Measured orbital period:", earth_orbital_period / (60 * 60 * 24), "days")

    KL3_proportionality_constant = earth_orbital_period ** 2 / (calculated_semi_major_axis ** 3)

    if np.isclose(KL3_proportionality_constant, 4 * np.pi ** 2 / (G * sun_mass), rtol=1e-7):
        print("Kepler's Third Law is satisfied.")
    else:
        print("Kepler's Third Law is not satisfied.")

    # Kinetic Energy

    kinetic_energies = []

    for i in range(len(verlet_vx)):
        kinetic_energies.append(0.5 * earth_mass * (verlet_vx[i] ** 2 + verlet_vy[i] ** 2))

    print("Initial kinetic energy:", kinetic_energies[0], "J")
    print("Final kinetic energy:", kinetic_energies[-1], "J")

    # Gravitational Potential Energy

    potential_energies = []

    for i in range(len(verlet_x)):
        distance = np.sqrt(verlet_x[i] ** 2 + verlet_y[i] ** 2)
        potential_energy = -G * sun_mass * earth_mass / distance
        potential_energies.append(potential_energy)

    print("Initial gravitational potential energy:", potential_energies[0], "J")
    print("Final gravitational potential energy:", potential_energies[-1], "J")

    # Total Orbital Energy (Checking Conservation of Energy)

    total_energies = []
    
    for i in range(len(kinetic_energies)):
        total_energies.append(kinetic_energies[i] + potential_energies[i])

    percentage_difference_total_energy = (np.max(total_energies) - np.min(total_energies)) / abs(np.mean(total_energies)) * 100
    print("Percentage difference between maximum and minimum total energies:", percentage_difference_total_energy, "%")

    if np.isclose(np.max(total_energies), np.min(total_energies), rtol=1e-10):
        print("Total orbital energy is conserved.")
    else:
        print("Total orbital energy is not conserved.")

    # Theoretical Orbital Energy (Verifying the Total Energy of the Orbit)

    theoretical_total_energy = -G * sun_mass * earth_mass / (2 * semi_major_axis)

    print("Theoretical total energy:", theoretical_total_energy, "J")
    print("Simulated mean total energy:", np.mean(total_energies), "J")

    percentage_difference_energy = (abs(theoretical_total_energy - np.mean(total_energies)) / abs(theoretical_total_energy)) * 100

    print("Percentage difference between simulated and theoretical energy:", percentage_difference_energy, "%")

    # Energy Variation Plot

    plot_energy(kinetic_energies, potential_energies, total_energies, dt)

    # Escape Velocity

    escape_velocities = []

    for i in range(len(verlet_x)):
        distance = np.sqrt(verlet_x[i] ** 2 + verlet_y[i] ** 2)
        escape_velocity = np.sqrt(2 * G * sun_mass / distance)
        escape_velocities.append(escape_velocity)

    calculated_earth_orbital_speed = np.sqrt(verlet_vx ** 2 + verlet_vy ** 2)

    print("Initial orbital speed:", calculated_earth_orbital_speed[0], "m/s")
    print("Initial escape velocity:", escape_velocities[0], "m/s")

    if np.all(calculated_earth_orbital_speed < escape_velocities):
        print("Earth remains below escape velocity.")
    else:
        print("Earth exceeds escape velocity.")

    # Orbit Classification

    if np.all(np.array(total_energies) < 0):
        print("Orbit is bound.")
    elif np.all(np.isclose(total_energies, 0)):
        print("Orbit is at the escape boundary.")
    else:
        print("Orbit is unbound.")

    # Convert metres to astronomical units for plotting

    verlet_x_AU = verlet_x / AU
    verlet_y_AU = verlet_y / AU

    # Plot

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(verlet_x_AU, verlet_y_AU, color="blue", label="Earth")
    ax.scatter([0], [0], s=200, color="orange", label="Sun")

    ax.set_xlabel("x position (AU)")
    ax.set_ylabel("y position (AU)")
    ax.set_title("Earth's Orbit")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    plt.show()

if __name__ == "__main__":
    main()