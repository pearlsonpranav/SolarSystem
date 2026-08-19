# Import Libraries

import numpy as np
import matplotlib.pyplot as plt

from dynamics import (G, calculate_distance, calculate_gravitational_force, calculate_gravitational_force_vector, calculate_acceleration)
from integrators import simulate_euler, simulate_verlet

# Constants

AU = 1.496e11
sun_mass = 1.989e30
earth_mass = 5.972e24

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

    earth_velocity = np.array([0.0, earth_orbital_speed])
    
    # Simulation

    dt = 10.0 # Time step in seconds - the smaller the time step, the more accurate the simulation, but it will take longer to run.
    simulation_time = 1000 * 24 * 60 * 60 # Total simulation time in seconds (1000 days) - the longer the simulation time, the more accurate the simulation, but it will take longer to run.
    number_of_steps = round(simulation_time / dt)

    verlet_x, verlet_y, verlet_vx, verlet_vy = simulate_verlet(sun_position, earth_position, sun_mass, earth_mass, earth_velocity, dt, number_of_steps)

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

    percentage_difference = (np.max(areas) - np.min(areas)) / np.mean(areas) * 100
    print("Percentage difference between maximum and minimum areas:", percentage_difference, "%")

    if np.isclose(np.min(areas), np.max(areas), rtol=1e-5):
        print("Kepler's Second Law is satisfied.")
    else:
        print("Kepler's Second Law is not satisfied.")

    # Kepler's Third Law

    earth_speed = np.sqrt(verlet_vx ** 2 + verlet_vy ** 2)

    perihelion_indices = []

    for n in range(1, len(earth_speed) - 1):
        if earth_speed[n] > earth_speed[n - 1] and earth_speed[n] > earth_speed[n + 1]:
            perihelion_indices.append(n)

    earth_orbital_period = (perihelion_indices[1] - perihelion_indices[0]) * dt

    print("Measured orbital period:", earth_orbital_period / (60 * 60 * 24), "days")

    KL3_proportionality_constant = earth_orbital_period ** 2 / (calculated_semi_major_axis ** 3)

    if np.isclose(KL3_proportionality_constant, 4 * np.pi ** 2 / (G * sun_mass), rtol=1e-7):
        print("Kepler's Third Law is satisfied.")
    else:
        print("Kepler's Third Law is not satisfied.")

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