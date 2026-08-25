# Import Libraries

import numpy as np
import matplotlib.pyplot as plt

from initial_conditions import (AU, SUN_MASS, SUN_POSITION, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS, MERCURY_INITIAL_POSITION, VENUS_INITIAL_POSITION, EARTH_INITIAL_POSITION, MARS_INITIAL_POSITION, JUPITER_INITIAL_POSITION, SATURN_INITIAL_POSITION, URANUS_INITIAL_POSITION, NEPTUNE_INITIAL_POSITION, SUN_INITIAL_VELOCITY, MERCURY_INITIAL_VELOCITY, VENUS_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY, MARS_INITIAL_VELOCITY, JUPITER_INITIAL_VELOCITY, SATURN_INITIAL_VELOCITY, URANUS_INITIAL_VELOCITY, NEPTUNE_INITIAL_VELOCITY)
from mechanics import (calculate_total_kinetic_energy, calculate_total_potential_energy, calculate_total_energy, calculate_total_angular_momentum)
from integrators import (simulate_euler, simulate_verlet, simulate_rk4)

# Solar System Data

MASSES = np.array([SUN_MASS, MERCURY_MASS, VENUS_MASS, EARTH_MASS, MARS_MASS, JUPITER_MASS, SATURN_MASS, URANUS_MASS, NEPTUNE_MASS])

INITIAL_POSITIONS = np.array([SUN_POSITION, MERCURY_INITIAL_POSITION, VENUS_INITIAL_POSITION, EARTH_INITIAL_POSITION, MARS_INITIAL_POSITION, JUPITER_INITIAL_POSITION, SATURN_INITIAL_POSITION, URANUS_INITIAL_POSITION, NEPTUNE_INITIAL_POSITION])

INITIAL_VELOCITIES = np.array([SUN_INITIAL_VELOCITY, MERCURY_INITIAL_VELOCITY, VENUS_INITIAL_VELOCITY, EARTH_INITIAL_VELOCITY, MARS_INITIAL_VELOCITY, JUPITER_INITIAL_VELOCITY, SATURN_INITIAL_VELOCITY, URANUS_INITIAL_VELOCITY, NEPTUNE_INITIAL_VELOCITY])

# Integrators

INTEGRATORS = {"euler": simulate_euler, "verlet": simulate_verlet, "rk4": simulate_rk4}

# Functions

def calculate_position_error(positions, reference_positions):
    return np.linalg.norm(positions - reference_positions, axis=2)

def calculate_velocity_error(velocities, reference_velocities):
    return np.linalg.norm(velocities - reference_velocities, axis=2)

def calculate_energy_and_angular_momentum(positions, velocities):
    kinetic_energies = []
    potential_energies = []
    total_energies = []
    angular_momenta = []

    for position, velocity in zip(positions, velocities):

        kinetic_energy = calculate_total_kinetic_energy(MASSES, velocity)
        potential_energy = calculate_total_potential_energy(MASSES, position)
        total_energy = calculate_total_energy(kinetic_energy, potential_energy)
        angular_momentum = calculate_total_angular_momentum(MASSES, position, velocity)

        kinetic_energies.append(kinetic_energy)
        potential_energies.append(potential_energy)
        total_energies.append(total_energy)
        angular_momenta.append(angular_momentum)

    return np.array(kinetic_energies), np.array(potential_energies), np.array(total_energies), np.array(angular_momenta)

def calculate_percentage_variation(values):
    return (np.max(values) - np.min(values)) / abs(np.mean(values)) * 100

# Core

def compare_integrators():

    simulation_time = 370 * 24 * 60 * 60
    dt_values = [10.0, 50.0, 100.0, 500.0, 1000.0]

    results = {}

    print("Running integrator comparison...")
    print()

    # Reference Solution

    reference_dt = 10.0
    reference_steps = round(simulation_time / reference_dt)

    print("Running RK4 reference solution...")
    reference_positions, reference_velocities = simulate_rk4(MASSES, INITIAL_POSITIONS, INITIAL_VELOCITIES, reference_dt, reference_steps)

    # Compare Integrators

    for dt in dt_values:

        print(f"Testing dt = {dt} s...")

        number_of_steps = round(simulation_time / dt)

        euler_positions, euler_velocities = simulate_euler(MASSES, INITIAL_POSITIONS, INITIAL_VELOCITIES, dt, number_of_steps)
        verlet_positions, verlet_velocities = simulate_verlet(MASSES, INITIAL_POSITIONS, INITIAL_VELOCITIES, dt, number_of_steps)
        rk4_positions, rk4_velocities = simulate_rk4(MASSES, INITIAL_POSITIONS, INITIAL_VELOCITIES, dt, number_of_steps)

        results[dt] = {"euler": (euler_positions, euler_velocities), "verlet": (verlet_positions, verlet_velocities), "rk4": (rk4_positions, rk4_velocities)}

    # Error Analysis

    position_errors = {}
    velocity_errors = {}

    for dt in dt_values:

        euler_positions, euler_velocities = results[dt]["euler"]
        verlet_positions, verlet_velocities = results[dt]["verlet"]
        rk4_positions, rk4_velocities = results[dt]["rk4"]

        sampling_interval = round(dt / reference_dt)

        reference_positions_sampled = reference_positions[::sampling_interval]
        reference_velocities_sampled = reference_velocities[::sampling_interval]

        number_of_points = min(len(reference_positions_sampled), len(euler_positions), len(verlet_positions), len(rk4_positions))

        reference_positions_sampled = reference_positions_sampled[:number_of_points]
        reference_velocities_sampled = reference_velocities_sampled[:number_of_points]

        euler_positions = euler_positions[:number_of_points]
        euler_velocities = euler_velocities[:number_of_points]

        verlet_positions = verlet_positions[:number_of_points]
        verlet_velocities = verlet_velocities[:number_of_points]

        rk4_positions = rk4_positions[:number_of_points]
        rk4_velocities = rk4_velocities[:number_of_points]

        position_errors[dt] = {"euler": calculate_position_error(euler_positions, reference_positions_sampled), "verlet": calculate_position_error(verlet_positions, reference_positions_sampled), "rk4": calculate_position_error(rk4_positions, reference_positions_sampled)}

        velocity_errors[dt] = {"euler": calculate_velocity_error(euler_velocities, reference_velocities_sampled), "verlet": calculate_velocity_error(verlet_velocities, reference_velocities_sampled), "rk4": calculate_velocity_error(rk4_velocities, reference_velocities_sampled)}

    # Print Error Results

    print()
    print("Maximum Position Errors")
    print("-----------------------")

    for dt in dt_values:

        print(f"dt = {dt} s")
        print("Euler:", np.max(position_errors[dt]["euler"]), "m")
        print("Verlet:", np.max(position_errors[dt]["verlet"]), "m")
        print("RK4:", np.max(position_errors[dt]["rk4"]), "m")
        print()

    print("Final Position Errors")
    print("---------------------")

    for dt in dt_values:

        print(f"dt = {dt} s")
        print("Euler:", position_errors[dt]["euler"][-1, 3], "m")
        print("Verlet:", position_errors[dt]["verlet"][-1, 3], "m")
        print("RK4:", position_errors[dt]["rk4"][-1, 3], "m")
        print()

    print("Maximum Velocity Errors")
    print("-----------------------")

    for dt in dt_values:

        print(f"dt = {dt} s")
        print("Euler:", np.max(velocity_errors[dt]["euler"]), "m/s")
        print("Verlet:", np.max(velocity_errors[dt]["verlet"]), "m/s")
        print("RK4:", np.max(velocity_errors[dt]["rk4"]), "m/s")
        print()

    # Stability Analysis

    stability_results = {}

    for dt in dt_values:

        euler_positions, euler_velocities = results[dt]["euler"]
        verlet_positions, verlet_velocities = results[dt]["verlet"]
        rk4_positions, rk4_velocities = results[dt]["rk4"]

        euler_energy = calculate_energy_and_angular_momentum(euler_positions, euler_velocities)
        verlet_energy = calculate_energy_and_angular_momentum(verlet_positions, verlet_velocities)
        rk4_energy = calculate_energy_and_angular_momentum(rk4_positions, rk4_velocities)

        stability_results[dt] = {"euler": euler_energy, "verlet": verlet_energy, "rk4": rk4_energy}

    print("Energy and Angular Momentum Variation")
    print("-------------------------------------")

    for dt in dt_values:

        print(f"dt = {dt} s")

        for integrator in ["euler", "verlet", "rk4"]:

            total_energy = stability_results[dt][integrator][2]
            angular_momentum = stability_results[dt][integrator][3]

            angular_momentum_magnitudes = np.linalg.norm(angular_momentum, axis=1)

            energy_variation = calculate_percentage_variation(total_energy)
            angular_momentum_variation = calculate_percentage_variation(angular_momentum_magnitudes)

            print(integrator.capitalize(), "energy variation:", energy_variation, "%")
            print(integrator.capitalize(), "angular momentum variation:", angular_momentum_variation, "%")

        print()

    # Earth Position Error Plot

    earth_index = 3

    plt.figure(figsize=(10, 6))

    for dt in dt_values:

        time = np.arange(len(position_errors[dt]["euler"])) * dt / (60 * 60 * 24)

        plt.plot(time, position_errors[dt]["euler"][:, earth_index] / AU, label=f"Euler dt={dt}s")
        plt.plot(time, position_errors[dt]["verlet"][:, earth_index] / AU, label=f"Verlet dt={dt}s")
        plt.plot(time, position_errors[dt]["rk4"][:, earth_index] / AU, label=f"RK4 dt={dt}s")

    plt.xlabel("Time (days)")
    plt.ylabel("Earth Position Error (AU)")
    plt.title("Integrator Earth Position Error")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.show()

    # Earth Velocity Error Plot

    plt.figure(figsize=(10, 6))

    for dt in dt_values:

        time = np.arange(len(velocity_errors[dt]["euler"])) * dt / (60 * 60 * 24)

        plt.plot(time, velocity_errors[dt]["euler"][:, earth_index], label=f"Euler dt={dt}s")
        plt.plot(time, velocity_errors[dt]["verlet"][:, earth_index], label=f"Verlet dt={dt}s")
        plt.plot(time, velocity_errors[dt]["rk4"][:, earth_index], label=f"RK4 dt={dt}s")

    plt.xlabel("Time (days)")
    plt.ylabel("Earth Velocity Error (m/s)")
    plt.title("Integrator Earth Velocity Error")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.show()

if __name__ == "__main__":
    compare_integrators()