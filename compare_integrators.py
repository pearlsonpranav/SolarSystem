# Import Libraries

import numpy as np
import matplotlib.pyplot as plt

from initial_conditions import (AU, SUN_MASS, EARTH_MASS, SUN_POSITION, EARTH_INITIAL_POSITION, EARTH_INITIAL_VELOCITY)
from mechanics import (calculate_kinetic_energy, calculate_potential_energy, calculate_total_energy, calculate_angular_momentum_2D)
from integrators import (simulate_euler, simulate_verlet, simulate_rk4)

# Functions

def calculate_position_error(x_positions, y_positions, reference_x, reference_y):
    return np.sqrt((x_positions - reference_x) ** 2 + (y_positions - reference_y) ** 2)

def calculate_velocity_error(vx, vy, reference_vx, reference_vy):
    return np.sqrt((vx - reference_vx) ** 2 + (vy - reference_vy) ** 2)

def calculate_energy_and_angular_momentum(x_positions, y_positions, x_velocities, y_velocities):
    kinetic_energies = []
    potential_energies = []
    total_energies = []
    angular_momenta = []

    for i in range(len(x_positions)):
        position = np.array([x_positions[i], y_positions[i]])
        velocity = np.array([x_velocities[i], y_velocities[i]])

        distance = np.sqrt(x_positions[i] ** 2 + y_positions[i] ** 2)

        kinetic_energy = calculate_kinetic_energy(EARTH_MASS, velocity)
        potential_energy = calculate_potential_energy(SUN_MASS, EARTH_MASS, distance)
        total_energy = calculate_total_energy(kinetic_energy, potential_energy)
        angular_momentum = calculate_angular_momentum_2D(EARTH_MASS, position, velocity)

        kinetic_energies.append(kinetic_energy)
        potential_energies.append(potential_energy)
        total_energies.append(total_energy)
        angular_momenta.append(angular_momentum)

    return np.array(kinetic_energies), np.array(potential_energies), np.array(total_energies), np.array(angular_momenta)

def calculate_percentage_variation(values):
    return (np.max(values) - np.min(values)) / abs(np.mean(values)) * 100

# Core

def compare_integrators():
    simulation_time = 750 * 24 * 60 * 60
    dt_values = [10.0, 50.0, 100.0, 500.0, 1000.0]

    results = {}

    print("Running integrator comparison...")
    print()

    # Reference solution (RK4 with dt = 10 s is used as the reference solution because it showed negligible numerical error in main.py validation)

    reference_dt = 10.0
    reference_steps = round(simulation_time / reference_dt)

    print("Running RK4 reference solution...")
    reference_x, reference_y, reference_vx, reference_vy = simulate_rk4(SUN_POSITION, EARTH_INITIAL_POSITION, SUN_MASS, EARTH_MASS, EARTH_INITIAL_VELOCITY, reference_dt, reference_steps)

    # Compare integrators

    for dt in dt_values:
        print(f"Testing dt = {dt} s...")

        number_of_steps = round(simulation_time / dt)

        euler_x, euler_y, euler_vx, euler_vy = simulate_euler(SUN_POSITION, EARTH_INITIAL_POSITION, SUN_MASS, EARTH_MASS, EARTH_INITIAL_VELOCITY, dt, number_of_steps)
        verlet_x, verlet_y, verlet_vx, verlet_vy = simulate_verlet(SUN_POSITION, EARTH_INITIAL_POSITION, SUN_MASS, EARTH_MASS, EARTH_INITIAL_VELOCITY, dt, number_of_steps)
        rk4_x, rk4_y, rk4_vx, rk4_vy = simulate_rk4(SUN_POSITION, EARTH_INITIAL_POSITION, SUN_MASS, EARTH_MASS, EARTH_INITIAL_VELOCITY, dt, number_of_steps)

        results[dt] = {"euler": (euler_x, euler_y, euler_vx, euler_vy), "verlet": (verlet_x, verlet_y, verlet_vx, verlet_vy), "rk4": (rk4_x, rk4_y, rk4_vx, rk4_vy)}

    # Error Analysis

    position_errors = {}
    velocity_errors = {}

    for dt in dt_values:
        euler_x, euler_y, euler_vx, euler_vy = results[dt]["euler"]
        verlet_x, verlet_y, verlet_vx, verlet_vy = results[dt]["verlet"]
        rk4_x, rk4_y, rk4_vx, rk4_vy = results[dt]["rk4"]

        reference_indices = np.arange(0, len(reference_x), round(dt / reference_dt))

        reference_x_sampled = reference_x[reference_indices]
        reference_y_sampled = reference_y[reference_indices]
        reference_vx_sampled = reference_vx[reference_indices]
        reference_vy_sampled = reference_vy[reference_indices]

        number_of_points = min(len(reference_x_sampled), len(euler_x), len(verlet_x), len(rk4_x))

        reference_x_sampled = reference_x_sampled[:number_of_points]
        reference_y_sampled = reference_y_sampled[:number_of_points]
        reference_vx_sampled = reference_vx_sampled[:number_of_points]
        reference_vy_sampled = reference_vy_sampled[:number_of_points]

        euler_x = euler_x[:number_of_points]
        euler_y = euler_y[:number_of_points]
        euler_vx = euler_vx[:number_of_points]
        euler_vy = euler_vy[:number_of_points]

        verlet_x = verlet_x[:number_of_points]
        verlet_y = verlet_y[:number_of_points]
        verlet_vx = verlet_vx[:number_of_points]
        verlet_vy = verlet_vy[:number_of_points]

        rk4_x = rk4_x[:number_of_points]
        rk4_y = rk4_y[:number_of_points]
        rk4_vx = rk4_vx[:number_of_points]
        rk4_vy = rk4_vy[:number_of_points]

        position_errors[dt] = {"euler": calculate_position_error(euler_x, euler_y, reference_x_sampled, reference_y_sampled), "verlet": calculate_position_error(verlet_x, verlet_y, reference_x_sampled, reference_y_sampled), "rk4": calculate_position_error(rk4_x, rk4_y, reference_x_sampled, reference_y_sampled)}

        velocity_errors[dt] = {"euler": calculate_velocity_error(euler_vx, euler_vy, reference_vx_sampled, reference_vy_sampled), "verlet": calculate_velocity_error(verlet_vx, verlet_vy, reference_vx_sampled, reference_vy_sampled), "rk4": calculate_velocity_error(rk4_vx, rk4_vy, reference_vx_sampled, reference_vy_sampled)}

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
        print("Euler:", position_errors[dt]["euler"][-1], "m")
        print("Verlet:", position_errors[dt]["verlet"][-1], "m")
        print("RK4:", position_errors[dt]["rk4"][-1], "m")
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
        euler_x, euler_y, euler_vx, euler_vy = results[dt]["euler"]
        verlet_x, verlet_y, verlet_vx, verlet_vy = results[dt]["verlet"]
        rk4_x, rk4_y, rk4_vx, rk4_vy = results[dt]["rk4"]

        euler_energy = calculate_energy_and_angular_momentum(euler_x, euler_y, euler_vx, euler_vy)
        verlet_energy = calculate_energy_and_angular_momentum(verlet_x, verlet_y, verlet_vx, verlet_vy)
        rk4_energy = calculate_energy_and_angular_momentum(rk4_x, rk4_y, rk4_vx, rk4_vy)

        stability_results[dt] = {"euler": euler_energy, "verlet": verlet_energy, "rk4": rk4_energy}

    print("Energy and Angular Momentum Variation")
    print("-------------------------------------")

    for dt in dt_values:
        print(f"dt = {dt} s")

        for integrator in ["euler", "verlet", "rk4"]:
            total_energy = stability_results[dt][integrator][2]
            angular_momentum = stability_results[dt][integrator][3]

            energy_variation = calculate_percentage_variation(total_energy)
            angular_momentum_variation = calculate_percentage_variation(angular_momentum)

            print(integrator.capitalize(), "energy variation:", energy_variation, "%")
            print(integrator.capitalize(), "angular momentum variation:", angular_momentum_variation, "%")

        print()

    # Position Error Plot

    plt.figure(figsize=(10, 6))

    for dt in dt_values:
        time = np.arange(len(position_errors[dt]["euler"])) * dt / (60 * 60 * 24)

        plt.plot(time, position_errors[dt]["euler"] / AU, label=f"Euler dt={dt}s")
        plt.plot(time, position_errors[dt]["verlet"] / AU, label=f"Verlet dt={dt}s")
        plt.plot(time, position_errors[dt]["rk4"] / AU, label=f"RK4 dt={dt}s")

    plt.xlabel("Time (days)")
    plt.ylabel("Position Error (AU)")
    plt.title("Integrator Position Error")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.show()

    # Velocity Error Plot

    plt.figure(figsize=(10, 6))

    for dt in dt_values:
        time = np.arange(len(velocity_errors[dt]["euler"])) * dt / (60 * 60 * 24)

        plt.plot(time, velocity_errors[dt]["euler"], label=f"Euler dt={dt}s")
        plt.plot(time, velocity_errors[dt]["verlet"], label=f"Verlet dt={dt}s")
        plt.plot(time, velocity_errors[dt]["rk4"], label=f"RK4 dt={dt}s")

    plt.xlabel("Time (days)")
    plt.ylabel("Velocity Error (m/s)")
    plt.title("Integrator Velocity Error")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.show()

if __name__ == "__main__":
    compare_integrators()