# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants

G = 6.67430e-11
AU = 1.496e11
sun_mass = 1.989e30
earth_mass = 5.972e24

# Functions

def calculate_distance(position1, position2):
    displacement = position1 - position2
    distance = np.linalg.norm(displacement)
    return distance


def calculate_gravitational_force(mass1, mass2, distance):
    force_scalar = G * (mass1 * mass2) / (distance ** 2)
    return force_scalar


def calculate_gravitational_force_vector(mass1, mass2, position1, position2):
    displacement = position1 - position2
    distance = np.linalg.norm(displacement)
    force_magnitude = G * (mass1 * mass2) / (distance ** 2)
    force_direction = displacement / distance
    force_vector = force_magnitude * force_direction
    return force_vector


def acceleration(force, mass):
    acceleration = force / mass
    return acceleration


def update_velocity(velocity, acceleration, dt):
    new_velocity = velocity + acceleration * dt
    return new_velocity


def update_position(position, velocity, dt):
   new_position = position + velocity * dt
   return new_position

def update_position_verlet(position, velocity, acceleration, dt):
    new_position = position + velocity * dt + 0.5 * acceleration * dt ** 2
    return new_position


def update_velocity_verlet(velocity, acceleration, new_acceleration, dt):
    new_velocity = velocity + 0.5 * (acceleration + new_acceleration) * dt
    return new_velocity

def simulate_euler(sun_position, earth_position, earth_velocity, dt, number_of_steps):
    x_positions = []
    y_positions = []

    for step in range(number_of_steps):
        force_vector = calculate_gravitational_force_vector(sun_mass, earth_mass, sun_position, earth_position)
        earth_acceleration = acceleration(force_vector, earth_mass)

        earth_velocity = update_velocity(earth_velocity, earth_acceleration, dt)
        earth_position = update_position(earth_position, earth_velocity, dt)

        x_positions.append(earth_position[0])
        y_positions.append(earth_position[1])

    return np.array(x_positions), np.array(y_positions)


def simulate_verlet(sun_position, earth_position, earth_velocity, dt, number_of_steps):
    x_positions = []
    y_positions = []

    force_vector = calculate_gravitational_force_vector(sun_mass, earth_mass, sun_position, earth_position)
    earth_acceleration = acceleration(force_vector, earth_mass)

    for step in range(number_of_steps):
        earth_position = update_position_verlet(earth_position, earth_velocity, earth_acceleration, dt)

        x_positions.append(earth_position[0])
        y_positions.append(earth_position[1])

        force_vector = calculate_gravitational_force_vector(sun_mass, earth_mass, sun_position, earth_position)
        new_acceleration = acceleration(force_vector, earth_mass)

        earth_velocity = update_velocity_verlet(earth_velocity, earth_acceleration, new_acceleration, dt)

        earth_acceleration = new_acceleration

    return np.array(x_positions), np.array(y_positions)

# Core

def main():
    print("Solar System Simulation")

    # Initial Positions (m)

    sun_position = np.array([0.0, 0.0, 0.0])
    earth_position = np.array([1.496e11, 0.0, 0.0])

    print("Sun position:", sun_position)
    print("Earth position:", earth_position)

    # Initial Velocity (m/s)

    orbital_radius = calculate_distance(sun_position, earth_position)

    earth_orbital_speed = np.sqrt(G * sun_mass / orbital_radius)

    earth_velocity = np.array([0.0, earth_orbital_speed, 0.0])

    print("Earth velocity:", earth_velocity, "m/s")

    # Gravitational Force (N)

    distance = calculate_distance(sun_position, earth_position)

    force = calculate_gravitational_force(sun_mass, earth_mass, distance)

    print("Gravitational force between Sun and Earth:", force, "N")

    force_vector = calculate_gravitational_force_vector(sun_mass, earth_mass, sun_position, earth_position)

    print("Gravitational force vector between Sun and Earth:", force_vector, "N")

    # Acceleration (m/s^2)

    earth_acceleration = acceleration(force_vector, earth_mass)

    print("Earth acceleration due to Sun's gravity:", earth_acceleration, "m/s^2")

    # Orbital Period

    orbital_period = (2 * np.pi * np.sqrt(orbital_radius ** 3 / (G * sun_mass)))

    print("Calculated orbital period:", orbital_period, "s")

    print("Calculated orbital period:", orbital_period / (60 * 60 * 24), "days")

    # Simulation

    x_positions = []
    y_positions = []

    time = 0.0

    dt = 10.0 # Smaller timestep for better numerical accuracy

    number_of_steps = int(orbital_period / dt)

    euler_x, euler_y = simulate_euler(sun_position, earth_position, earth_velocity, dt, number_of_steps)

    verlet_x, verlet_y = simulate_verlet(sun_position, earth_position, earth_velocity, dt, number_of_steps)

    # Convert metres to astronomical units for plotting

    euler_x_AU = euler_x / AU
    euler_y_AU = euler_y / AU

    verlet_x_AU = verlet_x / AU
    verlet_y_AU = verlet_y / AU

    # Plot

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(euler_x_AU, euler_y_AU, color="red", label="Euler")
    ax.plot(verlet_x_AU, verlet_y_AU, color="blue", label="Velocity Verlet")
    ax.scatter([0], [0], color="orange", s=200, label="Sun")

    ax.set_xlabel("x position (AU)")
    ax.set_ylabel("y position (AU)")
    ax.set_title("Euler vs Velocity Verlet")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    plt.show()

if __name__ == "__main__":
    main()