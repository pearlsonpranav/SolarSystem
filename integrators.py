import numpy as np

from dynamics import calculate_gravitational_force_vector, calculate_acceleration

def update_position(position, velocity, dt):
    return position + velocity * dt

def update_velocity(velocity, acceleration, dt):
    return velocity + acceleration * dt

def update_position_verlet(position, velocity, acceleration, dt):
    return position + velocity * dt + 0.5 * acceleration * dt ** 2

def update_velocity_verlet(velocity, acceleration, new_acceleration, dt):
    return velocity + 0.5 * (acceleration + new_acceleration) * dt

def simulate_euler(position1, position2, mass1, mass2, velocity, dt, number_of_steps):
    x_positions = []
    y_positions = []

    position = position2

    for step in range(number_of_steps):
        force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position)

        current_acceleration = calculate_acceleration(force_vector, mass2)

        velocity = update_velocity(velocity, current_acceleration, dt)

        position = update_position(position, velocity, dt)

        x_positions.append(position[0])
        y_positions.append(position[1])

    return np.array(x_positions), np.array(y_positions)

def simulate_verlet(position1, position2, mass1, mass2, velocity, dt, number_of_steps):
    x_positions = []
    y_positions = []
    x_velocities = []
    y_velocities = []

    position = position2

    force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position)

    current_acceleration = calculate_acceleration(force_vector, mass2)

    for step in range(number_of_steps):
        position = update_position_verlet(position, velocity, current_acceleration, dt)

        force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position)

        new_acceleration = calculate_acceleration(force_vector, mass2)

        velocity = update_velocity_verlet(velocity, current_acceleration, new_acceleration, dt)

        current_acceleration = new_acceleration

        x_positions.append(position[0])
        y_positions.append(position[1])
        x_velocities.append(velocity[0])
        y_velocities.append(velocity[1])

    return np.array(x_positions), np.array(y_positions), np.array(x_velocities), np.array(y_velocities)