import numpy as np

from mechanics import calculate_gravitational_force_vector, calculate_acceleration

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
    x_velocities = []
    y_velocities = []

    position = position2

    # Initial position and velocity
    x_positions.append(position[0])
    y_positions.append(position[1])
    x_velocities.append(velocity[0])
    y_velocities.append(velocity[1])

    for step in range(number_of_steps):
        force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position)

        current_acceleration = calculate_acceleration(force_vector, mass2)

        velocity = update_velocity(velocity, current_acceleration, dt)

        position = update_position(position, velocity, dt)

        x_positions.append(position[0])
        y_positions.append(position[1])
        x_velocities.append(velocity[0])
        y_velocities.append(velocity[1])

    return np.array(x_positions), np.array(y_positions), np.array(x_velocities), np.array(y_velocities)

def simulate_verlet(position1, position2, mass1, mass2, velocity, dt, number_of_steps):
    x_positions = []
    y_positions = []
    x_velocities = []
    y_velocities = []

    position = position2

    force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position)

    current_acceleration = calculate_acceleration(force_vector, mass2)

    # Initial position and velocity
    x_positions.append(position[0])
    y_positions.append(position[1])
    x_velocities.append(velocity[0])
    y_velocities.append(velocity[1])

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

def simulate_rk4(position1, position2, mass1, mass2, velocity, dt, number_of_steps):
    x_positions = []
    y_positions = []
    x_velocities = []
    y_velocities = []

    position = position2

    # Initial position and velocity
    x_positions.append(position[0])
    y_positions.append(position[1])
    x_velocities.append(velocity[0])
    y_velocities.append(velocity[1])

    for step in range(number_of_steps):
        force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position)
        acceleration = calculate_acceleration(force_vector, mass2)

        k1_position = velocity
        k1_velocity = acceleration

        position_k2 = position + 0.5 * k1_position * dt
        velocity_k2 = velocity + 0.5 * k1_velocity * dt
        force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position_k2)
        acceleration_k2 = calculate_acceleration(force_vector, mass2)

        k2_position = velocity_k2
        k2_velocity = acceleration_k2

        position_k3 = position + 0.5 * k2_position * dt
        velocity_k3 = velocity + 0.5 * k2_velocity * dt
        force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position_k3)
        acceleration_k3 = calculate_acceleration(force_vector, mass2)

        k3_position = velocity_k3
        k3_velocity = acceleration_k3

        position_k4 = position + k3_position * dt
        velocity_k4 = velocity + k3_velocity * dt
        force_vector = calculate_gravitational_force_vector(mass1, mass2, position1, position_k4)
        acceleration_k4 = calculate_acceleration(force_vector, mass2)

        k4_position = velocity_k4
        k4_velocity = acceleration_k4

        position = position + (dt / 6) * (k1_position + 2 * k2_position + 2 * k3_position + k4_position)
        velocity = velocity + (dt / 6) * (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity)

        x_positions.append(position[0])
        y_positions.append(position[1])
        x_velocities.append(velocity[0])
        y_velocities.append(velocity[1])

    return np.array(x_positions), np.array(y_positions), np.array(x_velocities), np.array(y_velocities)