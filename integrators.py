import numpy as np

from mechanics import calculate_n_body_accelerations

def update_position(position, velocity, dt):
    return position + velocity * dt

def update_velocity(velocity, acceleration, dt):
    return velocity + acceleration * dt

def update_position_verlet(position, velocity, acceleration, dt):
    return position + velocity * dt + 0.5 * acceleration * dt ** 2

def update_velocity_verlet(velocity, acceleration, new_acceleration, dt):
    return velocity + 0.5 * (acceleration + new_acceleration) * dt

def simulate_euler(masses, positions, velocities, dt, number_of_steps):
    position_history = [positions.copy()]
    velocity_history = [velocities.copy()]

    for step in range(number_of_steps):
        accelerations = calculate_n_body_accelerations(masses, positions)

        velocities = velocities + accelerations * dt
        positions = positions + velocities * dt

        position_history.append(positions.copy())
        velocity_history.append(velocities.copy())

    return np.array(position_history), np.array(velocity_history)

def simulate_verlet(masses, positions, velocities, dt, number_of_steps):
    position_history = [positions.copy()]
    velocity_history = [velocities.copy()]

    accelerations = calculate_n_body_accelerations(masses, positions)

    for step in range(number_of_steps):
        positions = positions + velocities * dt + 0.5 * accelerations * dt ** 2

        new_accelerations = calculate_n_body_accelerations(masses, positions)

        velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt

        accelerations = new_accelerations

        position_history.append(positions.copy())
        velocity_history.append(velocities.copy())

    return np.array(position_history), np.array(velocity_history)

def simulate_rk4(masses, positions, velocities, dt, number_of_steps):
    position_history = [positions.copy()]
    velocity_history = [velocities.copy()]

    for step in range(number_of_steps):
        k1_position = velocities
        k1_velocity = calculate_n_body_accelerations(masses, positions)

        position_k2 = positions + 0.5 * k1_position * dt
        velocity_k2 = velocities + 0.5 * k1_velocity * dt
        k2_position = velocity_k2
        k2_velocity = calculate_n_body_accelerations(masses, position_k2)

        position_k3 = positions + 0.5 * k2_position * dt
        velocity_k3 = velocities + 0.5 * k2_velocity * dt
        k3_position = velocity_k3
        k3_velocity = calculate_n_body_accelerations(masses, position_k3)

        position_k4 = positions + k3_position * dt
        velocity_k4 = velocities + k3_velocity * dt
        k4_position = velocity_k4
        k4_velocity = calculate_n_body_accelerations(masses, position_k4)

        positions = positions + (dt / 6) * (k1_position + 2 * k2_position + 2 * k3_position + k4_position)
        velocities = velocities + (dt / 6) * (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity)

        position_history.append(positions.copy())
        velocity_history.append(velocities.copy())

    return np.array(position_history), np.array(velocity_history)