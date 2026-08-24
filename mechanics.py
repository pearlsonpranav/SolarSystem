import numpy as np

from initial_conditions import G

def calculate_distance(position1, position2):
    displacement = position1 - position2
    distance = np.linalg.norm(displacement)
    return distance

def calculate_distances_2D(x_positions, y_positions):
    distances = np.sqrt(x_positions ** 2 + y_positions ** 2)
    return distances

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

def calculate_acceleration(force, mass):
    acceleration = force / mass
    return acceleration

def calculate_n_body_accelerations(masses, positions):
    accelerations = np.zeros_like(positions, dtype=float)

    for i in range(len(masses)):
        for j in range(len(masses)):
            if i == j:
                continue

            displacement = positions[j] - positions[i]
            distance = np.linalg.norm(displacement)

            accelerations[i] += G * masses[j] * displacement / distance ** 3

    return accelerations

def calculate_kinetic_energy(mass, velocity):
    speed = np.linalg.norm(velocity)
    kinetic_energy = 0.5 * mass * speed ** 2
    return kinetic_energy

def calculate_total_kinetic_energy(masses, velocities):
    kinetic_energies = 0.5 * masses * np.sum(velocities ** 2, axis=1)
    total_kinetic_energy = np.sum(kinetic_energies)
    return total_kinetic_energy

def calculate_potential_energy(mass1, mass2, distance):
    potential_energy = -G * mass1 * mass2 / distance
    return potential_energy

def calculate_total_potential_energy(masses, positions):
    total_potential_energy = 0.0

    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            distance = calculate_distance(positions[i], positions[j])
            total_potential_energy += -G * masses[i] * masses[j] / distance

    return total_potential_energy

def calculate_total_energy(kinetic_energy, potential_energy):
    total_energy = kinetic_energy + potential_energy
    return total_energy

def calculate_escape_velocity(mass, distance):
    escape_velocity = np.sqrt(2 * G * mass / distance)
    return escape_velocity

def calculate_angular_momentum_2D(mass, position, velocity):
    angular_momentum = mass * (position[0] * velocity[1] - position[1] * velocity[0])
    return angular_momentum

def calculate_total_angular_momentum_2D(masses, positions, velocities):
    angular_momenta = masses * (positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0])
    total_angular_momentum = np.sum(angular_momenta)
    return total_angular_momentum

def calculate_centre_of_mass(masses, positions):
    total_mass = np.sum(masses)
    centre_of_mass = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
    return centre_of_mass