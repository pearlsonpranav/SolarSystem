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

def calculate_kinetic_energy(mass, velocity):
    speed = np.linalg.norm(velocity)
    kinetic_energy = 0.5 * mass * speed ** 2
    return kinetic_energy

def calculate_potential_energy(mass1, mass2, distance):
    potential_energy = -G * mass1 * mass2 / distance
    return potential_energy

def calculate_total_energy(kinetic_energy, potential_energy):
    total_energy = kinetic_energy + potential_energy
    return total_energy

def calculate_escape_velocity(mass, distance):
    escape_velocity = np.sqrt(2 * G * mass / distance)
    return escape_velocity

def calculate_angular_momentum_2D(mass, position, velocity):
    angular_momentum = mass * (position[0] * velocity[1] - position[1] * velocity[0])
    return angular_momentum