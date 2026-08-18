import numpy as np

G = 6.67430e-11  # Gravitational constant

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

def calculate_acceleration(force, mass):
    acceleration = force / mass
    return acceleration