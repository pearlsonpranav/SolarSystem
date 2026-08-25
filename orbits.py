import numpy as np

from initial_conditions import G
from mechanics import calculate_unit_orbital_normal

def calculate_orbital_parameters(distances):
    perihelion_distance = np.min(distances)
    aphelion_distance = np.max(distances)

    semi_major_axis = (perihelion_distance + aphelion_distance) / 2
    eccentricity = (aphelion_distance - perihelion_distance) / (aphelion_distance + perihelion_distance)

    return perihelion_distance, aphelion_distance, semi_major_axis, eccentricity

def calculate_eccentricity_vector(position, velocity, mass1, mass2):
    distance = np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    total_mass = mass1 + mass2
    gravitational_parameter = G * total_mass

    eccentricity_vector = np.cross(velocity, angular_momentum) / gravitational_parameter - position / distance

    return eccentricity_vector

def calculate_eccentricity_vectors(positions, velocities, mass1, mass2):
    distances = np.linalg.norm(positions, axis=1)
    angular_momenta = np.cross(positions, velocities)
    total_mass = mass1 + mass2
    gravitational_parameter = G * total_mass

    eccentricity_vectors = np.cross(velocities, angular_momenta) / gravitational_parameter - positions / distances[:, np.newaxis]

    return eccentricity_vectors

def calculate_eccentricity_from_vector(eccentricity_vector):
    eccentricity = np.linalg.norm(eccentricity_vector)
    return eccentricity

def check_keplers_first_law_trajectory(positions, velocities, mass1, mass2):
    ecc_vectors = calculate_eccentricity_vectors(positions, velocities, mass1, mass2)
    e_mags = np.linalg.norm(ecc_vectors, axis=1)
    
    specific_energies = calculate_specific_orbital_energy(positions, velocities, mass1, mass2)
    gravitational_parameter = G * (mass1 + mass2)
    a_elements = -gravitational_parameter / (2 * specific_energies)
    
    actual_distances = np.linalg.norm(positions, axis=1)
    
    dot_products = np.sum(positions * ecc_vectors, axis=1)
    cos_theta = dot_products / (actual_distances * e_mags)
    
    expected_distances = (a_elements * (1 - e_mags**2)) / (1 + e_mags * cos_theta)
    
    max_percentage_error = np.max(np.abs(actual_distances - expected_distances) / expected_distances) * 100
    
    return max_percentage_error < 0.1, max_percentage_error

def calculate_areal_velocities(positions, velocities):
    areal_velocities = 0.5 * np.linalg.norm(np.cross(positions, velocities), axis=1)
    return areal_velocities

def check_keplers_second_law(areal_velocities):
    return np.isclose(np.min(areal_velocities), np.max(areal_velocities), rtol=1e-3)

def find_perihelion_indices(positions, velocities):
    perihelion_indices = [0]

    previous_radial_motion = np.dot(positions[0], velocities[0])

    for n in range(1, len(positions)):
        current_radial_motion = np.dot(positions[n], velocities[n])

        if previous_radial_motion < 0 and current_radial_motion >= 0:
            perihelion_indices.append(n)

        previous_radial_motion = current_radial_motion

    return perihelion_indices

def calculate_orbital_period(perihelion_indices, dt):
    if len(perihelion_indices) < 2:
        return None

    orbital_period = (perihelion_indices[1] - perihelion_indices[0]) * dt
    return orbital_period

def check_keplers_third_law(orbital_period, semi_major_axis, mass1, mass2):
    total_mass = mass1 + mass2
    proportionality_constant = orbital_period ** 2 / semi_major_axis ** 3
    theoretical_constant = 4 * np.pi ** 2 / (G * total_mass)

    return np.isclose(proportionality_constant, theoretical_constant, rtol=1e-3)

def calculate_vis_viva_velocity(mass1, mass2, distance, semi_major_axis):
    total_mass = mass1 + mass2
    velocity = np.sqrt(G * total_mass * (2 / distance - 1 / semi_major_axis))
    return velocity

def check_vis_viva(perihelion_velocity, aphelion_velocity, theoretical_perihelion_velocity, theoretical_aphelion_velocity):
    perihelion_difference = abs(perihelion_velocity - theoretical_perihelion_velocity) / theoretical_perihelion_velocity
    aphelion_difference = abs(aphelion_velocity - theoretical_aphelion_velocity) / theoretical_aphelion_velocity

    return perihelion_difference < 1e-3 and aphelion_difference < 1e-3

def calculate_specific_orbital_energy(position, velocity, mass1, mass2):
    distance = np.linalg.norm(position, axis=-1)
    speed = np.linalg.norm(velocity, axis=-1)
    gravitational_parameter = G * (mass1 + mass2)

    specific_orbital_energy = speed ** 2 / 2 - gravitational_parameter / distance

    return specific_orbital_energy

def classify_orbit(specific_orbital_energies):
    if np.all(specific_orbital_energies < 0):
        return "bound"
    elif np.all(np.isclose(specific_orbital_energies, 0)):
        return "escape boundary"
    else:
        return "unbound"
    
def calculate_inclination(position, velocity, reference_normal):
    unit_orbital_normal = calculate_unit_orbital_normal(position, velocity)
    unit_reference_normal = reference_normal / np.linalg.norm(reference_normal)

    cosine_inclination = np.dot(unit_orbital_normal, unit_reference_normal)
    cosine_inclination = np.clip(cosine_inclination, -1.0, 1.0)

    inclination = np.arccos(cosine_inclination)

    return inclination