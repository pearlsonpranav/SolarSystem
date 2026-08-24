import numpy as np

from initial_conditions import G

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

def calculate_ellipse_centre(eccentricity_vector, semi_major_axis):
    centre = -semi_major_axis * eccentricity_vector
    return centre

def check_keplers_first_law(centre, semi_major_axis, eccentricity):
    focal_distance = np.linalg.norm(centre)
    expected_focal_distance = semi_major_axis * eccentricity

    return np.isclose(focal_distance, expected_focal_distance, rtol=1e-4)

def calculate_areal_velocities(positions, velocities):
    areal_velocities = 0.5 * np.linalg.norm(np.cross(positions, velocities), axis=1)
    return areal_velocities

def check_keplers_second_law(areal_velocities):
    return np.isclose(np.min(areal_velocities), np.max(areal_velocities), rtol=1e-4)

def find_perihelion_indices_3D(positions, velocities):
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

    return np.isclose(proportionality_constant, theoretical_constant, rtol=1e-7)

def calculate_vis_viva_velocity(mass1, mass2, distance, semi_major_axis):
    total_mass = mass1 + mass2
    velocity = np.sqrt(G * total_mass * (2 / distance - 1 / semi_major_axis))
    return velocity

def check_vis_viva(perihelion_velocity, aphelion_velocity, theoretical_perihelion_velocity, theoretical_aphelion_velocity):
    perihelion_difference = abs(perihelion_velocity - theoretical_perihelion_velocity) / theoretical_perihelion_velocity
    aphelion_difference = abs(aphelion_velocity - theoretical_aphelion_velocity) / theoretical_aphelion_velocity

    return perihelion_difference < 1e-4 and aphelion_difference < 1e-4

def classify_orbit(total_energies):
    if np.all(np.array(total_energies) < 0):
        return "bound"
    elif np.all(np.isclose(total_energies, 0)):
        return "escape boundary"
    else:
        return "unbound"