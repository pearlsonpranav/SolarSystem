import numpy as np

from initial_conditions import G
from mechanics import calculate_distance

def calculate_orbital_parameters_2D(distances):
    perihelion_distance = np.min(distances)
    aphelion_distance = np.max(distances)

    semi_major_axis = (perihelion_distance + aphelion_distance) / 2
    eccentricity = (aphelion_distance - perihelion_distance) / (aphelion_distance + perihelion_distance)

    return perihelion_distance, aphelion_distance, semi_major_axis, eccentricity

def calculate_ellipse_centre_2D(x_positions, y_positions):
    centre = np.array([(np.max(x_positions) + np.min(x_positions)) / 2, (np.max(y_positions) + np.min(y_positions)) / 2])
    return centre

def check_keplers_first_law_2D(centre, sun_position, semi_major_axis, eccentricity):
    focal_distance = calculate_distance(centre, sun_position)
    expected_focal_distance = semi_major_axis * eccentricity

    return np.isclose(focal_distance, expected_focal_distance, rtol=1e-5) and np.isclose(centre[1], 0.0, atol=1e3)

def calculate_swept_areas_2D(x_positions, y_positions):
    areas = []

    for n in range(len(x_positions) - 1):
        area = 0.5 * np.abs(x_positions[n] * y_positions[n + 1] - y_positions[n] * x_positions[n + 1])
        areas.append(area)

    return np.array(areas)

def check_keplers_second_law(areas):
    return np.isclose(np.min(areas), np.max(areas), rtol=1e-5)

def find_perihelion_indices_2D(x_positions, y_positions, x_velocities, y_velocities):
    perihelion_indices = []

    for n in range(1, len(x_positions)):
        previous_position = np.array([x_positions[n - 1], y_positions[n - 1]])
        previous_velocity = np.array([x_velocities[n - 1], y_velocities[n - 1]])

        current_position = np.array([x_positions[n], y_positions[n]])
        current_velocity = np.array([x_velocities[n], y_velocities[n]])

        previous_radial_motion = np.dot(previous_position, previous_velocity)
        current_radial_motion = np.dot(current_position, current_velocity)

        if previous_radial_motion < 0 and current_radial_motion > 0:
            perihelion_indices.append(n)

    return perihelion_indices

def calculate_orbital_period(perihelion_indices, dt):
    orbital_period = (perihelion_indices[1] - perihelion_indices[0]) * dt
    return orbital_period

def check_keplers_third_law(orbital_period, semi_major_axis, sun_mass):
    proportionality_constant = orbital_period ** 2 / semi_major_axis ** 3
    theoretical_constant = 4 * np.pi ** 2 / (G * sun_mass)

    return np.isclose(proportionality_constant, theoretical_constant, rtol=1e-7)

def calculate_vis_viva_velocity(mass, distance, semi_major_axis):
    velocity = np.sqrt(G * mass * (2 / distance - 1 / semi_major_axis))
    return velocity

def check_vis_viva(perihelion_velocity, aphelion_velocity, theoretical_perihelion_velocity, theoretical_aphelion_velocity):
    return np.isclose(perihelion_velocity, theoretical_perihelion_velocity, atol=1e-10) and np.isclose(aphelion_velocity, theoretical_aphelion_velocity, atol=1e-10)

def classify_orbit(total_energies):
    if np.all(np.array(total_energies) < 0):
        return "bound"
    elif np.all(np.isclose(total_energies, 0)):
        return "escape boundary"
    else:
        return "unbound"