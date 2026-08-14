import numpy as np

# Constants

G = 6.67430e-11
sun_mass = 1.989e30
earth_mass = 5.972e24

# Functions

def calculate_distance (position1, position2):
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

# Main Simulation

def main():
    print("Solar System Simulation")

    # Initial Positions (m)

    sun_position = np.array([0, 0, 0])
    earth_position = np.array([1.496e11, 0, 0])

    print("Sun position:", sun_position)
    print("Earth position:", earth_position)

    # Distance (m)

    distance = calculate_distance(sun_position, earth_position)
    print("Distance from Sun to Earth:", distance, "m")

    # Velocity (m/s)

    earth_velocity = np.array([0, 29.78e3, 0])
    print("Earth velocity:", earth_velocity, "m/s")

    # Gravitational Force (N)

    force = calculate_gravitational_force(sun_mass, earth_mass, distance)
    print("Gravitational force between Sun and Earth:", force, "N")

    force_vector = calculate_gravitational_force_vector(sun_mass, earth_mass, sun_position, earth_position)
    print("Gravitational force vector between Sun and Earth:", force_vector, "N")

    # Acceleration (m/s^2)

    earth_acceleration = acceleration(force_vector, earth_mass)
    print("Earth acceleration due to Sun's gravity:", earth_acceleration, "m/s^2")

     # Simulation Loop

    time = 0.0
    dt = 60.0
    for step in range(5):
        force_vector = calculate_gravitational_force_vector(sun_mass, earth_mass, sun_position, earth_position)
        earth_acceleration = acceleration(force_vector, earth_mass)
        earth_velocity = update_velocity(earth_velocity, earth_acceleration, dt)
        earth_position = update_position(earth_position, earth_velocity, dt)
        time += dt
        print(f"Time: {time} s")
        print("Earth position:", earth_position)
        print("Earth velocity:", earth_velocity)

if __name__ == "__main__":
    main()