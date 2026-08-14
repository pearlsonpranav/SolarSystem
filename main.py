import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants

G = 6.67430e-11
AU = 1.496e11
sun_mass = 1.989e30
earth_mass = 5.972e24

# Functions

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

    sun_position = np.array([0.0, 0.0, 0.0])
    earth_position = np.array([1.496e11, 0.0, 0.0])

    print("Sun position:", sun_position)
    print("Earth position:", earth_position)

    # Initial Velocity (m/s)

    orbital_radius = calculate_distance(sun_position, earth_position)

    earth_orbital_speed = np.sqrt(G * sun_mass / orbital_radius)

    earth_velocity = np.array([0.0, earth_orbital_speed, 0.0])

    print("Earth velocity:", earth_velocity, "m/s")

    # Gravitational Force (N)

    distance = calculate_distance(sun_position, earth_position)

    force = calculate_gravitational_force(sun_mass, earth_mass, distance)

    print("Gravitational force between Sun and Earth:", force, "N")

    force_vector = calculate_gravitational_force_vector(sun_mass, earth_mass, sun_position, earth_position)

    print("Gravitational force vector between Sun and Earth:", force_vector, "N")

    # Acceleration (m/s^2)

    earth_acceleration = acceleration(force_vector, earth_mass)

    print("Earth acceleration due to Sun's gravity:", earth_acceleration, "m/s^2")

    # Orbital Period

    orbital_period = (2 * np.pi * np.sqrt(orbital_radius ** 3 / (G * sun_mass)))

    print("Calculated orbital period:", orbital_period, "s")

    print("Calculated orbital period:", orbital_period / (60 * 60 * 24), "days")

    # Simulation

    x_positions = []
    y_positions = []

    time = 0.0

    dt = 10.0 # Smaller timestep for better numerical accuracy

    number_of_steps = int(orbital_period / dt)

    for step in range(number_of_steps):

        # Calculate current gravitational force
        force_vector = calculate_gravitational_force_vector(sun_mass, earth_mass, sun_position, earth_position)

        # Calculate current acceleration
        earth_acceleration = acceleration(force_vector, earth_mass)

        # Update velocity
        earth_velocity = update_velocity(earth_velocity, earth_acceleration, dt)

        # Update position
        earth_position = update_position(earth_position, earth_velocity, dt)

        # Store position
        x_positions.append(earth_position[0])
        y_positions.append(earth_position[1])

        time += dt

    print("Final time:", time, "s")
    print("Final Earth position:", earth_position)
    print("Final Earth velocity:", earth_velocity)

    # Convert metres to astronomical units for plotting

    x_positions_AU = np.array(x_positions) / AU
    y_positions_AU = np.array(y_positions) / AU

    # Plot

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x_positions_AU, y_positions_AU, color="blue", label="Earth orbit")

    ax.scatter([0], [0], color="orange", s=200, label="Sun")

    # Moving Earth marker

    earth_marker, = ax.plot([], [], marker="o", color="blue", markersize=8, linestyle="None", label="Earth")

    ax.set_xlabel("x position (AU)")
    ax.set_ylabel("y position (AU)")
    ax.set_title("Earth's Orbit")
    ax.axis("equal")
    ax.legend()

    # Animation

    animation_step = max(1, len(x_positions_AU) // 500)

    def animate(frame):
        index = frame * animation_step

        if index >= len(x_positions_AU):
            index = len(x_positions_AU) - 1

        earth_marker.set_data([x_positions_AU[index]], [y_positions_AU[index]])

        return earth_marker,

    number_of_frames = (len(x_positions_AU) // animation_step)

    animation = FuncAnimation(fig, animate, frames=number_of_frames, interval=20, blit=True, repeat=True)

    plt.show()

if __name__ == "__main__":
    main()