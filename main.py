import numpy as np

def calculate_distance (position1, position2):
    displacement = position2 - position1
    distance = np.linalg.norm(displacement)
    return distance

def main():
    print("Solar System Simulation")

    # Masses (kg)

    sun_mass = 1.989e30
    earth_mass = 5.972e24

    # Positions Vectors (m)

    sun_position = np.array([0, 0, 0])
    earth_position = np.array([1.496e11, 0, 0])

    print("Sun position:", sun_position)
    print("Earth position:", earth_position)

    # Distance (m)

    distance = calculate_distance(sun_position, earth_position)
    print("Distance from Sun to Earth:", distance, "m")

    # Veloctiy (m/s)

    earth_velocity = np.array([0, 29.78e3, 0])
    print("Earth velocity:", earth_velocity, "m/s")

    # Time Loop

    time = 0.0
    dt = 60.0
    for step in range(5):
        time += dt
        print(f"Time: {time} seconds")

if __name__ == "__main__":
    main()