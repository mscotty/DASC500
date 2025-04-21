# This script simulates basic aerodynamic-related checks for a quadcopter UAV,
# using inheritance for a more structured UAV representation.

import ambiance  # For atmospheric properties
import numpy as np

class UAV:
    """Base class representing a generic UAV."""
    def __init__(self, model: str, weight_kg: float):
        """
        Initializes a generic UAV object.

        Args:
            model (str): The UAV model name.
            weight_kg (float): The weight of the UAV in kilograms.
        """
        self.model = model
        self.weight_kg = weight_kg

    def get_weight(self) -> float:
        """Returns the weight of the UAV."""
        return self.weight_kg

    def display_info(self):
        """Displays basic UAV information."""
        print(f"\nUAV Information:")
        print(f"  Model: {self.model}")
        print(f"  Weight: {self.weight_kg:.2f} kg")


class QuadcopterUAV(UAV):
    """Represents a quadcopter UAV, inheriting from the base UAV class."""

    def __init__(self, model: str, weight_kg: float, total_rotor_area_m2: float,
                 max_thrust_per_rotor_n: float):
        """
        Initializes a quadcopter UAV object.

        Args:
            model (str): The UAV model name.
            weight_kg (float): The weight of the UAV in kilograms.
            total_rotor_area_m2 (float): The combined area of all rotors.
            max_thrust_per_rotor_n (float): The maximum thrust produced by each rotor.
        """
        super().__init__(model, weight_kg)  # Call the base class constructor
        self.total_rotor_area_m2 = total_rotor_area_m2
        self.max_thrust_per_rotor_n = max_thrust_per_rotor_n
        self.num_rotors: int = 4  # Assuming a standard quadcopter

    def check_thrust_limits(self, required_thrust_n: float) -> str:
        """
        Checks if the required thrust exceeds the UAV's capabilities.

        Args:
            required_thrust_n (float): The total thrust required for the UAV.

        Returns:
            str: A message indicating if the thrust is within limits.
        """
        max_total_thrust = self.num_rotors * self.max_thrust_per_rotor_n
        if required_thrust_n > max_total_thrust:
            return "DANGER: Required thrust exceeds maximum UAV thrust!"
        else:
            return "Required thrust is within safe limits."

    def estimate_induced_velocity_ms(self, thrust_n: float, air_density_kg_m3: float) -> float:
        """
        Estimates the induced velocity (downwash) of the rotors.
        (Simplified momentum theory calculation)

        Args:
            thrust_n (float): The total thrust produced by the UAV.
            air_density_kg_m3 (float): The density of the surrounding air.

        Returns:
            float: Estimated induced velocity in m/s.
        """
        # Simplified induced velocity calculation (momentum theory)
        induced_velocity_ms = np.sqrt(thrust_n / (2 * air_density_kg_m3 * self.total_rotor_area_m2))
        return induced_velocity_ms

    def display_quad_info(self):
        """Displays quadcopter UAV information."""
        super().display_info()  # Call the base class method
        print(f"  Total Rotor Area: {self.total_rotor_area_m2} m^2")
        print(f"  Max Thrust per Rotor: {self.max_thrust_per_rotor_n} N")
        print(f"  Number of Rotors: {self.num_rotors}")



def main():
    """Performs a simplified pre-flight thrust check for a quadcopter UAV."""

    # UAV Initialization (Example Quadcopter)
    quadcopter = QuadcopterUAV(
        model="DJI Inspire 2",  # Example
        weight_kg=4.0,
        total_rotor_area_m2=4 * 0.02,  # Approximate
        max_thrust_per_rotor_n=35.0
    )
    quadcopter.display_quad_info()

    # Get user input for flight conditions
    altitude_ft_str = input("Enter the UAV altitude (feet): ")

    try:
        altitude_ft = float(altitude_ft_str)
        altitude_m = altitude_ft * 0.3048
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return

    # Calculate required thrust (simplified: thrust = weight * gravity)
    gravity_ms2 = 9.81
    required_thrust_n = quadcopter.get_weight() * gravity_ms2 # Use the get_weight method

    # Get air properties
    air = ambiance.Atmosphere(altitude_m)
    air_density_kg_m3 = air.density[0]

    # Perform thrust check
    thrust_status = quadcopter.check_thrust_limits(required_thrust_n)
    print(f"\nThrust Check: {thrust_status}")

    # Estimate induced velocity
    induced_velocity = quadcopter.estimate_induced_velocity_ms(required_thrust_n, air_density_kg_m3)
    print(f"Estimated Induced Velocity: {induced_velocity:.2f} m/s")

    # Display air properties
    print("\nAir Properties:")
    print(f"  Air Density: {air.density[0]:.4f} kg/m^3")
    print(f"  Air Temperature: {air.temperature[0]:.2f} K")
    print(f"  Air Pressure: {air.pressure[0]:.2f} Pa")


if __name__ == "__main__":
    main()