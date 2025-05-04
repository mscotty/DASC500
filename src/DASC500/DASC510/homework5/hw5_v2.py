import math
from enum import Enum
from typing import Tuple, Optional

# Problem 1 (2 points)
# Assign the 'name' variable an object that is your name of type str.
name = "Mitchell Scott"

# Needed for the code below
# --- Constants (Standard Library Only) ---
GRAVITY = 9.81  # m/s^2
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m^3
STANDARD_ATMOSPHERE_SCALE_HEIGHT = 8500  # m
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
# Simplified position updates - acknowledging limitations without geodetic libraries
METERS_PER_DEG_LAT = 111132.0
METERS_PER_DEG_LON_AT_EQUATOR = 111320.0


def get_meters_per_deg_lon(latitude: float) -> float:
    """Approx longitudinal meters per degree at a given latitude."""
    if abs(latitude) > 90:
        latitude = math.copysign(90.0, latitude)  # Cap latitude
    return METERS_PER_DEG_LON_AT_EQUATOR * math.cos(latitude * DEG_TO_RAD)


# --- Enums (Standard Library) ---
class PropulsionType(Enum):
    ELECTRIC = "electric"
    FUEL_BASED = "fuel_based"


class ManeuverType(Enum):
    CRUISE = "Cruise"
    CLIMB = "Climb"
    DIVE = "Dive"
    TURN = "Turn"
    # NOTE: Glide, Accelerate, Decelerate not included for simplicity


class State:
    """Represents the simplified state of the UAV."""

    def __init__(
        self,
        latitude: float = 0.0,
        longitude: float = 0.0,
        altitude: float = 0.0,
        speed: float = 0.0,
        heading: float = 0.0,
        remaining_energy: float = 1.0e6,  # Default large energy
        current_weight: float = 10.0,
    ):  # Default weight
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.altitude = float(altitude)  # meters AMSL
        self.speed = float(
            speed
        )  # m/s (Treating as Ground Speed for simplified position updates)
        self.heading = float(heading % 360)  # degrees (Course over ground)
        self.remaining_energy = float(remaining_energy)  # Joules or kg
        self.weight = float(current_weight)  # kg

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.latitude, self.longitude, self.altitude)

    def copy(self) -> "State":
        """Creates a copy of the state."""
        return State(
            self.latitude,
            self.longitude,
            self.altitude,
            self.speed,
            self.heading,
            self.remaining_energy,
            self.weight,
        )

    def update_simple_pos(self, distance: float, course_deg: float):
        """Simplified position update using basic trig. Ignores curvature, etc."""
        course_rad = course_deg * DEG_TO_RAD
        delta_lat_m = distance * math.cos(course_rad)
        delta_lon_m = distance * math.sin(course_rad)

        m_per_lon = get_meters_per_deg_lon(self.latitude)
        self.latitude += delta_lat_m / METERS_PER_DEG_LAT
        # Prevent longitude calculation error near poles
        if abs(m_per_lon) > 1e-6:
            self.longitude += delta_lon_m / m_per_lon
        # Basic latitude clamping
        if self.latitude > 90.0:
            self.latitude = 90.0
        if self.latitude < -90.0:
            self.latitude = -90.0
        # Longitude wrap around can be complex, omitted for simplicity here

    def update_energy_weight(self, energy_consumed: float, weight_change: float):
        """Updates energy and weight."""
        self.remaining_energy -= energy_consumed
        self.remaining_energy = max(
            0.0, self.remaining_energy
        )  # Cannot have negative energy
        self.weight -= weight_change
        # Note: Cannot enforce min weight without access to UAV's dry mass easily here

    def __repr__(self) -> str:
        return (
            f"State(Lat={self.latitude:.3f}, Lon={self.longitude:.3f}, Alt={self.altitude:.1f}m, "
            f"Spd={self.speed:.1f}m/s, Hdg={self.heading:.1f}deg, "
            f"Energy={self.remaining_energy:.1f}, Wgt={self.weight:.2f}kg)"
        )


class Environment:
    """Simplified environment, returning constant values (No external data loading)."""

    def __init__(self, air_density_model: str = "standard"):
        self.air_density_model = air_density_model
        print("Environment Initialized (Simplified).")

    def get_elevation(self, latitude: float, longitude: float) -> float:
        """Returns constant zero elevation (flat Earth)."""
        return 0.0

    def get_wind_vector(
        self, latitude: float, longitude: float, altitude: float
    ) -> Tuple[float, float, float]:
        """Returns zero wind vector (no wind)."""
        return 0.0, 0.0, 0.0  # East, North, Up components

    def get_air_density(self, altitude: float) -> float:
        """Return air density (kg/m^3) based on simplified model."""
        alt_m = max(0.0, altitude)  # Prevent negative altitude in calculation
        if self.air_density_model == "standard":
            density = AIR_DENSITY_SEA_LEVEL * math.exp(
                -alt_m / STANDARD_ATMOSPHERE_SCALE_HEIGHT
            )
        else:
            density = AIR_DENSITY_SEA_LEVEL
        return max(1e-9, density)  # Prevent zero or negative density

    def is_in_no_fly_zone(
        self, latitude: float, longitude: float, altitude: float
    ) -> bool:
        """Returns False (no NFZs defined)."""
        return False


class Waypoint:
    """Represents a target location with optional constraints."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        altitude: Optional[float] = None,
        name: str = "",
    ):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.altitude_constraint = altitude if altitude is None else float(altitude)
        self.name = str(name)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)

    @property
    def altitude(self) -> Optional[float]:
        return self.altitude_constraint

    def __repr__(self) -> str:
        alt_str = (
            f"{self.altitude_constraint:.1f}m"
            if self.altitude_constraint is not None
            else "Any Alt"
        )
        name_str = f" '{self.name}'" if self.name else ""
        return f"Waypoint{name_str}(Lat={self.latitude:.4f}, Lon={self.longitude:.4f}, {alt_str})"


class UAV:
    """Simplified UAV properties and calculations."""

    def __init__(
        self,
        propulsion_type: PropulsionType = PropulsionType.ELECTRIC,
        dry_mass: float = 10.0,  # kg
        initial_energy: float = 1.0e7,  # Joules or kg
        max_thrust: float = 50.0,  # N
        aero_ref_area: float = 1.0,  # m^2
        cd0: float = 0.05,  # Parasitic drag coeff
        k: float = 0.06,  # Induced drag factor
        propulsion_eff: float = 0.75,  # Overall efficiency
        sfc: float = 1.0e-5,
    ):  # kg/(N*s) Specific Fuel Consumption
        self.propulsion_type = propulsion_type
        self.dry_mass = float(dry_mass)
        self.initial_energy = float(initial_energy)
        self.max_thrust = float(max_thrust)
        self.aero_ref_area = float(aero_ref_area)
        self.cd0 = float(cd0)  # Parasitic drag
        self.k = float(k)  # Induced drag factor
        self.propulsion_eff = max(1e-6, float(propulsion_eff))  # Avoid zero efficiency
        self.sfc = max(0.0, float(sfc))  # Specific fuel consumption

    def get_current_weight(self, state: State) -> float:
        """Calculates current weight based on state's remaining energy (if fuel)."""
        fuel_mass = 0.0
        if self.propulsion_type == PropulsionType.FUEL_BASED:
            fuel_mass = state.remaining_energy  # Assumes energy is fuel mass
        return (
            self.dry_mass + fuel_mass
        )  # Assumes payload = 0, battery mass in dry_mass

    def calculate_drag(
        self, state: State, environment: Environment, cl: float
    ) -> float:
        """Calculates drag using simplified parabolic polar."""
        cd = self.cd0 + self.k * cl**2
        density = environment.get_air_density(state.altitude)
        # Use state.speed (treating as airspeed for aero calcs)
        dynamic_pressure = 0.5 * density * state.speed**2
        return dynamic_pressure * self.aero_ref_area * cd

    def calculate_lift_coeff(
        self, state: State, environment: Environment, flight_path_angle_deg: float = 0.0
    ) -> float:
        """Calculates required lift coefficient for quasi-steady flight."""
        weight = self.get_current_weight(state)
        required_lift = weight * GRAVITY * math.cos(flight_path_angle_deg * DEG_TO_RAD)
        density = environment.get_air_density(state.altitude)
        dynamic_pressure = 0.5 * density * state.speed**2
        if dynamic_pressure * self.aero_ref_area < 1e-6:
            return 0.0  # Avoid division by zero
        cl = required_lift / (dynamic_pressure * self.aero_ref_area)
        return cl

    def calculate_thrust_energy(
        self,
        state: State,
        environment: Environment,
        flight_path_angle_deg: float = 0.0,
        acceleration: float = 0.0,
        duration: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Calculates required thrust, energy consumed, and weight change."""
        if (
            state.speed < 1e-6 and acceleration <= 0
        ):  # Cannot calculate drag/thrust at zero speed without accel
            return 0.0, 0.0, 0.0

        weight = self.get_current_weight(state)
        cl = self.calculate_lift_coeff(state, environment, flight_path_angle_deg)
        drag = self.calculate_drag(state, environment, cl)

        # Thrust = Drag + Weight*sin(gamma) + mass*accel
        required_thrust = (
            drag
            + weight * GRAVITY * math.sin(flight_path_angle_deg * DEG_TO_RAD)
            + weight * acceleration
        )
        applied_thrust = max(
            0.0, min(required_thrust, self.max_thrust)
        )  # Apply feasible thrust

        # Energy Calculation
        energy_consumed = 0.0
        weight_change = 0.0
        if duration > 0:
            if self.propulsion_type == PropulsionType.ELECTRIC:
                power_mech = applied_thrust * state.speed
                power_electric = power_mech / self.propulsion_eff
                energy_consumed = power_electric * duration  # Joules
            elif self.propulsion_type == PropulsionType.FUEL_BASED:
                fuel_rate = applied_thrust * self.sfc  # kg/s
                fuel_consumed_mass = fuel_rate * duration  # kg
                energy_consumed = fuel_consumed_mass  # Track fuel mass as 'energy'
                weight_change = fuel_consumed_mass

        return applied_thrust, energy_consumed, weight_change

    def calculate_energy_consumption(
        self,
        state: State,
        environment: Environment,
        required_thrust: float,
        duration: float,
    ) -> Tuple[float, float, float]:
        """Calculates energy consumption for a given thrust and duration."""
        energy_consumed = 0.0
        weight_change = 0.0

        if duration > 0:
            if self.propulsion_type == PropulsionType.ELECTRIC:
                power_mech = required_thrust * state.speed  # Mechanical power
                power_electric = (
                    power_mech / self.propulsion_eff
                )  # Electrical power (accounting for efficiency)
                energy_consumed = power_electric * duration  # Total energy consumed
            elif self.propulsion_type == PropulsionType.FUEL_BASED:
                fuel_rate = required_thrust * self.sfc  # Fuel consumption rate (kg/s)
                fuel_consumed_mass = fuel_rate * duration  # Total fuel consumed (kg)
                energy_consumed = fuel_consumed_mass  # We track fuel mass as 'energy' for fuel-based systems
                weight_change = (
                    fuel_consumed_mass  # Weight change due to fuel consumption
                )

        return energy_consumed, weight_change, required_thrust


# Problem 2 (2 points)
# Create a Parent class (not named Parent) and implement it for your problem of interest
# It must have at least one attribute and one method
class Maneuver:
    """Base class for maneuvers - Demonstrates Inheritance."""

    def __init__(self, maneuver_type: ManeuverType):
        self.maneuver_type = maneuver_type

    def execute(
        self, start_state: State, uav: UAV, environment: Environment
    ) -> Optional[Tuple[State, float]]:
        """Executes maneuver, returns (end_state, duration). Simplified return."""
        # Needs implementation in subclasses
        print(f"Warning: Base Maneuver execute called for {self.maneuver_type}")
        return None  # Subclasses should override

    def plotManeuver(self, start_state: State, end_state: State) -> None:
        """Placeholder for plotting maneuver."""
        print(f"Plotting maneuver from {start_state} to {end_state}.")
        # Actual plotting code would go here but it depends on external libraries

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Problem 3 (3 points)
# Create a Child class (not named Child) and implement it for your problem of interest
# It must have at least one attribute and one method
class Cruise(Maneuver):
    """Simplified cruise for a fixed duration."""

    def __init__(self, duration: float = 10.0):
        super().__init__(ManeuverType.CRUISE)
        self.duration = max(0.0, float(duration))

    def execute(
        self, start_state: State, uav: UAV, environment: Environment
    ) -> Optional[Tuple[State, float]]:
        if self.duration <= 0:
            return start_state.copy(), 0.0
        if start_state.speed < 1.0:  # Need some speed to cruise
            print("Cruise Fail: Low start speed.")
            return None

        # Assume level flight, constant speed
        thrust, energy, weight_change = uav.calculate_thrust_energy(
            start_state,
            environment,
            flight_path_angle_deg=0.0,
            acceleration=0.0,
            duration=self.duration,
        )

        if energy > start_state.remaining_energy + 1e-9:
            print("Cruise Fail: Insufficient energy.")
            return None  # Not enough energy

        end_state = start_state.copy()
        distance = end_state.speed * self.duration
        end_state.update_simple_pos(distance, end_state.heading)
        end_state.update_energy_weight(energy, weight_change)

        return end_state, self.duration


# Problem 4 (3 points)
# Create another Child class and implement it for your problem of interest
# It must have at least one attribute and one method
class Climb(Maneuver):
    """Simplified climb for a fixed duration."""

    def __init__(self, duration: float = 10.0, climb_rate: float = 1.0):  # m/s
        super().__init__(ManeuverType.CLIMB)
        self.duration = max(0.0, float(duration))
        self.climb_rate = float(
            climb_rate
        )  # Can be negative for descent via this class if needed

    def execute(
        self, start_state: State, uav: UAV, environment: Environment
    ) -> Optional[Tuple[State, float]]:
        if self.duration <= 0:
            return start_state.copy(), 0.0
        if (
            start_state.speed < abs(self.climb_rate) + 1.0
        ):  # Need enough speed to climb/descend
            print(
                f"Climb/Dive Fail: Airspeed {start_state.speed:.1f} too low for rate {self.climb_rate:.1f}."
            )
            return None

        # Calculate flight path angle
        try:
            angle_rad = math.asin(self.climb_rate / start_state.speed)
        except ValueError:
            print(
                f"Climb/Dive Fail: Invalid asin calculation (rate {self.climb_rate} vs speed {start_state.speed})."
            )
            return None
        angle_deg = angle_rad * RAD_TO_DEG

        thrust, energy, weight_change = uav.calculate_thrust_energy(
            start_state,
            environment,
            flight_path_angle_deg=angle_deg,
            acceleration=0.0,
            duration=self.duration,
        )

        if energy > start_state.remaining_energy + 1e-9:
            print("Climb/Dive Fail: Insufficient energy.")
            return None

        end_state = start_state.copy()
        # Horizontal speed component
        horizontal_speed = start_state.speed * math.cos(angle_rad)
        distance = horizontal_speed * self.duration
        end_state.update_simple_pos(distance, end_state.heading)
        # Vertical change
        end_state.altitude += self.climb_rate * self.duration
        end_state.update_energy_weight(energy, weight_change)

        return end_state, self.duration


# (optional) If desired you can create any additional classes or functions here as well
class Turn(Maneuver):
    """Simplified turn for a fixed duration."""

    def __init__(self, duration: float = 10.0, turn_rate: float = 3.0):  # deg/s
        super().__init__(ManeuverType.TURN)
        self.duration = max(0.0, float(duration))
        # Store magnitude, apply direction based on calculation if needed
        self.turn_rate = float(turn_rate)  # Assume positive rate for simplicity

    def execute(
        self, start_state: State, uav: UAV, environment: Environment
    ) -> Optional[Tuple[State, float]]:
        if self.duration <= 0:
            return start_state.copy(), 0.0
        if start_state.speed < 1.0:  # Need speed to turn
            print("Turn Fail: Low start speed.")
            return None

        # Calculate bank angle needed (simplified)
        omega_rad_s = self.turn_rate * DEG_TO_RAD
        if GRAVITY < 1e-6:
            return None  # Avoid division by zero
        # tan(phi) = V*omega/g
        tan_phi = (start_state.speed * omega_rad_s) / GRAVITY
        # Limit bank angle to avoid excessive load factor approximation issues
        max_bank_rad = 60.0 * DEG_TO_RAD
        phi_rad = math.atan(tan_phi)
        if abs(phi_rad) > max_bank_rad:
            print(
                f"Turn Fail: Required bank angle {phi_rad*RAD_TO_DEG:.1f} exceeds limit."
            )
            return None

        # Calculate load factor n = 1/cos(phi)
        cos_phi = math.cos(phi_rad)
        if cos_phi < 1e-6:
            return None  # Avoid division by zero
        load_factor = 1.0 / cos_phi

        # Estimate thrust needed for level turn (L=n*W, T=D)
        weight = uav.get_current_weight(start_state)
        required_lift = load_factor * weight * GRAVITY
        density = environment.get_air_density(start_state.altitude)
        dynamic_pressure = 0.5 * density * start_state.speed**2
        if dynamic_pressure * uav.aero_ref_area < 1e-6:
            return None
        cl_turn = required_lift / (dynamic_pressure * uav.aero_ref_area)
        drag_turn = uav.calculate_drag(start_state, environment, cl_turn)
        required_thrust = drag_turn  # For steady turn

        energy, weight_change, thrust = uav.calculate_energy_consumption(
            start_state, environment, required_thrust, self.duration
        )

        if energy > start_state.remaining_energy + 1e-9:
            print("Turn Fail: Insufficient energy.")
            return None

        end_state = start_state.copy()
        # Update heading
        delta_heading = self.turn_rate * self.duration
        end_state.heading = (start_state.heading + delta_heading) % 360
        # Simplified position update (straight line along average heading)
        avg_heading = (start_state.heading + delta_heading / 2.0) % 360
        distance = end_state.speed * self.duration
        end_state.update_simple_pos(distance, avg_heading)
        end_state.update_energy_weight(energy, weight_change)

        return end_state, self.duration


# Problem 5 (3 points)
#  Assign a variable named 'obj_1' an example instance of your Parent class
#  Using obj_1, access (and print) an attribute of the Parent class
#  Using obj_1, execute a method from the Parent class.
obj_1 = Maneuver(ManeuverType.CRUISE)
print(obj_1.maneuver_type)  # Accessing attribute of Parent class
obj_1.execute(State(), UAV(), Environment())  # Executing method

# Problem 6 (4 points)
#  Assign a variable named 'obj_2' an example instance of a Child class
#  Using obj_2, access (and print) an attribute of the Parent class
#  Using obj_2, access (and print) an attribute of the Child class
obj_2 = Cruise(duration=5.0)
print(obj_2.maneuver_type)  # Accessing attribute of Parent class
print(obj_2.duration)  # Accessing attribute of child class

# Problem 7 (4 points)
#  Assign a variable named 'obj_3' an example instance of your other Child class
#  Using obj_3, execute a method from the Parent class
#  Using obj_3, execute a method from the Child class
obj_3 = Climb(duration=5.0, climb_rate=2.0)
# Sorry had to do it out of order because I needed the output from the Child class to execute the Parent class
endState = obj_3.execute(State(), UAV(), Environment())  # Executing method from Child class
obj_3.plotManeuver(State(), endState)  # Executing method from Parent class



# Problems 8 through 14 are worth 3 points each.
#    For each problem you must implement a test method in the following
#     TestCase class. Each method name should be unique and start with 'test_'.
#     Each method should test something different about the classes you created
#     above. Use unittest. TestCase's assert methods to check your implementation.
#     You will get 2 points for each test method created. You will get 2
#     additional points for each of these tests that complete successfully.
#     See https://docs.python.org/3/library/unittest.html for examples

import unittest
import sys
import inspect


###################################################
# DO NOT MODIFY this class's name or what it extends
###################################################
class MyTestCases(unittest.TestCase):

    # Problem 8
    # add test case method
    def test_maneuver_creation(self):
        maneuver = Maneuver(ManeuverType.CRUISE)
        self.assertIsInstance(maneuver, Maneuver)
        self.assertEqual(maneuver.maneuver_type, ManeuverType.CRUISE)

    # Problem 9
    # add test case method
    def test_cruise_creation(self):
        cruise = Cruise(duration=15.0)
        self.assertIsInstance(cruise, Cruise)
        self.assertEqual(cruise.duration, 15.0)
        self.assertEqual(cruise.maneuver_type, ManeuverType.CRUISE)

    # Problem 10
    # add test case method
    def test_climb_creation(self):
        climb = Climb(duration=20.0, climb_rate=3.0)
        self.assertIsInstance(climb, Climb)
        self.assertEqual(climb.duration, 20.0)
        self.assertEqual(climb.climb_rate, 3.0)
        self.assertEqual(climb.maneuver_type, ManeuverType.CLIMB)

    # Problem 11
    # add test case method
    def test_turn_creation(self):
        turn = Turn(duration=8.0, turn_rate=5.0)
        self.assertIsInstance(turn, Turn)
        self.assertEqual(turn.duration, 8.0)
        self.assertEqual(turn.turn_rate, 5.0)
        self.assertEqual(turn.maneuver_type, ManeuverType.TURN)

    # Problem 12
    # add test case method
    def test_cruise_execute(self):
        state = State(speed=20.0, heading=45.0)
        uav = UAV()
        env = Environment()
        cruise = Cruise(duration=10.0)
        end_state, duration = cruise.execute(state, uav, env)
        self.assertIsNotNone(end_state)
        self.assertEqual(duration, 10.0)
        self.assertAlmostEqual(
            end_state.latitude, state.latitude + 0.00127, delta=0.0001
        )
        self.assertAlmostEqual(
            end_state.longitude, state.longitude + 0.00127, delta=0.0001
        )
        self.assertLess(end_state.remaining_energy, state.remaining_energy)
        self.assertAlmostEqual(end_state.speed, state.speed, delta=0.11)
        self.assertAlmostEqual(end_state.heading, state.heading, delta=0.11)

    # Problem 13
    # add test case method
    def test_climb_execute(self):
        state = State(speed=20.0, heading=45.0, altitude=100.0)
        uav = UAV()
        env = Environment()
        climb = Climb(duration=10.0, climb_rate=2.0)
        end_state, duration = climb.execute(state, uav, env)
        self.assertIsNotNone(end_state)
        self.assertEqual(duration, 10.0)
        self.assertAlmostEqual(
            end_state.latitude, state.latitude + 0.00127, delta=0.0001
        )
        self.assertAlmostEqual(
            end_state.longitude, state.longitude + 0.00127, delta=0.0001
        )
        self.assertAlmostEqual(end_state.altitude, state.altitude + 20.0, delta=0.1)
        self.assertLess(end_state.remaining_energy, state.remaining_energy)
        self.assertAlmostEqual(
            end_state.speed,
            state.speed * math.cos(math.asin(2.0 / state.speed)),
            delta=0.11,
        )
        self.assertAlmostEqual(end_state.heading, state.heading, delta=0.1)

    # Problem 14
    # add test case method
    def test_turn_execute(self):
        state = State(speed=15.0, heading=0.0)
        uav = UAV()
        env = Environment()
        turn = Turn(duration=5.0, turn_rate=10.0)
        end_state, duration = turn.execute(state, uav, env)
        self.assertIsNotNone(end_state)
        self.assertEqual(duration, 5.0)
        self.assertAlmostEqual(end_state.heading, 50.0, delta=0.1)
        self.assertLess(end_state.remaining_energy, state.remaining_energy)
        self.assertAlmostEqual(end_state.speed, state.speed, delta=0.1)

    ##################################################
    # DO NOT MODIFY any of these test case methods
    ##################################################

    def get_classes(self):
        clses = inspect.getmembers(
            sys.modules[__name__],
            lambda member: inspect.isclass(member)
            and member.__module__ == __name__
            and member is not MyTestCases,
        )
        return clses

    def test_name_assigned(self):
        m = sys.modules[__name__]
        name = getattr(m, "name", None)
        self.assertTrue(name is not None)
        self.assertTrue(isinstance(name, str))
        self.assertTrue(len(name) > 0)

    def test_one_class_created(self):
        self.assertTrue(len(self.get_classes()) >= 1)

    def test_two_classes_created(self):
        self.assertTrue(len(self.get_classes()) >= 2)

    def test_three_classes_created(self):
        self.assertTrue(len(self.get_classes()) >= 3)

    def test_obj_1_instance_created(self):
        clses = self.get_classes()
        m = sys.modules[__name__]
        obj_1 = getattr(m, "obj_1", None)
        self.assertTrue(obj_1 is not None)
        self.assertTrue(isinstance(obj_1, tuple([cls[1] for cls in clses])))

    def test_obj_2_instance_created(self):
        clses = self.get_classes()
        m = sys.modules[__name__]
        obj_2 = getattr(m, "obj_2", None)
        obj_1 = getattr(m, "obj_1", None)
        self.assertTrue(obj_2 is not None)
        self.assertTrue(isinstance(obj_2, tuple([cls[1] for cls in clses])))
        self.assertNotEqual(type(obj_2), type(obj_1))

    def test_obj_3_instance_created(self):
        clses = self.get_classes()
        m = sys.modules[__name__]
        obj_3 = getattr(m, "obj_3", None)
        self.assertTrue(obj_3 is not None)
        self.assertTrue(isinstance(obj_3, tuple([cls[1] for cls in clses])))
        bases = tuple(b for b in type(obj_3).__bases__ if b is not object)
        print(dir(obj_3))
        self.assertTrue(len(bases) > 0)


##################################################
# The code below triggers the unit tests for a .py environment
# Please comment out if you are using a .ipynb environment
##################################################

if __name__ == "__main__":
    unittest.main()

##################################################
# The code below triggers the unit tests for a . ipynb environment
# Please comment out if you are using a .py environment
##################################################

# unittest.main(argv=[''], verbosity=2, exit=False)
