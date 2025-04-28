# --- uav_planner_objects_runnable_demo.py ---
import os
import json
from pathlib import Path
import math
import datetime
import sys
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Union, Callable


# --- Enumeration Classes ---
class StatusCode(Enum):
    OK = 0
    WARNING = 1
    ERROR = 2
    UNDEFINED = 3

    def __str__(self) -> str:
        return self.name


class CheckType(Enum):
    MIN_THRESHOLD = auto()
    MAX_THRESHOLD = auto()
    EXACT_MATCH = auto()
    UNRECOGNIZED = auto()


# Default filenames
DEFAULT_CONFIG_FILE: str = "uav_config.json"
DEFAULT_STATE_FILE: str = "uav_system_state_enum.pkl"

# Type Alias
SpecsDict = Dict[str, Union[str, float, int, bool, Dict]]
SystemsDict = Dict[str, "UAVSystem"]


# --- Basic Data Structures ---
class Location:
    def __init__(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m_msl: Optional[float] = None,
    ):
        self.latitude_deg: float = latitude_deg
        self.longitude_deg: float = longitude_deg
        self.altitude_m_msl: Optional[float] = altitude_m_msl

    def __repr__(self) -> str:
        alt_str = (
            f"{self.altitude_m_msl:.1f}m" if self.altitude_m_msl is not None else "N/A"
        )
        return f"Location(lat={self.latitude_deg:.5f}, lon={self.longitude_deg:.5f}, alt={alt_str})"

    def distance_to(self, other: "Location") -> float:
        # Placeholder distance calc
        lat1, lon1 = math.radians(self.latitude_deg), math.radians(self.longitude_deg)
        lat2, lon2 = math.radians(other.latitude_deg), math.radians(other.longitude_deg)
        R = 6371000
        x = (lon2 - lon1) * math.cos((lat1 + lat2) / 2)
        y = lat2 - lat1
        return math.sqrt(x * x + y * y) * R


class VelocityVector:
    def __init__(self, speed_kts: float, heading_deg: float):
        self.speed_kts: float = speed_kts
        self.heading_deg: float = heading_deg

    def __repr__(self) -> str:
        return f"Velocity(spd={self.speed_kts:.1f}kts, hdg={self.heading_deg:.1f}deg)"


class Waypoint:
    def __init__(
        self,
        waypoint_id: str,
        location: Location,
        target_speed_kts: Optional[float] = None,
    ):
        self.waypoint_id: str = waypoint_id
        self.location: Location = location
        self.target_speed_kts: Optional[float] = target_speed_kts

    def __repr__(self) -> str:
        spd_str = (
            f"{self.target_speed_kts:.1f}kts"
            if self.target_speed_kts is not None
            else "N/A"
        )
        return (
            f"Waypoint(id='{self.waypoint_id}', loc={self.location}, speed={spd_str})"
        )

    def determine_required_conditions(
        self, uav_state: "UAVState", environment: "Environment", uav: "UAV"
    ) -> Dict[str, float]:
        """Docstring unchanged..."""
        print(
            f"[Method] Calculating required conditions at Waypoint {self.waypoint_id}..."
        )
        mock_thrust = uav_state.current_weight_kg * 9.81 * 0.2
        mock_lift = uav_state.current_weight_kg * 9.81
        return {"required_thrust_n": mock_thrust, "required_lift_n": mock_lift}

    def check_reachability(
        self, uav_state: "UAVState", uav: "UAV", environment: "Environment"
    ) -> bool:
        """Docstring unchanged..."""
        print(
            f"[Method] Checking reachability for Waypoint {self.waypoint_id} from state {uav_state}..."
        )
        dist_m = uav_state.location.distance_to(self.location)
        min_energy_estimate = dist_m * 0.1  # Very rough guess
        print(
            f"  Est. distance: {dist_m:.0f} m, Est. min energy: {min_energy_estimate:.1f}, Available: {uav_state.remaining_energy:.1f}"
        )
        if uav_state.remaining_energy > min_energy_estimate:
            return True
        else:
            print(f"  Waypoint {self.waypoint_id} likely unreachable due to energy.")
            return False


class UAVState:
    def __init__(
        self,
        location: Location,
        velocity: VelocityVector,
        current_weight_kg: float,
        remaining_energy: float,
    ):
        self.location: Location = location
        self.velocity: VelocityVector = velocity
        self.current_weight_kg: float = current_weight_kg
        self.remaining_energy: float = remaining_energy

    def __repr__(self) -> str:
        return f"UAVState(loc={self.location}, vel={self.velocity}, w={self.current_weight_kg:.2f}kg, E={self.remaining_energy:.2f})"

    def update_state_after_maneuver(self, maneuver_result: Dict[str, Any]) -> None:
        """Docstring unchanged..."""
        print(f"[Method] Updating UAV state after maneuver...")
        self.location = maneuver_result.get("next_location", self.location)
        self.velocity = maneuver_result.get("next_velocity", self.velocity)
        energy_cost = maneuver_result.get("energy_cost", 0)
        fuel_burned = maneuver_result.get("fuel_burned_kg", 0)
        self.remaining_energy -= energy_cost
        self.current_weight_kg -= fuel_burned
        print(f"  New State: {self}")
        # if self.remaining_energy < 0: print("  WARNING: Negative energy!")

    def calculate_energy_cost(
        self,
        maneuver: "Maneuver",
        uav: "UAV",
        environment: "Environment",
        duration_s: float,
    ) -> float:
        """Docstring unchanged..."""
        print(
            f"[Method] Calculating energy cost for {maneuver.maneuver_type} over {duration_s:.1f}s..."
        )
        avg_power_kw_guess = 5.0
        if uav.propulsion_type == "electric":
            energy_wh = (avg_power_kw_guess * 1000 * duration_s / 3600) / 0.85
            # print(f"  Estimated Electric Cost: {energy_wh:.2f} Wh")
            return energy_wh
        elif uav.propulsion_type == "fuel":
            fuel_rate_kg_per_hr_guess = 2.0
            fuel_kg = fuel_rate_kg_per_hr_guess * duration_s / 3600
            # print(f"  Estimated Fuel Cost: {fuel_kg:.4f} kg")
            return fuel_kg
        else:
            return 0.0


class UAV:
    def __init__(
        self,
        uav_id: str,
        propulsion_type: str,
        performance_model: SpecsDict,
        initial_state: UAVState,
    ):
        self.uav_id: str = uav_id
        self.propulsion_type: str = propulsion_type
        self.performance_model: SpecsDict = performance_model
        self.current_state: UAVState = initial_state
        self.trim_data: Optional[Any] = None
        self.load_trim_data()  # Load on init

    def __repr__(self) -> str:
        return f"UAV(id='{self.uav_id}', type='{self.propulsion_type}', state={self.current_state})"

    def load_trim_data(self) -> None:
        """Loads trim data. If file not found, sets up a dummy structure."""
        trim_data_path_str = self.performance_model.get("aerodynamic_trim_data_path")
        self.trim_data = None  # Reset
        if not trim_data_path_str:
            print("Warning: No aerodynamic_trim_data_path specified.")
            return

        trim_data_path = Path(trim_data_path_str)
        print(f"[Method] Loading trim data from {trim_data_path}...")
        if not trim_data_path.is_file():
            print(
                f"Warning: Trim data file not found at {trim_data_path}. Using dummy lookup."
            )
            # *** MODIFICATION: Setup dummy data if file not found ***
            self.trim_data = {
                "loaded": False,  # Indicate it's dummy data
                "path": str(trim_data_path),
                "lookup": lambda alt, spd, aoa: {
                    "Cl": 0.4 + aoa * 0.05,
                    "Cd": 0.05 + aoa * 0.005,
                    "Cm": 0.01,
                },  # Simplified dummy lookup
            }
            return
        try:
            print(f"  (Simulation) Parsing actual trim data file if it existed...")
            # Real implementation would load data here using pandas/numpy/scipy
            # For demo, setting up the dummy structure anyway if file existed but parsing failed (or wasn't implemented)
            self.trim_data = {
                "loaded": True,  # Indicate real data *attempted*
                "path": str(trim_data_path),
                "lookup": lambda alt, spd, aoa: {
                    "Cl": 0.4 + aoa * 0.05,
                    "Cd": 0.05 + aoa * 0.005,
                    "Cm": 0.01,
                },
            }
            print(f"  Trim data loaded successfully (simulated).")
        except Exception as e:
            print(f"Error loading or processing trim data from {trim_data_path}: {e}")
            self.trim_data = None  # Ensure it's None on error

    def get_aero_coefficients(
        self, altitude_m: float, speed_kts: float, aoa_deg: float
    ) -> Dict[str, float]:
        """
        Determine aerodynamic coefficients (Cl, Cd, Cm).
        Returns default values if trim data is not loaded/available.
        """
        default_coeffs = {"Cl": 0.3, "Cd": 0.06, "Cm": 0.0}  # Sensible defaults
        if not self.trim_data or not self.trim_data.get("lookup"):
            print(
                "Warning: Trim data/lookup not available. Returning default aero coefficients."
            )
            return default_coeffs
        # print(f"[Method] Interpolating trim data for state (alt={altitude_m:.0f}m, spd={speed_kts:.1f}kts, aoa={aoa_deg:.1f}deg)...")
        try:
            coeffs = self.trim_data["lookup"](altitude_m, speed_kts, aoa_deg)
            # print(f"  Interpolated Coeffs ({'dummy' if not self.trim_data.get('loaded') else 'real'}): {coeffs}")
            return coeffs
        except Exception as e:
            print(f"Error during trim data interpolation: {e}. Returning defaults.")
            return default_coeffs

    def calculate_required_thrust(
        self,
        environment: "Environment",
        altitude_m: float,
        speed_kts: float,
        flight_path_angle_deg: float = 0.0,
        bank_angle_deg: float = 0.0,
    ) -> float:
        """
        Determine thrust required. Returns estimated value even if trim data is missing.
        """
        print(
            f"[Method] Calculating required thrust for condition (alt={altitude_m:.0f}m, spd={speed_kts:.1f}kts, fpa={flight_path_angle_deg:.1f}deg, bank={bank_angle_deg:.1f}deg)..."
        )
        try:
            weight_n = self.current_state.current_weight_kg * 9.81
            rho = environment.get_air_density_kg_m3(altitude_m)
            speed_mps = speed_kts * 0.514444
            dynamic_pressure = 0.5 * rho * speed_mps**2
            wing_area_m2 = self.performance_model.get(
                "wing_area_m2", 1.0
            )  # Need wing area!

            # Use get_aero_coefficients to get Cd (will use defaults if needed)
            # We need an estimated AoA - for placeholder, assume a typical cruise AoA like 2 deg
            estimated_aoa = 2.0
            coeffs = self.get_aero_coefficients(altitude_m, speed_kts, estimated_aoa)
            cd = coeffs["Cd"]  # Use looked-up or default Cd

            drag = dynamic_pressure * wing_area_m2 * cd
            thrust_req = drag + weight_n * math.sin(math.radians(flight_path_angle_deg))
            # Add turn component T = q*S*Cd + W*sin(gamma) + L*sin(phi)*sin(chi_dot) roughly (ignore turn for simplicity)
            print(f"  Estimated Thrust Req (using Cd={cd:.3f}): {thrust_req:.2f} N")
            return thrust_req
        except Exception as e:
            print(f"Error calculating required thrust: {e}. Returning 0.0")
            return 0.0  # Return 0 on error

    def update_energy_reserves(self, energy_used: float) -> None:
        """Docstring unchanged..."""
        print(f"[Method] Updating energy reserves directly. Used: {energy_used}")
        if self.propulsion_type == "electric":
            self.current_state.remaining_energy -= energy_used
        elif self.propulsion_type == "fuel":
            fuel_burned_kg = energy_used
            self.current_state.remaining_energy -= fuel_burned_kg
            self.current_state.current_weight_kg -= fuel_burned_kg
        print(f"  New energy level: {self.current_state.remaining_energy}")


class Environment:
    def __init__(
        self,
        terrain_data_path: Optional[str],
        no_fly_zone_path: Optional[str],
        wind_model: Dict,
        atmosphere_model: str = "standard",
    ):
        self.wind_model: Dict = wind_model
        self.atmosphere_model: str = atmosphere_model
        self.terrain_data: Optional[Any] = None
        self.no_fly_zones: Optional[Any] = None
        if terrain_data_path:
            self.load_terrain(Path(terrain_data_path))
        if no_fly_zone_path:
            self.load_no_fly_zones(Path(no_fly_zone_path))

    def load_terrain(self, filepath: Path) -> None:
        """Loads terrain data. If file not found, sets up dummy structure."""
        print(f"[Method] Loading terrain data from {filepath}...")
        self.terrain_data = None  # Reset
        if not filepath.is_file():
            print(
                f"Warning: Terrain file not found at {filepath}. Using dummy elevation."
            )
            # *** MODIFICATION: Setup dummy data ***
            self.terrain_data = {
                "loaded": False,
                "path": str(filepath),
                "get_elevation": lambda lat, lon: 250.0 + (lat - 39.78) * 100,
            }  # Flat dummy terrain + slope
            return
        try:
            print(f"  (Simulation) Parsing actual terrain file if it existed...")
            # Real implementation uses rasterio/gdal
            self.terrain_data = {
                "loaded": True,
                "path": str(filepath),
                "get_elevation": lambda lat, lon: 250.0 + (lat - 39.78) * 100,
            }
            print("  Terrain data loaded successfully (simulated).")
        except Exception as e:
            print(f"Error loading terrain data: {e}")
            self.terrain_data = None

    def load_no_fly_zones(self, filepath: Path) -> None:
        """Loads NFZs. If file not found, sets up dummy structure."""
        print(f"[Method] Loading no-fly zones from {filepath}...")
        self.no_fly_zones = None  # Reset
        if not filepath.is_file():
            print(
                f"Warning: No-fly zone file not found at {filepath}. Assuming no NFZs."
            )
            # *** MODIFICATION: Setup dummy data ***
            self.no_fly_zones = {
                "loaded": False,
                "path": str(filepath),
                "geometries": [],
            }  # Empty list means no NFZs
            return
        try:
            print(f"  (Simulation) Parsing actual NFZ file if it existed...")
            # Real implementation uses geopandas/shapely
            self.no_fly_zones = {
                "loaded": True,
                "path": str(filepath),
                "geometries": [],
            }
            print("  No-fly zones loaded successfully (simulated - none defined).")
        except Exception as e:
            print(f"Error loading NFZ data: {e}")
            self.no_fly_zones = None

    def get_elevation_m_msl(self, location: Location) -> float:
        """Returns terrain elevation, using dummy data if real data not loaded."""
        if not self.terrain_data or not self.terrain_data.get("get_elevation"):
            return 0.0
        try:
            return self.terrain_data["get_elevation"](
                location.latitude_deg, location.longitude_deg
            )
        except Exception as e:
            print(f"Error querying terrain elevation: {e}")
            return 0.0

    def get_wind_vector(self, altitude_m: float) -> Tuple[float, float]:
        """Docstring unchanged..."""
        if self.wind_model.get("type") == "uniform":
            return (
                self.wind_model.get("speed_kts", 0.0),
                self.wind_model.get("direction_deg_from", 0.0),
            )
        return (0.0, 0.0)

    def is_in_no_fly_zone(self, location: Location) -> bool:
        """Checks NFZs, using dummy data if real data not loaded."""
        if not self.no_fly_zones or not self.no_fly_zones.get("geometries"):
            return False
        # Placeholder logic would involve shapely point-in-polygon tests here
        return False  # Assume safe

    def get_air_density_kg_m3(self, altitude_m: float) -> float:
        """Docstring unchanged..."""
        if self.atmosphere_model == "standard":
            try:
                T0 = 288.15
                L = 0.0065
                R = 287.058
                g = 9.80665
                temp_k = T0 - L * altitude_m
                rho = 1.225 * (temp_k / T0) ** ((g / (L * R)) - 1)
                return max(0.01, rho)
            except Exception:
                return 0.01
        else:
            return 1.225


class Maneuver:
    def __init__(self, maneuver_type: str):
        self.maneuver_type: str = maneuver_type

    def __repr__(self) -> str:
        return f"Maneuver(type={self.maneuver_type})"

    def is_feasible(
        self, current_state: UAVState, uav: UAV, environment: Environment
    ) -> bool:
        """Checks feasibility against operational limits."""
        print(
            f"[Method] Checking feasibility of {self.maneuver_type} from {current_state.velocity}..."
        )
        limits = uav.performance_model.get("operational_limits", {})
        min_speed = limits.get("min_airspeed_kts", 5.0)
        max_speed = limits.get("max_airspeed_kts", 100.0)

        # *** MODIFICATION: Allow feasibility check if speed is 0 (takeoff state) ***
        # This is a simplification; real takeoff needs specific checks.
        if (
            current_state.velocity.speed_kts < min_speed
            and current_state.velocity.speed_kts > 1e-6
        ):  # Check if non-zero but too slow
            print(
                f"  Feasibility failed: Speed {current_state.velocity.speed_kts} kts below min {min_speed} kts"
            )
            return False
        if current_state.velocity.speed_kts > max_speed:
            print(
                f"  Feasibility failed: Speed {current_state.velocity.speed_kts} kts above max {max_speed} kts"
            )
            return False
        # Add G-load, altitude checks etc. here
        print("  Feasibility check passed (basic).")
        return True

    def execute(
        self, start_state: UAVState, uav: UAV, environment: Environment, **kwargs
    ) -> Tuple[Optional[UAVState], Optional[Dict[str, Any]]]:
        """Simulates maneuver execution."""
        print(
            f"[Method] Executing maneuver {self.maneuver_type} from state {start_state}..."
        )
        if not self.is_feasible(start_state, uav, environment):
            print("  Maneuver is infeasible.")
            return None, None

        # Placeholder: Simple straight flight, assumes target state is achievable
        # Needs actual physics integration
        target_location = kwargs.get(
            "target_location", start_state.location
        )  # Get target if provided
        if (
            "target_speed_kts" in kwargs.keys()
            and kwargs.get("target_speed_kts") is not None
        ):
            target_speed_kts = kwargs.get("target_speed_kts")
        else:
            target_speed_kts = start_state.velocity.speed_kts or 30.0

        dist_m = start_state.location.distance_to(target_location)
        avg_speed_kts = (start_state.velocity.speed_kts + target_speed_kts) / 2.0
        avg_speed_mps = max(avg_speed_kts, 1.0) * 0.514444  # Ensure non-zero avg speed
        duration_s = dist_m / avg_speed_mps if avg_speed_mps > 0 else 0.0

        # Estimate costs
        energy_cost = start_state.calculate_energy_cost(
            self, uav, environment, duration_s
        )
        fuel_burned_kg = energy_cost if uav.propulsion_type == "fuel" else 0.0

        # Calculate next state (simplified)
        next_weight = start_state.current_weight_kg - fuel_burned_kg
        next_energy = start_state.remaining_energy - (
            energy_cost if uav.propulsion_type == "electric" else fuel_burned_kg
        )
        # Determine heading (simple)
        delta_lon = math.radians(
            target_location.longitude_deg - start_state.location.longitude_deg
        )
        start_lat_rad = math.radians(start_state.location.latitude_deg)
        target_lat_rad = math.radians(target_location.latitude_deg)
        y = math.sin(delta_lon) * math.cos(target_lat_rad)
        x = math.cos(start_lat_rad) * math.sin(target_lat_rad) - math.sin(
            start_lat_rad
        ) * math.cos(target_lat_rad) * math.cos(delta_lon)
        next_heading_rad = math.atan2(y, x)
        next_heading_deg = (math.degrees(next_heading_rad) + 360) % 360

        next_vel = VelocityVector(target_speed_kts, next_heading_deg)
        final_state = UAVState(target_location, next_vel, next_weight, next_energy)

        results = {
            "energy_cost": energy_cost,
            "fuel_burned_kg": fuel_burned_kg,
            "duration_s": duration_s,
            "next_location": target_location,
            "next_velocity": next_vel,
        }
        print(
            f"  Maneuver execution simulation complete. Result State: {final_state}, Results: {results}"
        )
        return final_state, results


class FlightPath:
    def __init__(self, waypoints: List[Waypoint]):
        self.waypoints: List[Waypoint] = waypoints
        self.maneuver_segments: List[Maneuver] = []
        self.states: List[UAVState] = []
        self.total_energy_cost: float = 0.0
        self.total_distance_m: float = 0.0
        self.total_flight_time_s: float = 0.0
        self.is_valid: bool = False

    def __repr__(self) -> str:
        valid_str = "Valid" if self.is_valid else "Invalid"
        return f"FlightPath(wpts={len(self.waypoints)}, E={self.total_energy_cost:.2f}, T={self.total_flight_time_s:.0f}s, {valid_str})"

    def compute_path_details(
        self, start_state: UAVState, uav: UAV, environment: Environment
    ) -> bool:
        """Calculates path details by simulating maneuvers between waypoints."""
        print("[Method] Computing full flight path details...")
        self.states = [start_state]
        self.maneuver_segments = []
        self.total_energy_cost = 0.0
        self.total_flight_time_s = 0.0
        self.total_distance_m = 0.0
        current_uav_state = start_state

        for i in range(len(self.waypoints)):
            target_waypoint = self.waypoints[i]
            print(
                f"  Simulating segment from {current_uav_state.location} to WP {target_waypoint.waypoint_id}..."
            )

            # Placeholder maneuver selection: use generic direct maneuver
            maneuver = Maneuver(f"DirectTo_{target_waypoint.waypoint_id}")
            self.maneuver_segments.append(maneuver)

            # Execute the placeholder maneuver targeting the waypoint
            next_state, results = maneuver.execute(
                current_uav_state,
                uav,
                environment,
                target_location=target_waypoint.location,  # Pass target location
                target_speed_kts=target_waypoint.target_speed_kts,  # Pass target speed if available
            )

            if next_state is None or results is None:
                print(
                    f"  Path computation failed: Maneuver to {target_waypoint.waypoint_id} infeasible."
                )
                self.is_valid = False
                return False

            segment_dist = current_uav_state.location.distance_to(
                next_state.location
            )  # Distance covered in segment
            self.total_energy_cost += results["energy_cost"]
            self.total_flight_time_s += results["duration_s"]
            self.total_distance_m += segment_dist
            self.states.append(next_state)
            current_uav_state = next_state

            if current_uav_state.remaining_energy < 0:
                print(
                    f"  Path computation failed: Ran out of energy reaching {target_waypoint.waypoint_id}."
                )
                self.is_valid = False
                return False

        print(
            f"  Path computation finished. Total Time: {self.total_flight_time_s:.0f}s, Energy: {self.total_energy_cost:.2f}, Dist: {self.total_distance_m:.0f}m"
        )
        self.is_valid = True
        return True

    def check_constraints(
        self, uav: UAV, environment: Environment, min_clearance: float
    ) -> bool:
        """Checks path against terrain, NFZ, and energy constraints."""
        print("[Method] Checking flight path constraints...")
        if not self.states:
            print("  Cannot check constraints: Path has no states.")
            return False

        for i, state in enumerate(self.states):
            # Skip start state for terrain/NFZ if desired, or check all
            terrain_alt = environment.get_elevation_m_msl(state.location)
            current_alt = state.location.altitude_m_msl
            if current_alt is None:
                print(f"  Constraint WARNING: State {i} has no altitude.")
                continue  # Skip check if alt is None

            clearance = current_alt - terrain_alt
            if clearance < min_clearance:
                print(
                    f"  Constraint VIOLATED: Terrain clearance {clearance:.1f}m < {min_clearance}m at state {i} {state.location}"
                )
                self.is_valid = False
                return False
            if environment.is_in_no_fly_zone(state.location):
                print(
                    f"  Constraint VIOLATED: Entered No-Fly Zone at state {i} {state.location}"
                )
                self.is_valid = False
                return False

        final_state = self.states[-1]
        # Allow zero energy exactly at the end for placeholder
        if final_state.remaining_energy < -1e-6:  # Use small tolerance
            print(
                f"  Constraint VIOLATED: Insufficient final energy ({final_state.remaining_energy:.2f})."
            )
            self.is_valid = False
            return False

        print("  All path constraints passed.")
        self.is_valid = True  # Set valid only if all checks pass
        return True


class PathPlanner:
    def __init__(self, environment: Environment, settings: SpecsDict):
        self.environment: Environment = environment
        self.settings: SpecsDict = settings

    def find_optimal_path(
        self, uav: UAV, waypoints: List[Waypoint]
    ) -> Optional[FlightPath]:
        """Finds an optimal path (placeholder: finds the first valid path)."""
        print("[Method] Searching for the optimal flight path...")
        # Placeholder: Just compute one direct path and check it
        candidate_path = FlightPath(waypoints)
        computation_ok = candidate_path.compute_path_details(
            uav.current_state, uav, self.environment
        )

        if not computation_ok:
            print("Optimal path search failed: Initial path computation unsuccessful.")
            return None

        constraints_ok = candidate_path.check_constraints(
            uav, self.environment, self.settings.get("min_terrain_clearance_m", 50.0)
        )

        if constraints_ok:
            print(
                f"Optimal path found (simulated) with cost: {candidate_path.total_energy_cost:.2f}"
            )
            return candidate_path
        else:
            print(
                "Optimal path search failed: Candidate path failed constraint checks."
            )
            return None


def load_specs_from_json(filepath: Path) -> Optional[Dict[str, SpecsDict]]:
    # ... (Keep previous implementation, error handling is good)
    print(f"Attempting to load configuration from: {filepath}")
    try:
        with filepath.open("r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                print("Configuration loaded successfully.")
                return data
            else:
                print("Error: JSON root is not a dictionary.")
                return None
    except FileNotFoundError:
        print(f"Error: Config file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading JSON: {e}")
        return None


# --- Main Execution Function & Example Usage ---
def main() -> None:
    config_path = Path(DEFAULT_CONFIG_FILE)
    state_path = Path(DEFAULT_STATE_FILE)

    print("--- UAV Pre-Flight Checker v3 (Runnable Demo) ---")
    uav_systems_state_representation = (
        {}
    )  # Using a placeholder name, conceptually holds UAV object(s)
    # NOTE: The current structure focuses on ONE UAV plan, not multiple systems check like before.
    # We'll initialize the single UAV directly here for the planning example.

    # 1. Load Config (Simplified for single UAV plan)
    print(f"Loading mission config from {config_path}...")
    config_data = load_specs_from_json(
        config_path
    )  # Reusing helper, though structure differs slightly
    if config_data is None:
        print("Fatal: Could not load config JSON. Creating dummy data for demo.")
        # Create dummy data if JSON load fails, using previous example structure
        mission_name = "Dummy_Mission"
        waypoints_data = [
            {
                "id": "WP0",
                "latitude_deg": 39.780,
                "longitude_deg": -84.050,
                "altitude_m_msl": 250,
            },
            {
                "id": "WP1",
                "latitude_deg": 39.785,
                "longitude_deg": -84.055,
                "altitude_m_msl": 300,
            },
        ]
        uav_config = {
            "uav_id": "Dummy_UAV",
            "propulsion_type": "electric",
            "initial_state": {
                "latitude_deg": 39.780,
                "longitude_deg": -84.050,
                "altitude_m_msl": 250,
                "speed_kts": 20.0,
                "heading_deg": 90,
            },
            "performance_model": {
                "dry_mass_kg": 10.0,
                "payload_mass_kg": 2.0,
                "wing_area_m2": 1.0,
                "battery_capacity_wh": 1000.0,
                "initial_charge_wh": 950.0,
                "aerodynamic_trim_data_path": "dummy_trim.csv",
                "operational_limits": {
                    "min_airspeed_kts": 15.0,
                    "max_airspeed_kts": 80.0,
                },
            },
        }
        env_config = {
            "terrain_data_path": "dummy_terrain.tif",
            "no_fly_zone_path": "dummy_nfz.geojson",
            "wind_model": {"type": "none"},
            "atmosphere_model": "standard",
        }
        planner_settings = {
            "optimization_goal": "min_energy",
            "min_terrain_clearance_m": 50.0,
        }
    else:
        # Extract from loaded JSON
        mission_name = config_data.get("mission_definition", {}).get(
            "mission_name", "Unnamed Mission"
        )
        waypoints_data = config_data.get("mission_definition", {}).get("waypoints", [])
        uav_config = config_data.get("uav_configuration", {})
        env_config = config_data.get("environment_data", {})
        planner_settings = config_data.get(
            "planner_settings",
            {"optimization_goal": "min_energy", "min_terrain_clearance_m": 50.0},
        )

    # 2. Create Objects
    if not waypoints_data or not uav_config:
        print("Fatal: Missing mission waypoints or UAV configuration in JSON/defaults.")
        sys.exit(1)

    waypoints = [
        Waypoint(
            wp["id"],
            Location(wp["latitude_deg"], wp["longitude_deg"], wp.get("altitude_m_msl")),
            wp.get("target_speed_kts"),
        )
        for wp in waypoints_data
    ]
    initial_state_data = uav_config.get("initial_state", {})
    perf_model = uav_config.get("performance_model", {})
    initial_location = Location(
        initial_state_data.get("latitude_deg", 0),
        initial_state_data.get("longitude_deg", 0),
        initial_state_data.get("altitude_m_msl"),
    )
    # *** MODIFICATION: Start airborne for demo ***
    initial_velocity = VelocityVector(
        initial_state_data.get("speed_kts", 20.0),
        initial_state_data.get("heading_deg", 0),
    )  # Default to 20 kts if 0
    initial_weight = perf_model.get("dry_mass_kg", 10.0) + perf_model.get(
        "payload_mass_kg", 0.0
    )
    initial_energy = (
        perf_model.get("initial_charge_wh")
        if uav_config.get("propulsion_type") == "electric"
        else perf_model.get("initial_fuel_kg", 10.0)
    )

    initial_uav_state = UAVState(
        initial_location, initial_velocity, initial_weight, initial_energy or 1000.0
    )  # Provide default energy if missing

    uav = UAV(
        uav_config.get("uav_id", "Default_UAV"),
        uav_config.get("propulsion_type", "electric"),
        perf_model,
        initial_uav_state,
    )
    environment = Environment(
        env_config.get("terrain_data_path"),
        env_config.get("no_fly_zone_path"),
        env_config.get("wind_model", {"type": "none"}),
        env_config.get("atmosphere_model", "standard"),
    )
    planner = PathPlanner(environment, planner_settings)

    print("\n--- Objects Initialized ---")
    print(f"Mission: {mission_name}")
    print(f"UAV: {uav}")
    print(
        f"Environment Wind: {environment.get_wind_vector(initial_location.altitude_m_msl or 250)}"
    )

    # 3. Run Planner
    print("\n--- Running Path Planner ---")
    optimal_flight_path = planner.find_optimal_path(uav, waypoints)

    # 4. Display Results
    print("\n--- Planning Complete ---")
    if optimal_flight_path:
        print(f"Optimal Path Found: {optimal_flight_path}")
        # You could add a method to FlightPath to print its details
        # optimal_flight_path.display_details()
    else:
        print("Failed to find a valid flight path.")


# --- Script Entry Point with Output Redirection ---
if __name__ == "__main__":
    output_filename = "discussion3_output.txt"

    # Store the original standard output
    original_stdout = sys.stdout

    print(
        f"--- Script execution started. Output will be saved to {output_filename} ---"
    )

    try:
        # Open the output file in write mode ('w').
        # This will overwrite the file if it already exists. Use 'a' to append.
        with open(output_filename, "w") as f:
            # Redirect standard output to the file
            sys.stdout = f

            # Execute the main application logic
            # All print() statements within main() and the functions it calls
            # will now write to the file instead of the console.
            main()

    except Exception as e:
        # If any error occurs, print it to the original stdout (console)
        # and also try to write it to the file before closing.
        sys.stdout = original_stdout  # Restore stdout to show error on console
        print(
            f"\n--- An error occurred during execution: ---", file=sys.stderr
        )  # Print error to stderr
        print(str(e), file=sys.stderr)
        # Try to write the error to the file as well
        try:
            with open(output_filename, "a") as f_err:  # Append error
                print(f"\n--- An error occurred during execution: ---", file=f_err)
                print(str(e), file=f_err)
        except Exception:
            pass  # Ignore errors during error logging

    finally:
        # --- Crucial Step: Restore original standard output ---
        # This ensures that any print statements *after* this block,
        # or if you run other commands in the same terminal,
        # will go back to the console.
        sys.stdout = original_stdout

    print(f"--- Script execution finished. Output saved to {output_filename} ---")
