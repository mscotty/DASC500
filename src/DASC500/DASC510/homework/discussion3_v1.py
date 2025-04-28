# Import necessary types for type hinting
from typing import List, Dict, Optional, Any
import math

from DASC500.utilities.print.redirect_print import redirect_print

# --- Function 1: Calculate Reynolds Number ---
def calculate_reynolds(
    density: float,
    velocity: float,
    characteristic_length: float,
    dynamic_viscosity: float
) -> float:
    """
    Calculates the Reynolds number (Re), a dimensionless quantity used to
    predict flow patterns in fluid dynamics.

    The formula used is: $Re = \frac{\rho V L}{\mu}$

    Args:
        density: Density of the fluid (e.g., in $kg/m^3$). Must be positive.
        velocity: Characteristic velocity of the flow (e.g., in $m/s$).
                  Typically positive magnitude.
        characteristic_length: A characteristic linear dimension
                               (e.g., pipe diameter, airfoil chord, in $m$).
                               Must be positive.
        dynamic_viscosity: Dynamic viscosity (mu) of the fluid
                           (e.g., in $Pa \cdot s$ or $kg/(m \cdot s)$).
                           Must be strictly positive.

    Returns:
        The calculated Reynolds number (dimensionless). Returns `float('nan')`
        if any input is invalid (non-positive density, length, or viscosity).
    """
    if density <= 0:
        print(f"Warning: Density ({density}) must be positive.")
        return float('nan')
    if characteristic_length <= 0:
        print(f"Warning: Characteristic length ({characteristic_length}) must be positive.")
        return float('nan')
    if dynamic_viscosity <= 0:
        print(f"Warning: Dynamic viscosity ({dynamic_viscosity}) must be strictly positive.")
        return float('nan')

    # Calculate Reynolds number
    reynolds_num = (density * velocity * characteristic_length) / dynamic_viscosity
    return reynolds_num

# --- Function 2: Determine Flow Properties ---
def determine_flow_properties(
    reynolds_number: float
) -> Optional[Dict[str, str]]:
    """
    Determines the likely flow regime and suggests a general turbulence model
    category based on the Reynolds number.

    Note: The thresholds used (2300, 4000) are typical for flow inside a
    circular pipe. Thresholds for other geometries (e.g., flow over a flat plate)
    can differ significantly.

    Args:
        reynolds_number: The Reynolds number of the flow.

    Returns:
        A dictionary containing the 'flow_regime' (str) and
        'turbulence_model_suggestion' (str), or None if the input
        Reynolds number is invalid (e.g., NaN).
    """
    # Check for invalid Reynolds number (NaN or negative, though negative Re is unusual)
    if math.isnan(reynolds_number) or reynolds_number < 0:
        print(f"Cannot determine flow properties for invalid Reynolds number: {reynolds_number}")
        return None

    # Define flow regime thresholds (typical for internal pipe flow)
    laminar_limit: float = 2300.0
    turbulent_limit: float = 4000.0

    # Use if/elif/else to determine regime and suggest model type
    if reynolds_number < laminar_limit:
        flow_regime = 'Laminar'
        # For very low Re, DNS might be feasible. Otherwise, assume laminar flow equations apply.
        model_suggestion = 'None (Laminar flow equations apply)'
    elif reynolds_number >= laminar_limit and reynolds_number < turbulent_limit:
        flow_regime = 'Transitional'
        # Transitional flows are complex and often require specialized models.
        model_suggestion = 'Specialized Transitional models (e.g., Gamma-Re_theta) or Scale-Resolving Simulations (LES/DES)'
    else: # reynolds_number >= turbulent_limit
        flow_regime = 'Turbulent'
        # Standard choice for many engineering applications. Model choice depends on specifics.
        model_suggestion = 'Reynolds-Averaged Navier-Stokes (RANS) (e.g., k-epsilon, k-omega SST) or Scale-Resolving Simulations (LES/DES)'

    # Use a dictionary to store and return the results
    properties: Dict[str, str] = {
        'flow_regime': flow_regime,
        'turbulence_model_suggestion': model_suggestion
    }
    return properties

# --- Function 3: Process CFD Simulation Zones ---
def process_cfd_zones(
    zone_names_list: List[str],
    default_settings_by_type: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Processes a list of CFD zone names, attempts to determine their type based
    on common naming conventions, and associates them with provided default settings.

    Args:
        zone_names_list: A list of strings, where each string is a zone name
                           (e.g., ['inlet_velocity', 'outlet_pressure', 'wall_wing']).
        default_settings_by_type: A dictionary where keys are simplified zone types
                                  (e.g., 'inlet', 'outlet', 'wall', 'symmetry')
                                  and values are dictionaries of settings to apply.

    Returns:
        A dictionary where keys are the original zone names from the input list,
        and values are dictionaries containing the 'detected_type' (str) and
        'applied_settings' (dict or None if no match found).
    """
    print("\n--- Processing CFD Zones ---")
    processed_zones_info: Dict[str, Dict[str, Any]] = {}

    # Input validation
    if not isinstance(zone_names_list, list):
        print("Error: zone_names_list must be a list.")
        return processed_zones_info # Return empty dict on error
    if not isinstance(default_settings_by_type, dict):
        print("Error: default_settings_by_type must be a dictionary.")
        return processed_zones_info # Return empty dict on error

    # Use a for loop to iterate through the list of zone names
    for zone_name in zone_names_list:
        print(f"Processing zone: '{zone_name}'")
        detected_type: str = 'unknown'
        applied_settings: Optional[Dict[str, Any]] = None

        # Simple logic to guess zone type from name (case-insensitive)
        # More robust implementation might use regex or specific prefix/suffix rules.
        zone_name_lower = zone_name.lower()
        if 'inlet' in zone_name_lower:
            detected_type = 'inlet'
        elif 'outlet' in zone_name_lower:
            detected_type = 'outlet'
        elif 'wall' in zone_name_lower:
            detected_type = 'wall'
        elif 'symmetry' in zone_name_lower:
            detected_type = 'symmetry'
        elif 'interface' in zone_name_lower:
            detected_type = 'interface'
        # Add more rules as needed...

        # Use the dictionary parameter to get settings based on detected type
        if detected_type in default_settings_by_type:
            applied_settings = default_settings_by_type[detected_type]
            print(f"  Type detected: '{detected_type}' -> Applying settings: {applied_settings}")
        else:
            print(f"  Type detected: '{detected_type}'. No default settings found in provided dictionary.")

        # Store results for this zone
        processed_zones_info[zone_name] = {
            'detected_type': detected_type,
            'applied_settings': applied_settings
        }

    print("--- Zone Processing Complete ---")
    return processed_zones_info


if __name__ == "__main__":
    redirect_print('discussion3_logger.txt', also_to_stdout=True)
    
    # --- Example 1: Water Flow in a Pipe ---
    print("--- Example 1: Water Flow in a Pipe ---")
    density_water = 998.2 # kg/m^3
    viscosity_water = 1.002e-3 # Pa*s
    velocity_pipe = 0.5 # m/s
    diameter_pipe = 0.1 # m

    # Call function 1 and assign returned value
    re_water_pipe = calculate_reynolds(density_water, velocity_pipe, diameter_pipe, viscosity_water)
    print(f"Calculated Reynolds number for water in pipe: {re_water_pipe:.2f}")

    # Call function 2 using the result from function 1, assign returned value
    flow_props_water = determine_flow_properties(re_water_pipe)
    if flow_props_water: # Check if the function returned a valid dictionary
        print(f"Flow properties: {flow_props_water}")
    else:
        print("Could not determine flow properties for water.")

    # --- Example 2: Air Flow over a Wing Section ---
    print("\n--- Example 2: Air Flow over a Wing Section ---")
    density_air = 1.225 # kg/m^3
    viscosity_air = 1.81e-5 # Pa*s
    velocity_wing = 50 # m/s
    chord_length_wing = 1.5 # m

    # Call function 1 and assign returned value
    re_air_wing = calculate_reynolds(density_air, velocity_wing, chord_length_wing, viscosity_air)
    # Using f-string formatting for scientific notation
    print(f"Calculated Reynolds number for air over wing: {re_air_wing:.3e}")

    # Call function 2 using the result from function 1, assign returned value
    flow_props_air = determine_flow_properties(re_air_wing)
    if flow_props_air:
        print(f"Flow properties: {flow_props_air}")
    else:
        print("Could not determine flow properties for air.")


    # --- Example 3: Processing Simulation Zones ---
    # Define the list of zone names for function 3
    simulation_zones: List[str] = ['VELOCITY_INLET_MAIN', 'PRESSURE_OUTLET_EXIT', 'WALL_AEROFOIL', 'SYMMETRY_MIDPLANE', 'FLUID_INTERIOR']

    # Define the dictionary of default settings for function 3
    boundary_defaults: Dict[str, Dict[str, Any]] = {
        'inlet': {'BC_Type': 'velocity-inlet', 'Velocity': [10.0, 0.0, 0.0], 'TurbulenceModel': 'k-omega SST'},
        'outlet': {'BC_Type': 'pressure-outlet', 'GaugePressure': 0.0},
        'wall': {'BC_Type': 'wall', 'Motion': 'Stationary', 'Shear': 'No-slip'},
        'symmetry': {'BC_Type': 'symmetry'},
        # Note: No default for 'interface' or 'unknown' types defined here
    }

    # Call function 3 with the list and dictionary, assign returned value
    processed_zone_data = process_cfd_zones(simulation_zones, boundary_defaults)
    print("\n--- Returned Zone Data ---")
    # Pretty print the returned dictionary
    import json
    print(json.dumps(processed_zone_data, indent=2))


    # --- Example 4: Invalid Input Check ---
    print("\n--- Example 4: Invalid Input Checks ---")
    # Test invalid viscosity
    re_invalid_visc = calculate_reynolds(1000, 10, 1, 0)
    print(f"Result with zero viscosity: {re_invalid_visc}")
    # Test invalid length
    re_invalid_len = calculate_reynolds(1000, 10, -1, 0.01)
    print(f"Result with negative length: {re_invalid_len}")

    # Test passing NaN to flow properties
    flow_props_invalid = determine_flow_properties(re_invalid_visc) # Pass the NaN result
    if flow_props_invalid is None:
        print("Flow properties determination handled NaN input correctly.")

    # Test passing invalid list to zone processing
    invalid_zone_call = process_cfd_zones("not_a_list", boundary_defaults)
    print(f"Result from calling process_cfd_zones with invalid list: {invalid_zone_call}")