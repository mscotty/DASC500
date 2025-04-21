class FluidObject:
    def __init__(self, area, drag_coeff, fluid_density):
        self.area_m2 = area  # Cross-sectional area (m^2)
        self.drag_coeff = drag_coeff  # Drag coefficient (dimensionless)
        self.fluid_density_kg_m3 = fluid_density  # Fluid density (kg/m^3)

    def compute_drag(self, velocity_m_s):
        return 0.5 * self.fluid_density_kg_m3 * velocity_m_s**2 * self.drag_coeff * self.area_m2

# Example usage
airfoil = FluidObject(area=0.5, drag_coeff=0.3, fluid_density=1.225)
drag_force = airfoil.compute_drag(50)  # Velocity in m/s
print(f"Drag Force: {drag_force:.2f} [N]")
