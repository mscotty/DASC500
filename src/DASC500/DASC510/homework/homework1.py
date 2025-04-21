from graphviz import Digraph

# Create a top-down activity diagram
diagram = Digraph(format='png')
diagram.attr(rankdir='TB', size='10,20')

# Nodes
diagram.node("Start", "Start: Initialize Mission Profile")
diagram.node("LoadEnv", "Load Environment Data\n(Terrain, Wind, No-Fly Zones)")
diagram.node("SelectUAV", "Select UAV Model\n(Electric or Gas-Powered)")
diagram.node("CheckPowerType", "Is UAV Electric?")
diagram.node("ModelBattery", "Estimate Battery Usage via Trim Data")
diagram.node("ModelFuel", "Estimate Fuel Usage via Trim Data")
diagram.node("DefineProfile", "Define Mission Profile\n(Waypoints + Maneuvers)")
diagram.node("GeneratePaths", "Generate Candidate Paths")
diagram.node("EvaluatePath", "Evaluate Path: Energy, Constraints, Efficiency")
diagram.node("PathValid", "Is Path Valid?")
diagram.node("RevisePath", "Revise Parameters or Try Alternative Path")
diagram.node("Success", "Success: Optimal Path Found")
diagram.node("Fail", "Fail: No Feasible Path After Attempts")
diagram.node("End", "End")

# Edges
diagram.edge("Start", "LoadEnv")
diagram.edge("LoadEnv", "SelectUAV")
diagram.edge("SelectUAV", "CheckPowerType")
diagram.edge("CheckPowerType", "ModelBattery", label="Yes")
diagram.edge("CheckPowerType", "ModelFuel", label="No")
diagram.edge("ModelBattery", "DefineProfile")
diagram.edge("ModelFuel", "DefineProfile")
diagram.edge("DefineProfile", "GeneratePaths")
diagram.edge("GeneratePaths", "EvaluatePath")
diagram.edge("EvaluatePath", "PathValid")
diagram.edge("PathValid", "Success", label="Yes")
diagram.edge("PathValid", "RevisePath", label="No")
diagram.edge("RevisePath", "GeneratePaths")
diagram.edge("Success", "End")
diagram.edge("Fail", "End")
diagram.edge("RevisePath", "Fail", label="Too Many Failed Attempts?", style="dashed")

# Save the diagram
file_path = "UAV_FlightPath_Optimized"
diagram.render(file_path, cleanup=True)
