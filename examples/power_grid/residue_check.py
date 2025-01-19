import pandapower as pp
import pandapower.networks as pn
import numpy as np

# Helper function to calculate P and Q injections using exact formula
def calculate_injections_exact(net):
    voltage = net.res_bus.vm_pu.values
    angle_rad = np.deg2rad(net.res_bus.va_degree.values)
    V = voltage

    # Get system data
    num_buses = len(voltage)
    P_calc = np.zeros(num_buses)
    Q_calc = np.zeros(num_buses)

    # Get branch data for conductance (G) and susceptance (B)
    Ybus = net._ppc["internal"]["Ybus"].todense()
    G = np.real(Ybus)
    B = np.imag(Ybus)

    # Calculate P and Q using exact formulas
    for i in range(num_buses):
        for j in range(num_buses):
            P_calc[i] += (
                V[i]
                * V[j]
                * (
                    G[i, j] * np.cos(angle_rad[i] - angle_rad[j])
                    + B[i, j] * np.sin(angle_rad[i] - angle_rad[j])
                )
            )
            Q_calc[i] += (
                V[i]
                * V[j]
                * (
                    G[i, j] * np.sin(angle_rad[i] - angle_rad[j])
                    - B[i, j] * np.cos(angle_rad[i] - angle_rad[j])
                )
            )

    # Scale from per-unit to MW using system base power
    S_base = net.sn_mva
    P_calc *= S_base
    Q_calc *= S_base

    return P_calc, Q_calc


# Helper function to calculate P and Q injections using exact formula
def compute_power_flow_residual(net):

    # Extract power flow results from pandapower
    P_flow = -net.res_bus.p_mw.values  # Adjust sign for injection convention
    Q_flow = -net.res_bus.q_mvar.values

    voltage = net.res_bus.vm_pu.values
    angle_rad = np.deg2rad(net.res_bus.va_degree.values)
    V = voltage

    # Get system data
    num_buses = len(voltage)
    P_calc = np.zeros(num_buses)
    Q_calc = np.zeros(num_buses)

    # Get branch data for conductance (G) and susceptance (B)
    Ybus = net._ppc["internal"]["Ybus"].todense()
    G = np.real(Ybus)
    B = np.imag(Ybus)

    # Calculate P and Q using exact formulas
    for i in range(num_buses):
        for j in range(num_buses):
            P_calc[i] += (
                V[i]
                * V[j]
                * (
                    G[i, j] * np.cos(angle_rad[i] - angle_rad[j])
                    + B[i, j] * np.sin(angle_rad[i] - angle_rad[j])
                )
            )
            Q_calc[i] += (
                V[i]
                * V[j]
                * (
                    G[i, j] * np.sin(angle_rad[i] - angle_rad[j])
                    - B[i, j] * np.cos(angle_rad[i] - angle_rad[j])
                )
            )

    # Scale from per-unit to MW using system base power
    S_base = net.sn_mva
    P_calc *= S_base
    Q_calc *= S_base

    r_P = P_flow - P_calc
    r_Q = Q_flow - Q_calc

    return r_P, r_Q


# Helper function to run power flow and compare results
def run_and_compare(net, load_factor):
    # Scale loads for the new condition
    net.load.p_mw *= load_factor
    net.load.q_mvar *= load_factor

    # Run power flow
    pp.runpp(net)

    # Calculate P and Q injections using exact formula
    P_calc, Q_calc = calculate_injections_exact(net)

    # Extract power flow results from pandapower
    P_flow = -net.res_bus.p_mw.values  # Adjust sign for injection convention
    Q_flow = -net.res_bus.q_mvar.values

    r_P_flow, r_Q_flow = compute_power_flow_residual(net)

    # Compare calculated and power flow results
    print("Comparison for P injections (MW):")
    print(f"Calculated: {P_calc}")
    print(f"Power Flow (adjusted for injection convention): {P_flow}\n")
    print(f"Residual for Power Flow (adjusted for injection convention): {r_P_flow}\n")

    print("Comparison for Q injections (MVAR):")
    print(f"Calculated: {Q_calc}")
    print(f"Power Flow (adjusted for injection convention): {Q_flow}\n")
    print(f"Power Flow (adjusted for injection convention): {P_flow}\n")
    print(f"Residual for Power Flow (adjusted for injection convention): {r_Q_flow}\n")

    # Reset loads for next simulation
    net.load.p_mw /= load_factor
    net.load.q_mvar /= load_factor


# Load IEEE 14-bus and IEEE 300-bus systems
net_14 = pn.case14()

# Disable the shunt element at bus 8
net_14.shunt.loc[net_14.shunt.bus == 8, "p_mw"] = 0
net_14.shunt.loc[net_14.shunt.bus == 8, "q_mvar"] = 0

print("Running for IEEE 14-bus case:")
run_and_compare(net_14, load_factor=1.2)  # 20% increase in load
