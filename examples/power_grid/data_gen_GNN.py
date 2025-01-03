import pandapower as pp
import pandas as pd
import pandapower.networks as nw
import os
import numpy as np
from pandapower.pypower.makeYbus import makeYbus

from scipy.sparse import csc_matrix

# Function to generate pandapower data and print Ybus matrix
def generate_pandapower_data(num_cases, output_dir="output_files"):

    cases = ['case14', 'case300', 'case6470orte']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    count = 1

    for case in cases:

        os.makedirs(output_dir + '/' + case, exist_ok=True)

        for case_num in range(1, num_cases + 1):

            # Reset the network to the original case
            if case == "case14":
                net = nw.case14()
            elif case == "case300":
                net = nw.case300()
            elif case == "case6470rte":
                net = nw.case6470rte()

            # Identify bus types
            slack_bus = net.ext_grid.bus.values  # Slack bus indices
            gen_buses = net.gen.bus.values  # Generator bus indices
            load_buses = net.load.bus.values  # Load bus indices

            # Randomly vary the load (Pd and Qd) within ±10%
            for idx in load_buses:
                net.load.loc[net.load.bus == idx, 'p_mw'] *= (0.9 + 0.2 * np.random.rand())
                net.load.loc[net.load.bus == idx, 'q_mvar'] *= (0.9 + 0.2 * np.random.rand())

            try:
                # Run power flow
                pp.runpp(net)

                # Generate the admittance matrix Ybus
                baseMVA, bus, branch = net._ppc['baseMVA'], net._ppc['bus'], net._ppc['branch']
                Ybus, _, _ = makeYbus(baseMVA, bus, branch)

                # Extract rows and cols (non-zero entries)
                rows, cols = Ybus.nonzero()

                # Compute the impedance matrix (Zbus) by calculating the entry-wise inverse of the admittance matrix Zbus
                Zbus_numpyarray = 1.0 / Ybus.data

                # Compute the conductance matrix, which is the real part of the admittance
                Gbus_numpyarray = Ybus.data.real
                Gbus = csc_matrix((Gbus_numpyarray, Ybus.indices, Ybus.indptr), shape=Ybus.shape)

                # Compute the susceptance matrix, which is the imaginary part of the admittance
                Bbus_numpyarray = Ybus.data.imag
                Bbus = csc_matrix((Bbus_numpyarray, Ybus.indices, Ybus.indptr), shape=Ybus.shape)

                # Compute the resistance matrix Rbus by calculating the real part of each entry of Zbus
                Rbus_numpyarray = Zbus_numpyarray.real
                Rbus = csc_matrix((Rbus_numpyarray, Ybus.indices, Ybus.indptr), shape=Ybus.shape)

                # Compute the reactance matrix Xbus by calculating the real part of each entry of Zbus
                Xbus_numpyarray = Zbus_numpyarray.imag
                Xbus = csc_matrix((Xbus_numpyarray, Ybus.indices, Ybus.indptr), shape=Ybus.shape)

                # Filter out self-loops
                mask = rows != cols
                rows_no_self_loops = rows[mask]
                cols_no_self_loops = cols[mask]

                # Save binary adjacency matrix rows and cols to a .npz file
                np.savez(os.path.join(output_dir, case, f"{case}_Instance{case_num}_adjacency_binary_matrix.npz"), rows=rows_no_self_loops, cols=cols_no_self_loops)

                # Save conductance matrix rows and cols to a .npz file
                np.savez(os.path.join(output_dir, case, f"{case}_Instance{case_num}_conductance_matrix.npz"), data=Gbus.data,
                         indices=Gbus.indices, indptr=Gbus.indptr, shape=Gbus.shape)

                # Save susceptance matrix rows and cols to a .npz file
                np.savez(os.path.join(output_dir, case, f"{case}_Instance{case_num}_susceptance_matrix.npz"), data=Bbus.data,
                         indices=Bbus.indices, indptr=Bbus.indptr, shape=Bbus.shape)

                # Save resistance matrix rows and cols to a .npz file
                np.savez(os.path.join(output_dir, case, f"{case}_Instance{case_num}_resistance_matrix.npz"), data=Rbus.data, indices=Rbus.indices, indptr=Rbus.indptr, shape=Rbus.shape)

                # Save reactance matrix rows and cols to a .npz file
                np.savez(os.path.join(output_dir, case, f"{case}_Instance{case_num}_reactance_matrix.npz"), data=Xbus.data, indices=Xbus.indices, indptr=Xbus.indptr, shape=Xbus.shape)

                # Collect data for each bus
                bus_data = []
                for bus_idx in range(len(net.bus)):
                    # Determine bus type
                    if bus_idx in slack_bus:
                        bus_type = "Slack"
                    elif bus_idx in gen_buses:
                        bus_type = "PV Bus"  # Generator buses (active generation)
                    elif bus_idx in load_buses:
                        bus_type = "PQ Bus"  # Load buses
                    else:
                        bus_type = "PQ Bus"  # Transit buses, treated as PQ for simplicity

                    # Get power injections and voltage data
                    pin = net.res_bus.p_mw.at[bus_idx]
                    qin = net.res_bus.q_mvar.at[bus_idx]
                    vmag = net.res_bus.vm_pu.at[bus_idx]
                    vang = net.res_bus.va_degree.at[bus_idx]

                    # Append data for this bus
                    bus_data.append([bus_type, pin, qin, vmag, vang])

                # Create a DataFrame for the current case
                df = pd.DataFrame(bus_data, columns=["Bus Type", "Pin", "Qin", "Vmag", "Vang"], index=[f"Node{i+1}" for i in range(len(bus_data))])

                # Save the DataFrame to a CSV file
                output_file = os.path.join(output_dir, case, f"{case}_Instance{case_num}.csv")
                df.to_csv(output_file, index_label="Node")
                print(f"{case}: Saved results for Instance {case_num} to {output_file}/{case}")

            except pp.powerflow.LoadflowNotConverged:
                print(f"{case}: Power flow did not converge for Instance {case_num}")

            count = count+1

# Parameters
num_cases = 1000  # Number of runs

# Run the function
generate_pandapower_data(num_cases)
