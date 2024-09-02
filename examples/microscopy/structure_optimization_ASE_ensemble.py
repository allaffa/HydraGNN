import numpy
import torch
from torch_geometric.data import Data
from ase.calculators.calculator import Calculator, all_changes
from hydragnn.preprocess.utils import get_radius_graph_pbc
from torch_geometric.transforms import LocalCartesian
import json, os
import logging
from mpi4py import MPI
import argparse
from ase.optimize import BFGS, FIRE  # or any other optimizer
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.io import read, write
import hydragnn
from hydragnn.utils.time_utils import Timer
from copy import deepcopy

from ensemble_utils import model_ensemble

transform_coordinates = LocalCartesian(norm=False, cat=False)


def get_log_name_config(config):
    return (
        config["NeuralNetwork"]["Architecture"]["model_type"]
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-ncl-"
        + str(config["NeuralNetwork"]["Architecture"]["num_conv_layers"])
        + "-hd-"
        + str(config["NeuralNetwork"]["Architecture"]["hidden_dim"])
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
        + "-node_ft-"
        + "".join(
            str(x)
            for x in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        )
        + "-task_weights-"
        + "".join(
            str(weigh) + "-"
            for weigh in config["NeuralNetwork"]["Architecture"]["task_weights"]
        )
    )


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


class PyTorchCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, hydragnn_model):
        Calculator.__init__(self)
        self.model = hydragnn_model

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = atoms.get_positions()
        positions_tensor = torch.tensor(
            positions, requires_grad=False, dtype=torch.float
        )

        # Extract atomic numbers
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_numbers_torch = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(
            1
        )

        x = torch.cat((atomic_numbers_torch, positions_tensor), dim=1)

        # Create the torch_geometric data object
        data_object = Data(
            pos=positions_tensor,
            x=x,
            supercell_size=torch.tensor(atoms.cell.array).float(),
        )

        add_edges_pbc = get_radius_graph_pbc(
            radius=config["NeuralNetwork"]["Architecture"]["radius"], max_neighbours=20
        )
        data_object = add_edges_pbc(data_object)

        data_object = transform_coordinates(data_object)

        pred_mean, pred_std = self.model(data_object, meanstd=True)

        self.results["energy"] = pred_mean[0].item()
        self.results["forces"] = pred_mean[1].detach().numpy()


if __name__ == "__main__":

    modelname = "MO2"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir_folder", help="folder of trained models", type=str, default=None
    )
    parser.add_argument("--log", help="log name", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pickle",
        help="Pickle gan_dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )

    args = parser.parse_args()

    modeldirlist = [
        os.path.join(args.models_dir_folder, name)
        for name in os.listdir(args.models_dir_folder)
        if os.path.isdir(os.path.join(args.models_dir_folder, name))
    ]

    var_config = None
    for modeldir in modeldirlist:
        input_filename = os.path.join(modeldir, "config.json")
        with open(input_filename, "r") as f:
            config = json.load(f)
        if var_config is not None:
            assert var_config == config["NeuralNetwork"]["Variables_of_interest"], (
                "Inconsistent variable config in %s" % input_filename
            )
        else:
            var_config = config["NeuralNetwork"]["Variables_of_interest"]
    verbosity = config["Verbosity"]["level"]
    log_name = "GFM_EnsembleInference" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD

    comm.Barrier()

    timer = Timer("load_data")
    timer.start()

    hydragnn_model_ens = model_ensemble(modeldirlist)
    hydragnn_model = hydragnn.utils.get_distributed_model(hydragnn_model_ens, verbosity)
    hydragnn_model_ens.eval()

    # Create an instance of your custom ASE calculator
    calculator = PyTorchCalculator(hydragnn_model_ens)

    # Read the POSCAR file
    poscar_filename = "structures/mos2-B_Vacancy-Anion-03_PBE"

    atoms = read(poscar_filename + ".vasp", format="vasp")

    # Attach the calculator to the ASE atoms object
    atoms.set_calculator(calculator)

    maxstep = 1e-2
    maxiter = 200
    relative_increase_threshold = 0.2
    minimum_value_max_force = None  # To store the maximum force from the previous step
    minimum_value_mean_force = None  # To store the maximum force from the previous step
    prev_max_force = None  # To store the maximum force from the previous step
    prev_mean_force = None  # To store the maximum force from the previous step
    prev_positions = None  # To store the positions from the previous step

    check_1 = False
    check_2 = False

    add_random_displacement = False

    if add_random_displacement:
        print("ADDING RANDOM PERTURBATIONS TO INITIAL STRUCTURE")
        random_displacement = numpy.random.uniform(
            -0.1, 0.1, atoms.get_positions().shape
        )
        atoms.set_positions(atoms.get_positions() + random_displacement)
        write(poscar_filename + "_randomly_perturbed_structure" + ".vasp", atoms)

    # Perform structure optimization with a custom stopping and reverting criterion
    optimizer = FIRE(atoms, maxstep=maxstep)
    # optimizer = BFGSLineSearch(atoms, maxstep=maxstep)
    for step in range(maxiter):
        optimizer.step()

        # Calculate energy and forces
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = (forces ** 2).sum(axis=1).max() ** 0.5
        mean_force = (forces ** 2).sum(axis=1).mean() ** 0.5

        # Print energy and maximum force
        print(
            f"Step {step + 1}: Energy = {energy:.6f} eV, Max Force = {max_force:.6f} eV/Å, Mean Force = {mean_force:.6f} eV/Å"
        )

        check_1 = (minimum_value_mean_force is None) and (minimum_value_max_force is None)
        if not check_1:
            check_2 = (mean_force < minimum_value_mean_force) and (max_force < minimum_value_max_force)

        # Write the optimized geometry to a file
        if check_1 or check_2:
            minimum_value_mean_force = mean_force
            minimum_value_max_force = max_force
            print("Overwriting updated optimized geometry")
            if add_random_displacement:
                optimized_filename = (
                        poscar_filename
                        + "_optimized_structure_from_initial_randomly_perturbed_structure.vasp"
                )
            else:
                optimized_filename = poscar_filename + "_optimized_structure.vasp"
            write(optimized_filename, atoms)

        if (prev_max_force is not None) and (prev_mean_force is not None):
            relative_increase = (mean_force - prev_mean_force) / prev_mean_force
            if relative_increase > relative_increase_threshold:
                print(
                    f"Reverting to previous step at step {step + 1} due to a relative force increase of {relative_increase:.2%}."
                )
                atoms.set_positions(prev_positions)  # Revert to previous positions
                break

        # Save current positions and force for the next iteration
        prev_max_force = max_force
        prev_mean_force = mean_force
        prev_positions = deepcopy(atoms.get_positions())
