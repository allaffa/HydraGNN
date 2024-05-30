import torch
from torch_geometric.data import Data

from ase.calculators.calculator import Calculator, all_changes

from hydragnn.preprocess.utils import get_radius_graph_pbc

from torch_geometric.transforms import LocalCartesian
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
    implemented_properties = ['energy', 'forces']

    def __init__(self, hydragnn_model):
        Calculator.__init__(self)
        self.model = hydragnn_model

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = atoms.get_positions()
        positions_tensor = torch.tensor(positions, requires_grad=False, dtype=torch.float)

        # Extract atomic numbers
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_numbers_torch = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(1)

        x = torch.cat((atomic_numbers_torch, positions_tensor), dim=1)

        # Create the torch_geometric data object
        data_object = Data(pos=positions_tensor, x=x, supercell_size=torch.tensor(atoms.cell.array).float())

        add_edges_pbc = get_radius_graph_pbc(radius=config["NeuralNetwork"]["Architecture"]["radius"],
                                             max_neighbours=20)
        data_object = add_edges_pbc(data_object)

        data_object = transform_coordinates(data_object)

        energy, forces = self.model(data_object)

        self.results['energy'] = energy.item()
        self.results['forces'] = forces.detach().numpy()


class PyTorchCalculatorSelfConsistent(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, hydragnn_model):
        Calculator.__init__(self)
        self.model = hydragnn_model

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = atoms.get_positions()
        positions_tensor = torch.tensor(positions, requires_grad=False, dtype=torch.float)

        # Extract atomic numbers
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_numbers_torch = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(1)

        x = torch.cat((atomic_numbers_torch, positions_tensor), dim=1)

        # Create the torch_geometric data object
        data_object = Data(pos=positions_tensor, x=x, supercell_size=torch.tensor(atoms.cell.array).float())

        add_edges_pbc = get_radius_graph_pbc(radius=config["NeuralNetwork"]["Architecture"]["radius"],
                                             max_neighbours=20)
        data_object = add_edges_pbc(data_object)

        data_object = transform_coordinates(data_object)

        data_object.pos.requires_grad = True

        energy, _ = self.model(data_object)
        grads_energy = torch.autograd.grad(outputs=energy, inputs=data_object.pos,
                                           grad_outputs=torch.ones_like(energy),
                                           retain_graph=False)[0]

        grad_energy_post_scaling_factor = positions_tensor.shape[0] * torch.ones(positions_tensor.shape[0], 1)

        grads_energy_rescaled = grad_energy_post_scaling_factor * grads_energy

        self.results['energy'] = energy.item()
        self.results['forces'] = grads_energy_rescaled.detach().numpy()

import json, os
import logging
from mpi4py import MPI
import argparse


from ase.optimize import BFGS, FIRE # or any other optimizer
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.io import read, write

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.model import load_existing_model
from hydragnn.models.create import create_model_config


if __name__ == "__main__":

    modelname = "MO2"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="./logs/MO2/config.json"
    )
    group = parser.add_mutually_exclusive_group()

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD

    comm.Barrier()

    timer = Timer("load_data")
    timer.start()

    hydragnn_model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    hydragnn_model = torch.nn.parallel.DistributedDataParallel(
        hydragnn_model
    )

    load_existing_model(hydragnn_model, modelname, path="./logs/")
    hydragnn_model.eval()

    # Create an instance of your custom ASE calculator
    calculator = PyTorchCalculatorSelfConsistent(hydragnn_model)
    #calculator = PyTorchCalculator(hydragnn_model)

    # Read the POSCAR file
    poscar_filename = "./test.vasp"

    atoms = read(poscar_filename, format='vasp')

    # Attach the calculator to the ASE atoms object
    atoms.set_calculator(calculator)

    maxstep = 1e-1
    maxiter = 10000

    # Perform structure optimization
    optimizer = BFGS(atoms, maxstep=maxstep)
    optimizer = BFGSLineSearch(atoms, maxstep=maxstep)
    #optimizer = FIRE(atoms, maxstep=maxstep)
    optimizer.run()
    #optimizer.run(fsteps=maxiter)  # adjust convergence criteria as needed


