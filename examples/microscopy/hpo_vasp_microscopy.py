import os, re, json
import logging
import sys
from mpi4py import MPI
import argparse

import random
import numpy as np

import torch
from torch import tensor
from torch_geometric.data import Data

from torch_geometric.transforms import Spherical, LocalCartesian

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.preprocess.utils import RadiusGraphPBC
from hydragnn.preprocess.utils import gather_deg

import optuna
import pandas as pd

from hydragnn.utils.distributed import nsplit, get_device
import hydragnn.utils.tracer as tr

from hydragnn.utils.print_utils import iterate_tqdm, log

from ase.io.vasp import read_vasp_out

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

#transform_coordinates = Spherical(norm=False, cat=False)
transform_coordinates = LocalCartesian(norm=False, cat=False)


def extract_atom_species(outcar_path):
    # Read the OUTCAR file using ASE's read_vasp_outcar function
    outcar = read_vasp_out(outcar_path)

    # Extract atomic positions and forces for each atom
    torch_atom_numbers = torch.tensor(outcar.numbers).unsqueeze(1)

    return torch_atom_numbers


def extract_supercell(section):

    # Define the pattern to match the direct lattice vectors
    lattice_pattern = re.compile(r'\s*([-\d.]+\s+[-\d.]+\s+[-\d.]+)\s+([-\d.]+\s+[-\d.]+\s+[-\d.]+)', re.MULTILINE)

    direct_lattice_matrix = []

    # Iterate through lines in the subsection
    for line in section:
        match_supercell = lattice_pattern.match(line)
        # Extract the matched group
        if match_supercell:
            lattice_vectors = match_supercell.group(1).strip().split()
            # Convert the extracted values to floats
            lattice_vectors = list(map(float, lattice_vectors))

            # Reshape the list into a 3x3 matrix
            direct_lattice_matrix.extend([lattice_vectors[i:i + 3] for i in range(0, len(lattice_vectors), 3)])

    # I need to exclude the length vector
    direct_lattice_matrix.pop()

    return torch.tensor(direct_lattice_matrix)


def extract_positions_forces_energy(section):
    # Define regular expression patterns for POSITION and TOTAL-FORCE
    pos_force_pattern = re.compile(r'\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')
    energy_pattern = re.compile(r'\s+energy\s+without\s+entropy=\s+(-?\d+\.\d+)\s+energy\(sigma->0\) =\s+(-?\d+\.\d+)')

    # Initialize lists to store POSITION and TOTAL-FORCE
    positions_list = []
    forces_list = []
    energy = []

    # Iterate through lines in the subsection
    for line in section:
        match_pos_force = pos_force_pattern.match(line)
        match_energy = energy_pattern.match(line)
        if match_pos_force:
            # Extract values and convert to float
            position_values = [float(match_pos_force.group(i)) for i in range(1, 4)]
            force_values = [float(match_pos_force.group(i)) for i in range(4, 7)]

            # Append to lists
            positions_list.append(position_values)
            forces_list.append(force_values)

        if match_energy:
            # Extract values and convert to float
            # Define the regular expression pattern to match floating-point numbers
            pattern = re.compile(r'-?\d+\.\d+')

            # Find all matches in the input string
            matches = pattern.findall(line)

            # Extract the last match as a float
            if matches:
                last_float = float(matches[-1])
                energy = last_float
            else:
                print("No floating-point number found.")

    # Convert lists to PyTorch tensors
    positions_tensor = torch.tensor(positions_list)
    forces_tensor = torch.tensor(forces_list)
    energy_tensor = torch.tensor([energy])/positions_tensor.shape[0]

    return positions_tensor, forces_tensor, energy_tensor


def read_sections_between(file_path, start_marker, end_marker):
    sections = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            in_section = False
            current_section = []

            for line in lines:
                # Check if the current line contains the start marker
                if start_marker in line:
                    in_section = True
                    current_section = []
                    current_section.append(line)

                # Check if the current line contains the end marker
                elif end_marker in line:
                    in_section = False
                    current_section.append(line)
                    sections.append(current_section)

                # If we're in a section, append the line to the current section
                elif in_section:
                    current_section.append(line)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return sections


def read_outcar(file_path, world_size, rank):
    # Replace these with your file path, start marker, and end marker
    supercell_start_marker = 'VOLUME and BASIS-vectors are now :'
    supercell_end_marker = 'FORCES acting on ions'
    atomic_structure_start_marker = 'POSITION                                       TOTAL-FORCE (eV/Angst)'
    #atomic_structure_end_marker = 'stress matrix after NEB project (eV)'
    atomic_structure_end_marker = 'ENERGY OF THE ELECTRON-ION-THERMOSTAT SYSTEM (eV)'

    dataset = []

    full_string = file_path
    filename = full_string.split("/")[-1]

    # Read sections between specified markers
    result_supercell = read_sections_between(file_path, supercell_start_marker,
                                                             supercell_end_marker)

    # Read sections between specified markers
    result_atomic_structure_sections = read_sections_between(file_path, atomic_structure_start_marker, atomic_structure_end_marker)

    local_result_supercell = list(nsplit(result_supercell, world_size))[rank]
    local_result_atomic_structure_sections = list(nsplit(result_atomic_structure_sections, world_size))[rank]

    # Extract POSITION and TOTAL-FORCE from each section
    for i, (supercell_section, atomic_structure_section) in enumerate(zip(local_result_supercell, local_result_atomic_structure_sections), start=1):

        # Extract POSITION and TOTAL-FORCE into PyTorch tensors
        supercell = extract_supercell(supercell_section)
        positions, forces, energy = extract_positions_forces_energy(atomic_structure_section)

        data_object = Data()

        data_object.pos = positions
        data_object.supercell_size = supercell
        data_object.forces = forces
        data_object.energy = energy
        data_object.y = energy
        atom_numbers = extract_atom_species(file_path)
        data_object.atom_numbers = atom_numbers
        data_object.x = torch.cat((atom_numbers, positions, forces), dim=1)

        dataset.append(data_object)

        #print("optimization step: i = ", i)

    #plot_forces(filename, dataset)

    return dataset, atom_numbers.flatten().tolist()



class VASPDataset(AbstractBaseDataset):

    def __init__(self, dirpath, var_config, dist=False):
        super().__init__()

        self.var_config = var_config
        self.radius_graph = RadiusGraphPBC(
            5.0,
            loop=False,
            max_num_neighbors=20
        )
        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        global_files_list = os.listdir(dirpath)
        #local_files_list = list(nsplit(global_files_list, self.world_size))[self.rank]
        #local_dir_list = list(nsplit(global_files_list, self.world_size))[self.rank]
        local_dir_list = global_files_list

        # Iterate over the files
        """
        for file_name in local_files_list:
            # If you want to work with the full path, you can join the directory path and file name
            file_path = os.path.join(dirpath, file_name)

            temp_dataset, _ = read_outcar(file_path)

            for data_object in temp_dataset:
                if data_object is not None:
                    data_object = self.radius_graph(data_object)
                    data_object = transform_coordinates(data_object)
                    self.dataset.append(data_object)
        """
        for dir_name in local_dir_list:
            if os.path.isdir(os.path.join(dirpath, dir_name)):
                files = os.listdir(os.path.join(dirpath, dir_name))
                outcar_files = [file for file in files if file.startswith("OUTCAR")]
                for file_name in outcar_files:
                    # If you want to work with the full path, you can join the directory path and file name
                    file_path = os.path.join(dirpath, dir_name, file_name)

                    temp_dataset, _ = read_outcar(file_path, self.world_size, self.rank)

                    for data_object in temp_dataset:
                        if data_object is not None:
                            data_object = self.radius_graph(data_object)
                            data_object = transform_coordinates(data_object)
                            self.dataset.append(data_object)


        random.shuffle(self.dataset)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]

def objective(trial):

    global config, best_trial_id, best_validation_loss

    # Extract the unique trial ID
    trial_id = trial.number  # or trial_id = trial.trial_id

    log_name = "MO2" if args.log is None else args.log
    log_name = log_name + '_' + str(trial_id)
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    # log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    # Define the search space for hyperparameters
    model_type = trial.suggest_categorical('model_type', ['EGNN', 'PNA', 'SchNet'])
    hidden_dim = trial.suggest_int('hidden_dim', 50, 300)
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 5)
    num_headlayers = trial.suggest_int('num_headlayers', 1, 3)
    dim_headlayers = [trial.suggest_int(f'dim_headlayer_{i}', 50, 300) for i in range(num_headlayers)]

    # Update the config dictionary with the suggested hyperparameters
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = hidden_dim
    config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = num_conv_layers
    #config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["num_headlayers"] = num_headlayers
    #config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["dim_headlayers"] = dim_headlayers
    config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"]["num_headlayers"] = num_headlayers
    config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"]["dim_headlayers"] = dim_headlayers

    if model_type not in ['EGNN', 'SchNet', 'DimeNet']:
        config["NeuralNetwork"]["Architecture"]["equivariance"] = False


    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.save_config(config, log_name)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    """
    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    """

    # Return the metric to minimize (e.g., validation loss)
    validation_loss, tasks_loss = hydragnn.train.validate(val_loader, model, verbosity, reduce_ranks=True)

    # Move validation_loss to the CPU and convert to NumPy object
    validation_loss = validation_loss.cpu().detach().numpy()

    # Append trial results to the DataFrame
    trial_results.loc[trial_id] = [trial_id, hidden_dim, num_conv_layers, num_headlayers, dim_headlayers, model_type, validation_loss]

    # Update information about the best trial
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_trial_id = trial_id

    # Return the metric to minimize (e.g., validation loss)
    return validation_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="vasp.json"
    )
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates", "forces"]
    node_feature_dims = [1, 3, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/aimd_baseline")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD


    modelname = "MO2"
    if args.preonly:

        ## local data
        total = VASPDataset(
            os.path.join(datadir),
            var_config,
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print("Local splitting: ", len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        setnames = ["trainset", "valset", "testset"]

        ## pickle
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        attrs = dict()
        attrs["pna_deg"] = deg
        SimplePickleWriter(
            trainset,
            basedir,
            "trainset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
            attrs=attrs,
        )
        SimplePickleWriter(
            valset,
            basedir,
            "valset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
        SimplePickleWriter(
            testset,
            basedir,
            "testset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
        sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(basedir=basedir, label="trainset", var_config=var_config)
        valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=var_config)
        testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=var_config)
        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            # trainset.minmax_node_feature = minmax_node_feature
            # trainset.minmax_graph_feature = minmax_graph_feature
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    timer.stop()

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    sampler = optuna.samplers.TPESampler(consider_prior=True, consider_magic_clip=False)
    # sampler = optuna.samplers.RandomSampler()
    # sampler = optuna.samplers.CmaEsSampler(cma_stds=[1.0, 1.0, 1.0], consider_pruned_trials=True, consider_prior=False)
    # sampler = optuna.samplers.GridSampler(consider_prior=False)
    # sampler = optuna.samplers.NSGAIISampler(pop_size=100, crossover_prob=0.9, mutation_prob=0.1)

    # Create an empty DataFrame to store trial results
    trial_results = pd.DataFrame(columns=['Trial_ID', 'Hidden_Dim', 'Num_Conv_Layers', 'Num_Headlayers', 'Dim_Headlayers', 'Model_Type', 'Validation_Loss'])

    # Variables to store information about the best trial
    best_trial_id = None
    best_validation_loss = float('inf')

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    # Update the best trial information directly within the DataFrame
    best_trial_info = pd.Series({'Trial_ID': best_trial_id, 'Best_Validation_Loss': best_validation_loss})
    trial_results = trial_results.append(best_trial_info, ignore_index=True)

    # Save the trial results to a CSV file
    trial_results.to_csv('hpo_results.csv', index=False)

    # Get the best hyperparameters and corresponding trial ID
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    best_trial_id = study.best_trial.number  # or best_trial_id = study.best_trial.trial_id

    # Print information about the best trial
    print("Best Trial ID:", best_trial_id)
    print("Best Validation Loss:", best_validation_loss)

    sys.exit(0)


