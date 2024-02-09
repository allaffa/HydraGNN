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
from hydragnn.utils.serializeddataset import SerializedWriter, SerializedDataset
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.preprocess.utils import RadiusGraphPBC

from hydragnn.utils.distributed import nsplit, get_device

from hydragnn.utils.print_utils import iterate_tqdm, log

from ase.io.vasp import read_vasp_out

from generate_dictionaries_pure_elements import generate_dictionary_bulk_energies, generate_dictionary_elements

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


# transform_coordinates = Spherical(norm=False, cat=True)
transform_coordinates = LocalCartesian(norm=False, cat=True)

energy_bulk_metals = generate_dictionary_bulk_energies()
periodic_table = generate_dictionary_elements()


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
    pos_force_pattern = re.compile(
        r'\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')
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
    energy_tensor = torch.tensor([energy]) / positions_tensor.shape[0]

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


def read_outcar_pure_elements_ground_state(file_path):
    # Replace these with your file path, start marker, and end marker
    supercell_start_marker = 'VOLUME and BASIS-vectors are now :'
    supercell_end_marker = 'FORCES acting on ions'
    atomic_structure_start_marker = 'POSITION                                       TOTAL-FORCE (eV/Angst)'
    atomic_structure_end_marker = 'stress matrix after NEB project (eV)'
    # atomic_structure_end_marker = 'ENERGY OF THE ELECTRON-ION-THERMOSTAT SYSTEM (eV)'

    dataset = []

    full_string = file_path
    filename = full_string.split("/")[-1]

    # Read sections between specified markers
    result_supercell = read_sections_between(file_path, supercell_start_marker,
                                             supercell_end_marker)

    # Read sections between specified markers
    result_atomic_structure_sections = read_sections_between(file_path, atomic_structure_start_marker,
                                                             atomic_structure_end_marker)

    supercell_section = result_supercell[-1]
    atomic_structure_section = result_atomic_structure_sections[-1]

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

    return dataset, atom_numbers.flatten().tolist()


def read_outcar(file_path):
    # Replace these with your file path, start marker, and end marker
    supercell_start_marker = 'VOLUME and BASIS-vectors are now :'
    supercell_end_marker = 'FORCES acting on ions'
    atomic_structure_start_marker = 'POSITION                                       TOTAL-FORCE (eV/Angst)'
    atomic_structure_end_marker = 'stress matrix after NEB project (eV)'
    # atomic_structure_end_marker = 'ENERGY OF THE ELECTRON-ION-THERMOSTAT SYSTEM (eV)'

    dataset = []

    full_string = file_path
    filename = full_string.split("/")[-1]

    # Read sections between specified markers
    result_supercell = read_sections_between(file_path, supercell_start_marker,
                                             supercell_end_marker)

    # Read sections between specified markers
    result_atomic_structure_sections = read_sections_between(file_path, atomic_structure_start_marker,
                                                             atomic_structure_end_marker)

    # Extract POSITION and TOTAL-FORCE from each section
    for i, (supercell_section, atomic_structure_section) in enumerate(
            zip(result_supercell, result_atomic_structure_sections), start=1):
        # Extract POSITION and TOTAL-FORCE into PyTorch tensors
        supercell = extract_supercell(supercell_section)
        positions, forces, energy = extract_positions_forces_energy(atomic_structure_section)

        data_object = Data()

        data_object.pos = positions
        data_object.supercell_size = supercell
        data_object.forces = forces
        atom_numbers = extract_atom_species(file_path)
        data_object.atom_numbers = atom_numbers
        data_object.x = torch.cat((atom_numbers, positions, forces), dim=1)

        # compute the cohesive energy by removing the linear term of the energy from the total energy of the system
        # Count occurrences of each element
        element_counts = {}
        for atom_number in atom_numbers:
            if atom_number.item() not in element_counts:
                element_counts[atom_number.item()] = 0
            element_counts[atom_number.item()] += 1

        # Calculate total number of atoms
        total_atoms = data_object.pos.shape[0]

        # Calculate ratio for each element
        element_ratios = {element: count / total_atoms for element, count in element_counts.items()}

        for item in element_ratios.keys():
            energy -= element_ratios[item] * energy_bulk_metals[periodic_table[item]]

        data_object.energy = energy
        data_object.y = energy

        data_object.energy = energy
        data_object.y = energy

        dataset.append(data_object)

        # print("optimization step: i = ", i)

    # plot_forces(filename, dataset)

    return dataset, atom_numbers.flatten().tolist()


class VASPDataset(AbstractBaseDataset):

    def __init__(self, dirpath, var_config, dist=False):
        super().__init__()

        self.var_config = var_config
        self.radius_graph = RadiusGraphPBC(
            self.var_config["NeuralNetwork"]["Architecture"]["radius"],
            loop=False,
            max_num_neighbors=self.var_config["NeuralNetwork"]["Architecture"]["max_neighbours"]
        )
        self.dist = dist

        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        # Extract information about pure elements
        for name in iterate_tqdm(os.listdir(os.path.join(dirpath, "../", "pure_elements")), verbosity_level=2, desc="Load"):
            files = os.listdir(os.path.join(dirpath, "../", "pure_elements", name))
            outcar_files = [file for file in files if file.startswith("OUTCAR")]
            for file_name in outcar_files:
                # If you want to work with the full path, you can join the directory path and file name
                file_path = os.path.join(os.path.join(dirpath, "../", "Bulk-M-and-X/", file_name))

                element = name
                dataset, _ = read_outcar_pure_elements_ground_state(file_path)

                assert len(dataset) == 1, 'bulk metal calculation not imported correctly'

                energy_bulk_metals[element] = dataset[0].y.item()

                print(element, " - ", str(dataset[0].y.item()))

        # Extract information about alloys
        for name in os.listdir(dirpath):
            if name == ".DS_Store":
                continue
            dir_name = os.path.join(dirpath, name)
            for subname in iterate_tqdm(os.listdir(dir_name), verbosity_level=2, desc="Load"):
                if subname == ".DS_Store":
                    continue
                subdir_name = os.listdir(os.path.join(dir_name, subname))
                subdir_name = list(nsplit(subdir_name, self.world_size))[
                    self.rank]
                for subsubname in subdir_name:

                    files = os.listdir(os.path.join(dir_name, subname, subsubname))

                    for file in files:

                        if file == "OUTCAR" or file == "OUTCAR-bis":

                            temp_dataset = read_outcar(os.path.join(dir_name, subname, subsubname) + '/' + file)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="loading from existing pickle/adios files with train/test/validate splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios or pickle saving and no train",
    )
    parser.add_argument("--inputfile", help="input file", type=str, default="vasp.json")
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
    parser.set_defaults(format="pickle")

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
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    datasetname = config["Dataset"]["name"]
    fname_adios = dirpwd + "/dataset/%s.bp" % (datasetname)
    config["Dataset"]["name"] = "%s_%d" % (datasetname, rank)
    if not args.loadexistingsplit:
        total = VASPDataset(dirpwd + "/dataset/binaries", config, dist=True)

        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )
        print(len(total), len(trainset), len(valset), len(testset))

        if args.format == "adios":
            fname = os.path.join(
                os.path.dirname(__file__), "./dataset/%s.bp" % datasetname
            )
            adwriter = AdiosWriter(fname, MPI.COMM_SELF)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
            adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
            adwriter.save()
        elif args.format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "serialized_dataset"
            )
            SerializedWriter(
                trainset,
                basedir,
                datasetname,
                "trainset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
            )
            SerializedWriter(
                valset,
                basedir,
                datasetname,
                "valset",
            )
            SerializedWriter(
                testset,
                basedir,
                datasetname,
                "testset",
            )
    comm.Barrier()
    if args.preonly:
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        info("Adios load")
        opt = {
            "preload": True,
            "shmem": False,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % datasetname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "serialized_dataset"
        )
        trainset = SerializedDataset(basedir, datasetname, "trainset")
        valset = SerializedDataset(basedir, datasetname, "valset")
        testset = SerializedDataset(basedir, datasetname, "testset")
    else:
        raise ValueError("Unknown data format: %d" % args.format)
    ## Set minmax
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = trainset.minmax_node_feature
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = trainset.minmax_graph_feature

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    hydragnn.utils.save_config(config, log_name)

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
        create_plots=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)
