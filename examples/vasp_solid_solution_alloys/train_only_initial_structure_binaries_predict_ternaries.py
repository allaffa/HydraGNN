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

from hydragnn.utils.distributed import nsplit, get_device
import hydragnn.utils.tracer as tr

from hydragnn.utils.print_utils import iterate_tqdm, log

from ase.io import read

from generate_dictionaries_pure_elements import (
    generate_dictionary_bulk_energies,
    generate_dictionary_elements,
)

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
atom_number_dict = {23: 0, 41: 1, 73: 2}  # Dictionary mapping atom numbers to classes


def create_one_hot(atom_numbers, atom_number_dict):
    num_classes = len(atom_number_dict)
    one_hot = torch.zeros(
        atom_numbers.size(0), num_classes
    )  # Initialize one-hot tensor

    # Convert atom_numbers to class indices using the dictionary
    class_indices = [atom_number_dict[atom.item()] for atom in atom_numbers]

    # Use scatter_ to fill the one-hot tensor
    one_hot.scatter_(1, torch.tensor(class_indices).unsqueeze(1), 1)

    return one_hot


def extract_atom_species(outcar_path):
    # Read the OUTCAR file using ASE's read_vasp_outcar function
    outcar = read_vasp_out(outcar_path)

    # Extract atomic positions and forces for each atom
    torch_atom_numbers = torch.tensor(outcar.numbers).unsqueeze(1)

    return torch_atom_numbers


def read_initial_geometry_energy_rmsd_deformation(file_path):

    data_object = Data()

    # Read the POSCAR file using ASE
    ase_object = read(os.path.join(file_path, "0.POSCAR"), format="vasp")

    data_object.supercell_size = tensor(ase_object.cell.array).float()
    data_object.pos = tensor(ase_object.arrays["positions"]).float()
    proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
    data_object.x = tensor(proton_numbers).float()

    # Read information about deformation tensor
    deformation_lattice_vectors_file = open(
        os.path.join(file_path, "deformation_lattice_vectors.txt"), "r"
    )
    Lines = deformation_lattice_vectors_file.readlines()

    # Strips the newline character
    deformation_vectors_list = [list(map(float, line.split())) for line in Lines]

    # Convert list to a PyTorch tensor
    deformation_vectors_tensor = torch.tensor(deformation_vectors_list)
    data_object.deformation_vectors_tensor = deformation_vectors_tensor

    # Read information about root mean squared displacement
    root_mean_squared_displacement_file = open(
        os.path.join(file_path, "root_mean_squared_displacement.txt"), "r"
    )
    Lines = root_mean_squared_displacement_file.readlines()
    # Strips the newline character
    for line in Lines:
        rmsd = tensor([float(line.strip())])
    data_object.rmsd = rmsd

    # Read information about formation energy
    formation_energy_file = open(os.path.join(file_path, "formation_energy.txt"), "r")
    Lines = formation_energy_file.readlines()
    # Strips the newline character
    for line in Lines:
        formation_energy = tensor([float(line.strip())])
    data_object.formation_energy = formation_energy

    data_object.y = torch.cat(
        [
            data_object.formation_energy,
            data_object.rmsd,
            data_object.deformation_vectors_tensor.flatten(),
        ]
    )

    return data_object


class VASPDataset(AbstractBaseDataset):
    def __init__(self, dirpath, var_config, dist=False, cases_filter=None):
        super().__init__()

        self.var_config = var_config
        self.radius_graph = RadiusGraphPBC(
            self.var_config["NeuralNetwork"]["Architecture"]["radius"],
            loop=False,
            max_num_neighbors=self.var_config["NeuralNetwork"]["Architecture"][
                "max_neighbours"
            ],
        )
        self.dist = dist

        self.cases_filter = cases_filter

        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        # Extract information about pure elements
        for name in iterate_tqdm(
            os.listdir(os.path.join(dirpath, "../", "pure_elements")),
            verbosity_level=2,
            desc="Load",
        ):
            if name == ".DS_Store":
                continue
            # If you want to work with the full path, you can join the directory path and file name
            file_path = os.path.join(
                dirpath, "../", "pure_elements", name, name + "128", "case-1"
            )

            data_object = read_initial_geometry_energy_rmsd_deformation(file_path)
            data_object = self.radius_graph(data_object)
            data_object = transform_coordinates(data_object)

            self.dataset.append(data_object)

        # Extract information about alloys
        for name in os.listdir(dirpath):
            if name == ".DS_Store":
                continue
            dir_name = os.path.join(dirpath, name)
            for subdir_name in iterate_tqdm(
                os.listdir(dir_name), verbosity_level=2, desc="Load"
            ):
                if subdir_name == ".DS_Store":
                    continue
                subdir_global_list = os.listdir(os.path.join(dir_name, subdir_name))
                subdir_local_list = list(nsplit(subdir_global_list, self.world_size))[
                    self.rank
                ]

                if self.cases_filter is not None:
                    assert isinstance(
                        self.cases_filter, list
                    ), "cases_filter is not a list"
                    assert (
                        len(self.cases_filter) == 2
                    ), "cases_filter does not have two entries"
                    N1 = self.cases_filter[0]
                    N2 = self.cases_filter[1]
                    assert (
                        N1 <= N2
                    ), f"lower bound {N1} of cases_filter is not smaller than upper bound {N2}"
                    filtered_cases = [
                        case
                        for case in subdir_local_list
                        if N1 <= int(case.split("-")[1]) <= N2
                    ]
                    subdir_local_list = filtered_cases

                # print("MASSI - ", str(self.rank), " - total list: ", len(subdir_global_list))
                # print("MASSI - ", str(self.rank), " - about to read: ", subdir_name)
                # print("MASSI - ", str(self.rank), " - about to read: ", subdir_local_list)

                count = 0

                for subsubdir_name in subdir_local_list:

                    if subsubdir_name == ".DS_Store":
                        continue

                    count = count + 1

                    file_path = os.path.join(dir_name, subdir_name, subsubdir_name)
                    # print("MASSI - ", str(self.rank), " - about to read: ", os.path.join(dir_name, subdir_name, subsubdir_name))

                    try:
                        data_object = read_initial_geometry_energy_rmsd_deformation(
                            file_path
                        )
                        if data_object is not None:
                            data_object = self.radius_graph(data_object)
                            data_object = transform_coordinates(data_object)
                            self.dataset.append(data_object)

                    except ValueError as e:
                        pass
                    except Exception as e:
                        print(self.rank, "Exception:", file_path, e, file=sys.stderr)
                        # traceback.print_exc()
                        pass

                torch.distributed.barrier()
                # print("MASSI - ", str(self.rank), " - finished reading: ", count, " of ", len(subdir_local_list), " - ", os.path.join(dir_name, subdir_name, subsubdir_name))

                # print("MASSI - before barrier ", str(self.rank), " - finished reading: ", subdir_name)

                # print("MASSI - after barrier ", str(self.rank), " - finished reading: ", subdir_name)

        random.shuffle(self.dataset)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


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
        "--inputfile", help="input file", type=str, default="vasp_energy.json"
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

    graph_feature_names = ["energy", "rmsd", "deformation_lattice_vectors"]
    graph_feature_dims = [1, 1, 9]
    node_feature_names = ["atomic_number", "cartesian_coordinates"]
    node_feature_dims = [1, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/VASP_calculations")
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

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "NbTaV" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "NbTaV"
    if args.preonly:
        ## local data
        binaries_dataset = VASPDataset(
            os.path.join(datadir, "binaries"),
            config,
            dist=True,
        )
        ternaries_train_validate_dataset = VASPDataset(
            os.path.join(datadir, "ternaries"), config, dist=True, cases_filter=[1, 2]
        )
        ternaries_test_dataset = VASPDataset(
            os.path.join(datadir, "ternaries"), config, dist=True, cases_filter=[3, 100]
        )
        ## This is a local split
        binaries_trainset, binaries_valset, binaries_testset = split_dataset(
            dataset=binaries_dataset,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )
        ternaries_trainset, ternaries_valset1, ternaries_valset2 = split_dataset(
            dataset=ternaries_train_validate_dataset,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )
        binaries_valset.extend(binaries_testset)
        ternaries_valset = [*ternaries_valset1, *ternaries_valset2]
        trainset = [*binaries_trainset, *ternaries_trainset]
        valset = [*binaries_valset, *ternaries_valset]
        testset = ternaries_test_dataset[:]
        print("Local splitting: ", len(trainset), len(valset), len(testset))

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
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
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

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.save_config(config, log_name)

    timer.stop()

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

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
