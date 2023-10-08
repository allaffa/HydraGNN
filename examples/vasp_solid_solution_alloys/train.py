import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import random
import numpy as np

import torch
from torch import tensor
from torch_geometric.data import Data

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.serializeddataset import SerializedWriter, SerializedDataset

from hydragnn.utils.distributed import nsplit, get_device

from hydragnn.utils.print_utils import iterate_tqdm, log

from ase.io.vasp import read_vasp_out

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


class VASPDataset(AbstractBaseDataset):
    def __init__(self, config, dist=False, sampling=None):
        super().__init__(config, dist, sampling)

    def __init__(self, dirpath, var_config, dist=False):
        super().__init__()

        self.var_config = var_config
        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        for name in os.listdir(dirpath):
            if name == ".DS_Store":
                continue
            # if the directory contains file, iterate over them
            if os.path.isfile(os.path.join(dirpath, name)):
                data_object = self.transform_input_to_data_object_base(
                    filepath=os.path.join(dirpath, name)
                )
                if not isinstance(data_object, type(None)):
                    self.dataset.append(data_object)
            # if the directory contains subdirectories, explore their content
            # if the directory contains subdirectories, explore their content
            elif os.path.isdir(os.path.join(dirpath, name)):
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
                        data_object = (
                            self.transform_input_to_data_object_base(
                                filepath=os.path.join(dir_name, subname, subsubname)+'/'+'OUTCAR'
                            )
                        )
                        if not isinstance(data_object, type(None)):
                            self.dataset.append(data_object)

            if self.dist:
                torch.distributed.barrier()

    def transform_input_to_data_object_base(self, filepath):
        data_object = self.__transform_VASP_input_to_data_object_base(filepath=filepath)
        return data_object

    def __transform_VASP_input_to_data_object_base(self, filepath):
        """Transforms lines of strings read from the raw data EAM file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        if "OUTCAR" in filepath:

            ase_object = read_vasp_out(filepath)

            dirpath = filepath.split("OUTCAR")[0]
            data_object = self.__transform_ASE_object_to_data_object(
                dirpath, ase_object
            )

            return data_object

        else:
            return None

    def __transform_ASE_object_to_data_object(self, filepath, ase_object):
        # FIXME:
        #  this still assumes bulk modulus is specific to the CFG format.
        #  To deal with multiple files across formats, one should generalize this function
        #  by moving the reading of the .bulk file in a standalone routine.
        #  Morevoer, this approach assumes tha there is only one global feature to look at,
        #  and that this global feature is specicially retrieveable in a file with the string *bulk* inside.

        data_object = Data()

        data_object.supercell_size = tensor(ase_object.cell.array).float()
        data_object.pos = tensor(ase_object.arrays["positions"]).float()
        proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
        forces = ase_object.calc.results["forces"]
        stress = ase_object.calc.results["stress"]
        fermi_energy = ase_object.calc.eFermi
        free_energy = ase_object.calc.results["free_energy"]
        energy = ase_object.calc.results["energy"]
        node_feature_matrix = np.concatenate((proton_numbers, forces), axis=1)
        data_object.x = tensor(node_feature_matrix).float()

        formation_energy_file = open(filepath + 'formation_energy.txt', 'r')
        Lines = formation_energy_file.readlines()

        # Strips the newline character
        for line in Lines:
            formation_energy = tensor([float(line.strip())])
            # convert formation energy from eV to meV/atom
            data_object.y = formation_energy * 1000/data.num_nodes

        return data_object

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
        total = VASPDataset(dirpwd + "/dataset/bcc_enthalpy", config, dist=True)

        trainset = total
        valset = total
        testset = total
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
                minmax_node_feature=total.minmax_node_feature,
                minmax_graph_feature=total.minmax_graph_feature,
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

    if dist.is_initialized():
        dist.barrier()

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
