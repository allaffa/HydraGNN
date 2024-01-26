import shutil
import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json
import random
import pickle

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time

from torch_geometric.transforms import Spherical

from hydragnn.utils.atomicdescriptors import atomicdescriptors

from ase import io

from rdkit.Chem.rdmolfiles import MolFromPDBFile

import hydragnn
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.time_utils import Timer

# from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_rdkit_molecule,
)
from hydragnn.utils.distributed import get_device
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils.tar_utils import extract_tar_file, get_mol_dir_list

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

from hydragnn.utils import nsplit
import hydragnn.utils.tracer as tr

# FIXME: this works fine for now because we train on GDB-9 molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
dftb_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


from hydragnn.utils.abstractbasedataset import AbstractBaseDataset

spherical_coordinates = Spherical(norm=False, cat=True)

def dftb_to_graph(moldir, dftb_node_types, var_config):
    pdb_filename = os.path.join(moldir, "smiles.pdb")
    mol = MolFromPDBFile(
        pdb_filename, sanitize=False, proximityBonding=True, removeHs=True
    )  # , sanitize=False , removeHs=False)
    spectrum_filename = os.path.join(moldir, "EXC-smooth.DAT")
    ytarget = np.loadtxt(spectrum_filename, usecols=1, dtype=np.float32)
    ytarget = torch.tensor(ytarget)
    data = generate_graphdata_from_rdkit_molecule(
        mol, ytarget, dftb_node_types, var_config
    )
    data.ID = torch.tensor((int(os.path.basename(moldir).replace("mol_", "")),))
    return data


class DFTBDataset(AbstractBaseDataset):
    """DFTBDataset dataset class"""

    def __init__(self, dirpath, dftb_node_types, var_config, dist=False, sampling=None, tmpfs=None):
        super().__init__()

        # atomic descriptors
        atomicdescriptor = atomicdescriptors(
            "./embedding_onehot.json",
            overwritten=True,
            element_types=list(dftb_node_types.keys()),
            one_hot=True,
        )
        self.valence_electrons = atomicdescriptor.get_valence_electrons()
        self.electron_affinity = atomicdescriptor.get_electron_affinity()
        self.atomic_weight = atomicdescriptor.get_atomic_weight()
        self.ion_energies = atomicdescriptor.get_ionenergies()


        self.dftb_node_types = dftb_node_types
        self.var_config = var_config
        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        if os.path.isdir(dirpath):
            dirlist = sorted(os.listdir(dirpath))
        else:
            filelist = dirpath
            info("Reading filelist:", filelist)
            dirpath = os.path.dirname(dirpath)
            dirlist = list()
            with open(filelist, "r") as f:
                lines = f.readlines()
                for line in lines:
                    dirlist.append(line.rstrip())

        if self.dist:
            ## Random shuffle dirlist to avoid the same test/validation set
            random.seed(43)
            random.shuffle(dirlist)
            if sampling is not None:
                dirlist = np.random.choice(dirlist, int(len(dirlist) * sampling))

            x = torch.tensor(len(dirlist), requires_grad=False).to(get_device())
            y = x.clone().detach().requires_grad_(False)
            torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
            assert x == y
            dirlist = list(nsplit(dirlist, self.world_size))[self.rank]
            log("local dirlist", len(dirlist))

        # Iterate through molecule directories or tar files and create graph objects
        for item in iterate_tqdm(dirlist, verbosity_level=2, desc="Load"):
            if 'readme' in item.lower(): continue
            fullpath = os.path.join(dirpath, item)

            # Verify that files in the directory are either tar.gz files or molecule directories
            if (not fullpath.endswith(".tar.gz")) and (not os.path.isdir(fullpath)):
                raise RuntimeError("Can't understand if {} is a molecule directory or tar.gz file".format(fullpath))

            # If tar.gz files, extract tar file in a tmp dir
            if fullpath.endswith(".tar.gz"):
                if 'unprocessed' in os.path.basename(fullpath): continue
                extract_path = extract_tar_file(fullpath, tmpfs)
                mdirs = get_mol_dir_list(extract_path)
                assert len(mdirs) > 0, f"No molecules found in tar file {fullpath} extracted at {extract_path}"
                for mdir in iterate_tqdm(mdirs, verbosity_level=2, desc="Processing"):
                    data_object = dftb_to_graph(mdir, dftb_node_types, var_config)
                    if data_object is not None:
                        self.dataset.append(data_object)

                # Remove the temporary directory where the tar file was extracted
                shutil.rmtree(extract_path)

            # else if items in dirlist are molecule directories, parse them and create graph objects
            else:
                data_object = dftb_to_graph(mdir, dftb_node_types, var_config)
                if data_object is not None:
                    self.dataset.append(data_object)
                    

    def transform_input_to_data_object_base(self, dir):
        data_object = self.__transform_DFTB_UV_input_to_data_object_base(dir)

        return data_object


    def __transform_DFTB_UV_input_to_data_object_base(self, dir):
        """Transforms lines of strings read from the raw data DFTB+ file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        data_object = None

        # collect information about molecular structure and chemical composition
        try:
            pdb_filename = dir + '/' + 'smiles.pdb'
            mol = MolFromPDBFile(pdb_filename, sanitize=False, proximityBonding=True,
                                 removeHs=True)  # , sanitize=False , removeHs=False)
        # file not found -> exit here
        except IOError:
            print(f"'{pdb_filename}'" + " not found")
            return None

        try:
            spectrum_filename = dir + '/' + 'EXC-smooth.DAT'
            spectrum_energies = list()
            with open(spectrum_filename, "r") as input_file:
                count_line = 0
                for line in input_file:
                    spectrum_energies.append(float(line.strip().split()[1]))
        except:
            print(f"EXC-smooth.DAT not found for {dir}")
            return None

        valence_electrons_list = []
        for atom in mol.GetAtoms():
            valence_electrons_list.append(
                self.valence_electrons[dftb_node_types[atom.GetSymbol()]].item()
            )
        electron_affinity_list = []
        for atom in mol.GetAtoms():
            electron_affinity_list.append(
                self.electron_affinity[dftb_node_types[atom.GetSymbol()]].item()
            )
        atomic_weight_list = []
        for atom in mol.GetAtoms():
            atomic_weight_list.append(
                self.atomic_weight[dftb_node_types[atom.GetSymbol()]].item()
            )
        ion_energies_list = []
        for atom in mol.GetAtoms():
            ion_energies_list.append(
                self.ion_energies[dftb_node_types[atom.GetSymbol()]].item()
            )

        atomic_descriptors_list = [valence_electrons_list, electron_affinity_list, atomic_weight_list, ion_energies_list]
        num_manually_constructed_atomic_descriptors = len({len(i) for i in atomic_descriptors_list})

        # The list is empty if there are no atomic descirptors manually constructed, otherwise it shoulb have length=1
        assert num_manually_constructed_atomic_descriptors <= 1, "manually constructed lists of atomic descriptors are not consistent in length"

        if num_manually_constructed_atomic_descriptors == 1:
            atomicdescriptors_torch_tensor = torch.cat([torch.tensor([descriptor]) for descriptor in atomic_descriptors_list],
                           dim=0).t().contiguous()

        data_object = generate_graphdata_from_rdkit_molecule(mol, torch.tensor(spectrum_energies), dftb_node_types, atomicdescriptors_torch_tensor)
        atoms = io.read(dir + '/' + 'geo_end.xyz', parallel=False)
        data_object.pos = torch.from_numpy(atoms.positions)
        try:
            data_object = spherical_coordinates(data_object)
        except Exception as error:
            print(f"Spherical coordinates fails for {dir}: ", error)
            print(f"Spherical coordinates fails for {dir} - data_object.edge_index.shape: ", data_object.edge_index.shape)
            print(f"Spherical coordinates fails for {dir} - data_object.edge_attr.shape: ", data_object.edge_attr.shape)
            print(f"Spherical coordinates fails for {dir} - data.pos.shape: ", data_object.pos.shape)
            return None
        data_object.pos = data_object.pos.to(torch.float32)
        data_object.x = data_object.x.to(torch.float32)
        data_object.edge_attr = data_object.edge_attr.to(torch.float32)
        data_object.ID = torch.tensor((int(os.path.basename(dir).replace("mol_", "")),))

        return data_object
    

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
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--input_data", default=None,
                        help="Input dataset to read in. Must be either a .txt file listing molecule directories or a "
                             "directory containing .tar.gz files. Provide --tmpfs to extract tar files into fast, "
                             "local storage.")
    parser.add_argument("--tmpfs", default=None,
                        help="Transient storage space such as /tmp/$USER which can be used as a temporary "
                             "scratch space for caching and/or extracting data. "
                             "The location must exist before use by HydraGNN.")

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

    graph_feature_names = ["spectrum"]
    graph_feature_dim = [37500]
    node_feature_names = ["atomic_features"]
    node_feature_dims = [16]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datafile = args.input_data
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, "dftb_smooth_uv_spectrum.json")
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["output_names"] = [
        graph_feature_names[item]
        for ihead, item in enumerate(var_config["output_index"])
    ]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dim
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

    modelname = "dftb_smooth_uv_spectrum"
    log_name = modelname if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    if args.preonly:
        ## local data
        total = DFTBDataset(
            datafile,
            dftb_node_types,
            var_config,
            dist=True,
            tmpfs=args.tmpfs,
        )
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print(len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        ## adios
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
        # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

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
    if args.format == "adios":
        info("Adios load")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(basedir, "trainset", var_config=var_config)
        valset = SimplePickleDataset(basedir, "valset", var_config=var_config)
        testset = SimplePickleDataset(basedir, "testset", var_config=var_config)
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

    if args.mae:
        import matplotlib.pyplot as plt

        ##################################################################################################################
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for isub, (loader, setname) in enumerate(
            zip([train_loader, val_loader, test_loader], ["train", "val", "test"])
        ):
            error, rmse_task, true_values, predicted_values = hydragnn.train.test(
                loader, model, verbosity
            )
            ihead = 0
            head_true = np.asarray(true_values[ihead].cpu()).squeeze()
            head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = graph_feature_names[ifeat]

            ax = axs[isub]

            num_test_samples = len(test_loader.dataset)
            error_mae = 0.0
            error_mse = 0.0

            for sample_id in range(0, num_test_samples):
                error_mae += np.sum(
                    np.abs(
                        head_pred[
                            (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                            * graph_feature_dim[0]
                        ]
                        - head_true[
                            (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                            * graph_feature_dim[0]
                        ]
                    )
                )
                error_mse += np.sum(
                    (
                        head_pred[
                            (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                            * graph_feature_dim[0]
                        ]
                        - head_true[
                            (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                            * graph_feature_dim[0]
                        ]
                    )
                    ** 2
                )

                fig, ax = plt.subplots()
                true_sample = head_true[
                    (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                    * graph_feature_dim[0]
                ]
                pred_sample = head_pred[
                    (sample_id * graph_feature_dim[0]) : (sample_id + 1)
                    * graph_feature_dim[0]
                ]
                ax.plot(true_sample)
                ax.plot(pred_sample)
                plt.draw()
                plt.tight_layout()
                plt.ylim([-0.2, max(true_sample) + 0.2])
                plt.savefig(f"logs/sample_{sample_id}.png")
                plt.close(fig)

            error_mae /= num_test_samples
            error_mse /= num_test_samples
            error_rmse = np.sqrt(error_mse)

            print(varname, ": ev/cm, mae=", error_mae, ", rmse= ", error_rmse)

            ax.scatter(
                head_true,
                head_pred,
                s=7,
                linewidth=0.5,
                edgecolor="b",
                facecolor="none",
            )
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname + " (eV/cm)", fontsize=16)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2f}".format(error_mae),
            )
        if rank == 0:
            fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
        plt.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
