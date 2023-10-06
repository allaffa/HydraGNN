import os
import numpy as np
import random

import torch
from torch import tensor
from torch_geometric.data import Data
from hydragnn.utils.abstractrawdataset import AbstractRawDataset

from hydragnn.utils import nsplit
from hydragnn.utils.print_utils import iterate_tqdm, log

from ase.io.vasp import read_vasp_out
import subprocess


class VASPDataset(AbstractRawDataset):
    def __init__(self, config, dist=False, sampling=None):
        super().__init__(config, dist, sampling)

    def _AbstractRawDataset__load_raw_data(self):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.
        """

        for dataset_type, raw_data_path in self.path_dictionary.items():
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(os.getcwd(), raw_data_path)
            if not os.path.exists(raw_data_path):
                raise ValueError("Folder not found: ", raw_data_path)

            assert (
                len(os.listdir(raw_data_path)) > 0
            ), "No data files provided in {}!".format(raw_data_path)

            filelist = sorted(os.listdir(raw_data_path))
            if self.dist:
                ## Random shuffle filelist to avoid the same test/validation set
                random.seed(43)
                random.shuffle(filelist)
                if self.sampling is not None:
                    filelist = np.random.choice(
                        filelist, int(len(filelist) * self.sampling)
                    )

                x = torch.tensor(len(filelist), requires_grad=False).to(get_device())
                y = x.clone().detach().requires_grad_(False)
                torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
                assert x == y
                filelist = list(nsplit(filelist, self.world_size))[self.rank]
                log("local filelist", len(filelist))

            for name in iterate_tqdm(filelist, verbosity_level=2, desc="Local files"):
                if name == ".DS_Store":
                    continue
                # if the directory contains file, iterate over them
                if os.path.isfile(os.path.join(raw_data_path, name)):
                    data_object = self.transform_input_to_data_object_base(
                        filepath=os.path.join(raw_data_path, name)
                    )
                    if not isinstance(data_object, type(None)):
                        self.dataset.append(data_object)
                # if the directory contains subdirectories, explore their content
                # if the directory contains subdirectories, explore their content
                elif os.path.isdir(os.path.join(raw_data_path, name)):
                    if name == ".DS_Store":
                        continue
                    dir_name = os.path.join(raw_data_path, name)
                    for subname in os.listdir(dir_name):
                        if subname == ".DS_Store":
                            continue
                        subdir_name = os.path.join(dir_name, subname)
                        for subsubname in os.listdir(subdir_name):
                            subsubdir_name = os.path.join(subdir_name, subsubname)
                            for subsubsubname in os.listdir(subsubdir_name):
                                if os.path.isfile(
                                    os.path.join(subsubdir_name, subsubsubname)
                                ):
                                    data_object = (
                                        self.transform_input_to_data_object_base(
                                            filepath=os.path.join(
                                                subsubdir_name, subsubsubname
                                            )
                                        )
                                    )
                                    if not isinstance(data_object, type(None)):
                                        self.dataset.append(data_object)

            if self.dist:
                torch.distributed.barrier()

        # scaled features by number of nodes
        self._AbstractRawDataset__scale_features_by_num_nodes()

        self._AbstractRawDataset__normalize_dataset()

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
            data_object.y = tensor([float(line.strip())])

        return data_object
