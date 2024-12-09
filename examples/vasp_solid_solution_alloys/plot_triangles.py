import json, os
import sys
import logging
import pickle
from tqdm import tqdm
from mpi4py import MPI
import argparse

import ternary

import torch
import numpy as np

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.distributed import get_device
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.config_utils import (
    update_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.preprocess import create_dataloaders

from scipy.interpolate import griddata

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.rcParams.update({"font.size": 16})

from collections import defaultdict


atomic_number_to_symbol = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B",
    6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br",
    36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh",
    46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
    56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb",
    66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re",
    76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
    86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am",
    96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db",
    106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
    111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc",
    116: "Lv", 117: "Ts", 118: "Og"
}


def normalize_composition(atom_counts):
    """
    Normalize atom counts to sum to 1.
    """
    total = np.sum(atom_counts)
    return atom_counts / total if total > 0 else atom_counts


def error_map(model, test_data_loader, selected_atom_types):
    """
    Process a list of PyTorch Geometric data objects to extract chemical composition
    and maximum scalar property for each composition.
    """
    composition_dict = defaultdict(lambda: -np.inf)  # Stores max y for each composition

    # for data_id, data in enumerate(tqdm(testset)):
    for data in tqdm(test_data_loader):
        predicted = model(data.to(get_device()))
        predicted = predicted[variable_index].flatten()
        start = data.y_loc[0][variable_index].item()
        end = data.y_loc[0][variable_index + 1].item()
        true = data.y.squeeze(1)  # [start:end, 0]

        atom_types = data.x[:, 0].numpy()  # Assuming atom types are in the first column
        atom_counts = [np.sum(atom_types == atom) for atom in selected_atom_types]
        composition = tuple(normalize_composition(atom_counts))

        # Update the max scalar property for this composition
        abs_error = abs(predicted - true)
        composition_dict[composition] = max(composition_dict[composition], abs_error.detach().numpy())

        predicted_values.extend(predicted.tolist())
        true_values.extend(true.tolist())

    return composition_dict


def get_atom_type_mapping(data_loader):
    """
    Determine the atom types in the data and map them to triangle corners.
    """
    all_atom_types = set()
    for data in data_loader:
        atom_types = data.x[:, 0].numpy()  # Extract atom types
        all_atom_types.update(atom_types)
    all_atom_types = sorted(all_atom_types)
    if len(all_atom_types) > 3:
        raise ValueError("This script supports systems with at most 3 atom types.")
    return all_atom_types


def plot_ternary_field(composition_dict, atom_type_mapping):
    """
    Plot the ternary diagram field using the maximum values from the compositions.
    Handles cases with fewer than 3 atom types.
    """

    output_file = "triangle_plot"

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))  # Ensure a square figure to preserve aspect ratio

    # Initialize the ternary plot
    scale = 1.0
    figure, tax = ternary.figure(scale=scale)
    tax.boundary(linewidth=2.0)

    # Remove the external box by disabling axes spines
    tax.clear_matplotlib_ticks()
    for spine in tax.get_axes().spines.values():
        spine.set_visible(False)

    # Customize gridlines
    tax.gridlines(multiple=0.1, color="gray", linewidth=0.5)

    # Set tick formats for each axis to display one decimal place
    tick_formats = {
        'b': "%.1f",  # Bottom axis
        'l': "%.1f",  # Left axis
        'r': "%.1f"  # Right axis
    }
    tax.ticks(axis='lbr', multiple=0.1, linewidth=1, offset=0.02, tick_formats=tick_formats)

    # Set the title
    # tax.set_title("Maximum MAE", fontsize=15)

    points = list(composition_dict.keys())
    values = [composition_dict[point] for point in points]

    vmin = 0.0
    vmax = max(values)

    cmap = cm.get_cmap("viridis")  # Select colormap

    # Plot each composition with color intensity based on the value
    for point, value in zip(points, values):
        tax.scatter([point], c=value, marker='o', s=175, vmin=vmin, vmax=vmax, cmap=cmap)

    tax.get_axes().set_aspect('equal', adjustable='box')

    # Add color bar to the right
    # Create a normalization object with the desired range
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Create a color bar with the specified normalization
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=tax.get_axes(),
        orientation="vertical",
        pad=0.1
    )
    #cbar.set_label("Maximum MAE", fontsize=12)

    # Set corner labels
    labels = [atom_type_mapping.get(i, "") for i in range(3)]
    tax.left_axis_label(labels[0], fontsize=12, offset=0.15)   # Component A
    tax.right_axis_label(labels[1], fontsize=12, offset=0.15)  # Component B
    tax.bottom_axis_label(labels[2], fontsize=12, offset=0.15)    # Component C

    # Force redraw of labels to ensure they appear
    tax._redraw_labels()

    # Save the plot to a .png file
    plt.savefig(output_file, dpi=300)
    print(f"Ternary plot saved to {output_file}")

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


def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":

    modelname = "NbTaV"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        help="input file",
        type=str,
        default=f"./logs/{modelname}/config.json",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios gan_dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle gan_dataset",
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

    datasetname = "NbTaV"

    comm.Barrier()

    timer = Timer("load_data")
    timer.start()
    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir,
            label="trainset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        valset = SimplePickleDataset(
            basedir=basedir,
            label="valset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        testset = SimplePickleDataset(
            basedir=basedir,
            label="testset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        pna_deg = trainset.pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = torch.nn.parallel.DistributedDataParallel(model)

    load_existing_model(model, modelname, path="./logs/")
    model.eval()

    variable_index = 0
    for output_name, output_type, output_dim in zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
    ):

        test_MAE = 0.0

        num_samples = len(testset)
        true_values = []
        predicted_values = []

        # Batch size
        from torch_geometric.loader import DataLoader
        batch_size = 1
        test_data_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        # Determine atom type mapping
        atom_types = get_atom_type_mapping(test_data_loader)

        # Process the data and plot the ternary field
        composition_dict = error_map(model, test_data_loader, atom_types)

        atom_type_mapping = {i: f"{atomic_number_to_symbol[atom]}" for i, atom in enumerate(atom_types)}

        plot_ternary_field(composition_dict, atom_type_mapping)



