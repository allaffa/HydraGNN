import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import adios2 as ad2
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
plt.rcParams.update({'font.size': 22})

data_list = list() 
for fn in ["ANI1x-v2", "MPTrj-v2", "qm7x-v2", "OC2022-v2", "OC2020-v2"]:
    dirname = "examples/multidataset/dataset"
    with ad2.open(os.path.join(dirname, fn+".bp"), "r") as f:
        f.__next__()
        x = f.read("trainset/x")
        vcount = f.read("trainset/x/variable_count")
        voffset = f.read("trainset/x/variable_offset")
    print("Adios reading done:", fn)

    for i in tqdm(range(len(vcount))):
        cnt = vcount[i]
        off = voffset[i]
        sample = x[off:off+cnt,]
        data_list.append(sample)

# Initialize a dictionary to store frequencies of each element
element_frequencies = {}

# Iterate over each data object and count atomic numbers
for data in data_list:
    atomic_numbers = data[:, 0].tolist()  # Extract atomic numbers
    for atomic_number in atomic_numbers:
        if atomic_number in element_frequencies:
            element_frequencies[atomic_number] += 1
        else:
            element_frequencies[atomic_number] = 1

# Periodic Table Element Symbols
element_symbols = [
    "H", "He", 
    "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", 
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", 
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

# Define the layout of the periodic table
periodic_table_layout = {
    1: (0, 0),  2: (0, 17),
    3: (1, 0),  4: (1, 1),  5: (1, 12),  6: (1, 13),  7: (1, 14),  8: (1, 15),  9: (1, 16), 10: (1, 17),
    11: (2, 0), 12: (2, 1), 13: (2, 12), 14: (2, 13), 15: (2, 14), 16: (2, 15), 17: (2, 16), 18: (2, 17),
    19: (3, 0), 20: (3, 1), 21: (3, 2),  22: (3, 3),  23: (3, 4),  24: (3, 5),  25: (3, 6),  26: (3, 7),
    27: (3, 8), 28: (3, 9), 29: (3, 10), 30: (3, 11), 31: (3, 12), 32: (3, 13), 33: (3, 14), 34: (3, 15),
    35: (3, 16), 36: (3, 17),
    37: (4, 0), 38: (4, 1), 39: (4, 2),  40: (4, 3),  41: (4, 4),  42: (4, 5),  43: (4, 6),  44: (4, 7),
    45: (4, 8), 46: (4, 9), 47: (4, 10), 48: (4, 11), 49: (4, 12), 50: (4, 13), 51: (4, 14), 52: (4, 15),
    53: (4, 16), 54: (4, 17),
    55: (5, 0), 56: (5, 1), 57: (5, 2),  72: (5, 3),  73: (5, 4),  74: (5, 5),  75: (5, 6),  76: (5, 7),
    77: (5, 8), 78: (5, 9), 79: (5, 10), 80: (5, 11), 81: (5, 12), 82: (5, 13), 83: (5, 14), 84: (5, 15),
    85: (5, 16), 86: (5, 17),
    87: (6, 0), 88: (6, 1), 89: (6, 2), 104: (6, 3), 105: (6, 4), 106: (6, 5), 107: (6, 6), 108: (6, 7),
    109: (6, 8), 110: (6, 9), 111: (6, 10), 112: (6, 11), 113: (6, 12), 114: (6, 13), 115: (6, 14), 116: (6, 15),
    117: (6, 16), 118: (6, 17),
    58: (7, 2),  59: (7, 3),  60: (7, 4),  61: (7, 5),  62: (7, 6),  63: (7, 7),  64: (7, 8),  65: (7, 9),
    66: (7, 10), 67: (7, 11), 68: (7, 12), 69: (7, 13), 70: (7, 14), 71: (7, 15),
    90: (8, 2),  91: (8, 3),  92: (8, 4),  93: (8, 5),  94: (8, 6),  95: (8, 7),  96: (8, 8),  97: (8, 9),
    98: (8, 10), 99: (8, 11), 100: (8, 12), 101: (8, 13), 102: (8, 14), 103: (8, 15)
}

# Prepare a 2D grid with the same size as the periodic table layout
max_row = max([pos[0] for pos in periodic_table_layout.values()]) + 1
max_col = max([pos[1] for pos in periodic_table_layout.values()]) + 1
heatmap = np.zeros((max_row, max_col))

# Fill in the heatmap grid with element frequencies
for atomic_number, freq in element_frequencies.items():
    if atomic_number in periodic_table_layout:
        row, col = periodic_table_layout[atomic_number]
        heatmap[row, col] = freq

# Create a custom colormap that includes white for the background 
cmap = ListedColormap(['white'] + plt.cm.YlGnBu(np.linspace(0, 1, 256)).tolist())


# Plotting the heatmap
plt.figure(figsize=(18, 10))
plt.imshow(heatmap, cmap=cmap, interpolation='nearest')
plt.colorbar(label='Frequency')
plt.xticks([])  # Remove x-axis ticks for a cleaner look
plt.yticks([])  # Remove y-axis ticks for a cleaner look

# Adding borders around each element 
ax = plt.gca() 
for atomic_number, (row, col) in periodic_table_layout.items(): 
    rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
                                                                                                        
# Annotating elements in the heatmap
for atomic_number, (row, col) in tqdm(periodic_table_layout.items()):
    freq = heatmap[row, col]
    element_symbol = element_symbols[atomic_number - 1]
    annotation = f"{atomic_number}\n{element_symbol}"
    plt.text(col, row, annotation, ha='center', va='center', 
             color='black' if freq < heatmap.max()//2 else 'white', fontsize=10)

plt.title('Periodic Table Heatmap of Element Frequencies')
plt.show()
plt.savefig('periodic_table_heatmap.png', dpi=300, bbox_inches='tight')
