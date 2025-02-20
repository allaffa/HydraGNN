import sys
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 20})



if __name__ == "__main__":
    cases=["EL", "EL_ternary", "EL_cross_ternary2binary", "EL_cross_binary2ternary"]
    caseplots=["InD_Bin","InD_Tern", "OoD_Bin","OoD_Tern"]
    var_case=["_energy3", "_rmsd3"]
    variables=["formation_enthalpy","rmsd"]
    paren_dir="/lustre/orion/cph161/proj-shared/zhangp/HydraGNN_EL/logs"
    lines=["-","--"]
    colors=["r","g","b"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ivar in range(2):
        ax=axs[ivar]
        for icase, case in enumerate(cases):
            datafile=paren_dir+"/"+case+var_case[ivar]+"/uncertainty.npz"
            data = np.load(datafile)
            uncer_bins=data["uncer_bins"]
            count_ratio=data["count_ratio"]
            ax.plot(uncer_bins, count_ratio, colors[icase%2]+lines[icase//2],label=caseplots[icase],linewidth=2.0)
            ax.set_title(variables[ivar])
            ax.set_ylabel("Count Ratio")
            ax.set_xlabel("Uncertainties")
            ax.set_xscale("log")
        ax.legend(fontsize=18) 
    ##################################################################################################################
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.925, wspace=0.3, hspace=0.3)
    fig.savefig("./crosscheck_post_uncertainty.png",dpi=500)
    fig.savefig("./crosscheck_post_uncertainty.pdf")
    plt.close()
    ##################################################################################################################
    sys.exit(0)
