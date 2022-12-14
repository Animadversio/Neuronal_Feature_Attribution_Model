import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from core.neural_data_loader import load_score_mat, uniformize_psth

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('Agg')
# mpl.use('module://backend_interagg')
#%%
rootdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
sumdir  = join(rootdir, "summary")
figdir  = join(sumdir, "classic_figs")
os.makedirs(figdir, exist_ok=True)
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"


#%%
bslwdw = slice(0, 45)
evkwdw = slice(50, 200)
for Animal in ["Beto"]: # ["Alfa", "Beto"]:
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats) + 1):
        #%%
        S = EStats[Expi - 1]
        pref_chan = S.units.pref_chan
        unit_in_chan = S.units.unit_in_pref_chan if hasattr(S.units, "unit_in_pref_chan") else 1
        expstr = f"{Animal} Exp{Expi:02d} Chan{pref_chan:02d} Unit{unit_in_chan:02d}\n" \
                 f"img size {S.evol.imgsize}deg  at {S.evol.imgpos}\n{S.meta.ephysFN}"
        evoke_fr = []
        ref_evk_fr = []
        baseline_fr = []
        generation = []
        ref_gen = []
        Ngen = len(S.evol.psth)
        for igen in range(Ngen-1):
            evoke_fr.append(S.evol.psth[igen][evkwdw, :].mean(axis=0))
            baseline_fr.append(S.evol.psth[igen][bslwdw, :].mean(axis=0))
            generation.append((1+igen) * np.ones(S.evol.psth[igen].shape[1]))
            ref_evk_fr.append(S.ref.psth[igen][evkwdw, :].mean(axis=0))
            ref_gen.append((1+igen) * np.ones(S.ref.psth[igen].shape[1]))
        evoke_fr = np.concatenate(evoke_fr, axis=0)
        baseline_fr = np.concatenate(baseline_fr, axis=0)
        ref_evk_fr = np.concatenate(ref_evk_fr, axis=0)
        generation = np.concatenate(generation, axis=0)
        ref_gen = np.concatenate(ref_gen, axis=0)
        plt.figure(figsize=(6, 5.5))
        sns.lineplot(x=generation, y=evoke_fr, label="Evoked")
        sns.lineplot(x=generation, y=baseline_fr, label="Baseline")
        sns.lineplot(x=ref_gen, y=ref_evk_fr, label="Reference")
        plt.title(f"Evolution {expstr}", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Firing Rate (Hz)", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(figdir, f"{Animal}_Exp{Expi:02d}_Evolution.png"))
        plt.show()
        #%%%
        MS = MStats[Expi - 1]
        expstr = f"{Animal} Exp{Expi:02d} Chan{pref_chan:02d} Unit{unit_in_chan:02d}\n" \
                    f"img size {S.evol.imgsize}deg  at {S.evol.imgpos}\n{MS.meta.ephysFN}"
        # idxgrid = np.vectorize(lambda x: x[0])(MS.manif.idx_grid)
        # imgname_grid = MS.imageName[idxgrid-1]
        PC2ticks = np.arange(-90, 90+1, 18)
        PC3ticks = np.arange(-90, 90+1, 18)
        evk_act_func = np.vectorize(lambda x: x[evkwdw, :].mean(axis=0), otypes="O")
        bsl_act_func = np.vectorize(lambda x: x[bslwdw, :].mean(axis=0), otypes="O")
        evk_mean_func = np.vectorize(lambda x: x[evkwdw, :].mean(axis=(0, 1)), )
        if MS.manif.psth.shape == (11, 11):
            psth_arrs = [MS.manif.psth]
        elif MS.manif.psth.shape == (3, ):
            psth_arrs = MS.manif.psth
        else:
            raise ValueError("Unrecognized psth format {}".format(MS.manif.psth.shape))
        axesdict = {"":("PC2", "PC3"), "PC4950":("PC49", "PC50"), "RND12":("RND1", "RND2")}
        for psth_arr, spacelabel in zip(psth_arrs, ["", "PC4950", "RND12"]):
            format_psth = uniformize_psth(psth_arr, unit_in_chan)  # uniform the squeezed dimensions
            evkvec_mat = evk_act_func(format_psth)
            evk_mat = evk_mean_func(format_psth)
            bslvec_mat = bsl_act_func(format_psth)
            bslvec_all = np.concatenate(tuple(bslvec_mat.flatten()))
            bslvec_mean = bslvec_all.mean()
            plt.figure(figsize=(5.5, 5.5))
            sns.heatmap(evk_mat, xticklabels=PC3ticks, yticklabels=PC2ticks, cmap="viridis")
            plt.title(f"Manifold {expstr}", fontsize=14)
            plt.axis("image")
            vec1lab, vec2lab = axesdict[spacelabel]
            plt.xlabel(vec2lab, fontsize=12)
            plt.ylabel(vec1lab, fontsize=12)
            plt.tight_layout()
            plt.savefig(join(figdir, f"{Animal}_Exp{Expi:02d}_Manifold{spacelabel}.png"))
            # plt.savefig(join(figdir, f"{Animal}_Exp{Expi:02d}_Manifold.pdf"))
            plt.show()
            # raise NotImplementedError

#%% Export the prototyes images
for Animal in ["Alfa", "Beto"]:
    ReprStats = \
    loadmat(join(mat_path, Animal + "_ImageRepr.mat"),
            struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
            'ReprStats']
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        protoimg = ReprStats[Expi - 1].Manif.BestImg
        plt.imsave(join(sumdir, "proto", f"{Animal}_Exp{Expi:02d}_manif_proto.png"), protoimg)
