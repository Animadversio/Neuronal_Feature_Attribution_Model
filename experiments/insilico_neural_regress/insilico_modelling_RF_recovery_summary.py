"""
Summarize statistics about the RF recovery for in silico modelling experiments
"""
import os
import re
from glob import glob
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.plot_utils import saveallforms

rootdir = r"E:\OneDrive - Harvard University\CNN_neural_regression\insilico_final\resnet50_linf8-resnet50"
sumdir = join(rootdir, "summary")
figdir = join(rootdir, "summary_figs")
os.makedirs(figdir, exist_ok=True)
#%%
def summarize_RF_exp(sumdir):
    csvlist = glob(join(sumdir, "*.csv"))
    df_all = None
    for fpath in csvlist:
        fname = os.path.basename(fpath)
        parts = fname.split("-")
        layer_s, chan, x, y = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        match = re.findall("L(\d)Btn(\d)", layer_s)
        layer_i, sublayer_i = int(match[0][0]), int(match[0][1])
        df_part = pd.read_csv(fpath, index_col=0)
        df_part["layerstr"] = layer_s
        df_part["layer"] = layer_i
        df_part["sublayer"] = sublayer_i
        df_part["chan"] = chan
        df_part["x"] = x
        df_part["y"] = y
        df_all = pd.concat([df_all, df_part]) if df_all is not None else df_part
    return df_all


df_all = summarize_RF_exp(sumdir)
#%%
df_all.to_csv(join(figdir, "all_target_summary.csv"))
#%%
df_all = pd.read_csv(join(figdir, "all_target_summary.csv"))
#%%
df_all.groupby(["layer", "sublayer", "featred"])["cval", "cval_fit", "iou"].agg(["mean","sem"]) # ,"count"
#%%
""" summary statistics of RF recovery as a function of Feature reducer and regressor
ploted against layers - sublayer
"""
for yval in ["cval", "cval_fit", "iou"]:
    sns.FacetGrid(df_all, col="layer", row="sublayer", hue="regressor", size=5)\
        .map(sns.barplot, "featred", yval, alpha=0.4).add_legend()
    saveallforms(figdir, f"{yval}_bar_by_layer-sublayer")
    # plt.savefig(join(figdir, f"{yval}_bar_by_layer-sublayer.png"))
    # plt.savefig(join(figdir, f"{yval}_bar_by_layer-sublayer.pdf"))
    plt.show()
#%%
for yval in ["cval", "cval_fit", "iou"]:
    g = sns.FacetGrid(df_all, col="layer", row="sublayer", hue="regressor",
                      size=5, palette="Set2", )
    g.map(sns.barplot, "featred", yval, alpha=0.4, dodge=True,
          order=['spmask3', 'featvec3', 'factor3', 'pca', 'srp']).add_legend()
    # plt.savefig(join(figdir, f"{yval}_bar_by_layer-sublayer.png"))
    # plt.savefig(join(figdir, f"{yval}_bar_by_layer-sublayer.pdf"))
    plt.show()
    raise Exception("stop")
#%%
for yval in ["cval", ]: # "cval_fit", "iou"
    g = sns.FacetGrid(df_all, col="layer", row="sublayer", hue="regressor", size=5)
    g.map(sns.barplot, "featred", yval, alpha=0.4).add_legend()
    plt.savefig(join(figdir, f"{yval}_pnt_by_layer-sublayer-part1.pdf"))
    plt.show()
    g = sns.FacetGrid(df_all, col="layer", row="sublayer", size=5)
    g.map(sns.stripplot, "featred", yval, "regressor", alpha=0.4, palette="Set2", ).add_legend()
    # g = g.map(sns.stripplot, "featred", yval, "regressor", alpha=0.4).add_legend()
    g.set(ylim=(-0.1, 1.0))
    # plt.savefig(join(figdir, f"{yval}_pnt_by_layer-sublayer-part2.png"))
    plt.savefig(join(figdir, f"{yval}_pnt_by_layer-sublayer-part2.pdf"))
    plt.show()
#%%
for yval in ["cval", "cval_fit", "iou"]: #
    g = sns.FacetGrid(df_all, col="layer", row="sublayer", size=5)
    g.map(sns.stripplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True, ).add_legend()
    g.map(sns.pointplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True, join=True).add_legend()
    g.set(ylim=(-0.1, 1.0))
    plt.savefig(join(figdir, f"{yval}_strip_by_layer-sublayer.png"))
    plt.savefig(join(figdir, f"{yval}_strip_by_layer-sublayer.pdf"))
    plt.show()
#%%
for yval in ["cval", "cval_fit", "iou"]: #
    g = sns.FacetGrid(df_all, col="layer", size=5, aspect=0.8)
    g.map(sns.stripplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True, order=["factor3","spmask3","featvec3","pca","srp"]).add_legend()
    g.map(sns.pointplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True, order=["factor3","spmask3","featvec3","pca","srp"], join=True).add_legend()
    g.set(ylim=(-0.1, 1.0))
    plt.savefig(join(figdir, f"{yval}_strip_by_layer.png"))
    plt.savefig(join(figdir, f"{yval}_strip_by_layer.pdf"))
    plt.show()
#%%
for yval in ["cval", "cval_fit", "iou"]: #
    g = sns.FacetGrid(df_all, size=5)
    g.map(sns.stripplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True, order=["factor3","spmask3","featvec3","pca","srp"]).add_legend()
    g.map(sns.pointplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True, order=["factor3","spmask3","featvec3","pca","srp"], join=True).add_legend()
    g.set(ylim=(-0.1, 1.0))
    plt.savefig(join(figdir, f"{yval}_strip_pooled.png"))
    plt.savefig(join(figdir, f"{yval}_strip_pooled.pdf"))
    plt.show()



#%%
plt.figure(figsize=(10,10))
# sns.pointplot(data=df_all, x="featred", y=yval)
sns.stripplot(data=df_all, x="featred", y=yval, hue="regressor", alpha=0.4)
plt.show()
#%%
import pickle as pkl
expdir = r"E:\OneDrive - Harvard University\CNN_neural_regression\insilico_final\resnet50_linf8-resnet50\L3Btn3-25-3-3"
gradmap_dict = pkl.load(open(join(expdir, "model_gradmap_rfdict.pkl"), "wb"))
gradAmpmap, fitdict = pkl.load(open(join(expdir, "groundtruth_gradmap_rfdict.pkl"), "wb"))
df_rfstat, _, _, _ = summarize_rf_cmp(gradAmpmap, fitdict, gradmap_dict, expstr, expdir=[expdir, sumdir])