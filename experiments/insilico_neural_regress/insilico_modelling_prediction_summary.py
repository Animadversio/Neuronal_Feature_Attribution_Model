"""
Summarize statistics about the prediction for in silico modelling experiments
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
def summarize_pred_exp():
    df_all = None
    subdirs = [dir for dir in os.listdir(join(rootdir)) if "Btn" in dir]
    for subdir in subdirs:
        try:
            df_INet = pd.read_csv(join(rootdir, subdir, f"eval_predict_-layer3-ImageNet.csv"), index_col=[0,1])
            df_evol = pd.read_csv(join(rootdir, subdir, f"evol_regress_results.csv"), index_col=[0,1])
        except FileNotFoundError:
            print(f"{join(rootdir, subdir, f'eval_predict_-layer3-ImageNet.csv')} not found")
            continue
        df_part = df_INet.merge(df_evol, left_index=True, right_index=True)
        df_part.rename(columns={"train_score": "evol_train_D2", "test_score": "evol_test_D2"}, inplace=True)
        # csvlist = glob(join(sumdir, "*.csv"))
        # df_all = None
        # for fpath in csvlist:
        #     fname = os.path.basename(fpath)
        # return df_INet, df_evol, df_part
        parts = subdir.split("-")
        layer_s, chan, x, y = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        match = re.findall("L(\d)Btn(\d)", layer_s)
        layer_i, sublayer_i = int(match[0][0]), int(match[0][1])
        # df_part = pd.read_csv(fpath, index_col=0)
        df_part["layerstr"] = layer_s
        df_part["layer"] = layer_i
        df_part["sublayer"] = sublayer_i
        df_part["chan"] = chan
        df_part["x"] = x
        df_part["y"] = y
        df_all = pd.concat([df_all, df_part]) if df_all is not None else df_part
    return df_all


# df_INet, df_evol, df_part = summarize_pred_exp()
df_all = summarize_pred_exp()
#%%
df_all.to_csv(join(figdir, "all_target_prediction_summary.csv"))
#%%
"L4Btn0-10-6-6"
"eval_predict_-layer3-ImageNet.csv"
"evol_regress_results.csv"
#%%
df_all.groupby(by="layer", level=(0,1)).agg({"rho_p": ["mean", "sem"], "D2":["mean","sem"], "n_feat":"mean"})
#%%
#%%
df_all.reset_index(level=(0,1)).groupby(by=["layer", "level_0", "level_1", ])\
    .agg({"rho_p": ["mean", "sem"], "D2":["mean","sem"], "n_feat":["mean","count"], "evol_test_D2":["mean","sem"]})
#%%
df_all.reset_index(level=(0,1)).groupby(by=["layer", "level_0", "level_1", ])\
    .agg({"D2":["mean","sem"], "evol_test_D2":["mean","sem"], "n_feat":["mean","count"], })
#%%
df_all.reset_index(level=(0,1)).groupby(by=["level_0", "level_1", ], sort=False)\
    .agg({"D2":["mean","sem"], "evol_test_D2":["mean","sem"], "n_feat":["mean","count"], })
#%%
df_all.reset_index(level=(0,1)).groupby(by=["level_0", "level_1", ], sort=False)\
    .agg({"rho_p": ["mean", "sem"], "evol_test_D2":["mean","sem"], "n_feat":["mean","count"], })
#%%

df_allcol = df_all.reset_index(level=(0,1)).rename(columns={"level_0": "featred","level_1":"regressor"}, )
#%%
for yval in ["evol_test_D2", "D2", "rho_p"]:
    g = sns.FacetGrid(df_allcol, col="layer", row="sublayer", hue="regressor",
                      size=5, palette="Set2", )
    g.map(sns.barplot, "featred", yval, alpha=0.4, dodge=True,
          order=["factor3","spmask3","featvec3","pca","srp"]).add_legend()
    plt.savefig(join(figdir, f"{yval}_bar_by_layer-sublayer.png"))
    plt.savefig(join(figdir, f"{yval}_bar_by_layer-sublayer.pdf"))
    plt.show()
    # raise Exception("stop")
#%%
for yval in ["evol_test_D2", "D2", "rho_p"]: #
    g = sns.FacetGrid(df_allcol, col="layer", row="sublayer", size=5)
    g.map(sns.stripplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True,
          order=["factor3","spmask3","featvec3","pca","srp"]).add_legend()
    g.map(sns.pointplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True,
          order=["factor3","spmask3","featvec3","pca","srp"], join=True).add_legend()
    g.set(ylim=(-0.1, 1.0))
    plt.savefig(join(figdir, f"{yval}_strip_by_layer-sublayer.png"))
    plt.savefig(join(figdir, f"{yval}_strip_by_layer-sublayer.pdf"))
    plt.show()
#%%
for yval in ["evol_test_D2", "D2", "rho_p"]: #
    g = sns.FacetGrid(df_allcol, col="layer", size=5)
    g.map(sns.stripplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True,
          order=["factor3","spmask3","featvec3","pca","srp"]).add_legend()
    g.map(sns.pointplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True,
          order=["factor3","spmask3","featvec3","pca","srp"], join=True).add_legend()
    g.set(ylim=(-0.1, 1.0))
    plt.savefig(join(figdir, f"{yval}_strip_by_layer.png"))
    plt.savefig(join(figdir, f"{yval}_strip_by_layer.pdf"))
    plt.show()
#%%
for yval in ["evol_test_D2", "D2", "rho_p"]:  #
    g = sns.FacetGrid(df_allcol, size=5)
    g.map(sns.stripplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True,
          order=["factor3","spmask3","featvec3","pca","srp"]).add_legend()
    g.map(sns.pointplot, "featred", yval, "regressor", alpha=0.6, palette="Set2", dodge=True,
          order=["factor3","spmask3","featvec3","pca","srp"], join=True).add_legend()
    g.set(ylim=(-0.1, 1.0))
    plt.savefig(join(figdir, f"{yval}_strip_pooled.png"))
    plt.savefig(join(figdir, f"{yval}_strip_pooled.pdf"))
    plt.show()
