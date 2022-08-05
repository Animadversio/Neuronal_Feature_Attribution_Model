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
"L4Btn0-10-6-6"
"eval_predict_-layer3-ImageNet.csv"
"evol_regress_results.csv"
#%%
df_all.groupby(by="layer", level=(0,1)).agg({"rho_p": ["mean", "sem"], "D2":["mean","sem"], "n_feat":"mean"})
#%%
df_all.reset_index(level=(0,1)).groupby(by=["layer", "level_0", "level_1", ])\
    .agg({"rho_p": ["mean", "sem"], "D2":["mean","sem"], "n_feat":["mean","count"],
           "evol_test_D2":["mean","sem"]})
