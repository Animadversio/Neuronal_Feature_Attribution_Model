"""
 * load in linear weights and the linear projection weights
 * port the linear weights to a torch model.
 * visualize the model by back propagation.
 * Sumarize the images.
"""
# %% Newer version pipeline
import os
import pickle as pkl
from os.path import join

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

from core.CNN_scorers import load_featnet
from core.plot_utils import save_imgrid
from core.GAN_utils import upconvGAN
from core.layer_hook_utils import featureFetcher
from core.montage_utils import crop_from_montage, make_grid_np
from core.neural_regress.sklearn_torchify_lib import \
    LinearRegression_torch, PLS_torch, PCA_torch, SpatialAvg_torch, SRP_torch


saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
#%%
G = upconvGAN()
G.eval().cuda()
G.requires_grad_(False)
featnet, net = load_featnet("resnet50_linf8")
net.eval().cuda()
RGBmean = torch.tensor([0.485, 0.456, 0.406]).cuda().reshape(1, 3, 1, 1)
RGBstd = torch.tensor([0.229, 0.224, 0.225]).cuda().reshape(1, 3, 1, 1)

#%%
# featlayer = ".layer3.Bottleneck5"
# featlayer = ".layer4.Bottleneck2"
featlayer = ".layer2.Bottleneck3"
Xtfm_fn = f"{featlayer}_regress_Xtransforms.pkl"
data = pkl.load(open(join(saveroot, Xtfm_fn), "rb"))
#%%
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        regr_data = pkl.load(open(join(expdir, f"{featlayer}_regress_models.pkl"), "rb"))
        featvis_dir = join(expdir, "featvis")
        os.makedirs(featvis_dir, exist_ok=True)
        #%%
        regr_cfgs = [*regr_data.keys()]
        for regrlabel in regr_cfgs:
            Xtype, regressor = regrlabel
            print(f"Processing {Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}")
            clf = regr_data[regrlabel]
            if Xtype == 'pca':
                Xtfm_th = PCA_torch(data['pca'], device="cpu")
            elif Xtype == 'srp':
                Xtfm_th = SRP_torch(data['srp'], device="cpu")
            elif Xtype == 'sp_avg':
                Xtfm_th = SpatialAvg_torch()
            if isinstance(clf, GridSearchCV):
                clf_th = LinearRegression_torch(clf, device="cpu")
            elif isinstance(clf, PLSRegression):
                clf_th = PLS_torch(clf, device="cpu")
            else:
                raise ValueError("Unknown regressor type")
            #%% Visualize the model
            B = 5
            fetcher = featureFetcher(net, input_size=(3, 227, 227))
            fetcher.record(featlayer, ingraph=True)
            zs = torch.randn(B, 4096).cuda()
            zs.requires_grad_(True)
            optimizer = Adam([zs], lr=0.1)
            for i in range(100):
                optimizer.zero_grad()
                imgtsrs = G.visualize(zs)
                imgtsrs_rsz = F.interpolate(imgtsrs, size=(227, 227), mode='bilinear', align_corners=False)
                imgtsrs_rsz = (imgtsrs_rsz - RGBmean) / RGBstd
                featnet(imgtsrs_rsz)
                activations = fetcher[featlayer]
                featmat = Xtfm_th(activations.cpu()) # .flatten(start_dim=1)
                scores = clf_th(featmat)
                loss = - scores.sum()
                loss.backward()
                optimizer.step()
                print(f"{i}, {scores.sum().item():.3f}")

            # show_imgrid(imgtsrs)
            save_imgrid(imgtsrs, join(featvis_dir,
                      f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}_vis.png"))
#%%
"""Summarize the results into a montage acrsoo methods"""
# featlayer = ".layer3.Bottleneck5"
featlayer = ".layer4.Bottleneck2"

for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        featvis_dir = join(expdir, "featvis")
        proto_col = []
        for regrlabel in regr_cfgs:
            Xtype, regressor = regrlabel
            mtg = plt.imread(join(featvis_dir, f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}_vis.png"))
            proto_first = crop_from_montage(mtg, (0, 0))
            proto_col.append(proto_first)
        method_mtg = make_grid_np(proto_col, nrow=3)
        plt.imsave(join(featvis_dir, f"{Animal}-Exp{Expi:02d}-{featlayer}-regr_merge_vis.png"), method_mtg, )

#%%
# regr_cfgs = [ ('srp', 'Ridge'),
#  ('srp', 'Lasso'),
#  ('srp', 'PLS'),
#              ('sp_avg', 'Ridge'),
#              ('sp_avg', 'Lasso'),
#              ('sp_avg', 'PLS'),
#              ('pca', 'Ridge'),
#              ('pca', 'Lasso'),
#              ('pca', 'PLS'),
# ]