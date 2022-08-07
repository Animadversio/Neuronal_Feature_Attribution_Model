"""

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

import pickle as pkl
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr
from core.neural_data_loader import load_score_mat

# #%%
# for Animal in ["Alfa", "Beto"]:
#     MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
#     EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
#     ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
#     for Expi in range(1, len(EStats) + 1):
#         # if Animal == "Alfa" and Expi <= 10: continue
#         expdir = join(saveroot, f"{Animal}_{Expi:02d}")
#         os.makedirs(expdir, exist_ok=True)
#         # load the image paths and the response vector
#         score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="N")

#%%
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from easydict import EasyDict as edict
from core.plot_utils import saveallforms
from core.dataset_utils import ImagePathDataset, DataLoader
from core.CNN_scorers import load_featnet
from core.layer_hook_utils import featureFetcher
from core.neural_regress.cnn_readout_model import FactorizedConv2D, SeparableConv2D, MultilayerCnn
from core.neural_regress.regress_lib import train_test_split, calc_reduce_features, calc_reduce_features_dataset
from core.neural_regress.cnn_train_utils import Weight_Laplacian, grad_diagnose


def test_model_dataset(featnet, featFetcher, model, loader_test, label=None, modelid=None):
    model.eval()
    scores_vec = []
    pred_vec = []
    for j, (imgs, scores) in tqdm(enumerate(loader_test)):
        imgs = imgs.cuda()
        scores_norm = ((scores - score_m) / score_s).float().cuda().unsqueeze(1)
        with torch.no_grad():
            featnet(imgs)
            feat = featFetcher[regresslayer]
            pred = model(feat, )
        scores_vec.append(scores_norm.cpu().numpy())
        pred_vec.append(pred.cpu().numpy())
    scores_vec = np.concatenate(scores_vec, axis=0)
    pred_vec = np.concatenate(pred_vec, axis=0)
    S = edict()
    S.mean_err = ((scores_vec - pred_vec)**2).mean()
    S.FEV = 1 - np.var(scores_vec - pred_vec) / np.var(scores_vec)
    S.cval, S.pval = pearsonr(scores_vec[:, 0], pred_vec[:, 0])
    if label is not None:
        S.label = label
    if modelid is not None:
        S.modelid = modelid
    return scores_vec, pred_vec, S


def test_multimodel_dataset(featnet, featFetcher, model_col, loader_test, score_m, score_s, label=None):
    for k, model in model_col.items():
        model.eval()
    scores_vec = []
    pred_col = defaultdict(list)
    for j, (imgs, scores) in tqdm(enumerate(loader_test)):
        imgs = imgs.cuda()
        scores_norm = ((scores - score_m) / score_s).float().cuda().unsqueeze(1)
        with torch.no_grad():
            featnet(imgs)
            feat = featFetcher[regresslayer]
            for k, model in model_col.items():
                pred = model(feat, )
                pred_col[k].append(pred.cpu().numpy())

        scores_vec.append(scores_norm.cpu().numpy())

    scores_vec = np.concatenate(scores_vec, axis=0)
    pred_dict = {}
    S_dict = {}
    for k, model in model_col.items():
        pred_vec = np.concatenate(pred_col[k], axis=0)
        pred_dict[k] = pred_vec
        S = edict()
        # S.pred_vec = pred_vec
        S.mean_err = ((scores_vec - pred_vec)**2).mean()
        S.FEV = 1 - np.var(scores_vec - pred_vec) / np.var(scores_vec)
        S.cval, S.pval = pearsonr(scores_vec[:, 0], pred_vec[:, 0])
        S.modelid = k
        if label is not None:
            S.label = label
        S_dict[k] = S
    return scores_vec, pred_dict, S_dict


def test_model(featnet, featFetcher, model, imgfps, scores, img_dim=(224, 224), label=None, modelid=None):
    dataset_tmp = ImagePathDataset(imgfps, scores, img_dim=img_dim)
    loader_tmp = DataLoader(dataset_tmp, batch_size=120, shuffle=False, num_workers=8, drop_last=False)
    scores_vec, pred_vec, S = test_model_dataset(featnet, featFetcher, model, loader_tmp, label=label, modelid=modelid)
    return scores_vec, pred_vec, S

#%%
regresslayer = ".layer3.Bottleneck5"
featnet, net = load_featnet("resnet50_linf8")
featFetcher = featureFetcher(featnet, input_size=(3, 224, 224),
                             device="cuda", print_module=False)
featFetcher.record(regresslayer, ingraph=True)
#%%
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress_SGD"
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"

Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
#%%
Expi = 3
# load the image paths and the response vector
expdir = join(saveroot, f"{Animal}_{Expi:02d}")
os.makedirs(expdir, exist_ok=True)

score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="N")
score_m, score_s = torch.tensor(score_vect.mean()), torch.tensor(score_vect.std())

score_train, score_test, imgfp_train, imgfp_test = train_test_split(
    score_vect, imgfullpath_vect, test_size=0.2, random_state=42)
dataset = ImagePathDataset(imgfp_train, score_train, img_dim=(224, 224))
dataset_test = ImagePathDataset(imgfp_test, score_test, img_dim=(224, 224))
loader = DataLoader(dataset, batch_size=120, shuffle=True, num_workers=8, drop_last=True)
loader_test = DataLoader(dataset_test, batch_size=120, shuffle=False, num_workers=8, drop_last=False)

#%% Model training pipeline
beta_lpls = 0.01
beta_L2 = 0.0001
for triali in range(0, 5):
    model = FactorizedConv2D(1024, 1, kernel_size=(14, 14), factors=3, bn=False).cuda()
    optim = Adam(model.parameters(), lr=0.001, )  # weight_decay=0.001)
    # optim = SGD(model.parameters(), lr=0.001, )  # weight_decay=0.001)
    trial_dir = join(expdir, "FactorConv_lpls001_l200001_nobn_tr%d" % triali)
    tsbd = SummaryWriter(trial_dir, comment="FactorConv_tr%d_001_00001_nobn" % triali)
    best_val_loss = 1e10
    for epoch in tqdm(range(0, 20)): # 60 is too much will overfit. 20 is enough to get the best result
        cum_loss = 0
        img_cnt = 0
        model.train()
        for i, (imgs, scores) in enumerate(loader): # tqdm
            imgs = imgs.cuda()
            scores_norm = ((scores - score_m) / score_s).float().cuda().unsqueeze(1)
            scores = (scores / score_s).float().cuda().unsqueeze(1)
            with torch.no_grad():
                featnet(imgs)
                feat = featFetcher[regresslayer]
            pred = model(feat, )
            # pred_nl = nn.Softplus()(pred)
            # loss = (pred_nl - scores_norm * torch.log(pred_nl)).mean()
            MSEloss = torch.mean((pred - scores_norm)**2)
            lplsloss = Weight_Laplacian(model.depth_conv)
            featvecL2 = model.point_conv.weight.norm()**2
            loss = MSEloss + lplsloss * beta_lpls + featvecL2 * beta_L2  # 0.01 0.0001
            optim.zero_grad()
            loss.backward()
            optim.step()
            cum_loss += loss.item() * imgs.shape[0]
            img_cnt += imgs.shape[0]
            if pred.std().item() < 0.001:
                raise ValueError("pred.std() < 0.001, stop training")
            if i % 10 == 0:
                print("loss %.3f avg loss %.3f laplace %.3f  featL2 %.3f   pred std %.3f score std %.3f"%
                      (loss.item(), (cum_loss / img_cnt), lplsloss.item(), featvecL2.item(),
                       pred.std(), scores_norm.std(),))
                tsbd.add_scalar("loss", loss.item(), epoch * len(loader) + i)
                tsbd.add_scalar("avg_loss", (cum_loss / img_cnt), epoch * len(loader) + i)
                tsbd.add_scalar("MSE_loss", MSEloss.item(), epoch * len(loader) + i)
                tsbd.add_scalar("laplace_loss", lplsloss.item(), epoch * len(loader) + i)
                tsbd.add_scalar("featvec_L2", featvecL2.item(), epoch * len(loader) + i)
                tsbd.add_scalar("pred_std", pred.std().item(), epoch * len(loader) + i)
                grad_diagnose(model, tsbd, epoch * len(loader) + i)
            # raise Exception("stop")
        # 0.224 at 60 gen epochs
        scores_vec, pred_vec, S = test_model_dataset(featnet, featFetcher, model, loader_test)
        mean_test_err = S.mean_err
        tsbd.add_scalar("test_MSE", mean_test_err, epoch * len(loader) + i)
        print("epoc test MSE %.3f" % (mean_test_err,))
        if mean_test_err < best_val_loss:
            best_val_loss = mean_test_err
            torch.save(model.state_dict(), join(trial_dir, "model_best.pt"))
            torch.save({"epoch": epoch, "step": epoch * len(loader) + i, "test_err": mean_test_err,
                        "beta_lpls": beta_lpls, "beta_L2": beta_L2},
                       join(trial_dir, "model_best_info.pt"))
            print("best model saved")

    torch.save(model.state_dict(), join(trial_dir, "model_last.pt"))


#%%
score_manif, imgfp_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="N")
score_pasu, imgfp_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="N")
score_gab, imgfp_gab = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="N")
score_natref, imgfp_natref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="N")
manif_loader = DataLoader(ImagePathDataset(imgfp_manif, score_manif, img_dim=(224, 224), ), num_workers=8, batch_size=120, shuffle=False)
pasu_loader = DataLoader(ImagePathDataset(imgfp_pasu, score_pasu, img_dim=(224, 224), ), num_workers=8, batch_size=120, shuffle=False)
gab_loader = DataLoader(ImagePathDataset(imgfp_gab, score_gab, img_dim=(224, 224), ), num_workers=8, batch_size=120, shuffle=False)
natref_loader = DataLoader(ImagePathDataset(imgfp_natref, score_natref, img_dim=(224, 224), ), num_workers=8, batch_size=120, shuffle=False)
#%%
model_col = {}
for triali in range(0, 6):
    trial_dir = join(expdir, "FactorConv_lpls001_l200001_nobn_tr%d" % triali)
    # expdir_tr = join(expdir, "FactorConv_tr%d" % triali)
    try:
        model = FactorizedConv2D(1024, 1, kernel_size=(14, 14), factors=3, bn=False).cuda()
        model.load_state_dict(torch.load(join(trial_dir, "model_best.pt")))
        model_col[triali] = model
    except FileNotFoundError:
        continue

scores_vec, preddict_vec, Scol = test_multimodel_dataset(featnet, featFetcher, model_col, loader_test, label="Evoltest", )
scores_manif, preddict_manif, Scol_manif = test_multimodel_dataset(featnet, featFetcher, model_col, manif_loader, label="Manif", )
scores_pasu, preddict_pasu, Scol_pasu = test_multimodel_dataset(featnet, featFetcher, model_col, pasu_loader, label="Pasu", )
scores_gabor, preddict_gabor, Scol_gabor = test_multimodel_dataset(featnet, featFetcher, model_col, gab_loader, label="Gabor", )
scores_natref, preddict_natref, Scol_natref = test_multimodel_dataset(featnet, featFetcher, model_col, natref_loader, label="NatRef", )
# Scol.extend([S, S_manif, S_pasu, S_gabor, S_natref])
#%%
pkl.dump({"scores_evol":scores_vec, "preddict_evol":preddict_vec, "Scol_evol":Scol,
        "scores_manif":scores_manif, "preddict_manif":preddict_manif, "Scol_manif":Scol_manif,
        "scores_pasu":scores_pasu, "preddict_pasu":preddict_pasu, "Scol_pasu":Scol_pasu,
        "scores_gabor":scores_gabor, "preddict_gabor":preddict_gabor, "Scol_gabor":Scol_gabor,
        "scores_natref":scores_natref, "preddict_natref":preddict_natref, "Scol_natref":Scol_natref,},
         open(join(expdir, "all_tr_all_imgset_pred_perform.pkl"), "wb"))
df_all = pd.concat([pd.DataFrame(Scol).T,
           pd.DataFrame(Scol_manif).T,
           pd.DataFrame(Scol_pasu).T,
           pd.DataFrame(Scol_gabor).T,
           pd.DataFrame(Scol_natref).T], axis=0)
df_all.to_csv(join(expdir, "all_tr_all_imgset_pred_perform.csv"))
#%%
Scol = []
for triali in range(0, 6):
    trial_dir = join(expdir, "FactorConv_lpls001_l200001_nobn_tr%d" % triali)
    # expdir_tr = join(expdir, "FactorConv_tr%d" % triali)
    try:
        model = FactorizedConv2D(1024, 1, kernel_size=(14, 14), factors=3, bn=False).cuda()
        model.load_state_dict(torch.load(join(trial_dir, "model_best.pt")))
    except FileNotFoundError:
        continue
    # info = torch.load(join(expdir, "FactorConv_tr%d_best_info.pt"))
    scores_vec, pred_vec, S = test_model_dataset(featnet, featFetcher, model, loader_test,
                                                 label="Evoltest", modelid=triali)
    scores_manif_, pred_manif, S_manif = test_model_dataset(featnet, featFetcher, model, manif_loader,
                                                 label="Manif", modelid=triali)
    scores_pasu_, pred_pasu, S_pasu = test_model_dataset(featnet, featFetcher, model, pasu_loader,
                                                 label="Pasu", modelid=triali)
    scores_gabor_, pred_gabor, S_gabor = test_model_dataset(featnet, featFetcher, model, gab_loader,
                                                 label="Gabor", modelid=triali)
    scores_natref_, pred_natref, S_natref = test_model_dataset(featnet, featFetcher, model, natref_loader,
                                                 label="NatRef", modelid=triali)
    Scol.extend([S, S_manif, S_pasu, S_gabor, S_natref])

    plt.figure(figsize=(5, 5))
    plt.scatter(scores_vec[:, 0], pred_vec[:, 0], s=1, c="k")
    plt.axline([0, 0], [1, 1], color="r", alpha=0.5)
    plt.xlabel("scores")
    plt.ylabel("pred")
    plt.title("Trial %d FEV %.3f\ntest MSE %.3f corr %.3f (%.1e)" %
              (triali, S.FEV, S.mean_err, S.cval, S.pval))
    saveallforms(trial_dir, "test_scatter_lpls_best")
    plt.show()

    wtsr = model.depth_conv.weight.data.cpu().numpy()
    wmat = wtsr.squeeze().transpose((1, 2, 0))
    plt.figure(figsize=(5, 5))
    plt.imshow(wmat / wmat.max())
    saveallforms(trial_dir, "depth_conv_weights_lpls_best")
    plt.show()

#%%
df = pd.DataFrame(Scol)
df.to_csv(join(expdir, "all_tr_all_imgset_pred_perform.csv"))
#%%



#%%
#%% Fit the factorized or PCA model on the same data.
from core.CorrFeatFactor.CorrFeatTsr_data_load import load_NMF_factors
from core.neural_regress.regress_lib import sweep_regressors, Ridge, Lasso, evaluate_prediction

bdr = 1
layer = "layer3"
featlayer = ".layer3.Bottleneck5"
Hmat3, Hmaps3, ccfactor3, FactStat = load_NMF_factors(Animal, Expi, layer, NF=3)
padded_mask3 = np.pad(Hmaps3[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
Xfeat_transformer = {"spmask3": lambda tsr: np.einsum("BCHW,HWF->BFC", tsr, padded_mask3).reshape(tsr.shape[0], -1),
                     "featvec3": lambda tsr: np.einsum("BCHW,CF->BFHW", tsr, ccfactor3).reshape(tsr.shape[0], -1),
                     "factor3": lambda tsr: np.einsum("BCHW,CF,HWF->BF", tsr, ccfactor3, padded_mask3)
                            .reshape(tsr.shape[0], -1),
                     "facttsr3": lambda tsr: np.einsum("BCHW,CF,HWF->B", tsr, ccfactor3, padded_mask3)
                            .reshape(tsr.shape[0], -1),
                     }

#%%
score_train, score_test, imgfp_train, imgfp_test = train_test_split(
    score_vect, imgfullpath_vect, test_size=0.2, random_state=42)
dataset = ImagePathDataset(imgfp_train, score_train, img_dim=(224, 224), )
loader = DataLoader(dataset, batch_size=120, shuffle=True, num_workers=8, drop_last=True)
dataset_test = ImagePathDataset(imgfp_test, score_test, img_dim=(224, 224), )
loader_test = DataLoader(dataset_test, batch_size=120, shuffle=False, num_workers=8, drop_last=False)

#%%
model = FactorizedConv2D(1024, 1, kernel_size=(14, 14), factors=3).cuda()
model.depth_conv.weight.data = torch.from_numpy(padded_mask3).float().cuda().permute([2, 0, 1]).unsqueeze(1)
model.point_conv.weight.data = torch.from_numpy(ccfactor3).float().cuda().permute([1, 0]).unsqueeze(2).unsqueeze(3)
model.linear.weight.data = torch.ones_like(model.linear.weight.data)
scores_vec, pred_vec, S = test_model_dataset(featnet, featFetcher, model, loader_test)

#%%
Xfeat_dict = calc_reduce_features_dataset(dataset, Xfeat_transformer, featnet, featlayer,
                                          img_dim=(224, 224), batch_size=120, workers=8, )
Xfeat_test_dict = calc_reduce_features_dataset(dataset_test, Xfeat_transformer, featnet, featlayer,
                                          img_dim=(224, 224), batch_size=120, workers=8, )
#%%
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
result_df, model_list = sweep_regressors(Xfeat_dict, (score_train - score_m.numpy()) / score_s.numpy(),
                                         [ridge, lasso], ["Ridge", "Lasso"],)
#%%
df_pred, eval_dict, y_pred_dict = evaluate_prediction(model_list, Xfeat_test_dict,
                                      (score_test - score_m.numpy()) / score_s.numpy(),
                                                 label="layer3-Evol", savedir=expdir)
print(df_pred)
#%%
def transfer_weight2model():
    model.depth_conv.weight.data = torch.from_numpy(padded_mask3).float().cuda().permute([2, 0, 1]).unsqueeze(1)
    model.point_conv.weight.data = torch.from_numpy(ccfactor3).float().cuda().permute([1, 0]).unsqueeze(2).unsqueeze(3)
    model.linear.weight.data = torch.ones_like(model.linear.weight.data)
    return model

#%%
#%%
#%%



#%% visualize the model factors for factor regression model
wtsr = model_list[('featvec3', 'Ridge')].best_estimator_.coef_.reshape(3,14,14)
wtsr = wtsr.transpose(1,2,0)
plt.figure()
plt.imshow(wtsr / wtsr.max(), )
plt.show()
