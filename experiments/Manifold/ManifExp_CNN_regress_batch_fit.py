"""
With the same base network, train K readout networks on top of it.
Study the ensemble performance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

import pickle as pkl
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr

from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from easydict import EasyDict as edict
from core.neural_data_loader import load_score_mat
from core.plot_utils import saveallforms
from core.dataset_utils import ImagePathDataset, DataLoader
from core.CNN_scorers import load_featnet
from core.layer_hook_utils import featureFetcher
from core.neural_regress.cnn_readout_model import FactorizedConv2D, SeparableConv2D, MultilayerCnn
from core.neural_regress.regress_lib import train_test_split, calc_reduce_features, calc_reduce_features_dataset
from core.neural_regress.cnn_train_utils import Weight_Laplacian, grad_diagnose, \
    test_model_dataset, test_multimodel_dataset
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress_SGD"
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"

regresslayer = ".layer3.Bottleneck5"
featnet, net = load_featnet("resnet50_linf8")
featFetcher = featureFetcher(featnet, input_size=(3, 224, 224),
                             device="cuda", print_module=False)
featFetcher.record(regresslayer, ingraph=True)

#%%
class ModelTrainer:

    def __init__(self, expstr, beta_lpls=0.01, beta_L2=0.0001, expdir=""): #" FactorConv_lpls001_l200001_nobn_tr%d" % triali
        self.model = FactorizedConv2D(1024, 1, kernel_size=(14, 14), factors=3, bn=False).cuda()
        self.optim = Adam(self.model.parameters(), lr=0.001, )  # weight_decay=0.001)
        # optim = SGD(model.parameters(), lr=0.001, )  # weight_decay=0.001)
        self.trial_dir = join(expdir, expstr)
        self.tsbd = SummaryWriter(self.trial_dir, comment=expstr)
        self.beta_lpls = beta_lpls
        self.beta_L2 = beta_L2
        self.global_step = 0
        self.img_cnt = 0
        self.cum_loss = 0
        self.best_val_loss = 1e10

    def new_epoch(self):
        self.img_cnt = 0
        self.cum_loss = 0
        self.model.train()

    def learn_step(self, feat, scores_norm, record=False):
        self.model.train()
        pred = self.model(feat, )
        # pred_nl = nn.Softplus()(pred)
        # loss = (pred_nl - scores_norm * torch.log(pred_nl)).mean()
        MSEloss = torch.mean((pred - scores_norm) ** 2)
        lplsloss = Weight_Laplacian(self.model.depth_conv)
        featvecL2 = self.model.point_conv.weight.norm() ** 2
        loss = MSEloss + lplsloss * self.beta_lpls + featvecL2 * self.beta_L2  # 0.01 0.0001
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.global_step += 1
        self.img_cnt += feat.shape[0]
        self.cum_loss += loss.item() * feat.shape[0]
        if record:
            print("loss %.3f avg loss %.3f laplace %.3f  featL2 %.3f   pred std %.3f score std %.3f" %
                  (loss.item(), (self.cum_loss / self.img_cnt), lplsloss.item(), featvecL2.item(),
                   pred.std(), scores_norm.std(),))
            self.tsbd.add_scalar("loss", loss.item(), self.global_step)
            self.tsbd.add_scalar("avg_loss", (self.cum_loss / self.img_cnt), self.global_step)
            self.tsbd.add_scalar("MSE_loss", MSEloss.item(), self.global_step)
            self.tsbd.add_scalar("laplace_loss", lplsloss.item(), self.global_step)
            self.tsbd.add_scalar("featvec_L2", featvecL2.item(), self.global_step)
            self.tsbd.add_scalar("pred_std", pred.std().item(), self.global_step)
            grad_diagnose(self.model, self.tsbd, self.global_step)

        return pred, loss.item()

    def test_step(self, feat, scores_norm):
        pass

    def save_curbest_model(self, mean_test_err):
        self.tsbd.add_scalar("test_MSE", mean_test_err, self.global_step)
        print("epoc test MSE %.3f" % (mean_test_err,))
        if mean_test_err < self.best_val_loss:
            self.best_val_loss = mean_test_err
            torch.save(self.model.state_dict(), join(self.trial_dir, "model_best.pt"))
            torch.save({"step": self.global_step, "test_err": mean_test_err,
                        "beta_lpls": self.beta_lpls, "beta_L2": self.beta_L2},
                       join(self.trial_dir, "model_best_info.pt"))
            print("best model saved")

Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']

#%%
for Expi in range(11, len(MStats)+1):
    # load the image paths and the response vector
    expdir = join(saveroot, f"{Animal}_{Expi:02d}")

    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="N")
    score_m, score_s = torch.tensor(score_vect.mean()), torch.tensor(score_vect.std())

    score_train, score_test, imgfp_train, imgfp_test = train_test_split(
        score_vect, imgfullpath_vect, test_size=0.2, random_state=42)
    dataset = ImagePathDataset(imgfp_train, score_train, img_dim=(224, 224))
    dataset_test = ImagePathDataset(imgfp_test, score_test, img_dim=(224, 224))
    loader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=8, drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=200, shuffle=False, num_workers=8, drop_last=False)
    # prepare the models
    expstr = "FactorConv_lpls05_l200001_nobn"
    trainers = []
    for triali in range(0, 10):
        trialstr = expstr+"_tr%d" % triali
        trainers.append(ModelTrainer(trialstr, beta_lpls=0.5, beta_L2=0.0001, expdir=expdir))
    #%%
    for epoch in tqdm(range(0, 20)):
        for trainer in trainers:
            trainer.new_epoch()
        for i, (imgs, scores) in enumerate(loader):  # tqdm
            imgs = imgs.cuda()
            scores_norm = ((scores - score_m) / score_s).float().cuda().unsqueeze(1)
            scores = (scores / score_s).float().cuda().unsqueeze(1)
            with torch.no_grad():
                featnet(imgs)
                feat = featFetcher[regresslayer]
            for trainer in trainers:
                pred, loss = trainer.learn_step(feat, scores_norm, record=(i % 10 == 0))
        scores_vec, pred_dict, S_dict = test_multimodel_dataset(featnet, featFetcher,
                                                                {i: trainer.model for i, trainer in enumerate(trainers)},
                                                                loader_test, score_m, score_s, regresslayer, label="evol_test")
        for i, trainer in enumerate(trainers):
            mean_test_err = S_dict[i].mean_err
            trainer.save_curbest_model(mean_test_err)

    for trainer in trainers:
        torch.save(trainer.model.state_dict(), join(trainer.trial_dir, "model_last.pt"))
    # %% Load the best models and test them
    model_col = {}
    for triali, trainer in enumerate(trainers):
        trainer.model.eval()
        trainer.model.load_state_dict(torch.load(join(trainer.trial_dir, "model_best.pt")))
        model_col[triali] = trainer.model
    #%%
    score_manif, imgfp_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="N")
    score_pasu, imgfp_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="N")
    score_gab, imgfp_gab = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="N")
    score_natref, imgfp_natref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="N")
    manif_loader = DataLoader(ImagePathDataset(imgfp_manif, score_manif, img_dim=(224, 224), ), num_workers=8, batch_size=200, shuffle=False)
    pasu_loader = DataLoader(ImagePathDataset(imgfp_pasu, score_pasu, img_dim=(224, 224), ), num_workers=8, batch_size=200, shuffle=False)
    gab_loader = DataLoader(ImagePathDataset(imgfp_gab, score_gab, img_dim=(224, 224), ), num_workers=8, batch_size=200, shuffle=False)
    natref_loader = DataLoader(ImagePathDataset(imgfp_natref, score_natref, img_dim=(224, 224), ), num_workers=8, batch_size=200, shuffle=False)

    scores_vec, preddict_vec, Scol = test_multimodel_dataset(featnet, featFetcher, model_col, loader_test, score_m, score_s, regresslayer, label="Evoltest", )
    scores_manif, preddict_manif, Scol_manif = test_multimodel_dataset(featnet, featFetcher, model_col, manif_loader, score_m, score_s, regresslayer, label="Manif", )
    scores_pasu, preddict_pasu, Scol_pasu = test_multimodel_dataset(featnet, featFetcher, model_col, pasu_loader, score_m, score_s, regresslayer, label="Pasu", )
    scores_gabor, preddict_gabor, Scol_gabor = test_multimodel_dataset(featnet, featFetcher, model_col, gab_loader, score_m, score_s, regresslayer, label="Gabor", )
    scores_natref, preddict_natref, Scol_natref = test_multimodel_dataset(featnet, featFetcher, model_col, natref_loader, score_m, score_s, regresslayer, label="NatRef", )

    pkl.dump({"scores_evol":scores_vec, "preddict_evol":preddict_vec, "Scol_evol":Scol,
              "scores_manif":scores_manif, "preddict_manif":preddict_manif, "Scol_manif":Scol_manif,
              "scores_pasu":scores_pasu, "preddict_pasu":preddict_pasu, "Scol_pasu":Scol_pasu,
              "scores_gabor":scores_gabor, "preddict_gabor":preddict_gabor, "Scol_gabor":Scol_gabor,
              "scores_natref":scores_natref, "preddict_natref":preddict_natref, "Scol_natref":Scol_natref,},
             open(join(expdir, "all_tr_all_imgset_pred_perform_%s.pkl"%expstr), "wb"))
    df_all = pd.concat([pd.DataFrame(Scol).T,
                        pd.DataFrame(Scol_manif).T,
                        pd.DataFrame(Scol_pasu).T,
                        pd.DataFrame(Scol_gabor).T,
                        pd.DataFrame(Scol_natref).T], axis=0)
    df_all.to_csv(join(expdir, "all_tr_all_imgset_pred_perform_%s.csv"%expstr))
    for triali, trainer in enumerate(trainers):
        wtsr = trainer.model.depth_conv.weight.data.cpu().numpy()
        wmat = wtsr.squeeze().transpose((1, 2, 0))
        plt.figure(figsize=(5, 5))
        plt.imshow(wmat / wmat.max())
        saveallforms(trainer.trial_dir, "depth_conv_weights_lpls_best")
        plt.show()
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
        for i in range(3):
            im = axs[i].imshow(wmat[:, :, i] / wmat.max(), cmap="bwr")
            plt.colorbar(im, ax=axs[i])
        plt.tight_layout()
        saveallforms(trainer.trial_dir, "depth_conv_weights_lpls_best_sep")
        plt.show()

#%%

