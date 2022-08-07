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

model_fun = lambda: FactorizedConv2D(1024, 1, kernel_size=(14, 14), factors=3, bn=False).cuda()
regresslayer = ".layer3.Bottleneck5"
featnet, net = load_featnet("resnet50_linf8")
featFetcher = featureFetcher(featnet, input_size=(3, 224, 224),
                             device="cuda", print_module=False)
featFetcher.record(regresslayer, ingraph=True)

Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']

for Expi in range(1, 11):#len(MStats)+1):
    expdir = join(saveroot, f"{Animal}_{Expi:02d}")

    expstr = "FactorConv_lpls05_l200001_nobn"

    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="N")
    score_m, score_s = torch.tensor(score_vect.mean()), torch.tensor(score_vect.std())

    score_train, score_test, imgfp_train, imgfp_test = train_test_split(
        score_vect, imgfullpath_vect, test_size=0.2, random_state=42)
    dataset = ImagePathDataset(imgfp_train, score_train, img_dim=(224, 224))
    dataset_test = ImagePathDataset(imgfp_test, score_test, img_dim=(224, 224))
    loader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=8, drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=200, shuffle=False, num_workers=8, drop_last=False)


    score_manif, imgfp_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="N")
    score_pasu, imgfp_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="N")
    score_gab, imgfp_gab = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="N")
    score_natref, imgfp_natref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="N")
    manif_loader = DataLoader(ImagePathDataset(imgfp_manif, score_manif, img_dim=(224, 224), ), num_workers=8,
                              batch_size=200, shuffle=False)
    pasu_loader = DataLoader(ImagePathDataset(imgfp_pasu, score_pasu, img_dim=(224, 224), ), num_workers=8,
                             batch_size=200, shuffle=False)
    gab_loader = DataLoader(ImagePathDataset(imgfp_gab, score_gab, img_dim=(224, 224), ), num_workers=8, batch_size=200,
                            shuffle=False)
    natref_loader = DataLoader(ImagePathDataset(imgfp_natref, score_natref, img_dim=(224, 224), ), num_workers=8,
                               batch_size=200, shuffle=False)


    model_col = {}
    for triali in range(10):
        trial_dir = join(expdir, expstr+"_tr%d" % triali)
        model = model_fun().eval()
        model.load_state_dict(torch.load(join(trial_dir, "model_best.pt")))
        model_col[triali] = model

    scores_vec, preddict_vec, Scol = test_multimodel_dataset(featnet, featFetcher, model_col, loader_test, score_m,
                                                             score_s, regresslayer, label="Evoltest", )
    scores_manif, preddict_manif, Scol_manif = test_multimodel_dataset(featnet, featFetcher, model_col, manif_loader,
                          score_m, score_s, regresslayer, label="Manif", )
    scores_pasu, preddict_pasu, Scol_pasu = test_multimodel_dataset(featnet, featFetcher, model_col, pasu_loader,
                          score_m, score_s, regresslayer, label="Pasu", )
    scores_gabor, preddict_gabor, Scol_gabor = test_multimodel_dataset(featnet, featFetcher, model_col, gab_loader,
                          score_m, score_s, regresslayer, label="Gabor", )
    scores_natref, preddict_natref, Scol_natref = test_multimodel_dataset(featnet, featFetcher, model_col, natref_loader,
                          score_m, score_s, regresslayer, label="NatRef", )

    pkl.dump({"scores_evol": scores_vec, "preddict_evol": preddict_vec, "Scol_evol": Scol,
              "scores_manif": scores_manif, "preddict_manif": preddict_manif, "Scol_manif": Scol_manif,
              "scores_pasu": scores_pasu, "preddict_pasu": preddict_pasu, "Scol_pasu": Scol_pasu,
              "scores_gabor": scores_gabor, "preddict_gabor": preddict_gabor, "Scol_gabor": Scol_gabor,
              "scores_natref": scores_natref, "preddict_natref": preddict_natref, "Scol_natref": Scol_natref, },
             open(join(expdir, "all_tr_all_imgset_pred_perform_%s.pkl" % expstr), "wb"))
    df_all = pd.concat([pd.DataFrame(Scol).T,
                        pd.DataFrame(Scol_manif).T,
                        pd.DataFrame(Scol_pasu).T,
                        pd.DataFrame(Scol_gabor).T,
                        pd.DataFrame(Scol_natref).T], axis=0)
    df_all.to_csv(join(expdir, "all_tr_all_imgset_pred_perform_%s.csv" % expstr))