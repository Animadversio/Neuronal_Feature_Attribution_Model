"""This library provides higher level api over CorrFeatTsr_visualize_lib
Specifically, it provides functions that visualize a feature vector / tensor in a given layer of CNN
"""
import os
from os.path import join

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import torch
from easydict import EasyDict
from numpy.linalg import norm as npnorm
from sklearn.decomposition import NMF
from torch import nn
from torchvision.transforms import ToPILImage
from core.CorrFeatFactor.CorrFeatTsr_visualize_lib import CorrFeatScore, corr_GAN_visualize, corr_visualize, \
    preprocess
from core.CorrFeatFactor.CorrFeatTsr_visualize_lib import visualize_cctsr_simple, rectify_tsr, tsr_factorize, \
    posneg_sep, tsr_posneg_factorize, vis_featvec, vis_featvec_point, vis_featvec_wmaps, vis_feattsr, vis_feattsr_factor, \
    pad_factor_prod, vis_featmap_corr
mpl.rcParams['pdf.fonttype'] = 42




#%%
if __name__ == "__main__":

    from core.CNN_scorers import load_featnet
    from core.GAN_utils import upconvGAN
    from core.neural_data_loader import mat_path, loadmat
    exp_suffix = "_nobdr_alex"
    netname = "alexnet"
    G = upconvGAN("fc6").cuda()
    G.requires_grad_(False)
    featnet, net = load_featnet(netname)
    #%%
    Animal = "Beto"; Expi = 11
    corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)), allow_pickle=True)#
    cctsr_dict = corrDict.get("cctsr").item()
    Ttsr_dict = corrDict.get("Ttsr").item()
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    show_img(ReprStats[Expi-1].Manif.BestImg)
    #%%
    figroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis"
    figdir = join(figroot, "%s_Exp%02d"%(Animal, Expi))
    os.makedirs(figdir, exist_ok=True)
    #%%
    layer = "conv3"
    Ttsr = Ttsr_dict[layer]
    cctsr = cctsr_dict[layer]
    bdr = 1; NF = 5
    Ttsr_pp = rectify_tsr(Ttsr, "abs")  # "mode="thresh", thr=(-5,5))
    Hmat, Hmaps, Tcomponents, ccfactor, Stat = tsr_factorize(Ttsr_pp, cctsr, bdr=bdr, Nfactor=NF, figdir=figdir, savestr="%s-%s"%(netname, layer))
    
    finimgs, mtg, score_traj = vis_feattsr(cctsr, net, G, layer, netname=netname, Bsize=5, figdir=figdir, savestr="")
    finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, Bsize=5,
                                                  bdr=bdr, figdir=figdir, savestr="")
    finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname=netname, featnet=featnet,
                                   Bsize=5, figdir=figdir, savestr="", imshow=False)
    finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname,
                                 featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="", imshow=False)
    finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor, Hmaps, net, G, layer, netname=netname,
                                 featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="", imshow=False)


    #%% Development zone for feature map visualization
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        H, W, _ = Hmaps.shape
        sp_mask = np.pad(np.ones([2, 2, 1]), ((H//2-1+bdr, H-H//2-1+bdr), (W//2-1+bdr, W-W//2-1+bdr),(0,0)), mode="constant", constant_values=0)
        fact_Chtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci+1], sp_mask))
        scorer.register_weights({layer: fact_Chtsr})
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, lr=0.05, MAXSTEP=100, use_adam=True, Bsize=6, langevin_eps=0,
                  imshow=False, verbose=False)
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="corr", imgscores=score_traj[-1, :])
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    # scorer.clear_hook()
    #%%
    featnet(finimgs.cuda())
    #%%
    ci=4
    maptype = "cov"

    act_feattsr = scorer.feat_tsr[layer].cpu()
    target_vec = torch.from_numpy(ccfactor[:, ci:ci+1]).reshape([1,-1,1,1]).float()
    cov_map = (act_feattsr * target_vec).mean(dim=1, keepdim=False)
    z_feattsr = (act_feattsr - act_feattsr.mean(dim=1, keepdim=True)) / act_feattsr.std(dim=1, keepdim=True)
    z_featvec = (target_vec - target_vec.mean(dim=1, keepdim=True)) / target_vec.std(dim=1, keepdim=True)
    corr_map = (z_feattsr * z_featvec).mean(dim=1)

    map2show = cov_map if maptype == "cov" else corr_map
    NS = map2show.shape[0]
    #%%
    Mcol = []
    [figh, axs] = plt.subplots(2, NS, figsize=[NS*2.5, 5.3])
    for ci in range(NS):
        plt.sca(axs[0, ci])  # show the map correlation
        M = plt.imshow((map2show[ci, :, :] / map2show.max()).numpy())
        plt.axis("off")
        plt.title("%.2e"%score_traj[-1,ci].item())
        plt.sca(axs[1, ci])
        plt.imshow(ToPILImage()(finimgs[ci, :,:,:]))
        plt.axis("off")
        Mcol.append(M)
    align_clim(Mcol)
    plt.show()
    # show the resulting feature map that match the current feature descriptor
    #%%


