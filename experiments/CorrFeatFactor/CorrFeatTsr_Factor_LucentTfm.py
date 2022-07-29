""" Test transform robustness to feature visualization """
import pickle as pkl
from os.path import join
from core.CNN_scorers import load_featnet
from core.GAN_utils import upconvGAN
from core.featvis_lib import vis_feattsr_factor
from lucent.optvis.transform import random_rotate, jitter

G = upconvGAN().cuda().eval()

expdir = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis\models\resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV"
#%%
tfms = [
    jitter(8),
    # random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]
#%%
saveDict = pkl.load(open(join(expdir, "Alfa_Exp03_factors.pkl"),'rb'))
ccfactor = saveDict.ccfactor
Hmaps = saveDict.Hmaps
netname = saveDict.netname
layer = saveDict.layer
featvis_mode = saveDict.featvis_mode
bdr = saveDict.bdr
featnet, net = load_featnet(netname)
tsrimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, tfms=tfms,#[],
              score_mode=featvis_mode, featnet=featnet, Bsize=5, saveImgN=1, bdr=bdr, figdir=expdir, savestr="corr",
              saveimg=False, imshow=True)

