"""Visualize the correlated units for a given evolution.
Basic building blocks of a feature visualization
    CorrFeatScore: a Scorer or objective function based on neural network
    corr_visualize: Visualize feature based on pixel parametrization
    corr_GAN_visualize: Visualize feature based on GAN parametrization
All these components are heavily used in higher level api in featvis_lib

"""
import os
from os.path import join
from easydict import EasyDict
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
from numpy.linalg import norm as npnorm
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from kornia.filters import gaussian_blur2d
from scipy.io import loadmat
from skimage.transform import resize
from torchvision import models, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

layername_dict={"alexnet":["conv1", "conv1_relu", "pool1",
                            "conv2", "conv2_relu", "pool2",
                            "conv3", "conv3_relu",
                            "conv4", "conv4_relu",
                            "conv5", "conv5_relu", "pool3",
                            "dropout1", "fc6", "fc6_relu",
                            "dropout2", "fc7", "fc7_relu",
                            "fc8",],
                "vgg16":['conv1_1', 'conv1_1_relu',
                         'conv1_2', 'conv1_2_relu', 'pool1',
                         'conv2_1', 'conv2_1_relu',
                         'conv2_2', 'conv2_2_relu', 'pool2',
                         'conv3_1', 'conv3_1_relu',
                         'conv3_2', 'conv3_2_relu',
                         'conv3_3', 'conv3_3_relu', 'pool3',
                         'conv4_1', 'conv4_1_relu',
                         'conv4_2', 'conv4_2_relu',
                         'conv4_3', 'conv4_3_relu', 'pool4',
                         'conv5_1', 'conv5_1_relu',
                         'conv5_2', 'conv5_2_relu',
                         'conv5_3', 'conv5_3_relu', 'pool5',
                         'fc6', 'fc6_relu', 'dropout6',
                         'fc7', 'fc7_relu', 'dropout7',
                         'fc8'],
                "densenet121":['conv1',
                                 'bn1',
                                 'bn1_relu',
                                 'pool1',
                                 'denseblock1', 'transition1',
                                 'denseblock2', 'transition2',
                                 'denseblock3', 'transition3',
                                 'denseblock4',
                                 'bn2',
                                 'fc1'],
                "resnet50": ["layer1", "layer2", "layer3", "layer4"]}

# ensure compatibility with multiple versions of torch.
if hasattr(torch, "norm"):
    norm_torch = torch.norm
    def norm_torch(tsr, dim=[], keepdim=False):
        norms = (tsr ** 2).sum(dim=dim, keepdim=keepdim).sqrt()
        return norms
    
elif hasattr(torch, "linalg"):
    norm_torch = torch.linalg.norm
else:
    raise ModuleNotFoundError("Torch version non supported yet, check it. ")


class CorrFeatScore:
    """ Util class to compute a score from an image using pretrained model and read out matrix.

    Use a fixed weight matrix (deduced by correlated features) to read out from features of a layer.
    This score could be used as objective to visualize the model.
    This score could also be used as activation prediction for other images.

    Example:

    """
    def __init__(self):
        self.feat_tsr = {}
        self.weight_tsr = {}
        self.mask_tsr = {}
        self.weight_N = {}
        self.hooks = []
        self.layers = []
        self.scores = {}
        self.netname = None
        self.mode = "dot"  # use dot product by default. This could work as linear model over features.

    def hook_forger(self, layer, grad=True):
        # this function is important, or layer will be redefined in the same scope!
        def activ_hook(module, fea_in, fea_out):
            # print("Extract from hooker on %s" % module.__class__)
            # ref_feat = fea_out.detach().clone().cpu()
            # ref_feat.requires_grad_(False)
            self.feat_tsr[layer] = fea_out
            return None
        return activ_hook

    def register_hooks(self, net, layers, netname="vgg16"):
        if isinstance(layers, str):
            layers = [layers]

        for layer in layers:
            if netname in ["vgg16","alexnet"]:
                layer_idx = layername_dict[netname].index(layer)
                if layer_idx > 30:
                    targmodule = net.classifier[layer_idx-31]
                elif layer_idx < 30:
                    targmodule = net.features[layer_idx]
                else:
                    targmodule = net.avgpool
            elif "resnet50" in netname: # in ["resnet50", "resnet50_linf"]:
                targmodule = net.__getattr__(layer)
            else:
                raise NotImplementedError
            actH = targmodule.register_forward_hook(self.hook_forger(layer))
            self.hooks.append(actH)
            self.layers.append(layer)
        self.netname = netname

    def register_weights(self, weight_dict, mask_dict=None):
        for layer, weight in weight_dict.items():
            self.weight_tsr[layer] = torch.tensor(weight).float().cuda()
            self.weight_tsr[layer].requires_grad_(False)
            self.weight_N[layer] = (weight > 0).sum()
            if mask_dict is not None and layer in mask_dict:
                mask = mask_dict[layer]
                self.mask_tsr[layer] = torch.tensor(mask, requires_grad=False).bool().cuda()

    def corrfeat_score(self, layers=None, Nnorm=True):
        if layers is None: layers = self.layers
        if isinstance(layers, str):
            layers = [layers]
        for layer in layers:
            acttsr = self.feat_tsr[layer]
            if acttsr.ndim == 2: # fc layers
                sumdims = [1]
            elif acttsr.ndim == 4: # conv layers
                sumdims = [1, 2, 3]
            else:
                raise ValueError

            if self.mode == "dot":
                if self.weight_tsr[layer].ndim == 4:
                    # if multiple weight tensors are combined along the 0 dim, (Nweights, C, H, W)
                    # we can compute scores for them all at once.
                    # score will be of shape (B, Nweights)
                    score = (acttsr.unsqueeze(-1) * self.weight_tsr[layer].permute([1, 2, 3, 0]) ).sum(dim=sumdims)
                else:
                    score = (self.weight_tsr[layer] * acttsr).sum(dim=sumdims)
                if Nnorm: score = score / self.weight_N[layer]
            elif self.mode == "MSE":
                score = (self.weight_tsr[layer] - acttsr).pow(2).mean(dim=sumdims)
            elif self.mode == "MSEmask":
                score = torch.sum(self.mask_tsr[layer]*(self.weight_tsr[layer] - acttsr).pow(2), dim=sumdims) / \
                        self.mask_tsr[layer].count_nonzero().float()#(~torch.isnan(self.weight_tsr[layer])).float().sum()
            elif self.mode == "L1":
                score = (self.weight_tsr[layer] - acttsr).abs().mean(dim=sumdims)
            elif self.mode == "L1mask":
                score = torch.sum(self.mask_tsr[layer]*(self.weight_tsr[layer] - acttsr).abs(), dim=sumdims) / \
                        self.mask_tsr[layer].count_nonzero().float()
            elif self.mode == "corr":
                w_mean = self.weight_tsr[layer].mean()
                w_std = self.weight_tsr[layer].std()
                act_mean = acttsr.mean(dim=sumdims, keepdim=True)
                act_std = acttsr.std(dim=sumdims, keepdim=True)
                score = ((self.weight_tsr[layer] - w_mean) / w_std * (acttsr - act_mean) / act_std).mean(dim=sumdims)
            elif self.mode == "cosine":
                # w_norm = torch.linalg.norm(self.weight_tsr[layer])
                # act_norm = torch.linalg.norm(acttsr, dim=sumdims, keepdim=True)
                w_norm = norm_torch(self.weight_tsr[layer]) # .pow(2).sum().sqrt()
                act_norm = norm_torch(acttsr, dim=sumdims, keepdim=True)
                score = ((self.weight_tsr[layer] * acttsr) / w_norm / act_norm).sum(dim=sumdims) # validate
            elif self.mode == "corrmask":
                msk = self.mask_tsr[layer]
                weightvec = self.weight_tsr[layer][msk]
                w_mean = weightvec.mean()
                w_std = weightvec.std()
                actmat = acttsr[:, msk]
                act_mean = actmat.mean(dim=1, keepdim=True)
                act_std = actmat.std(dim=1, keepdim=True)
                score = ((weightvec - w_mean) / w_std * (actmat - act_mean) / act_std).mean(dim=1)
            else:
                raise NotImplementedError("Check `mode` of `scorer` ")
            self.scores[layer] = score
        if len(layers) > 1:
            return self.scores
        else:
            return self.scores[layers[0]]

    def featvec_corrmap(self, layer:str, featvec):
        act_feattsr = self.feat_tsr[layer].cpu()
        target_vec = torch.from_numpy(featvec).reshape([1, -1, 1, 1]).float()
        cov_map = (act_feattsr * target_vec).mean(dim=1, keepdim=False) # torch.tensor (B, H, W)
        z_feattsr = (act_feattsr - act_feattsr.mean(dim=1, keepdim=True)) / act_feattsr.std(dim=1, keepdim=True)
        z_featvec = (target_vec - target_vec.mean(dim=1, keepdim=True)) / target_vec.std(dim=1, keepdim=True)
        corr_map = (z_feattsr * z_featvec).mean(dim=1) # torch.tensor (B, H, W)
        return cov_map, corr_map

    def load_from_npy(self, savedict, net, netname, thresh=0, layers=[]):
        # imgN = savedict["imgN"]
        if savedict["cctsr"].shape == ():
            cctsr = savedict["cctsr"].item()
            Ttsr = savedict["Ttsr"].item()
        else:
            cctsr = savedict["cctsr"]
            Ttsr = savedict["Ttsr"]

        weight_dict = {}
        for layer in (cctsr.keys() if len(layers) == 0 else layers):
            weight = cctsr[layer]
            weight[np.abs(Ttsr[layer]) < thresh] = 0
            weight_dict[layer] = weight
            if layer not in self.layers:
                self.register_hooks(net, layer, netname=netname)
        self.register_weights(weight_dict)

    def clear_hook(self):
        for h in self.hooks:
            h.remove()
        self.layers = []

    def __del__(self):
        self.clear_hook()
        print('Feature Correlator Destructed, Hooks deleted.')

#%%
RGBmean = torch.tensor([0.485, 0.456, 0.406]).float().reshape([1,3,1,1])
RGBstd = torch.tensor([0.229, 0.224, 0.225]).float().reshape([1,3,1,1])
def save_imgtsr(finimgs, figdir:str ="", savestr:str =""):
    """
    finimgs: a torch tensor on cpu with shape B,C,H,W. 
    """
    B = finimgs.shape[0]
    for imgi in range(B):
        ToPILImage()(finimgs[imgi,:,:,:]).save(join(figdir, "%s_%02d.png"%(savestr, imgi)))


def preprocess(img: torch.tensor):
    """ clamp range to 0, 1; Blur the image; Centralize the tensor to go into CNN."""
    img = torch.clamp(img,0,1)
    img = gaussian_blur2d(img, (5,5), sigma=(3, 3))
    img = (img - RGBmean.to(img.device)) / RGBstd.to(img.device)
    return img


def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def corr_visualize(scorer, CNNnet, preprocess, layername, tfms=[],
    lr=0.01, imgfullpix=224, MAXSTEP=100, Bsize=4, saveImgN=None, use_adam=True, langevin_eps=0, 
    savestr="", figdir="", imshow=False, PILshow=False, verbose=True, saveimg=False, score_mode="dot", maximize=True):
    """ similar to `corr_GAN_visualize` but search for preferred images in pixel space instead of GAN space. """
    scorer.mode = score_mode
    score_sgn = -1 if maximize else 1
    x = 0.5+0.01*torch.randn((Bsize,3,imgfullpix,imgfullpix)).cuda() # fixed Feb.5th
    x.requires_grad_(True)
    optimizer = Adam([x], lr=lr) if use_adam else SGD([x], lr=lr)
    tfms_f = compose(tfms)
    score_traj = []
    pbar = tqdm(range(MAXSTEP))
    for step in pbar:
        ppx = preprocess(x)
        optimizer.zero_grad()
        CNNnet(tfms_f(ppx))
        score = score_sgn * scorer.corrfeat_score(layername)
        score.sum().backward()
        x.grad = x.norm() / x.grad.norm() * x.grad
        optimizer.step()
        score_traj.append(score.detach().clone().cpu())
        if langevin_eps > 0:
            # if > 0 then add noise to become Langevin gradient descent jump minimum
            x.data.add_(torch.randn(x.shape, device="cuda") * langevin_eps)
        if verbose and step % 10 == 0:
            print("step %d, score %s"%(step, " ".join("%.1f"%s for s in score_sgn * score)))
        pbar.set_description("step %d, score %s"%(step, " ".join("%.2f" % s for s in score_sgn * score)))

    final_score = score_sgn * score.detach().clone().cpu()
    del score
    torch.cuda.empty_cache()
    if maximize:
        idx = torch.argsort(final_score, descending=True)
    else:
        idx = torch.argsort(final_score, descending=False)
    score_traj = score_sgn * torch.stack(tuple(score_traj))[:, idx]
    finimgs = x.detach().clone().cpu()[idx, :, :, :]  # finimgs are generated by z before preprocessing.
    finimgs = torch.clamp(finimgs, 0, 1)
    print("Final scores %s"%(" ".join("%.2f" % s for s in final_score[idx])))
    mtg = ToPILImage()(make_grid(finimgs))
    if PILshow: mtg.show()
    mtg.save(join(figdir, "%s_pix_%s.png"%(savestr, layername)))
    np.savez(join(figdir, "%s_pix_%s.npz"%(savestr, layername)), score_traj=score_traj.numpy())
    if imshow:
        plt.figure(figsize=[Bsize*2, 2.3])
        plt.imshow(mtg)
        plt.axis("off")
        plt.show()
    if saveimg:
        os.makedirs(join(figdir, "img"), exist_ok=True)
        if saveImgN is None:
            save_imgtsr(finimgs, figdir=join(figdir, "img"), savestr="%s"%(savestr))
        else:
            save_imgtsr(finimgs[:saveImgN,:,:,:], figdir=join(figdir, "img"), savestr="%s"%(savestr))
    return finimgs, mtg, score_traj


def corr_GAN_visualize(G, scorer, CNNnet, preprocess, layername, tfms=[],
    lr=0.01, imgfullpix=224, MAXSTEP=100, Bsize=4, saveImgN=None, use_adam=True, langevin_eps=0, 
    savestr="", figdir="", imshow=False, PILshow=False, verbose=True, saveimg=False, score_mode="dot", maximize=True):
    """ Visualize the features carried by the scorer.  """
    scorer.mode = score_mode
    score_sgn = -1 if maximize else 1
    z = 0.5*torch.randn([Bsize, 4096]).cuda()
    z.requires_grad_(True)
    optimizer = Adam([z], lr=lr) if use_adam else SGD([z], lr=lr)
    tfms_f = compose(tfms)
    score_traj = []
    pbar = tqdm(range(MAXSTEP))
    for step in pbar:
        x = G.visualize(z, scale=1.0)
        ppx = preprocess(x)
        ppx = F.interpolate(ppx, [imgfullpix, imgfullpix], mode="bilinear", align_corners=True)
        optimizer.zero_grad()
        CNNnet(tfms_f(ppx))
        score = score_sgn * scorer.corrfeat_score(layername)
        score.sum().backward()
        z.grad = z.norm(dim=1, keepdim=True) / z.grad.norm(dim=1, keepdim=True) * z.grad  # this is a gradient normalizing step 
        optimizer.step()
        score_traj.append(score.detach().clone().cpu())
        if langevin_eps > 0: 
            # if > 0 then add noise to become Langevin gradient descent jump minimum
            z.data.add_(torch.randn(z.shape, device="cuda") * langevin_eps)
        if verbose and step % 10 == 0:
            print("step %d, score %s"%(step, " ".join("%.2f" % s for s in score_sgn * score)))
        pbar.set_description("step %d, score %s"%(step, " ".join("%.2f" % s for s in score_sgn * score)))

    final_score = score_sgn * score.detach().clone().cpu()
    del score
    torch.cuda.empty_cache()
    if maximize:
        idx = torch.argsort(final_score, descending=True)
    else:
        idx = torch.argsort(final_score, descending=False)
    score_traj = score_sgn * torch.stack(tuple(score_traj))[:, idx]
    finimgs = x.detach().clone().cpu()[idx, :, :, :]  # finimgs are generated by z before preprocessing.
    print("Final scores %s"%(" ".join("%.2f" % s for s in final_score[idx])))
    mtg = ToPILImage()(make_grid(finimgs))
    if PILshow: mtg.show()
    mtg.save(join(figdir, "%s_G_%s.png"%(savestr, layername)))
    np.savez(join(figdir, "%s_G_%s.npz"%(savestr, layername)), z=z.detach().cpu().numpy(), score_traj=score_traj.numpy())
    if imshow:
        plt.figure(figsize=[Bsize*2, 2.3])
        plt.imshow(mtg)
        plt.axis("off")
        plt.show()
    if saveimg:
        os.makedirs(join(figdir, "img"), exist_ok=True)
        if saveImgN is None:
            save_imgtsr(finimgs, figdir=join(figdir, "img"), savestr="%s"%(savestr))
        else:
            save_imgtsr(finimgs[:saveImgN,:,:,:], figdir=join(figdir, "img"), savestr="%s"%(savestr))
            mtg_sel = ToPILImage()(make_grid(finimgs[:saveImgN,:,:,:]))
            mtg_sel.save(join(figdir, "%s_G_%s_best.png" % (savestr, layername)))
    return finimgs, mtg, score_traj


def align_clim(Mcol: mpl.image.AxesImage):
    """Util function to align the color axes of a bunch of imshow maps."""
    cmin = np.inf
    cmax = -np.inf
    for M in Mcol:
        MIN, MAX = M.get_clim()
        cmin = min(MIN, cmin)
        cmax = max(MAX, cmax)
    for M in Mcol:
        M.set_clim((cmin, cmax))
    return cmin, cmax


def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def visualize_cctsr_simple(featFetcher, layers2plot, imgcol=(), savestr="Evol", titstr="Alfa_Evol", figdir=""):
    """ Visualize correlated values in the feature tensor.
    copied from experiment_EvolFeatDecompose.py

    Example:
        ExpType = "EM_cmb"
        layers2plot = ['conv3_3', 'conv4_3', 'conv5_3']
        figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, )
        figh.savefig(join("S:\corrFeatTsr","VGGsummary","%s_Exp%d_%s_corrTsr_vis.png"%(Animal,Expi,ExpType)))
    """
    nlayer = max(4, len(layers2plot))
    figh, axs = plt.subplots(3,nlayer,figsize=[10/3*nlayer,8])
    if imgcol is not None:
        for imgi in range(len(imgcol)):
            axs[0,imgi].imshow(imgcol[imgi])
            axs[0,imgi].set_title("Highest Score Evol Img")
            axs[0,imgi].axis("off")
    for li, layer in enumerate(layers2plot):
        chanN = featFetcher.cctsr[layer].shape[0]
        tmp=axs[1,li].matshow(np.nansum(featFetcher.cctsr[layer].abs().numpy(),axis=0) / chanN)
        plt.colorbar(tmp, ax=axs[1,li])
        axs[1,li].set_title(layer+" mean abs cc")
        tmp=axs[2,li].matshow(np.nanmax(featFetcher.cctsr[layer].abs().numpy(),axis=0))
        plt.colorbar(tmp, ax=axs[2,li])
        axs[2,li].set_title(layer+" max abs cc")
    figh.suptitle("%s Exp Corr Feat Tensor"%(titstr))
    plt.show()
    figh.savefig(join(figdir, "%s_corrTsr_vis.png" % (savestr)))
    figh.savefig(join(figdir, "%s_corrTsr_vis.pdf" % (savestr)))
    return figh


def rectify_tsr(cctsr: np.ndarray, mode="abs", thr=(-5, 5), Ttsr: np.ndarray=None):
    """ Rectify tensor to prep for NMF """
    if mode == "pos":
        cctsr_pp = np.clip(cctsr, 0, None)
    elif mode == "abs":
        cctsr_pp = np.abs(cctsr)
    elif mode == "thresh":
        thr = list(thr)
        if thr[0] is None: thr[0] = - np.inf
        if thr[1] is None: thr[1] =   np.inf
        cctsr_pp = cctsr.copy()
        cctsr_pp[(cctsr < thr[1]) * (cctsr > thr[0])] = 0
        # cctsr_pp = np.abs(cctsr_pp)
    elif mode == "Tthresh":
        thr = list(thr)
        if thr[0] is None: thr[0] = - np.inf
        if thr[1] is None: thr[1] =   np.inf
        maskTsr = (Ttsr > thr[0]) * (Ttsr < thr[1])
        print("Sparsity after T threshold %.3f"%((~maskTsr).sum() / np.prod(maskTsr.shape)))
        cctsr_pp = cctsr.copy()
        cctsr_pp[maskTsr] = 0
        # ctsr_pp = np.abs(cctsr_pp)
    elif mode == "none":
        cctsr_pp = cctsr
    else:
        raise ValueError
    return cctsr_pp


def tsr_factorize(Ttsr_pp: np.ndarray, cctsr: np.ndarray, bdr=2, Nfactor=3, init="nndsvda", solver="cd",
                figdir="", savestr="", suptit="", show=True):
    """ Factorize the T tensor using NMF, compute the corresponding features for cctsr """
    C, H, W = Ttsr_pp.shape
    if bdr == 0:
        Tmat = Ttsr_pp.reshape(C, H * W)
        ccmat = cctsr.reshape(C, H * W)
    else:
        Tmat = Ttsr_pp[:, bdr:-bdr, bdr:-bdr].reshape(C, (H-2*bdr)*(W-2*bdr))
        ccmat = cctsr[:, bdr:-bdr, bdr:-bdr].reshape(C, (H-2*bdr)*(W-2*bdr))
    nmfsolver = NMF(n_components=Nfactor, init=init, solver=solver)  # mu
    Hmat = nmfsolver.fit_transform(Tmat.T)
    Hmaps = Hmat.reshape([H-2*bdr, W-2*bdr, Nfactor])
    Tcompon = nmfsolver.components_
    exp_var = 1-npnorm(Tmat.T - Hmat @ Tcompon) / npnorm(Tmat)
    print("NMF explained variance %.3f"%exp_var)
    ccfactor = (ccmat @ np.linalg.pinv(Hmat).T )
    # ccfactor = (ccmat @ Hmat)
    # Calculate norm of diff factors
    fact_norms = []
    for i in range(Hmaps.shape[2]):
        rank1_mat = Hmat[:, i:i+1]@Tcompon[i:i+1, :]
        matnorm = npnorm(rank1_mat, ord="fro")
        fact_norms.append(matnorm)
        print("Factor%d norm %.2f"%(i, matnorm))

    reg_cc = np.corrcoef((ccfactor @ Hmat.T).flatten(), ccmat.flatten())[0,1]
    print("Predictability of the corr coef tensor %.3f"%reg_cc)
    # Visualize maps as 3 channel image.
    plt.figure(figsize=[5, 5])
    plt.imshow(Hmaps[:,:,:3] / Hmaps[:,:,:3].max())
    plt.axis('off')
    plt.title("channel merged")
    plt.savefig(join(figdir, "%s_factor_merged.png" % (savestr))) # Indirect factorize
    plt.savefig(join(figdir, "%s_factor_merged.pdf" % (savestr)))
    if show: plt.show()
    else: plt.close()
    # Visualize maps and their associated channel vector
    [figh, axs] = plt.subplots(2, Nfactor, figsize=[Nfactor*2.7, 5.0])
    for ci in range(Hmaps.shape[2]):
        plt.sca(axs[0, ci])  # show the map correlation
        plt.imshow(Hmaps[:, :, ci] / Hmaps.max())
        plt.axis("off")
        plt.colorbar()
        plt.sca(axs[1, ci])  # show the channel association
        axs[1, ci].plot([0, ccfactor.shape[0]], [0, 0], 'k-.', alpha=0.4)
        axs[1, ci].plot(ccfactor[:, ci], alpha=0.5) # show the indirectly computed correlation the left.
        ax2 = axs[1, ci].twinx()
        ax2.plot(Tcompon.T[:, ci], color="C1", alpha=0.5) # show the directly computed factors for T tensor on the right.
        ax2.spines['left'].set_color('C0')
        ax2.spines['right'].set_color('C1')
    plt.suptitle("%s Separate Factors"%suptit)
    figh.savefig(join(figdir, "%s_factors.png" % (savestr)))
    figh.savefig(join(figdir, "%s_factors.pdf" % (savestr)))
    if show: plt.show()
    else: plt.close()
    Stat = EasyDict()
    for varnm in ["reg_cc", "fact_norms", "exp_var", "C", "H", "W", "bdr", "Nfactor", "init", "solver"]:
        Stat[varnm] = eval(varnm)
    return Hmat, Hmaps, Tcompon, ccfactor, Stat


def posneg_sep(tsr: np.ndarray, axis=0):
    """Separate the positive and negative entries of a matrix and concatenate along certain axis."""
    return np.concatenate((np.clip(tsr, 0, None), -np.clip(tsr, None, 0)), axis=axis)


def tsr_posneg_factorize(cctsr: np.ndarray, bdr=2, Nfactor=3,
                init="nndsvda", solver="cd", l1_ratio=0, alpha=0, beta_loss="frobenius",
                figdir="", savestr="", suptit="", show=True, do_plot=True, do_save=True):
    """ Factorize the cc tensor using NMF directly
    If any entries of cctsr is negative, it will use `posneg_sep` to create an augmented matrix with only positive entries.
    Then use NMF on that matrix. This process simulates the one sided NNMF.

    """
    C, H, W = cctsr.shape
    if bdr == 0:
        ccmat = cctsr.reshape(C, H * W)
    else:
        ccmat = cctsr[:, bdr:-bdr, bdr:-bdr].reshape(C, (H-2*bdr)*(W-2*bdr))
    if np.any(ccmat < 0):
        sep_flag = True
        posccmat = posneg_sep(ccmat, 0)
    else:
        sep_flag = False
        posccmat = ccmat
    nmfsolver = NMF(n_components=Nfactor, init=init, solver=solver, l1_ratio=l1_ratio, alpha=alpha, beta_loss=beta_loss)  # mu
    Hmat = nmfsolver.fit_transform(posccmat.T)
    Hmaps = Hmat.reshape([H-2*bdr, W-2*bdr, Nfactor])
    CCcompon = nmfsolver.components_  # potentially augmented CC components
    if sep_flag:  # reproduce the positive and negative factors back.
        ccfactor = (CCcompon[:, :C] - CCcompon[:, C:]).T
    else:
        ccfactor = CCcompon.T
    exp_var = 1-npnorm(posccmat.T - Hmat @ CCcompon) / npnorm(ccmat)
    print("NMF explained variance %.3f"%exp_var)
    # ccfactor = (ccmat @ np.linalg.pinv(Hmat).T )
    # ccfactor = (ccmat @ Hmat)
    # Calculate norm of diff factors
    fact_norms = []
    for i in range(Hmaps.shape[2]):
        rank1_mat = Hmat[:, i:i+1]@CCcompon[i:i+1, :]
        matnorm = npnorm(rank1_mat, ord="fro")
        fact_norms.append(matnorm)
        print("Factor%d norm %.2f"%(i, matnorm))

    reg_cc = np.corrcoef((ccfactor @ Hmat.T).flatten(), ccmat.flatten())[0,1]
    print("Correlation to the corr coef tensor %.3f"%reg_cc)
    # Visualize maps as 3 channel image.
    if Hmaps.shape[2] < 3: # Add zero channels if < 3 channels are there.
        Hmaps_plot = np.concatenate((Hmaps, np.zeros((*Hmaps.shape[:2], 3 - Hmaps.shape[2]))), axis=2)
    else:
        Hmaps_plot = Hmaps[:, :, :3]
    if do_plot:
        plt.imshow(Hmaps_plot / Hmaps_plot.max())
        plt.axis('off')
        plt.title("%s\nchannel merged"%suptit)
        if do_save:
            plt.savefig(join(figdir, "%s_dir_factor_merged.png" % (savestr))) # direct factorize
            plt.savefig(join(figdir, "%s_dir_factor_merged.pdf" % (savestr)))
        if show: plt.show()
        else: plt.close()
        # Visualize maps and their associated channel vector
        [figh, axs] = plt.subplots(2, Nfactor, figsize=[Nfactor*2.7, 5.0], squeeze=False)
        for ci in range(Hmaps.shape[2]):
            plt.sca(axs[0, ci])  # show the map correlation
            plt.imshow(Hmaps[:, :, ci] / Hmaps.max())
            plt.axis("off")
            plt.colorbar()
            plt.sca(axs[1, ci])  # show the channel association
            axs[1, ci].plot([0, ccfactor.shape[0]], [0, 0], 'k-.', alpha=0.4)
            axs[1, ci].plot(ccfactor[:, ci], alpha=0.5)
            axs[1, ci].plot(sorted(ccfactor[:, ci]), alpha=0.25)
        plt.suptitle("%s\nSeparate Factors"%suptit)
        if do_save:
            figh.savefig(join(figdir, "%s_dir_factors.png" % (savestr)))
            figh.savefig(join(figdir, "%s_dir_factors.pdf" % (savestr)))
        if show: plt.show()
        else: plt.close()
    Stat = EasyDict()
    for varnm in ["exp_var", "reg_cc", "fact_norms", "exp_var", "C", "H", "W", "bdr", "Nfactor", "init", "solver"]:
        Stat[varnm] = eval(varnm)
    return Hmat, Hmaps, ccfactor, Stat


def vis_featvec(ccfactor, net, G, layer, netname="alexnet", featnet=None, tfms=[],
        preprocess=preprocess, lr=0.05, MAXSTEP=100, use_adam=True, Bsize=4, saveImgN=None, langevin_eps=0,
        imshow=True, verbose=False, savestr="", figdir="", saveimg=False, save_featmap=True, show_featmap=True, score_mode="dot"):
    """Feature vector over the whole map"""
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        # fact_W = torch.from_numpy(ccfactor[:, ci]).reshape([1, -1, 1, 1])
        fact_W = torch.from_numpy(ccfactor[:, ci]).reshape([-1, 1, 1]) # debug at Feb.5th 2023, need 3d tensor not 4d
        scorer.register_weights({layer: fact_W})
        if G is None:
            finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms,
             lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
             imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_full_%s%s-%s"%(ci, savestr, netname, layer))
        else:
            finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms,
             lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
             imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_full_%s%s-%s"%(ci, savestr, netname, layer))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
             figdir=figdir, savestr="fac%d_full_%s%s_%s"%(ci, savestr, netname, "pix" if G is None else "G"), saveimg=save_featmap, showimg=show_featmap)
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_featvec_point(ccfactor: np.ndarray, Hmaps: np.ndarray, net, G, layer, netname="alexnet", featnet=None, bdr=2, tfms=[],
              preprocess=preprocess, lr=0.05, MAXSTEP=100, use_adam=True, Bsize=4, saveImgN=None, langevin_eps=0, pntsize=2,
              imshow=True, verbose=False, savestr="", figdir="", saveimg=False, save_featmap=True, show_featmap=True, score_mode="dot"):
    """ Feature vector at the centor of the map as spatial mask. """
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        H, W, _ = Hmaps.shape
        sp_mask = np.pad(np.ones([pntsize, pntsize, 1]), ((H//2-pntsize//2+bdr, H-H//2-pntsize+pntsize//2+bdr),
                                                          (W//2-pntsize//2+bdr, W-W//2-pntsize+pntsize//2+bdr),(0,0)),
                         mode="constant", constant_values=0)
        fact_Chtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci+1], sp_mask))
        scorer.register_weights({layer: fact_Chtsr})
        if G is None:
            finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms,
              lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
              imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_cntpnt_%s%s-%s"%(ci, savestr, netname, layer))
        else:
            finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms,
              lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
              imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_cntpnt_%s%s-%s"%(ci, savestr, netname, layer))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
                figdir=figdir, savestr="fac%d_cntpnt_%s%s_%s"%(ci, savestr, netname, "pix" if G is None else "G"), saveimg=save_featmap, showimg=show_featmap)
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_featvec_wmaps(ccfactor: np.ndarray, Hmaps: np.ndarray, net, G, layer, netname="alexnet", featnet=None, bdr=2, tfms=[],
             preprocess=preprocess, lr=0.1, MAXSTEP=100, use_adam=True, Bsize=4, saveImgN=None, langevin_eps=0,
             imshow=True, verbose=False, savestr="", figdir="", saveimg=False, save_featmap=True, show_featmap=True, score_mode="dot"):
    """ Feature vector at the centor of the map as spatial mask. """
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        padded_mask = np.pad(Hmaps[:, :, ci:ci + 1], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
        fact_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci + 1], padded_mask))
        if show_featmap or imshow: show_img(padded_mask[:, :, 0])
        scorer.register_weights({layer: fact_Wtsr})
        if G is None:
            finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms,
              lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
              imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_map_%s%s-%s"%(ci, savestr, netname, layer))
        else:
            finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms,
              lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
              imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_map_%s%s-%s"%(ci, savestr, netname, layer))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
                figdir=figdir, savestr="fac%d_map_%s%s_%s"%(ci, savestr, netname, "pix" if G is None else "G"), saveimg=save_featmap, showimg=show_featmap)
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_feattsr(cctsr, net, G, layer, netname="alexnet", featnet=None, bdr=2, tfms=[],
                preprocess=preprocess, lr=0.05, MAXSTEP=150, use_adam=True, Bsize=5, saveImgN=None, langevin_eps=0.03,
                imshow=True, verbose=False, savestr="", figdir="", saveimg=False, score_mode="dot"):
    """ Visualize the full feature tensor pattern. """
    if featnet is None: featnet = net.features
    # padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    # DR_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask))
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: cctsr})
    if G is None:
        finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms,
          lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
          imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="tsr_%s%s-%s"%(savestr, netname, layer))
    else:
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms,
          lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
          imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="tsr_%s%s-%s"%(savestr, netname, layer))
    scorer.clear_hook()
    return finimgs, mtg, score_traj


def vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname="alexnet", featnet=None, bdr=2, tfms=[],
                preprocess=preprocess, lr=0.05, MAXSTEP=150, use_adam=True, Bsize=5, saveImgN=None, langevin_eps=0.03,
                imshow=True, verbose=False, savestr="", figdir="", saveimg=False, score_mode="dot"):
    """ Visualize the factorized feature tensor """
    if featnet is None: featnet = net.features
    padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    DR_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask))
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: DR_Wtsr})
    if G is None:
        finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms,
          lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
          imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="facttsr_%s%s-%s"%(savestr, netname, layer))
    else:
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms,
            lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
            imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="facttsr_%s%s-%s"%(savestr, netname, layer))
    scorer.clear_hook()
    return finimgs, mtg, score_traj


def pad_factor_prod(Hmaps, ccfactor, bdr=0):
    padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    DR_Wtsr = np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask)
    return DR_Wtsr


def vis_featmap_corr(scorer: CorrFeatScore, featnet: nn.Module, finimgs: torch.tensor, targvect: np.ndarray, layer: str,
                     maptype="cov", imgscores=[], figdir="", savestr="", saveimg=True, showimg=True):
    """Given a feature vec, the feature map as projecting the feat tensor onto this vector."""
    featnet(finimgs.cuda())
    act_feattsr = scorer.feat_tsr[layer].cpu()
    target_vec = torch.from_numpy(targvect).reshape([1, -1, 1, 1]).float()

    cov_map = (act_feattsr * target_vec).mean(dim=1, keepdim=False) # torch.tensor (B, H, W)
    z_feattsr = (act_feattsr - act_feattsr.mean(dim=1, keepdim=True)) / act_feattsr.std(dim=1, keepdim=True)
    z_featvec = (target_vec - target_vec.mean(dim=1, keepdim=True)) / target_vec.std(dim=1, keepdim=True)
    corr_map = (z_feattsr * z_featvec).mean(dim=1) # torch.tensor (B, H, W)
    for maptype in ["cov", "corr"]:
        map2show = cov_map if maptype == "cov" else corr_map
        NS = map2show.shape[0]
        Mcol = []
        [figh, axs] = plt.subplots(2, NS, figsize=[NS * 2.5, 5.3])
        for ci in range(NS):
            plt.sca(axs[0, ci])  # show the map correlation
            M = plt.imshow((map2show[ci, :, :] / map2show.max()).numpy())
            plt.axis("off")
            plt.title("%.2e" % imgscores[ci])
            plt.sca(axs[1, ci])  # show the image itself
            plt.imshow(ToPILImage()(finimgs[ci, :, :, :]))
            plt.axis("off")
            Mcol.append(M)
        align_clim(Mcol)
        if saveimg:
            figh.savefig(join(figdir, "%s_%s_img_%smap.png" % (savestr, layer, maptype)))
            figh.savefig(join(figdir, "%s_%s_img_%smap.pdf" % (savestr, layer, maptype)))
        if showimg:
            figh.show()
        else:
            plt.close(figh)
    return cov_map, corr_map


if __name__ == "__main__":
    from core.GAN_utils import upconvGAN
    # Prepare the networks
    VGG = models.vgg16(pretrained=True)
    VGG.requires_grad_(False)
    VGG.features.cuda()
    G = upconvGAN("fc6").cuda()
    G.requires_grad_(False)
    #%%
    mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
    Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
    Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
    Animal = "Beto"
    # ManifDyn = loadmat(join(mat_path, Animal + "_ManifPopDynamics.mat"), struct_as_record=False, squeeze_me=True)['ManifDyn']
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    # %% Final Batch Processing of all Exp.
    figdir = r"S:\corrFeatTsr\VGG_featvis"
    Animal = "Alfa"; Expi = 3
    for Animal in ["Alfa",]: # "Beto"
        for Expi in range(27,46+1):
            D = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_EM_corrTsr.npz"%(Animal, Expi)), allow_pickle=True)
            scorer = CorrFeatScore()
            scorer.load_from_npy(D, VGG, netname="vgg16", thresh=4, layers=[])
            savefn = "%s_Exp%d_EM_corrVis"%(Animal, Expi)
            imgs5, mtg5, score_traj5 = corr_GAN_visualize(G, scorer, VGG.features, preprocess, "conv5_3", lr=0.080, imgfullpix=224, MAXSTEP=75, Bsize=4, savestr=savefn, figdir=figdir)
            imgs4, mtg4, score_traj4 = corr_GAN_visualize(G, scorer, VGG.features, preprocess, "conv4_3", lr=0.025, imgfullpix=224, MAXSTEP=75, Bsize=4, savestr=savefn, figdir=figdir)
            imgs3, mtg3, score_traj3 = corr_GAN_visualize(G, scorer, VGG.features, preprocess, "conv3_3", lr=0.01, imgfullpix=224, MAXSTEP=75, Bsize=4, savestr=savefn, figdir=figdir)
    #%% Try to use GradCam, to understand output for a single input.

    #%% Backprop based feature visualization
    Animal = "Beto"
    Expi = 11
    # for Expi in range(27,46+1):
    D = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_EM_corrTsr.npz" % (Animal, Expi)), allow_pickle=True)
    scorer = CorrFeatScore()
    scorer.load_from_npy(D, VGG, netname="vgg16", thresh=4, layers=[])
    img = ReprStats[Expi-1].Evol.BestBlockAvgImg
    x = transforms.ToTensor()(img)
    x = preprocess(x.unsqueeze(0))
    x = F.interpolate(x, [224,224])
    x.requires_grad_(True)
    VGG.features(x.cuda())
    score = scorer.corrfeat_score("conv5_3")
    #%% Visualize feature related to these masks
    GradMask = x.grad.clone()
    GradMask /= GradMask.abs().max()
    GradMaskNP = GradMask[0].permute([1,2,0]).numpy()
    # GradMaskNP_rsz = resize(img, (224,224))
    maskimg = resize(img, (224,224))*(np.minimum(1, 3*np.abs(GradMaskNP).mean(axis=2,keepdims=True)))
    plt.imshow(maskimg)
    plt.show()
    #%%
    Animal="Alfa"; Expi = 3
    D = np.load(join("S:\corrFeatTsr","%s_Exp%d_EM_corrTsr.npz"%(Animal,Expi)), allow_pickle=True)
    scorer = CorrFeatScore()
    scorer.load_from_npy(D, VGG, netname="vgg16", thresh=4, layers=[])
    #%% Maximize to the scorer using pixel parametrization.
    imgfullpix = 224
    MAXSTEP = 50
    Bsize = 4
    x = 0.5+0.01*torch.rand((Bsize,3,imgfullpix,imgfullpix)).cuda()
    x.requires_grad_(True)
    optimizer = SGD([x], lr=0.01)
    for step in range(MAXSTEP):
        ppx = preprocess(x)
        optimizer.zero_grad()
        VGG.features(ppx)
        score = -scorer.corrfeat_score("conv3_3")
        score.sum().backward()
        x.grad = x.norm() / x.grad.norm() * x.grad
        optimizer.step()
        if step % 10 == 0:
            print("step %d, score %.s"%(step, -score.item()))

    ToPILImage()(torch.clamp(x[0], 0, 1).cpu()).show()
    del score
    torch.cuda.empty_cache()
    #%%
    def preprocess(img, res=224):
        img = F.interpolate(img, [res,res], mode="bilinear", align_corners=True)
        img = torch.clamp(img,0,1)
        img = gaussian_blur2d(img, (5,5), sigma=(3, 3))
        img = (img - RGBmean.to(img.device)) / RGBstd.to(img.device)
        return img
    #%
    imgfullpix = 224
    MAXSTEP = 50
    Bsize = 4
    G = upconvGAN("fc6").cuda()
    G.requires_grad_(False)
    z = 0.5*torch.randn([Bsize, 4096]).cuda()
    z.requires_grad_(True)
    optimizer = Adam([z], lr=0.1)
    for step in range(MAXSTEP):
        x = G.visualize(z, scale=1.0)
        ppx = preprocess(x)
        optimizer.zero_grad()
        VGG.features(ppx)
        score = -scorer.corrfeat_score("conv5_3")
        score.sum().backward()
        z.grad = z.norm() / z.grad.norm() * z.grad
        optimizer.step()
        if step % 10 ==0:
            print("step %d, score %s"%(step, " ".join("%.1f"%s for s in -score)))
    # ToPILImage()(torch.clamp(x[0],0,1).cpu()).show()
    ToPILImage()(make_grid(x).cpu()).show()
    del score
    torch.cuda.empty_cache()
    #%%
