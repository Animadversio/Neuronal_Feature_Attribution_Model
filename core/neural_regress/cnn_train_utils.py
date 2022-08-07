
import torch
import torch.nn.functional as F
from core.neural_regress.cnn_readout_model import FactorizedConv2D, SeparableConv2D, MultilayerCnn


kerMat = torch.tensor([[0.5,  1.0, 0.5],
                       [1.0, -6.0, 1.0],
                       [0.5,  1.0, 0.5]]).reshape(1, 1, 3, 3,).cuda()


def Weight_Laplacian(depth_conv, ):
    weighttsr = depth_conv.weight
    filtered_wtsr = F.conv2d(weighttsr, kerMat, groups=weighttsr.shape[1])
    return (filtered_wtsr**2).mean()


def grad_diagnose(model, tb_writer=None, global_step=0):
    if isinstance(model, MultilayerCnn):
        print("\tGrad w norm %.1e wg norm %.1e Lw norm %.1e Lwg norm %.1e" % \
            (model.model.layer2_conv.depth_conv.weight.norm(), model.model.layer2_conv.depth_conv.weight.grad.norm(),
                   model.model.Linear.weight.norm(), model.model.Linear.weight.grad.norm()))

    elif isinstance(model, FactorizedConv2D):
        print("\tGrad w norm %.1e wg norm %.1e Lw norm %.1e Lwg norm %.1e" % \
            (model.depth_conv.weight.norm(), model.depth_conv.weight.grad.norm(),
                    model.linear.weight.norm(), model.linear.weight.grad.norm()))
        if tb_writer is not None:
            tb_writer.add_scalar("spmask_weight_norm", model.depth_conv.weight.norm(), global_step)
            tb_writer.add_scalar("spmask_grad_norm", model.depth_conv.weight.grad.norm(), global_step)
            tb_writer.add_scalar("featvec_weight_norm", model.point_conv.weight.norm(), global_step)
            tb_writer.add_scalar("featvec_grad_norm", model.point_conv.weight.grad.norm(), global_step)
            tb_writer.add_scalar("linear_weight_norm", model.linear.weight.norm(), global_step)
            tb_writer.add_scalar("linear_grad_norm", model.linear.weight.grad.norm(), global_step)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.grad.norm())



from tqdm import tqdm
from collections import defaultdict
from easydict import EasyDict as edict
from scipy.stats import spearmanr, pearsonr
import numpy as np
def test_model_dataset(featnet, featFetcher, model, loader_test, score_m, score_s, label=None, modelid=None):
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


def test_multimodel_dataset(featnet, featFetcher, model_col, loader_test,
                            score_m, score_s, regresslayer, label=None):
    """  """
    if len(loader_test) == 0:
        return np.array([[]]), {}, {}
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
        S.regresslayer = regresslayer
        if label is not None:
            S.label = label
        S_dict[k] = S
    return scores_vec, pred_dict, S_dict

