"""
Many small utils functions useful for Correlated Feature Visualization Analysis etc.
"""
from easydict import EasyDict
import numpy as np
from core.plot_utils import saveallforms, off_axes, showimg


def area_mapping(num):
    if num <= 32: return "IT"
    elif num <= 48 and num >= 33: return "V1"
    elif num >= 49: return "V4"


def add_suffix(D: dict, sfx: str=""):
    newdict = EasyDict()
    for k, v in D.items():
        newdict[k + sfx] = v
    return newdict


def merge_dicts(dicts: list):
    newdict = EasyDict()
    for D in dicts:
        newdict.update(D)
    return newdict


def multichan2rgb(Hmaps):
    """Util function to summarize multi channel array to show as rgb"""
    if Hmaps.ndim == 2:
        Hmaps_plot = np.repeat(Hmaps[:,:,np.newaxis], 3, axis=2)
    elif Hmaps.shape[2] < 3:
        Hmaps_plot = np.concatenate((Hmaps, np.zeros((*Hmaps.shape[:2], 3 - Hmaps.shape[2]))), axis=2)
    else:
        Hmaps_plot = Hmaps[:, :, :3]
    Hmaps_plot = Hmaps_plot/Hmaps_plot.max()
    return Hmaps_plot

