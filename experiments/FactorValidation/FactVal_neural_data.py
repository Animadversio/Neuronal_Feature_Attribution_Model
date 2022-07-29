
import os
from os.path import join
import numpy as np
from scipy.io import loadmat
from hdf5storage import loadmat
from easydict import EasyDict as edict
dataroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatVis_FactorPredict"
expdirs = [dir for dir in os.listdir(dataroot) if "decomp" in dir]

#%%
#%%
def recursive_matdata2dict(matdata):
    if type(matdata) is not np.ndarray:
        return matdata
    else:
        if matdata.dtype is np.dtype("O"):  # object type
            if matdata.shape is ():
                return recursive_matdata2dict(matdata.item())
            else:
                return list(matdata)
        elif np.issubdtype(matdata.dtype, np.number):  # number data could be ported directly
            return matdata
        else:  # composite type
            mdict = edict()
            for name in matdata.dtype.names:
                try:
                    mdict[name] = recursive_matdata2dict(matdata[name])
                except: # debugging information
                    print(matdata.dtype, name, type(matdata[name]), matdata[name])
            return mdict


expfdr = '2021-06-25-Alfa-01-decomp-Ch51'
for expfdr in expdirs:
    matdata = loadmat(join(dataroot, expfdr, "ExpStat.mat"), struct_as_record=True, squeeze_me=True, )
    mdict = recursive_matdata2dict(matdata["S"])
#%%
import matplotlib.pyplot as plt
# plt.plot(mdict.prefresp.img_mean)
# plt.show()
plt.figure()
plt.plot(mdict.prefresp.group_mean)
plt.xticks(ticks=range(len(mdict.prefresp.group_mean)),
           labels=mdict.stim.grouplabs, rotation=30)
plt.tight_layout()
plt.show()
#%%
matdata = loadmat(join(dataroot, expfdr, "ExpStat.mat"), struct_as_record=False, squeeze_me=True, )
expS = matdata["S"]
#%%
# import h5py
# Data = h5py.File(join(dataroot, expfdr, "ExpStat.mat"))
# self.imgnms = np.array([''.join(chr(i) for i in rspData[ref]) for ref in imgnms_refs])

# matdict = edict(matdata["S"])
# matdata["S"]["unit"]
# matdata["S"]["meta"]
# matdata["S"]["stim"]
# matdata["S"]["imageName"]
# matdata["S"]["prefresp"]

mdict.stim.group_uniqidxs
mdict.stim.group_idxarr
#%%
# bytes(f.get('stime')[:]).decode('utf-8')
bytes(mdict.Animal[:]).decode('utf-8')