from skimage import data
from skimage.io import imsave
import numpy as np
import torch
from torch.nn import functional as F

def sample_data(im:np.ndarray, new_shape):
    data_type=im.dtype
    
    im=torch.Tensor(im)
    if len(im.shape)==3:
        im=im.unsqueeze(0)
    print(im.size(),im.dtype)
    im=im.permute(0,3,1,2)    
    im= F.interpolate(im, size=new_shape, mode='bilinear', align_corners=False)
    im=im.permute(0,2,3,1)
    return im.numpy().astype(data_type)


 

def mesage_range(t: np.ndarray):
    t_amp=t.abs()
    return "({},{})".format(t_amp.min(), t_amp.max())