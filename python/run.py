
from skimage import data
from skimage.io import imsave
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skimage.transform import resize
from skimage.io import imread
import custom_op as cop
import pdb

def sample_data(im:torch.Tensor):
    im=im.permute(0,3,1,2)
    print(im.size())
    im= F.interpolate(im, size=(128,128), mode='bilinear', align_corners=False)
    im=im.permute(0,2,3,1)
    return im


def sample_process():
    im=data.chelsea()# shape (300,451,3)
    im= torch.Tensor(im).float().unsqueeze(0)
    print(im.size())
    s=sample_data(im)
    s = s.numpy().astype(np.uint8)
    imsave('sample.jpg', s)


class SUnit(nn.Module):
    def __init__(self,lamda, beta):
        super(SUnit, self).__init__()
        self.lamda=lamda
        self.beta=beta
        self.sample=cop.Sample(3,3,(128,128),(256,256))
        self.sample_t=cop.SampleTranpose(self.sample)
        self.dwt=cop.DWT()
        self.idwt=cop.IDWT()
        self.shrink=cop.Shrink(1/beta)    
        
    def forward(self, s,image, v, alpha):
        coeff= self.lamda/self.beta
        phi_im=self.dwt(image)+v                
        new_alpha=self.shrink(phi_im)
        new_v= phi_im-alpha
        idwtx=self.idwt(new_alpha-new_v)
        y1=torch.fft.fft2(idwtx)
        y2= coeff*self.sample_t(s)
        print( image.size(), idwtx.size(),y1.size(), y2.size())
        y=y1+y2        
        y=y+coeff*self.sample_t(self.sample(y))
        return torch.fft.ifft2(y), new_v, new_alpha

def process():
    signal = imread("sample.jpg")
    signal = torch.Tensor(signal).float().unsqueeze(0)
    signal=signal.permute(0,3,1,2)
    print(signal.size())
    net= SUnit(0.5,0.5)
    
    im=net.sample_t(signal)
    tmp=net.dwt(im)
    print(tmp.size())
    alpha=net.shrink(tmp)
    v=torch.zeros_like(alpha)    
    # im,v,alpha=net(signal,im,0,0)
    print(alpha.size(), v.size())
    for i in range(1):
        im,v,alpha=net(signal,im,v,alpha)        
    im=im.permute(0,2,3,1)
    # TODO: to be training
    imsave('reconstructed.jpg', im.detach().numpy().astype(np.uint8)[0])
    

def test_sample():    
    x=np.random.randn(1,256,512,3)
    x=torch.Tensor(x).permute(0,3,1,2)
    sampler=cop.Sample(32,3,(64,128),(256,512))
    sampler_t=cop.SampleTranpose(sampler)
    y=sampler(torch.Tensor(x))
    xr=sampler_t(y)
    print(y.size(),    xr.size())
    
if __name__ == '__main__':
    # test_sample()
    process()
    