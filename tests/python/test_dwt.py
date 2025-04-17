from skimage import data

import pytorch_wavelets as wt
import matplotlib.pyplot as plt
import torch
import pdb

class MyDwt2d(torch.nn.Module):
    """docstring for MyDwt2d."""
    def __init__(self, wave_type="db1"):
        super(MyDwt2d, self).__init__()
        self.dwt=wt.DWTForward(1,wave_type)
    
    def forward(self,x )->torch.Tensor:
        # pdb.set_trace()
        s=self.dwt(x)
        LL,(LH,HL,HH)=self.dwt(x)
        h = torch.cat([LL, LH, HL, HH], dim=1)
        print(h.shape)
        return h


    

im_ori=data.chelsea()# shape (300,451,3)
im= torch.Tensor(im_ori).permute(2,0,1).unsqueeze(0)
dwt=MyDwt2d("db1")
s=dwt(im)
print(x.shape)

plt.imshow(s, cmap='gray')
plt.title("DWT Coefficients")
plt.show()
