import torch 
import pywt
import numpy as np
import inspect
class DWT(torch.nn.Module):
    def __init__(self, wavelet='haar'):
        super(DWT, self).__init__()
        self.wavelet = wavelet

    def forward(self, x: torch.Tensor):        
        # return pywt.dwt(x.detach().numpy(), self.wavelet)
        # return pywt.dwt2(x.detach().numpy(), self.wavelet)
        cA, (cH, cV, cD)= pywt.dwt2(x.detach().numpy(), self.wavelet)
        return torch.Tensor(np.array([cA, cH, cV, cD]))

class IDWT(torch.nn.Module):
    def __init__(self, wavelet='haar'):
        super(IDWT, self).__init__()
        self.wavelet = wavelet
    def forward(self, x: torch.Tensor):
        cA, cH, cV, cD = x.detach().numpy()
        coeffs=[cA, (cH, cV, cD)]
        # return pywt.idwtn(x.detach().numpy(), self.wavelet)
        return  torch.Tensor(pywt.idwt2(coeffs, self.wavelet))
    

class Shrink(torch.nn.Module):
    def __init__(self, threshold):
        super(Shrink, self).__init__()
        self.threshold = threshold
    def forward(self, x):
        return torch.sign(x)* torch.relu(torch.abs(x)-self.threshold)

class Sample(torch.nn.Module):
    # NCHW
    def __init__(self, channel_out,channel_in,sample_size, high_res_size):        
        # Initialize the Sample class with the given parameters
        # sm, sn: dimensions of the sample size
        # m, n: dimensions of the high resolution size
        # lshape: shape of the left matrix for transformation
        # rshape: shape of the right matrix for transformation
        # cshape: shape of the channel dimensions (output channels, input channels)
        # L: left transformation matrix
        # R: right transformation matrix
        super(Sample, self).__init__()
        sm,sn=sample_size
        m,n=high_res_size
        self.lshape=(sm,m)
        self.rshape=(n,sn)
        self.cshape=(channel_in,channel_out)
        # print(self.lshape,self.rshape,self.cshape)
        self.L=torch.nn.Parameter(torch.randn(self.lshape))
        self.R=torch.nn.Parameter(torch.randn(self.rshape))
        self.C=torch.nn.Parameter(torch.randn(self.cshape))        


    def forward(self, x:torch.Tensor):
        input_size=x.size() # [Batch,CI,M,N]
        assert(input_size[-3]==self.cshape[0]) # CI
        assert(input_size[-2]==self.lshape[1]) # M    
        assert(input_size[-1]==self.rshape[0])  #N
        # print( inspect.currentframe().f_lineno,self.L.size(), x.size())
        x=torch.matmul(self.L.to(x.dtype),x) # [sm,M]* [Batch,CI,M,N]  -> [Batch,CI,Sm,N]
        # print( inspect.currentframe().f_lineno, x.size(),self.R.size())
        x=torch.matmul(x,self.R.to(x.dtype)) # [Batch,CI,Sm,N] *[N,Sn] -> [Batch,CI,Sm,Sn]
        x=x.permute(0,2,3,1) # [Batch,Sm,Sn,CI]
        # print( inspect.currentframe().f_lineno, x.size())
        x=torch.matmul(x,self.C.to(x.dtype)) # [Batch,Sm,Sn,CI]* [CI,CO]-> [Batch,Sm,Sn,Co]
        # print(x.size())  
        x=x.permute(0,3,1,2) # [Batch,Co,Sm,Sn]      
        # print(x.size())  
        return  x
    
    def tranpose_forward(self, x:torch.Tensor):
        # x should be [Batch,Co,Sm,Sn]
        # ret [Batch,CI,Sm,Sn]
        input_size = x.size() # [Batch, CO, Sm, Sn]
        assert(input_size[-3] == self.cshape[1]) # CO
        assert(input_size[-2] == self.lshape[0]) # Sm
        assert(input_size[-1] == self.rshape[1]) # Sn
        print( inspect.currentframe().f_lineno, x.size())
        x=x.permute(0,2,3,1) # [Batch,Sm,Sn,CO]
        print( inspect.currentframe().f_lineno, x.size())
        x=torch.matmul(x,self.C.t().to(x.dtype)) # [Batch,Sm,Sn,CO]@[CI,CO]^T -> [Batch,Sm,Sn,CI]
        print( inspect.currentframe().f_lineno, x.size())
        x=x.permute(0,3,1,2) # [Batch,CI,Sm,Sn]
        print( inspect.currentframe().f_lineno, x.size())
        x=torch.matmul(x,self.R.t().to(x.dtype)) # [Batch,CI,Sm,Sn]*[N,SN]^T -> [Batch,CI,Sm,N]
        print( inspect.currentframe().f_lineno, x.size())
        x=torch.matmul(self.L.t().to(x.dtype),x) # [sm,M]^T * [Batch,CI,Sm,N] -> [Batch,CI,M,N]
        print( inspect.currentframe().f_lineno, x.size())
        print( inspect.currentframe().f_lineno, x.size())
        return x

class SampleTranpose(torch.nn.Module):
    def __init__(self, sample: Sample):
        super(SampleTranpose, self).__init__()
        self.sample=sample 
    def forward(self, x:torch.Tensor):
        return self.sample.tranpose_forward(x)
