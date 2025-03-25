import custom_op as cop


import torch
import torch.nn.functional as F
import numpy    as np


class Block(torch.nn.Module):
    def __init__(self, wave_type, param_shrink):
        super(Block, self).__init__()
        self._dwt=cop.DWT(wave_type)
        self._idwt=cop.IDWT(wave_type)
        # self._param=torch.nn.Parameter(torch.rand(1))
        self._shrink= torch.nn.Softshrink(param_shrink)
        
    def forward(self, x: torch.Tensor):  
        pre_alpha= self._dwt(x)     
        alpha= self._shrink(pre_alpha)
        return  self._idwt(alpha-pre_alpha)


class SuperResolve(torch.nn.Module):
    def __init__(self):
        super(SuperResolve, self).__init__()

        self._blocks=list()
        self._blocks.append(Block("db1",0.5))
        self._blocks.append(Block("db2",0.5))
        self._weight_params=torch.nn.Parameter(torch.rand((len(self._blocks),1)))
    def forward(self, x: torch.Tensor):         
        delta_xs=list()
        for b in self._blocks:
            delta_xs.append(b(x))
        delta_xs = torch.stack(delta_xs, dim=-1)
        delta=torch.matmul(delta_xs,self._weight_params)   
        # print(delta.size())            
        x= x+delta.reshape(x.size())
        return x
class SuperResolve1(torch.nn.Module):
    def __init__(self,p_lambda,p_beta):
        super(SuperResolve1, self).__init__()
        # Define layers here, for example:
        self.p1=p_lambda/(p_lambda+p_beta)
        self.p2=p_beta/(p_lambda+p_beta)
        self._wave=list()
        self._wave.append((cop.DWT("db1"),cop.IDWT("db1")))
        self._wave.append((cop.DWT("db2"),cop.IDWT("db2")))
        self._wave.append((cop.DWT("db3"),cop.IDWT("db3")))
        self._wave.append((cop.DWT("db4"),cop.IDWT("db4")))
        self._wave.append((cop.DWT("db5"),cop.IDWT("db5")))       
        self._shrink= torch.nn.Softshrink(1/p_beta)
        self.weight_params=torch.nn.Parameter(torch.rand((len(self._wave),1)))

        
    def wave_delta(self,x ,dwt, idwt):
        pre_alpha= dwt(x)     
        alpha= self._shrink(pre_alpha)
        return  idwt(alpha-pre_alpha)
        
    def forward(self, x: torch.Tensor):         
        total=torch.sum(self.weight_params)
        normal_weight_params=  self.weight_params/total
        
        delta_xs=list()
        for pair in self._wave:
            delta_xs.append(self.wave_delta( x, pair[0], pair[1]))
        delta_xs = torch.stack(delta_xs, dim=-1)
        # print(delta_xs.size())            
        # print(normal_weight_params.size())            
        delta=torch.matmul(delta_xs,normal_weight_params)   
        # print(delta.size())            
        x= x+delta.reshape(x.size())
        return x


class SRModel(torch.nn.Module):    
    def __init__(self, new_shape=(512, 512)):
        super(SRModel, self).__init__()
        self.new_shape=new_shape
        # self.sr= SuperResolve(p_lambda=0.3, p_beta=0.6)
        self.sr= SuperResolve( )
        self.softmax=torch.nn.Softmax(-1)
    def forward(self,s):
        x = F.interpolate(s, size=self.new_shape)
        for i in range(5):
            # pre=x
            x = self.sr(x)
            # x= self.softmax(x)
        return x

# def model(s: torch.Tensor, new_shape=(512, 512))->torch.Tensor:
    
    
#     # optimizer = torch.optim.Adam(sr_model.parameters(), lr=0.001)
#     # optimizer.zero_grad()
#     # loss.backward()
#     # optimizer.step()

#     return x, loss


    