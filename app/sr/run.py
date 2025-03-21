import data_util
import skimage
from skimage import data
from skimage import io
import numpy as np
import torch
import model 
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F


im=data.chelsea()# shape (300,451,3)
im=data_util.sample_data(im, (512,512))
print(im.shape)
s=data_util.sample_data(im, (128,128))

sp=torch.Tensor(s)
print(sp.size(),sp.dtype)
sp=sp.permute(0,3,1,2)   # NCHW
# print(sp.size())
sr_model=model.SRModel((512,512))


# train

optimizer = torch.optim.Adam(sr_model.parameters(), lr=0.001)
optimizer.zero_grad()
for epoch in range(4):
    pre,x=sr_model(sp)
    loss = F.mse_loss(x, pre)    
    loss.backward()
    optimizer.step()
    print(loss)
# torch.onnx.export(
#     sr_model,
#     sp,
#     "aaa.onnx",
#     input_names=["input"],   # 输入节点名称（用于部署时识别）
#     output_names=["output"], # 输出节点名称
#     opset_version=11,        # ONNX 算子集版本（建议 ≥11）
#     verbose=True             # 打印导出过程
# )



# x=F.interpolate(x, size=(300,451), mode='bilinear', align_corners=False)
x=(x-x.min())/(x.max()-x.min())*255
x=x.permute(0,2,3,1)
x=x.detach().numpy().astype(np.uint8)
io.imsave("sample_ori.jpg",im)
io.imsave("sample.jpg",s)
io.imsave("rec.jpg",x )

print(im.min(),im.max())
print(x.min(),x.max())
print(im.shape, x.shape)
coeff_ssim = ssim(im[0], x[0],channel_axis=-1)
print(coeff_ssim)