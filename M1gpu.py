import torch
import torchvision

torch.version
import torch
X= torch.arange(24 ,dtype=float)
X=X.reshape(2,3,4)
print(X)
mean = X.mean(dim=(0,2),keepdim=True)
print(mean)

Y=torch.arange(12, dtype=float)
Y=Y.reshape(3, 4)
mean = Y.mean(dim=0, keepdim=True)
print(mean)