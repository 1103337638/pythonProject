import torch
import torchvision

torch.version
import torch
device = torch.device("mps")
print(torch.cuda.is_available())
x = torch.randn(128, 128, device=device)
net = torchvision.models.resnet18().to(device)

print(x.device)
print(next(net.parameters()).device)