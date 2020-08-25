import torch
from rubiksnet.models import RubiksNet

num_frames = 8

net = RubiksNet(tier='large', num_classes=42, num_frames=num_frames)
net.cuda()

video = torch.randn((2, num_frames, 3, 224, 224), device='cuda')
prediction = net(video)

print('Random prediction:', prediction)
print('Installation successful!')