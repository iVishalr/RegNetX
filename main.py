from regnet import regnetx_032 as RegNet32
from regnet import XBlock
import torch
model = RegNet32(block=XBlock, num_classes=10)
print(model)
print(model.parameters())