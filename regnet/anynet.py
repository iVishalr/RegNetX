
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules import Stem, Head, Stage, XBlock, YBlock

class AnyNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(AnyNet, self).__init__()

        if kwargs:
            self._construct(
                stem_w=kwargs["stem_w"],
                depths=kwargs["depths"],
                widths=kwargs["widths"],
                strides=kwargs["strides"],
                block=kwargs["block"],
                bottlenecks=kwargs["bottlenecks"],
                group_widths=kwargs["group_widths"],
                se_ratio=kwargs["se_ratio"],
                num_classes=kwargs["num_classes"]
            )
    
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m ,nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0/fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def _construct(self, stem_w, depths, widths, strides, block, bottlenecks, group_widths, se_ratio, num_classes):
        bottlenecks = bottlenecks if bottlenecks else [None for _d in depths]
        group_widths = group_widths if group_widths else [None for _d in depths]

        self.stem = Stem(3, stem_w)
        prev_w = stem_w
        for i, (d, w, s, bottleneck, group_width) in enumerate(list(zip(depths, widths, strides, bottlenecks, group_widths))):
            self.add_module(f"stage_{i+1}", Stage(in_channels=prev_w, out_channels=w, stride=s, depth=d, block=block, bottleneck_ratio=bottleneck, group_width=group_width, se_ratio=se_ratio))
            prev_w = w
        self.head = Head(prev_w, num_classes)

    def forward(self, x, y=None):
        for layer in self.children():
            x = layer(x)

        if y is not None:
            loss = F.cross_entropy(x,y)

        return x, loss