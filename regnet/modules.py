import torch.nn as nn
import torch.nn.functional as F

class Stem(nn.Module):
    """
    Stem Layer as described in Section 3.2.

    Paper uses the following defaults
    in_channels = 3
    out_channels = 32
    kernel_size = 3
    stride = 2
    padding = 1
    bias = False
    """

    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False) -> None:
        super(Stem,self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class Head(nn.Module):
    """
    Head layer as described in Section 3.2

    Average pooling followed by a fully connected network
    Outputs predictions for n classes
    """

    def __init__(self, in_features, out_features) -> None:
        super(Head, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0),-1) #(batch_size, -1)
        x = self.fc(x)
        return x

class SE(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block
    """

    def __init__(self, in_channels, out_channels) -> None:
        super(SE, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc(out)
        out = out * x
        return out

class XBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, group_width=1, bottleneck_ratio=1, sr_ratio=None) -> None:
        super(XBlock, self).__init__()

        inter_channels = int(round(out_channels * bottleneck_ratio))
        groups = inter_channels // group_width

        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels!=out_channels or stride!=1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.sbn = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut = None
    
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
            shortcut = self.sbn(shortcut)
        else:
            shortcut = 0
        
        out = out + shortcut
        out = F.relu(out)
        return out


class YBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, group_width=1, bottleneck_ratio=1, se_ratio=(1/4)) -> None:
        super(YBlock, self).__init__()

        inter_channels = int(round(out_channels * bottleneck_ratio))
        groups = inter_channels // group_width

        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)

        if se_ratio is not None:
            se_channels = int(round(in_channels * se_ratio))
            self.se = SE(inter_channels, se_channels)

        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels!=out_channels or stride!=1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.sbn = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut = None
    
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
            shortcut = self.sbn(shortcut)
        else:
            shortcut = 0
        
        out = out + shortcut
        out = F.relu(out)
        return out

class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depth, block, bottleneck_ratio, group_width, se_ratio=None) -> None:
        super(Stage, self).__init__()

        for i in range(depth):
            block_stride = stride if i==0 else 1
            block_width = in_channels if i==0 else out_channels
            self.add_module(f"block_{i+1}", block(block_width, out_channels, block_stride, group_width, bottleneck_ratio, se_ratio))

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x
