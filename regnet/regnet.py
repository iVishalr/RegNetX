import torch.nn as nn
from .anynet import AnyNet
from .modules import YBlock
from .utils import *

__all__ = ['regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032',
           'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320']

class RegNet(AnyNet):
    def __init__(self, w_a, w_0, w_m, d, group_w, bot_mul, block=YBlock, se_r=0.25, num_classes=10, **kwargs) -> None:

        ws,num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        s_ws, s_ds = get_stages_from_blocks(ws,ws)

        s_gs = [group_w for _ in range(num_stages)]
        s_bs = [bot_mul for _ in range(num_stages)]
        s_ss = [2 for _ in range(num_stages)]

        s_ws, s_gs = adjust_widths_groups_compatibility(s_ws, s_bs, s_gs)

        kwargs = {
            "stem_w": 32,
            "depths":s_ds,
            "widths":s_ws,
            "strides":s_ss,
            "bottlenecks":s_bs,
            "block":block,
            "group_widths":s_gs,
            "se_ratio":se_r,
            "num_classes":num_classes
        }

        super(RegNet, self).__init__(**kwargs)
        print("number of parameters: ", sum(p.numel() for p in self.parameters()))

def regnetx_002(**kwargs):
    model = RegNet(w_a=36.44, w_0=24, w_m=2.49, d=13, group_w=8, bot_mul=1, **kwargs)
    return model


def regnetx_004(**kwargs):
    model = RegNet(w_a=24.48, w_0=24, w_m=2.54, d=22, group_w=16, bot_mul=1, **kwargs)
    return model


def regnetx_006(**kwargs):
    model = RegNet(w_a=36.97, w_0=48, w_m=2.24, d=16, group_w=24, bot_mul=1, **kwargs)
    return model


def regnetx_008(**kwargs):
    model = RegNet(w_a=35.73, w_0=56, w_m=2.28, d=16, group_w=16, bot_mul=1, **kwargs)
    return model


def regnetx_016(**kwargs):
    model = RegNet(w_a=34.01, w_0=80, w_m=2.25, d=18, group_w=24, bot_mul=1, **kwargs)
    return model


def regnetx_032(**kwargs):
    model = RegNet(w_a=26.31, w_0=88, w_m=2.25, d=25, group_w=48, bot_mul=1, **kwargs)
    return model


def regnetx_040(**kwargs):
    model = RegNet(w_a=38.65, w_0=96, w_m=2.43, d=23, group_w=40, bot_mul=1, **kwargs)
    return model


def regnetx_064(**kwargs):
    model = RegNet(w_a=60.83, w_0=184, w_m=2.07, d=17, group_w=56, bot_mul=1, **kwargs)
    return model


def regnetx_080(**kwargs):
    model = RegNet(w_a=49.56, w_0=80, w_m=2.88, d=23, group_w=120, bot_mul=1, **kwargs)
    return model


def regnetx_120(**kwargs):
    model = RegNet(w_a=73.36, w_0=168, w_m=2.37, d=19, group_w=112, bot_mul=1, **kwargs)
    return model


def regnetx_160(**kwargs):
    model = RegNet(w_a=55.59, w_0=216, w_m=2.1, d=22, group_w=128, bot_mul=1, **kwargs)
    return model


def regnetx_320(**kwargs):
    model = RegNet(w_a=69.86, w_0=320, w_m=2.0, d=23, group_w=168, bot_mul=1, **kwargs)
    return model