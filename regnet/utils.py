
import numpy as np

def quantize_float(f, q):
    """
    Converts a float to closest non-zero int divisible by q
    """
    return int(round(f/q) * q)

def adjust_widths_groups_compatibility(widths, bottlenecks, group_widths):
    """
    Adjusts the compatibility of widths and groups.
    """
    ws_bot = [int(w*b) for w,b, in zip(widths, bottlenecks)]
    gs = [min(g, w_bot) for g,w_bot in zip(group_widths,ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, group_widths)]
    ws = [int(w_bot/b) for w_bot,b in zip(ws_bot, bottlenecks)]
    return ws, gs

def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))       # ks = [0,1,2...,3...]
    ws = w_0 * np.power(w_m, ks)                             # float channel for 4 stages
    ws = np.round(np.divide(ws, q)) * q                      # make it divisible by 8
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont