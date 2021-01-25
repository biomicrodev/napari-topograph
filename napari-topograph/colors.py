from typing import List, Tuple

import colorcet as cc

__all__ = ["cc_cmaps", "hex2uint8"]


cc_cmaps: List[str] = sorted(list(cc.palette.keys()))


def hex2uint8(v: str) -> Tuple[int, int, int]:
    if v.startswith("#"):
        v = v[1:]

    r, g, b = v[0:2], v[2:4], v[4:6]
    r, g, b = int(r, 16), int(g, 16), int(b, 16)
    return r, g, b
