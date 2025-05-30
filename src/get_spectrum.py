import os
import struct
import hashlib
from pathlib import Path
from typing import Literal
import numpy as np
from .spectrum import spectrum


def get_spectrum(
    ag: np.ndarray,
    dt: float,
    T: np.ndarray,
    zeta: float,
    algorithm: Literal['NJ', 'NM']='NM',
    cache_dir: Path | str=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算反应谱，如果有缓存则从缓存中读取"""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        bdata = T.tobytes() + ag.tobytes() + struct.pack('d', dt) + struct.pack('d', zeta)
        hash_val = hashlib.sha256(bdata).hexdigest()
        if (cache_dir / f"{hash_val}.npy").exists():
            spec_data = np.load(cache_dir / f"{hash_val}.npy")
            RSA, RSV, RSD = spec_data.T
        else:
            RSA, RSV, RSD = spectrum(ag, dt, T, zeta, algorithm)
            spec_data = np.column_stack((RSA, RSV, RSD))
            if not cache_dir.exists():
                os.makedirs(cache_dir)
            np.save(cache_dir / f"{hash_val}.npy", spec_data)
    else:
        RSA, RSV, RSD = spectrum(ag, dt, T, zeta, algorithm)
    return RSA, RSV, RSD
    
