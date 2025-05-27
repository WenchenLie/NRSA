import numpy as np


def newmark_solver(
    T: float,
    ag: np.ndarray,  # 只能一维
    dt: float,
    materials: dict[str, tuple],
    uy: float=0.0,
    fv_duration: float=0.0,
    sf: float=1.0,
    P: float=0.0,
    h: float=1.0,
    zeta: float=0.05,
    m: float=1.0,
    g: float=9800,
    collapse_disp: float=1e14,
    maxAnalysis_disp: float=1e15,
    tol: float = 1e-5,
    max_iter: float = 100,
    beta: float = 0.25,
    gamma: float = 0.5,
    display_info: bool=True
): ...