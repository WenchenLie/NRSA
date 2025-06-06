import numpy as np


def newmark_solver(
    Ti: float,
    ag: np.ndarray,  # 只能一维
    dt: float,
    materials: dict[str, tuple],
    uy: float,
    fv_duration: float,
    sf: float,
    P: float,
    h: float,
    zeta: float,
    m: float,
    g: float = 9800,
    collapse_disp: float = 1e14,
    maxAnalysis_disp: float=1e15,
    tol: float = 1e-5,
    max_iter: float = 100,
    beta: float = 0.25,
    gamma: float = 0.5,
    display_info: bool=True,
    record_res: bool=False,
    **kwargs
) -> dict[str, float] | tuple[dict[str, float], tuple[np.ndarray, ...]]:
    ...