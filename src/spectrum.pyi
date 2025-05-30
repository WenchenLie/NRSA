from typing import Literal
import numpy as np


def spectrum(ag: float, dt: float, T: np.ndarray, zeta: float=0.05, algorithm: Literal['NJ', 'NM']='NM'):
    """计算地震动弹性反应谱

    Args:
        ag (np.ndarray): 加速度时程
        dt (float): 时间间隔
        T (np.ndarray): 周期序列
        zeta (float, optional): 阻尼比，默认0.05.
        algorithm (Literal['NJ', 'NM'], optional): 算法，NJ: Nigam-Jennings精确解，NM: Newmark-β直接积分
    """
    ...