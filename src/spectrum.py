from math import pi
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
    if algorithm == 'NJ':
        return _spectrum_NJ(ag, dt, T, zeta)
    elif algorithm == 'NM':
        return _spectrum_NM(ag, dt, T, zeta)
    else:
        assert False, 'algorithm must be NJ or NM'

def _spectrum_NJ(ag: float, dt: float, T: np.ndarray, zeta: float):
    """Nigam-Jennings精确解"""
    if T[0] == 0:
        T = T[1:]
        mark = 1
    else:
        mark = 0
    N = len(T)
    omg = 2 * np.pi / T
    wd = omg * np.sqrt(1 - zeta**2)
    n = len(ag)
    u = np.zeros((N, n))
    v = np.zeros((N, n))
    B1 = np.exp(-zeta * omg * dt) * np.cos(wd * dt)
    B2 = np.exp(-zeta * omg * dt) * np.sin(wd * dt)
    w_2 = 1.0 / omg ** 2
    w_3 = 1.0 / omg ** 3
    for i in range(n - 1):
        u_i = u[:, i]
        v_i = v[:, i]
        p_i = -ag[i]
        alpha_i = (-ag[i + 1] + ag[i]) / dt
        A0 = p_i * w_2 - 2.0 * zeta * alpha_i * w_3
        A1 = alpha_i * w_2
        A2 = u_i - A0
        A3 = (v_i + zeta * omg * A2 - A1) / wd
        u[:, i+1] = A0 + A1 * dt + A2 * B1 + A3 * B2
        v[:, i+1] = A1 + (wd * A3 - zeta * omg * A2) * B1 - (wd * A2 + zeta * omg * A3) * B2
    RSD = np.amax(abs(u), axis=1)
    RSV = RSD * omg
    RSA = RSD * omg ** 2
    if mark == 1:
        RSA = np.insert(RSA, 0, np.max(abs(ag)))
        RSV = np.insert(RSV, 0, 0)
        RSD = np.insert(RSD, 0, 0)
    return RSA, RSV, RSD

def _spectrum_NM(ag: float, dt: float, T: np.ndarray, zeta: float, gamma: float=0.5, beta: float=0.25):
    """Newmark-β"""
    if T[0] == 0:
        T = T[1:]
        mark = 1
    else:
        mark = 0
    N = len(T)
    n = len(ag)
    omg = 2 * pi / T
    u = np.zeros((N, n))
    v = np.zeros((N, n))
    a = np.zeros((N, n))
    a[0] = -ag[0]
    for i in range(n - 1):
        v_pred = v[:, i] + (1 - gamma) * dt * a[:, i]
        u_pred = u[:, i] + dt * v[:, i] + (0.5 - beta) * dt**2 * a[:, i]
        numerator = -ag[i+1] - 2*zeta*omg*v_pred - omg**2*u_pred
        denominator = 1 + 2*zeta*omg*gamma*dt + omg**2*beta*dt**2
        a_next = numerator / denominator
        v_next = v_pred + gamma * dt * a_next
        u_next = u_pred + beta * dt**2 * a_next
        u[:, i+1] = u_next
        v[:, i+1] = v_next
        a[:, i+1] = a_next
    RSD = np.max(np.abs(u), axis=1)
    RSV = RSD * omg
    RSA = RSD * omg ** 2
    if mark == 1:
        RSA = np.insert(RSA, 0, np.max(abs(ag)))
        RSV = np.insert(RSV, 0, 0)
        RSD = np.insert(RSD, 0, 0)
    return RSA, RSV, RSD


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ag = np.loadtxt(r"F:\Projects\OpenSAS\GMs\th1.th")
    dt = 0.01
    T = np.arange(0.0, 6, 0.02)
    RSA1, RSV, RSD = spectrum(ag, dt, T, zeta=0.05, algorithm='NM')
    RSA2, RSV, RSD = spectrum(ag, dt, T, zeta=0.05, algorithm='NJ')
    plt.plot(T, RSA1, label='NM')
    plt.plot(T, RSA2, label='NJ')
    plt.legend()
    plt.show()