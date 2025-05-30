# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# cython: language_level=3

import numpy as np
cimport numpy as cnp
import cython
from math import pi
from libc.math cimport exp, cos, sin

# 定义数据类型
cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

def spectrum(ag: cnp.ndarray, dt: float, T: cnp.ndarray, 
             zeta: float = 0.05, algorithm: str = 'NM'):
    """计算地震动弹性反应谱
    
    Args:
        ag (np.ndarray): 加速度时程
        dt (float): 时间间隔
        T (np.ndarray): 周期序列
        zeta (float, optional): 阻尼比，默认0.05
        algorithm (Literal['NJ', 'NM'], optional): 算法，NJ: Nigam-Jennings精确解，NM: Newmark-β直接积分
    
    Returns:
        tuple: (伪加速度谱, 伪速度谱, 位移谱)
    """
    if algorithm == 'NJ':
        return _spectrum_NJ(ag, dt, T, zeta)
    elif algorithm == 'NM':
        return _spectrum_NM(ag, dt, T, zeta)
    else:
        raise ValueError("algorithm must be 'NJ' or 'NM'")

cdef tuple _spectrum_NJ(
    cnp.ndarray[DTYPE_t, ndim=1] ag, 
    DTYPE_t dt, 
    cnp.ndarray[DTYPE_t, ndim=1] T, 
    DTYPE_t zeta):
    
    cdef:
        Py_ssize_t mark = 0
        Py_ssize_t N, n, i, j
        cnp.ndarray[DTYPE_t, ndim=1] omg, wd, RSD, RSV, RSA
        cnp.ndarray[DTYPE_t, ndim=2] u, v
        DTYPE_t p_i, alpha_i, A0, A1, A2, A3, max_ag
        DTYPE_t omg_i, wd_i, B1_i, B2_i, w_2_i, w_3_i
    
    # 检查零周期情况
    if T[0] == 0:
        T = T[1:]
        mark = 1
    
    N = T.shape[0]
    n = ag.shape[0]
    
    # 计算角频率
    omg = 2 * np.pi / T
    wd = omg * np.sqrt(1 - zeta**2)
    
    # 初始化位移和速度矩阵
    u = np.zeros((N, n), dtype=DTYPE)
    v = np.zeros((N, n), dtype=DTYPE)
    
    # Nigam-Jennings核心计算 - 优化版本
    for i in range(N):
        omg_i = omg[i]
        wd_i = wd[i]
        w_2_i = 1.0 / (omg_i * omg_i)
        w_3_i = 1.0 / (omg_i * omg_i * omg_i)
        B1_i = exp(-zeta * omg_i * dt) * cos(wd_i * dt)
        B2_i = exp(-zeta * omg_i * dt) * sin(wd_i * dt)
        
        for j in range(n - 1):
            p_i = -ag[j]
            alpha_i = (-ag[j+1] + ag[j]) / dt
            
            A0 = p_i * w_2_i - 2.0 * zeta * alpha_i * w_3_i
            A1 = alpha_i * w_2_i
            A2 = u[i, j] - A0
            A3 = (v[i, j] + zeta * omg_i * A2 - A1) / wd_i
            
            u[i, j+1] = A0 + A1 * dt + A2 * B1_i + A3 * B2_i
            v[i, j+1] = A1 + (wd_i * A3 - zeta * omg_i * A2) * B1_i - \
                        (wd_i * A2 + zeta * omg_i * A3) * B2_i
    
    # 计算反应谱
    RSD = np.max(np.abs(u), axis=1)
    RSV = RSD * omg
    RSA = RSD * omg**2
    
    # 处理零周期情况
    if mark:
        max_ag = np.max(np.abs(ag))
        RSA = np.insert(RSA, 0, max_ag)
        RSV = np.insert(RSV, 0, 0)
        RSD = np.insert(RSD, 0, 0)
    
    return RSA, RSV, RSD

cdef tuple _spectrum_NM(
    cnp.ndarray[DTYPE_t, ndim=1] ag, 
    DTYPE_t dt, 
    cnp.ndarray[DTYPE_t, ndim=1] T, 
    DTYPE_t zeta,
    DTYPE_t gamma = 0.5, 
    DTYPE_t beta = 0.25):
    
    cdef:
        Py_ssize_t mark = 0
        Py_ssize_t N, n, i, j
        cnp.ndarray[DTYPE_t, ndim=1] omg, RSD, RSV, RSA
        cnp.ndarray[DTYPE_t, ndim=2] u, v, a
        DTYPE_t max_ag
        DTYPE_t v_pred, u_pred, numerator, denominator
        DTYPE_t omg_i
    
    # 检查零周期情况
    if T[0] == 0:
        T = T[1:]
        mark = 1
    
    N = T.shape[0]
    n = ag.shape[0]
    
    # 计算角频率
    omg = 2 * np.pi / T
    
    # 初始化位移、速度和加速度矩阵
    u = np.zeros((N, n), dtype=DTYPE)
    v = np.zeros((N, n), dtype=DTYPE)
    a = np.zeros((N, n), dtype=DTYPE)
    a[:, 0] = -ag[0]
    
    # Newmark-β核心计算 - 优化版本
    for i in range(N):
        omg_i = omg[i]
        
        for j in range(n - 1):
            # 预测步
            v_pred = v[i, j] + (1 - gamma) * dt * a[i, j]
            u_pred = u[i, j] + dt * v[i, j] + (0.5 - beta) * dt**2 * a[i, j]
            
            # 修正步
            numerator = -ag[j+1] - 2 * zeta * omg_i * v_pred - omg_i**2 * u_pred
            denominator = 1 + 2 * zeta * omg_i * gamma * dt + omg_i**2 * beta * dt**2
            
            a[i, j+1] = numerator / denominator
            v[i, j+1] = v_pred + gamma * dt * a[i, j+1]
            u[i, j+1] = u_pred + beta * dt**2 * a[i, j+1]
    
    # 计算反应谱
    RSD = np.max(np.abs(u), axis=1)
    RSV = RSD * omg
    RSA = RSD * omg**2
    
    # 处理零周期情况
    if mark:
        max_ag = np.max(np.abs(ag))
        RSA = np.insert(RSA, 0, max_ag)
        RSV = np.insert(RSV, 0, 0)
        RSD = np.insert(RSD, 0, 0)
    
    return RSA, RSV, RSD
