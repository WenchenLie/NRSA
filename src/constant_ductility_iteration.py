import os, json
import time
import traceback
import multiprocessing
import importlib
from pathlib import Path
from typing import Callable
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd

from .config import SOLVER_TYPING
from .utils import SDOFHelper, SDOFError


def constant_ductility_iteration(*args, **kwargs):
    queue: multiprocessing.Queue = args[-4]
    hidden_prints: bool = args[-5]
    try:
        with SDOFHelper(getTime=False, suppress=hidden_prints):
            _constant_ductility_iteration(*args, **kwargs)
    except Exception as error:
        tb = traceback.format_exc()
        queue.put({'d': (error, tb)})
        return

def _constant_ductility_iteration(
    wkdir: Path,
    subfolder: str,
    periods_shm_name: str,
    num_period: int,
    material_function: Callable[[float, float, float, float],
                                tuple[dict[str, tuple | float], float, float]],
    material_paras: tuple[float],
    damping: float,
    target_ductility: float,
    thetaD: float,
    mass: float,
    height: float,
    GM_name: str,
    gm_shm_name: str,
    NPTS: int,
    scaling_factor: float,
    dt: float,
    fv_duration: float,
    R_init: float,
    R_incr: float,
    Sa_shm_name: str,
    solver: SOLVER_TYPING,
    tol_ductility: float,
    tol_R: float,
    max_iter: int,
    hidden_prints: bool,
    queue: multiprocessing.Queue,
    stop_event,
    pause_event,
    lock,
    **kwargs
):
    """等延性分析迭代函数

    Args:
        wkdir (Path): 工作路径
        subfolder (str): 子文件夹名称
        periods_shm_name (str): 周期序列共享内存名
        num_period (int): 周期数
        material_function (Callable[[float, float, float, float], tuple[str, list, float, float]]): opensees格式材料定义函数
        material_paras (tuple[float]): 滞回模型控制参数，用于输入material_function中获得材料参数
        damping (float): 阻尼比
        target_ductility (float): 目标延性
        thetaD (float): P-Delta系数
        mass (float): 质量
        height (float): 高度
        GM_name (str): 地震动名称
        gm_shm_name (str): 地震动共享内存名
        NPTS (int): 时间步数
        scaling_factor (float): 地震动时缩放系数
        dt (float): 时间步长
        fv_duration (float): 自由振动持续时间
        R_init (float): 初始强度折减系数
        R_incr (float): 强度折减系数增量
        Sa_shm_name (str): 加速度反应谱共享内存名
        solver (SOLVER_TYPES): SODF求解器类型
        tol_ductility (float): 延性(μ)收敛容差
        tol_R (float): 相邻强度折减系数(R)收敛容差
        max_iter (int): 最大迭代次数
        hidden_prints (bool): 是否屏蔽输出
        queue (multiprocessing.Queue): 进程通信
        stop_event: 停止事件
        pause_event: 暂停事件
        lock: 锁，文件读写时使用
        kwargs: 求解器参数

    Note:
    ----
        (1) 以`R=1`为起始条件，每次迭代增大R
        (2) 收敛准则：`abs(μ - μ_target) / μ_target < tol`
    """
    periods_shm = SharedMemory(name=periods_shm_name)
    periods = np.ndarray(shape=(num_period,), dtype=np.dtype('float64'), buffer=periods_shm.buf).copy()
    periods_shm.close()
    periods: list[float] = list(periods)
    Sa_shm = SharedMemory(name=Sa_shm_name)
    Sa_ls = np.ndarray(shape=(num_period,), dtype=np.dtype('float64'), buffer=Sa_shm.buf).copy()
    Sa_shm.close()
    gm_shm = SharedMemory(name=gm_shm_name)
    th = np.ndarray(shape=(NPTS,), dtype=np.dtype('float64'), buffer=gm_shm.buf).copy()
    gm_shm.close()
    num_ana = 0  # 计算该地震动所进行的总SDOF分析次数
    results = pd.DataFrame(None, columns=['T', 'Sa', 'E', 'Fy', 'uy', 'R', 'maxDisp', 'maxVel', 'maxAccel', 'Ec', 'Ev', 'maxReaction', 'CD', 'CPD','resDisp', 'solving_converge', 'iter_converge', 'n_iter', 'miu'])
    start_time = time.time()
    num_iters: list[int] = []  # 迭代次数
    solving_converge = 1  # 求解是否收敛，如果有不收敛则变为False
    iter_converge = 1  # 迭代是否收敛，如果超过迭代次数则变为False
    package_name = __package__
    module_attr = {
        'newmark': 'newmark_solver',
        'ops_solver': 'ops_solver'
    }
    for idx, Ti in enumerate(periods):
        R = R_init  # 初始强度折减系数 (R=1时结构弹性，R越大，结构强度越低，位移越大)
        R1, R2 = 0, 10000000  # 不超过目标延性的最大强度折减系数, 超过目标延性的最小强度折减系数
        best_R = 0  # 最优强度折减系数
        miu1, miu2 = 0, 0  # 不超过目标延性的最大延性, 超过目标延性的最小延性
        best_miu = 0  # 最接近目标延性的延性
        best_E, best_Fy = 0, 0  # 最优刚度、屈服力
        best_tol_ductility = 1e10  # 最优延性收敛容差
        best_tol_R = 1e10  # 最优R收敛容差
        miu_prev = 0  # 前一次迭代的延性
        iter_status = False  # 迭代状态
        best_res = None  # 最优结果
        Sa = Sa_ls[idx]  # 弹性谱加速度
        for n_iter in range(1, max_iter + 1):
            if stop_event.is_set():
                queue.put({'e': '中断计算'})
                return
            pause_event.wait()
            mat_paras, Fy, E = material_function(Ti, mass, R, Sa, *material_paras)
            uy = Fy / E
            if thetaD == 0:
                P = 0
            else:
                P = thetaD * E * height
            solver_paras = (Ti, th, dt, mat_paras, uy, fv_duration, scaling_factor, P, height, damping, mass)
            try:
                if solver == 'auto':
                    for module_name in ['newmark', 'ops_solver']:
                        solver_func = importlib.import_module(f".{module_name}", package=package_name).__getattribute__(module_attr[module_name])
                        res: dict = solver_func(*solver_paras, **kwargs)
                        if res['converge']:
                            break 
                elif solver == 'OPS':
                    solver_func = importlib.import_module(f".ops_solver", package=package_name).__getattribute__('ops_solver')
                    res: dict = solver_func(*solver_paras, **kwargs)
                elif solver == 'Newmark-Newton':
                    solver_func = importlib.import_module(f".newmark", package=package_name).__getattribute__('newmark_solver')
                    res: dict = solver_func(*solver_paras, **kwargs)
                else:
                    raise SDOFError(f'Wrong solver name: {solver}')
            except:
                raise SDOFError('Solver error!')
            num_ana += 1
            is_converged: bool = res['converge']  # 计算是否收敛
            if not is_converged:
                solving_converge = 0
                queue.put({'b': (GM_name, Ti, solver_paras[2:])})
                print('Warning: Unconvered analysis!')
                print(f'Ground motion: {GM_name}, Period: {Ti}, dt: {dt}, solver_paras:', *solver_paras[3:])
                unconverged_res = {
                    'period': Ti,
                    'ground motion': GM_name,
                    'dt': dt,
                    'mat_paras': mat_paras,
                    'uy': uy,
                    'fv_duration': fv_duration,
                    'scaling_factor': scaling_factor,
                    'P': P,
                    'height': height,
                    'damping': damping,
                    'mass': mass,
                    'E': E,
                    'Fy': Fy
                }
                current_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                lock.acquire()
                if not Path(wkdir / subfolder / f'warnings').exists():
                    os.makedirs(wkdir / subfolder / f'warnings')
                json.dump(unconverged_res, open(wkdir / subfolder / f'warnings/{current_date}_{GM_name}_UnconvergedAnalysis.json', 'w', encoding='utf8'), ensure_ascii=False, indent=4)
                lock.release()
            maxDisp: float = res['maxDisp']  # 最大相对位移
            miu = maxDisp / uy  # 延性
            current_tol_ductility = abs(miu - target_ductility) / target_ductility
            current_tol_R = abs(R1 - R2) / max(R1, R2)
            # print('--------------', miu, maxDisp, mat_paras)
            if current_tol_ductility < best_tol_ductility:
                # 更新最小容差，最优R，最优延性，最优结果
                best_tol_ductility = current_tol_ductility
                best_tol_R = current_tol_R
                best_R = R
                best_miu = miu
                best_res = res
                best_E = E
                best_Fy = Fy
            if current_tol_ductility < tol_ductility and current_tol_R < tol_R:
                # 迭代成功收敛
                break  # 当且仅当成功收敛才跳出循环
            if not iter_status:
                # 非迭代状态
                if miu < target_ductility:
                    R = R + R_incr  # 保持非迭代状态，且延性小于目标
                else:
                    iter_status = True  # 上一次为非迭代状态，延性大于目标，首次进入迭代状态
                    R1 = R - R_incr
                    R1 = max(R1, 0)  # 强度折减系数不能为负
                    R2 = R
                    miu1 = miu_prev
                    miu2 = miu
                    R = (R1 + R2) / 2  # 二分法
                    # R = (R2 - R1) / (miu2 - miu1) * (target_ductility - miu1) + R1  # 线性插值法
            else:
                # 迭代状态
                if miu < target_ductility:
                    R1 = R  # 延性小于目标，更新不超过目标延性的最大强度折减系数
                    miu1 = max(miu, miu1)
                else:
                    R2 = R  # 延性大于目标，更新超过目标延性的最小强度折减系数
                    miu2 = min(miu, miu2)
                R = (R1 + R2) / 2  # 二分法
                # R = (R2 - R1) / (miu2 - miu1) * (target_ductility - miu1) + R1  # 线性插值法
            miu_prev = miu  # 记录前一次迭代的延性
        else:
            # 迭代达到最大次数仍未收敛
            iter_converge = 0
            queue.put({'c': (GM_name, Ti, current_tol_ductility, current_tol_R)})
            print('Warning: Unconvered iteration!')
            print(f'Ground motion: {GM_name}, Period: {Ti}, dt: {dt}, bset_miu: {best_miu}, current_tol_ductility: {current_tol_ductility}, current_tol_R: {current_tol_R}')
            unconverged_res = {
                'period': Ti,
                'ground motion': GM_name,
                'current_tol_ductility': current_tol_ductility,
                'current_tol_R': current_tol_R,
                'current_R': R,
                'current_miu': miu,
                'best_miu': best_miu,
                'target_miu': target_ductility,
                'iter_num': n_iter,
            }
            current_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            lock.acquire()
            if not Path(wkdir / subfolder / f'warnings').exists():
                os.makedirs(wkdir / subfolder / f'warnings')
            json.dump(unconverged_res, open(wkdir / subfolder / f'warnings/{current_date}_{GM_name}_UnconvergedIteration.json', 'w', encoding='utf8'), ensure_ascii=False, indent=4)
            lock.release()
        num_iters.append(n_iter)
        row = len(results)
        # 无论迭代收敛或计算收敛是否成功，都记录结果
        results.loc[row] = best_res
        results.loc[row, 'Sa'] = Sa
        results.loc[row, 'E'] = best_E
        results.loc[row, 'Fy'] = best_Fy
        results.loc[row, 'uy'] = best_Fy / best_E
        results.loc[row, 'T'] = Ti
        results.loc[row, 'R'] = best_R
        results.loc[row, 'solving_converge'] = solving_converge
        results.loc[row, 'iter_converge'] = iter_converge
        results.loc[row, 'n_iter'] = n_iter
        results.loc[row, 'miu'] = best_miu
        results.loc[row, 'tol_ductility'] = best_tol_ductility
        results.loc[row, 'tol_R'] = best_tol_R
    mean_iter = np.mean(num_iters)
    mean_ana = num_ana / num_period
    end_time = time.time()
    lock.acquire()
    results.to_csv(wkdir / subfolder / f'{GM_name}.csv', index=False)
    lock.release()
    queue.put({'a': (GM_name, num_ana, num_period, iter_converge, solving_converge, start_time, end_time, mean_ana)})
    return None



