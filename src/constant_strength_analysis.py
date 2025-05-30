import os, json
import time
import traceback
import multiprocessing
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .config import SOLVER_TYPING
from .utils import SDOFHelper, SDOFError
from .ops_solver import ops_solver
from .newmark import newmark_solver


def constant_strength_analysis(*args, **kwargs):
    queue: multiprocessing.Queue = args[-4]
    hidden_prints: bool = args[-5]
    try:
        with SDOFHelper(getTime=False, suppress=hidden_prints):
            _constant_strength_analysis(*args, **kwargs)
    except Exception as error:
        tb = traceback.format_exc()
        queue.put({'d': (error, tb)})
        return

def _constant_strength_analysis(
    wkdir: Path,
    periods: np.ndarray,
    material_function: Callable[[float, float, float, float], tuple[dict[str, tuple | float], float, float]],
    material_paras: dict[str, float],
    damping: float,
    thetaD: float,
    mass: float,
    height: float,
    GM_name: str,
    th: np.ndarray,
    scaling_factor: float,
    dt: float,
    fv_duration: float,
    Sa_ls: np.ndarray,
    solver: SOLVER_TYPING,
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
        periods (np.ndarray): 周期序列
        material_function (Callable[[float, float, float, float], tuple[str, list, float, float]]): opensees格式材料定义函数
        material_paras (dict[str, float]): 滞回模型控制参数，用于输入material_function中获得材料参数
        damping (float): 阻尼比
        thetaD (float): P-Delta系数
        mass (float): 质量
        height (float): 高度
        th (np.ndarray): 加速度时程
        GM_name (str): 地震动名称
        dt (float): 时间步长
        fv_duration (float): 自由振动持续时间
        Sa_ls (np.ndarray): 加速度反应谱，应与periods等长
        solver (SOLVER_TYPES): SODF求解器类型
        hidden_prints (bool): 是否屏蔽输出
        queue (multiprocessing.Queue): 进程通信
        stop_event: 停止事件
        pause_event: 暂停事件
        lock: 锁，文件读写时使用
        kwargs: 求解器参数
    """
    periods: list[float] = list(periods)
    num_period = len(periods)
    num_ana = 0  # 计算该地震动所进行的总SDOF分析次数
    results = pd.DataFrame(None, columns=['T', 'E', 'Fy', 'uy', 'Sa', 'R', 'miu', 'maxDisp', 'maxVel', 'maxAccel', 'Ec', 'Ev', 'maxReaction', 'CD', 'CPD','resDisp', 'solving_converge'])
    start_time = time.time()
    for idx, Ti in enumerate(periods):
        solving_converge = 1  # 求解是否收敛，如果有不收敛则变为False
        if stop_event.is_set():
            queue.put({'e': '中断计算'})
            return
        pause_event.wait()
        Sa = Sa_ls[idx]  # 弹性谱加速度
        paras = material_paras.values()
        ops_paras, Fy, E = material_function(Ti, mass, Sa, *paras)
        uy = Fy / E
        if thetaD == 0:
            P = 0
        else:
            P = thetaD * E * height
        solver_paras = (Ti, th, dt, ops_paras, uy, fv_duration, scaling_factor, P, height, damping, mass)
        try:
            if solver == 'auto':
                for solver_func in [newmark_solver, ops_solver]:
                    res: dict = solver_func(*solver_paras, **kwargs)
                    if res['converge']:
                        break 
            elif solver == 'OPS':
                res: dict = ops_solver(*solver_paras, **kwargs)
            elif solver == 'Newmark-Newton':
                res: dict = newmark_solver(*solver_paras, **kwargs)
            else:
                raise SDOFError(f'Wrong solver name: {solver}')
        except:
            raise SDOFError('Solver error!')
        num_ana += 1
        is_converged: bool = res['converge']  # 计算是否收敛
        if not is_converged:
            solving_converge = 0
            queue.put({'b': (GM_name, Ti, solver_paras[3:])})
            print(f'Warning: Unconverged analysis! ')
            print(f'Ground motion: {GM_name}, Period: {Ti}, dt: {dt}, solver_paras:', *solver_paras[3:])
            unconverged_res = {
                'period': Ti,
                'ground motion': GM_name,
                'dt': dt,
                'ops_paras': ops_paras,
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
            lock.acquire()
            if not Path(wkdir / f'warnings').exists():
                os.makedirs(wkdir / f'warnings')
            json.dump(unconverged_res, open(wkdir / f'warnings/c.json', 'w', encoding='utf8'), ensure_ascii=False, indent=4)
            lock.release()
        row = len(results)
        maxDisp = res['maxDisp']
        results.loc[row] = res
        results.loc[row, 'T'] = Ti
        results.loc[row, 'E'] = E
        results.loc[row, 'Fy'] = Fy
        results.loc[row, 'uy'] = Fy / E
        results.loc[row, 'Sa'] = Sa
        results.loc[row, 'R'] = mass * Sa * 9800 / Fy
        results.loc[row, 'miu'] = maxDisp / uy
        results.loc[row, 'solving_converge'] = solving_converge
    end_time = time.time()
    queue.put({'a': (GM_name, num_ana, num_period, None, solving_converge, start_time, end_time, None)})
    lock.acquire()
    results.to_csv(wkdir / f'results/{GM_name}.csv', index=False)
    lock.release()
    return results



