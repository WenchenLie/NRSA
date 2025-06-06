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


def time_history_analysis(*args, **kwargs):
    queue: multiprocessing.Queue = args[-4]
    hidden_prints: bool = args[-5]
    try:
        with SDOFHelper(getTime=False, suppress=hidden_prints):
            _time_history_analysis(*args, **kwargs)
    except Exception as error:
        tb = traceback.format_exc()
        queue.put({'d': (error, tb)})
        return

def _time_history_analysis(
    wkdir: Path,
    Ti: float,
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
    get_Sa: interp1d,
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
        Ti (np.ndarray): 周期点
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
        get_Sa (interp1d): 根据周期获取谱加速度，默认为0.02-6s，间隔0.02
        solver (SOLVER_TYPES): SODF求解器类型
        hidden_prints (bool): 是否屏蔽输出
        queue (multiprocessing.Queue): 进程通信
        stop_event: 停止事件
        pause_event: 暂停事件
        lock: 锁，文件读写时使用
        kwargs: 求解器参数
    """
    num_ana = 1
    results = pd.DataFrame(None, columns=['T', 'E', 'Fy', 'uy', 'Sa', 'R', 'miu', 'maxDisp', 'maxVel', 'maxAccel', 'Ec', 'Ev', 'maxReaction', 'CD', 'CPD','resDisp', 'solving_converge'])
    start_time = time.time()
    solving_converge = 1  # 求解是否收敛，如果有不收敛则变为False
    if stop_event.is_set():
        queue.put({'e': '中断计算'})
        return
    pause_event.wait()
    if Ti is not None:
        Sa = get_Sa(Ti)  # 弹性谱加速度
    else:
        Sa = None
    paras = material_paras.values()
    ops_paras, Fy, E = material_function(Ti, mass, Sa, *paras)
    uy = Fy / E
    if thetaD == 0:
        P = 0
    else:
        P = thetaD * E * height
    solver_paras = (Ti, th, dt, ops_paras, uy, fv_duration, scaling_factor, P, height, damping, mass)
    try:
        res: dict[str, float]
        res_th: tuple[np.ndarray, ...]
        if solver == 'auto':
            for solver_func in [newmark_solver, ops_solver]:
                res, res_th = solver_func(*solver_paras, record_res=True, **kwargs)
                if res['converge']:
                    break 
        elif solver == 'OPS':
            res, res_th = ops_solver(*solver_paras, record_res=True, **kwargs)
        elif solver == 'Newmark-Newton':
            res, res_th = newmark_solver(*solver_paras, record_res=True, **kwargs)
        else:
            raise SDOFError(f'Wrong solver name: {solver}')
    except:
        raise SDOFError('Solver error!')
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
    results.loc[row, 'R']   = mass * Sa * 9800 / Fy
    results.loc[row, 'miu'] = maxDisp / uy
    results.loc[row, 'solving_converge'] = solving_converge
    end_time = time.time()
    queue.put({'a': (GM_name, num_ana, 1, None, solving_converge, start_time, end_time, None)})
    time_, ag_scaled, disp_th, vel_th, accel_th, Ec_th, Ev_th, CD_th, CPD_th, reaction_th, eleForce_th, dampingForce_th = res_th
    res_data = np.column_stack((time_, ag_scaled, disp_th, vel_th, accel_th, Ec_th, Ev_th, CD_th, CPD_th, reaction_th, eleForce_th, dampingForce_th))
    lock.acquire()
    results.to_csv(wkdir / f'results/{GM_name}.csv', index=False)
    np.save(wkdir / f'results/{GM_name}.npy', res_data)
    lock.release()
    return None



