"""
共包含三个进行非线性SDOF体系时程分析的函数，
分别为SDOF_solver、SDOF_batched_solver和PDtSDOF_batched_solver，
其中
SDOF_solver：            普通非线性单自由度体系
SDOF_batched_solver：    可在同一模型空间下同时建立多个SDOF以进行批量分析
PDtSDOF_batched_solver： 可进一步考虑P-Delta效应（同样可批量分析）
注：批量分析目前仅支持相同地震动，
但各个SDOF的周期、材料、屈服位移（用于计算累积塑性位移）、倒塌判定位移、最大分析位移均可单独指定。
各个函数的输入、输出参数可见对应的文档注释和类型注解。
"""

import sys
from math import pi
from typing import Dict, Tuple, List
from pathlib import Path
import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.utils import SDOF_Error, SDOF_Helper


if __name__ == "__main__":
    # 一些全局变量用于储存结构响应时程
    A = []
    A_BASE = []
    V = []
    U = []
    U_BASE = []
    t = []
    F_LINK = []
    EV = []
    EC = []
    CPD_GB = []
    V_BASE = []
    F_RAY = []
    F_ELE = []

# ---------------------------------------------------------------------------
# --------------------------------- 单个SDOF求解 -----------------------------
# ---------------------------------------------------------------------------

def SDOF_solver(
        T: int,
        gm: np.ndarray,
        dt: float,
        materials: Dict[str, tuple],
        uy: float=None,
        fv_duration: float=0,
        zeta: float=0.05,
        m: float=1,
        g: float=9800,
        collapse_disp: float=1e14,
        maxAnalysis_disp: float=1e15,
    ):
    """SDOF求解函数，每次调用对一个SDOF进行非线性时程分析。
    模型结构为两个具有相同位置的结点，中间采用zeroLength单元连接。

    Args:
        T (float): 周期
        gm (np.ndarray): 地震动加速度时程（单位为g）
        dt (float): 时程步长
        materials (Dict[str, tuple]): 材料属性，包括材料名和参数（不包括编号）
        uy (float, optional): 屈服位移，仅用于计算累积塑性应变，默认None即计算值为None
        fv_duration (float, optional): 自由振动时长，默认为0
        zeta (float, optional): 阻尼比，默认0.05
        m (float, optional): 质量，默认1
        g (float, optional): 重力加速度，默认9800
        collapse_disp (float, optional): 倒塌位移判定准则，默认1e14
        maxAnalysis_disp (float, optional): 最大分析位移，默认1e15

    Returns: Tuple[int, tuple]
        元组第一项为计算状态
        * 1 - 分析成功，结构不倒塌
        * 2 - 分析成功，结构倒塌
        * 3 - 分析不收敛，结构不倒塌
        * 4 - 分析不收敛，结构倒塌\n
        元组第二项为结构响应，依次为：
        * 最大相对位移
        * 最大绝对速度
        * 最大绝对加速度
        * 累积弹塑性耗能
        * 累积Rayleigh阻尼耗能
        * 最大基底反力
        * 累积位移
        * 累积塑性位移
        * 残余位移
    """

    omega = 2 * pi / T
    NPTS = len(gm)
    duration = (NPTS - 1) * dt + fv_duration
    a = 0
    b = 2 * zeta / omega

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    ops.node(1, 0, 0)
    ops.node(2, 0, 0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 1)
    ops.mass(2, m, 0, 0)
    matTag = 0
    for matType, paras in materials.items():
        matTag += 1
        ops.uniaxialMaterial(matType, matTag, *paras)
    ops.uniaxialMaterial('Parallel', matTag + 1, *range(1, matTag + 1))
    ops.element('zeroLength', 1, 1, 2, '-mat', matTag + 1, '-dir', 1, '-doRayleigh', 1)  # 弹塑性
    ops.region(1, '-ele', 1, '-rayleigh', a, 0, b, 0)  # Rayleigh阻尼
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *gm, '-factor', g)
    ops.pattern('MultipleSupport', 1)
    ops.groundMotion(1, 'Plain', '-accel', 1)
    ops.imposedMotion(1, 1, 1)
    # ops.recorder('Node', '-file', 'd.out',  '-node', 2, '-dof', 1, 'disp')
    state, result = _TimeHistoryAnalysis(dt, duration, 2, collapse_disp, maxAnalysis_disp, uy)
    # 分析
    return state, result



def _TimeHistoryAnalysis(
        dt_init: float,
        duration: float,
        ctrl_node: float,
        collapse_disp: float,
        maxAnalysis_disp: float,
        uy: float,
        min_factor: float=1e-6, max_factor: float=1) -> Tuple[int, Dict]:
    """自适应时程分析，可根据收敛状况自动调整步长和迭代算法

    Args:
        dt_init (float): 地震动步长
        duration (float): 地震动持时
        ctrl_node (int): 控制节点编号
        collapse_disp (float): 倒塌判定位移
        maxAnalysis_disp (float): 最大分析的位移
        uy (float): 屈服位移
        min_factor (float): 自适应步长的最小调整系数
        max_factor (float): 自适应步长的最大调整系数
    
    Return: int
        * 1 - 分析成功，结构不倒塌
        * 2 - 分析成功，结构倒塌
        * 3 - 分析不收敛，结构不倒塌
        * 4 - 分析不收敛，结构倒塌
    """

    result = (0,) * 100  # 用来储存结构响应
    algorithms = [("KrylovNewton",), ("NewtonLineSearch",), ("Newton",), ("SecantNewton",)]
    algorithm_id = 0
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("EnergyIncr", 1.0e-5, 30)
    ops.algorithm("KrylovNewton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    collapse_flag = False
    maxAna_flag = False
    factor = 1
    dt = dt_init
    while True:
        ok = ops.analyze(1, dt)
        if ok == 0:
            # 当前步收敛
            result = _get_result(ctrl_node, 1, result, uy)  # 计算当前步结构响应
            collapse_flag, maxAna_flag = _SDR_tester(ctrl_node, collapse_disp, maxAnalysis_disp)  # 判断当前步是否收敛
            if (ops.getTime() >= duration or (abs(ops.getTime() - duration) < 1e-5)) and not collapse_flag:
                return 1, result[: 9]  # 分析成功，结构不倒塌
            if ops.getTime() >= duration and collapse_flag:
                return 2, result[: 9]  # 分析成功，结构倒塌
            if maxAna_flag:
                return 2, result[: 9]  # 分析成功，结构倒塌
            factor *= 2
            factor = min(factor, max_factor)
            algorithm_id -= 1
            algorithm_id = max(0, algorithm_id)
        else:
            # 当前步不收敛
            factor *= 0.5
            if factor < min_factor:
                factor = min_factor
                algorithm_id += 1
                if algorithm_id == 4 and collapse_flag:
                    return 4, result[: 9]
                if algorithm_id == 4 and not collapse_flag:
                    return 3, result[: 9]
        dt = dt_init * factor
        if dt + ops.getTime() > duration:
            dt = duration - ops.getTime()
        ops.algorithm(*algorithms[algorithm_id])


def _SDR_tester(ctrl_node: list, collapse_disp: float, maxAnalysis_disp: float
               ) -> tuple[bool, bool]:
    """
    return (tuple[bool, bool]): 是否倒塌？是否超过最大计算位移？
    """
    if collapse_disp > maxAnalysis_disp:
        raise SDOF_Error('`MaxAnalysisDrift`应大于`CollapseDrift`')
    result = (False, False)
    u = abs(ops.nodeDisp(ctrl_node, 1) - ops.nodeDisp(1, 1))
    if u >= collapse_disp:
        result = (True, False)
    if u >= maxAnalysis_disp:
        result = (True, True)
    return result


def _get_result(
        ctrl_node: int,
        eleTag: int,
        input_result: tuple[float],
        uy: float=None
    ) -> Tuple:
    """获取分析结果
    """
    # t.append(ops.getTime())
    maxDisp, maxVel, maxAccel, Ec, Ev, maxReaction, CD, CPD, u_old,\
        F_Hys_old, F_Ray_old, u_cent, *_ = input_result
    # 最大相对位移
    u = ops.nodeDisp(ctrl_node, 1) - ops.nodeDisp(1, 1)
    du = u - u_old
    maxDisp = max(maxDisp, abs(u))
    # 最大绝对速度
    v = ops.nodeVel(ctrl_node, 1)
    maxVel = max(maxVel, abs(v))
    # 最大绝对加速度
    a = ops.nodeAccel(ctrl_node, 1)
    maxAccel = max(maxAccel, abs(a))
    # 累积弹塑性耗能
    F_Hys = ops.eleResponse(eleTag, 'material', 1, 'stress')[0]
    Si = 0.5 * (F_Hys + F_Hys_old) * du
    Ec += Si
    # 累积Rayleigh耗能
    F_Ray = ops.eleResponse(eleTag, 'rayleighForces')[0]
    Si = -0.5 * (F_Ray + F_Ray_old) * du
    Ev += Si
    # 最大基底反力
    F = F_Ray + ops.eleForce(eleTag, 1)
    maxReaction = max(maxReaction, abs(F))
    # 累积变形
    CD += abs(du)
    # 累积塑性变形
    if uy is None:
        CPD = None
    else:
        if u > u_cent + uy:
            # 正向屈服
            CPD += u - (u_cent + uy)
            u_cent += u - (u_cent + uy)
        elif u < u_cent - uy:
            # 负向屈服
            CPD += u_cent - uy - u
            u_cent -= u_cent - uy - u
        else:
            CPD += 0
    return maxDisp, maxVel, maxAccel,\
        Ec, Ev, maxReaction,\
        CD, CPD, u,\
        F_Hys, F_Ray, u_cent
    # 12个参数
    

# -------------------------------------------------------------------------------
# --------------------------------- 多个SDOF批量求解 -----------------------------
# -------------------------------------------------------------------------------

def SDOF_batched_solver(
        N_SDOFs: int,
        ls_T: tuple[float, ...],
        gm: np.ndarray,
        dt: float,
        ls_materials: tuple[Dict[str, tuple], ...],
        ls_uy: tuple[float, ...]=None,
        fv_duration: float=0,
        zeta: float=0.05,
        ls_m: tuple[float, ...]=(1,)*1000000,
        g: float=9800,
        ls_collapse_disp: tuple[float, ...]=(1e14,)*1000000,
        ls_maxAnalysis_disp: tuple[float, ...]=(1e15,)*1000000,
    ) -> Tuple[int, Tuple[bool, ...], Tuple[List[float], ...]]:
    """SDOF求解函数，每次调用可对多个SDOF在同一模型空间下进行非线性时程分析。
    每个SDOF的模型结构与`SDOF_solver`函数相同，但可批量创建。

    Args:
        N_SDOFs (int): SDOF体系的数量
        ls_T (tuple[float, ...]): 周期
        gm (np.ndarray): 地震动加速度时程（单位为g）
        dt (float): 时程步长
        ls_materials (tuple[Dict[str, tuple], ...]): 材料属性，包括材料名和参数（不包括编号）
        ls_uy (tuple[float, ...], optional): 屈服位移，仅用于计算累积塑性应变，默认None即计算值为None
        fv_duration (float, optional): 自由振动时长，默认为0
        zeta (float, optional): 阻尼比，默认0.05
        ls_m (tuple[float, ...], optional): 质量，默认1
        g (float, optional): 重力加速度，默认9800
        ls_collapse_disp (tuple[float, ...], optional): 倒塌位移判定准则，默认1e14
        ls_maxAnalysis_disp (tuple[float, ...], optional): 最大分析位移，默认1e15

    Returns: Tuple[int, tuple, list[tuple]]
        元组第一项为计算状态
        * 1 - 分析不收敛，结构不倒塌
        * 2 - 分析不收敛，结构倒塌\n
        元组第二项为SDOF倒塌状态响应，倒塌为True，不倒塌为False  
        元组第三项为各个SDOF的结构响应，每个SDOF的响应类型依次为：
        * 最大相对位移
        * 最大绝对速度
        * 最大绝对加速度
        * 累积弹塑性耗能
        * 累积Rayleigh阻尼耗能
        * 最大基底反力
        * 累积位移
        * 累积塑性位移
        * 残余变形
    """
    if not (N_SDOFs == len(ls_T) == len(ls_materials)):
        raise SDOF_Error(f'SDOF数量、周期数量、材料数量不等！({N_SDOFs}, {len(ls_T)}, {len(ls_materials)})')

    NPTS = len(gm)
    duration = (NPTS - 1) * dt + fv_duration

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    nodeTag = 1  # 当前节点编号
    baseNodes = []  # 基底节点编号
    ctrlNodes = []  # 控制节点的编号
    matTag = 1  # 当前材料编号
    ctrlMats = []  # 控制材料的编号
    eleTag = 1  # 当前单元编号
    ctrlEles = []  # 控制单元的编号
    for i in range(N_SDOFs):
        # 节点、约束、质量
        ops.node(nodeTag, 0, 0)
        ops.node(nodeTag + 1, 0, 0)
        ops.fix(nodeTag, 1, 1, 1)
        ops.fix(nodeTag + 1, 0, 1, 1)
        ops.mass(nodeTag + 1, ls_m[i], 0, 0)
        inode, jnode = nodeTag, nodeTag + 1
        baseNodes.append(nodeTag)
        ctrlNodes.append(nodeTag + 1)
        nodeTag += 2
        # 材料
        material = ls_materials[i]
        matTag_start = matTag
        for matType, paras in material.items():
            ops.uniaxialMaterial(matType, matTag, *paras)
            matTag += 1
        ops.uniaxialMaterial('Parallel', matTag, *range(matTag_start, matTag))
        ctrlMats.append(matTag)
        matTag += 1
        # 单元
        ops.element('zeroLength', eleTag, inode, jnode, '-mat', matTag - 1, '-dir', 1, '-doRayleigh', 1)  # 弹塑性
        T = ls_T[i]
        omega = 2 * pi / T
        b = 2 * zeta / omega
        ops.region(i + 1, '-ele', eleTag, '-rayleigh', 0, 0, b, 0)  # Rayleigh阻尼
        ctrlEles.append(eleTag)
        eleTag += 1

    ops.timeSeries('Path', 1, '-dt', dt, '-values', *gm, '-factor', g)
    ops.pattern('MultipleSupport', 1)
    ops.groundMotion(1, 'Plain', '-accel', 1)
    for tag in baseNodes:
        ops.imposedMotion(tag, 1, 1)
    state, collapse, result = _batchedTimeHistoryAnalysis(N_SDOFs, dt, duration, baseNodes, ctrlNodes, ctrlEles,
                                ls_collapse_disp, ls_maxAnalysis_disp, ls_uy)
    return state, collapse, result
    

def _batchedTimeHistoryAnalysis(
        N_SDOFs: int,
        dt_init: float,
        duration: float,
        baseNodes: tuple,
        ctrlNodes: list,
        ctrlEles: list,
        collapse_disp: tuple,
        maxAnalysis_disp: tuple,
        ls_uy: tuple,
        min_factor: float=1e-6, max_factor: float=1
        ) -> Tuple[int, Tuple[bool, ...], Tuple[List[float], ...]]:
    """自适应时程分析，可根据收敛状况自动调整步长和迭代算法

    Args:
        N_SDOFs (int): SDOF的数量
        dt_init (float): 地震动步长
        duration (float): 地震动持时
        base_node (tuple): 基底节点编号
        ctrl_node (tuple): 控制节点编号
        collapse_disp (tuple): 倒塌判定位移
        maxAnalysis_disp (tuple): 最大分析的位移
        uy (tuple): 屈服位移
        min_factor (float): 自适应步长的最小调整系数
        max_factor (float): 自适应步长的最大调整系数
    
    Return: Tuple[int, tuple, list[tuple]]
        元组第一项为计算状态
        * 1 - 分析收敛
        * 2 - 分析不收敛\n
        元组第二项为SDOF倒塌状态响应，倒塌为True，不倒塌为False\n
        元组第三项为各个SDOF的结构响应
    """
    result = tuple([0.0] * N_SDOFs for _ in range(100))  # 用来储存结构响应
    algorithms = [("KrylovNewton",), ("NewtonLineSearch",), ("Newton",), ("SecantNewton",)]
    algorithm_id = 0
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("EnergyIncr", 1.0e-5, 30)
    ops.algorithm("KrylovNewton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    collapse_flag = (False,) * N_SDOFs
    maxAna_flag = False
    factor = 1
    dt = dt_init
    while True:
        ok = ops.analyze(1, dt)
        if ok == 0:
            # 当前步收敛
            result = _get_batched_results(N_SDOFs, baseNodes, ctrlNodes, ctrlEles, result, ls_uy)  # 计算当前步结构响应
            collapse_flag, maxAna_flag = _SDR_batched_tester(N_SDOFs, baseNodes, ctrlNodes, collapse_disp, maxAnalysis_disp)  # 判断当前步是否收敛
            if (ops.getTime() >= duration) or maxAna_flag or (abs(ops.getTime() - duration) < 1e-5):
                return 1, collapse_flag, result[: 9]
            factor *= 2
            factor = min(factor, max_factor)
            algorithm_id -= 1
            algorithm_id = max(0, algorithm_id)
        else:
            # 当前步不收敛
            factor *= 0.5
            if factor < min_factor:
                factor = min_factor
                algorithm_id += 1
                if algorithm_id == 4:
                    return 2, collapse_flag, result[: 9]
        dt = dt_init * factor
        if dt + ops.getTime() > duration:
            dt = duration - ops.getTime()
        ops.algorithm(*algorithms[algorithm_id])


def _SDR_batched_tester(
        N_SDOFs: int,
        base_nodes: List[int],
        ctrl_nodes: list,
        collapse_disp: Tuple[bool, ...],
        maxAnalysis_disp: tuple
        ) -> Tuple[tuple[bool], bool]:
    """
    return Tuple[tuple[bool], bool]: 各个SDOF是否倒塌？是否全部超过最大计算位移？
    """
    results_collapse = []
    results_maxAna = []
    for i in range(N_SDOFs):
        if collapse_disp[i] > maxAnalysis_disp[i]:
            raise SDOF_Error('`MaxAnalysisDrift`应大于`CollapseDrift`')
        u = abs(ops.nodeDisp(ctrl_nodes[i], 1) - ops.nodeDisp(base_nodes[i], 1))
        if u >= collapse_disp[i]:
            results_collapse.append(True)
        else:
            results_collapse.append(False)  
        if u >= maxAnalysis_disp[i]:
            results_maxAna.append(True)
        else:
            results_maxAna.append(False)
    return tuple(results_collapse), all(results_maxAna)


def _get_batched_results(
        N_SDOFs: int,
        baseNodes: list[int],
        ctrlNodes: list[int],
        ctrlEles: list[int],
        input_result: Tuple[List[float], ...],
        ls_uy: Tuple[float] | None=None
    ) -> Tuple[list[float], ...]:
    """获取分析结果
    """
    maxDisp, maxVel, maxAccel, Ec, Ev, maxReaction, CD, CPD, ls_u_old,\
        ls_F_Hys_old, ls_F_Ray_old, ls_u_cent, *_ = input_result
    ls_u = []
    ls_F_Hys = []
    ls_F_Ray = []
    for i in range(N_SDOFs):
        ctrl_node = ctrlNodes[i]
        base_node = baseNodes[i]
        eleTag = ctrlEles[i]
        u_old = ls_u_old[i]
        F_Hys_old = ls_F_Hys_old[i]
        F_Ray_old = ls_F_Ray_old[i]
        u_cent = ls_u_cent[i]
        # 最大相对位移
        u: float = ops.nodeDisp(ctrl_node, 1) - ops.nodeDisp(base_node, 1)
        ls_u.append(u)
        du = u - u_old
        maxDisp[i] = max(maxDisp[i], abs(u))
        # 最大绝对速度
        v: float = ops.nodeVel(ctrl_node, 1)
        maxVel[i] = max(maxVel[i], abs(v))
        # 最大绝对加速度
        a: float = ops.nodeAccel(ctrl_node, 1)
        maxAccel[i] = max(maxAccel[i], abs(a))
        # 累积弹塑性耗能
        F_Hys: float = ops.eleResponse(eleTag, 'material', 1, 'stress')[0]
        ls_F_Hys.append(F_Hys)
        Si = 0.5 * (F_Hys + F_Hys_old) * du
        Ec[i] += Si
        # 累积Rayleigh耗能
        F_Ray: float = ops.eleResponse(eleTag, 'rayleighForces')[0]
        ls_F_Ray.append(F_Ray)
        Si = -0.5 * (F_Ray + F_Ray_old) * du
        Ev[i] += Si
        # 最大基底反力
        F: float = F_Ray + ops.eleForce(eleTag, 1)
        maxReaction[i] = max(maxReaction[i], abs(F))
        # 累积变形
        CD[i] += abs(du)
        # 累积塑性变形
        if ls_uy is None:
            CPD = None
        else:
            uy = ls_uy[i]
            if u > u_cent + uy:
                # 正向屈服
                CPD[i] += u - (u_cent + uy)
                u_cent += u - (u_cent + uy)
            elif u < u_cent - uy:
                # 负向屈服
                CPD[i] += u_cent - uy - u
                u_cent -= u_cent - uy - u
            else:
                CPD[i] += 0
            ls_u_cent[i] = u_cent
    return maxDisp, maxVel, maxAccel,\
        Ec, Ev, maxReaction,\
        CD, CPD, ls_u,\
        ls_F_Hys, ls_F_Ray, ls_u_cent
    # 12个参数


# -----------------------------------------------------------------------------------
# --------------------------------- SDOF批量求解 -------------------------------------
# ------------------------------- （可考虑P-Delta） ----------------------------------
# -----------------------------------------------------------------------------------

def PDtSDOF_batched_solver(
        N_SDOFs: int,
        h: float,
        ls_T: tuple[float, ...],
        ls_grav: tuple[float, ...],
        gm: np.ndarray,
        dt: float,
        ls_materials: tuple[Dict[str, tuple], ...],
        ls_uy: tuple[float, ...]=None,
        fv_duration: float=0,
        zeta: float=0.05,
        ls_m: tuple[float, ...]=(1,)*1000000,
        g: float=9800,
        ls_collapse_disp: tuple[float, ...]=(1e14,)*1000000,
        ls_maxAnalysis_disp: tuple[float, ...]=(1e15,)*1000000,
    ) -> Tuple[int, Tuple[bool, ...], Tuple[List[float], ...]]:
    """SDOF求解函数，每次调用可对多个SDOF在同一模型空间下进行非线性时程分析，
    可考虑P-Delta效应

    Args:
        N_SDOFs (int): SDOF体系的数量
        h (float): 结构等效高度
        ls_T (tuple[float, ...]): 周期
        ls_grav (tuple[float, ...]): 竖向荷载(需为正值)
        gm (np.ndarray): 地震动加速度时程（单位为g）
        dt (float): 时程步长
        ls_materials (tuple[Dict[str, tuple], ...]): 材料属性(弯矩-转角关系)，包括材料名和参数（不包括编号）
        uy (tuple[float, ...], optional): 屈服转角，仅用于计算累积塑性应变，默认None即计算值为None
        fv_duration (float, optional): 自由振动时长，默认为0
        zeta (float, optional): 阻尼比，默认0.05
        ls_m (tuple[float, ...], optional): 质量，默认1
        g (float, optional): 重力加速度，默认9800
        ls_collapse_disp (tuple[float, ...], optional): 倒塌转角判定准则，默认1e14
        ls_maxAnalysis_disp (tuple[float, ...], optional): 最大分析转角，默认1e15

    Returns: Tuple[int, tuple, list[tuple]]
        元组第一项为计算状态
        * 1 - 分析不收敛，结构不倒塌
        * 2 - 分析不收敛，结构倒塌\n
        元组第二项为SDOF倒塌状态响应，倒塌为True，不倒塌为False\n
        元组第三项为各个SDOF的结构响应，每个SDOF的响应类型依次为：
        * 最大相对位移
        * 最大绝对速度
        * 最大绝对加速度
        * 累积弹塑性耗能
        * 累积Rayleigh阻尼耗能
        * 最大基底反力
        * 累积位移
        * 累积塑性位移
        * 残余变形
    """
    """
    模型结构示意图：

        gravity
           ↓   非线性弹簧(zeroLength)
     jnode o-----/\/\/\/\/\/\----o knode (accel input)
           |
           | ← 刚性梁(ElasticBeamColumn)
           |
           |      边界条件:
           |      inode: U1=U2=1, U3=0
           |      jnode: U1=U2=U3=0
           |      knode: U1=U2=U3=1
           |
           |
     inode o (accel input)
    """
    if not (N_SDOFs == len(ls_T) == len(ls_materials)):
        raise SDOF_Error(f'SDOF数量、周期数量、材料数量不等！({N_SDOFs}, {len(ls_T)}, {len(ls_materials)})')

    NPTS = len(gm)
    duration = (NPTS - 1) * dt + fv_duration
    A_rigid = 1e10
    I_rigid = 1e10

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    nodeTag = 1  # 当前节点编号
    baseNodes = []  # 基底节点编号
    midNodes = []  # 中间节点（与刚性杆下部连接的节点）编号
    ctrlNodes = []  # 控制节点的编号
    matTag = 1  # 当前材料编号
    ctrlMats = []  # 控制材料的编号
    eleTag = 1  # 当前单元编号
    ctrlEles = []  # 控制单元的编号
    ops.geomTransf('PDelta', 1)
    for i in range(N_SDOFs):
        # 节点、约束、质量
        inode, jnode, knode = nodeTag, nodeTag + 1, nodeTag + 2
        ops.node(inode, 0, 0)
        ops.node(jnode, 0, h)
        ops.node(knode, 0, h)
        ops.fix(inode, 1, 1, 0)
        # ops.fix(jnode, 0, 1, 1)
        ops.fix(knode, 1, 1, 1)
        # ops.equalDOF(inode, jnode, 1, 2)
        ops.mass(jnode, ls_m[i], 0, 0)
        baseNodes.append(inode)
        ctrlNodes.append(jnode)
        midNodes.append(knode)
        nodeTag += 3
        # 材料
        material = ls_materials[i]
        matTag_start = matTag
        for matType, paras in material.items():
            ops.uniaxialMaterial(matType, matTag, *paras)
            matTag += 1
        ops.uniaxialMaterial('Parallel', matTag, *range(matTag_start, matTag))
        ctrlMats.append(matTag)
        matTag += 1
        # 单元
        ops.element('zeroLength', eleTag, jnode, knode, '-mat', matTag - 1, '-dir', 1, '-doRayleigh', 1)  # 零长度弹塑性弹簧
        ops.element('elasticBeamColumn', eleTag + 1, inode, jnode, A_rigid, 206000, I_rigid, 1)  # 刚性梁
        T = ls_T[i]
        omega = 2 * pi / T
        b = 2 * zeta / omega
        ops.region(i + 1, '-ele', eleTag, '-rayleigh', 0, 0, b, 0)  # Rayleigh阻尼
        ctrlEles.append(eleTag)
        eleTag += 2

    # 竖向荷载
    ops.timeSeries('Linear', 11)
    ops.pattern('Plain', 11, 11)
    for i in range(N_SDOFs):
        F = ls_grav[i]
        if F == 0:
            continue
        ops.load(ctrlNodes[i], 0, -F, 0)

    # 分析重力
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', 1.0e-5, 60)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 0.1)
    ops.analysis('Static')
    ops.analyze(10)
    ops.loadConst('-time', 0.0)

    # 时程分析
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *gm, '-factor', g)
    ops.pattern('MultipleSupport', 1)
    ops.groundMotion(1, 'Plain', '-accel', 1)
    for tag1, tag2 in zip(baseNodes, midNodes):
        ops.imposedMotion(tag1, 1, 1)
        ops.imposedMotion(tag2, 1, 1)
    state, collapse, result = _PDtBatchedTimeHistoryAnalysis(N_SDOFs, dt, duration, baseNodes, midNodes, ctrlNodes, ctrlEles,
                                ls_collapse_disp, ls_maxAnalysis_disp, ls_uy)
    return state, collapse, result
    

def _PDtBatchedTimeHistoryAnalysis(
        N_SDOFs: int,
        dt_init: float,
        duration: float,
        baseNodes: list[int],
        midNodes: list[int],
        ctrlNodes: list[int],
        ctrlEles: list[int],
        ls_collapse_disp: tuple,
        ls_maxAnalysis_disp: tuple,
        ls_uy: tuple[float, ...],
        min_factor: float=1e-6, max_factor: float=1
        ) -> Tuple[int, Tuple[bool, ...], Tuple[List[float], ...]]:
    """自适应时程分析，可根据收敛状况自动调整步长和迭代算法

    Args:
        N_SDOFs (int): SDOF的数量
        dt_init (float): 地震动步长
        duration (float): 地震动持时
        base_node (tuple): 基底节点编号
        ctrl_node (tuple): 控制节点编号
        ls_collapse_disp (tuple[float, ...]): 倒塌判定转角
        ls_maxAnalysis_disp (tuple[float, ...]): 最大分析的转角
        ls_uy (tuple): 屈服转角
        min_factor (float): 自适应步长的最小调整系数
        max_factor (float): 自适应步长的最大调整系数
    
    Return: Tuple[int, tuple, list[tuple]]
        元组第一项为计算状态
        * 1 - 分析收敛
        * 2 - 分析不收敛\n
        元组第二项为SDOF倒塌状态响应，倒塌为True，不倒塌为False\n
        元组第三项为各个SDOF的结构响应
    """
    result = tuple([0.0] * N_SDOFs for _ in range(100))  # 用来储存结构响应
    algorithms = [("KrylovNewton",), ("NewtonLineSearch",), ("Newton",), ("SecantNewton",)]
    algorithm_id = 0
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("EnergyIncr", 1.0e-5, 30)
    ops.algorithm("KrylovNewton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    collapse_flag = (False,) * N_SDOFs
    maxAna_flag = False
    factor = 1
    dt = dt_init
    while True:
        ok = ops.analyze(1, dt)
        if ok == 0:
            # 当前步收敛
            result = _PDt_get_batched_results(N_SDOFs, baseNodes, midNodes, ctrlNodes, ctrlEles, result, ls_uy)  # 计算当前步结构响应
            collapse_flag, maxAna_flag = _PDt_SDR_batched_tester(N_SDOFs, ctrlNodes, midNodes, ls_collapse_disp, ls_maxAnalysis_disp)  # 判断当前步是否收敛
            if (ops.getTime() >= duration) or maxAna_flag or (abs(ops.getTime() - duration) < 1e-5):
                return 1, collapse_flag, result[: 9]
            factor *= 2
            factor = min(factor, max_factor)
            algorithm_id -= 1
            algorithm_id = max(0, algorithm_id)
        else:
            # 当前步不收敛
            factor *= 0.5
            if factor < min_factor:
                factor = min_factor
                algorithm_id += 1
                if algorithm_id == 4:
                    return 2, collapse_flag, result[: 9]
        dt = dt_init * factor
        if dt + ops.getTime() > duration:
            dt = duration - ops.getTime()
        ops.algorithm(*algorithms[algorithm_id])


def _PDt_SDR_batched_tester(
        N_SDOFs: int,
        ctrlNodes: List[int],
        midNodes: List[int],
        ls_collapse_disp: Tuple[bool, ...],
        ls_maxAnalysis_disp: Tuple[bool, ...],
        ) -> Tuple[tuple[bool], bool]:
    """
    return Tuple[tuple[bool], bool]: 各个SDOF是否倒塌？是否全部超过最大计算位移？
    """
    results_collapse = []
    results_maxAna = []
    for i in range(N_SDOFs):
        if ls_collapse_disp[i] > ls_maxAnalysis_disp[i]:
            raise SDOF_Error('`ls_collapse_disp`应大于`ls_maxAnalysis_disp`')
        u = ops.nodeDisp(ctrlNodes[i], 1) - ops.nodeDisp(midNodes[i], 1)
        if abs(u) >= ls_collapse_disp[i]:
            results_collapse.append(True)
        else:
            results_collapse.append(False)  
        if abs(u) >= ls_maxAnalysis_disp[i]:
            results_maxAna.append(True)
        else:
            results_maxAna.append(False)
    return tuple(results_collapse), all(results_maxAna)


def _PDt_get_batched_results(
        N_SDOFs: int,
        baseNodes: list[int],
        midNodes: list[int],
        ctrlNodes: list[int],
        ctrlEles: list[int],
        input_result: Tuple[List[float], ...],
        ls_uy: Tuple[float] | None=None
    ) -> Tuple[list[float], ...]:
    """获取分析结果
    """
    # t.append(ops.getTime())
    maxDisp, maxVel, maxAccel, Ec, Ev, maxReaction, CD, CPD, ls_u_old,\
        ls_F_Hys_old, ls_F_Ray_old, ls_u_cent, *_ = input_result
    ls_u = []
    ls_F_Hys = []
    ls_F_Ray = []
    for i in range(N_SDOFs):
        ctrl_node = ctrlNodes[i]
        base_node = baseNodes[i]
        mid_node = midNodes[i]
        eleTag = ctrlEles[i]
        u_old = ls_u_old[i]
        F_Hys_old = ls_F_Hys_old[i]
        F_Ray_old = ls_F_Ray_old[i]
        u_cent = ls_u_cent[i]
        # 最大相对位移
        u: float = ops.nodeDisp(ctrl_node, 1) - ops.nodeDisp(mid_node, 1)
        ls_u.append(u)
        du = u - u_old
        maxDisp[i] = max(maxDisp[i], abs(u))
        # if i == 0:
        #     U.append(u)
        # 最大绝对速度
        v: float = ops.nodeVel(ctrl_node, 1)
        maxVel[i] = max(maxVel[i], abs(v))
        # if i == 0:
        #     V.append(v)
        # 最大绝对加速度
        a: float = ops.nodeAccel(ctrl_node, 1)
        maxAccel[i] = max(maxAccel[i], abs(a))
        if i == 0:
            A_BASE.append(ops.nodeAccel(base_node, 1))
            A.append(a)
        # 累积弹塑性耗能
        F_Hys: float = ops.eleResponse(eleTag, 'material', 1, 'stress')[0]
        ls_F_Hys.append(F_Hys)
        Si = 0.5 * (F_Hys + F_Hys_old) * du
        Ec[i] -= Si
        # if i == 0:
        #     EC.append(Ec[i])
        # 累积Rayleigh耗能
        F_Ray: float = ops.eleResponse(eleTag, 'rayleighForces')[0]
        ls_F_Ray.append(F_Ray)
        Si = -0.5 * (F_Ray + F_Ray_old) * du
        Ev[i] -= Si
        # if i == 0:
        #     EV.append(Ev[i])
        # 最大基底反力
        F: float = -(ops.eleResponse(eleTag, 'rayleighForces')[0] + ops.eleForce(eleTag, 1))
        maxReaction[i] = max(maxReaction[i], abs(F))
        # if i == 0:
        #     V_BASE.append(F)
        # 累积变形
        CD[i] += abs(du)
        # 累积塑性变形
        if ls_uy is None:
            CPD = None
        else:
            uy = ls_uy[i]
            if u > u_cent + uy:
                # 正向屈服
                CPD[i] += u - (u_cent + uy)
                u_cent += u - (u_cent + uy)
            elif u < u_cent - uy:
                # 负向屈服
                CPD[i] += u_cent - uy - u
                u_cent -= u_cent - uy - u
            else:
                CPD[i] += 0
            ls_u_cent[i] = u_cent
    return maxDisp, maxVel, maxAccel,\
        Ec, Ev, maxReaction,\
        CD, CPD, ls_u,\
        ls_F_Hys, ls_F_Ray, ls_u_cent
    # 12个参数



if __name__ == "__main__":
    ls_T = tuple(0.62831853 for _ in range(100))
    T = 0.62831853
    h = 1000
    ls_grav = (0,) * 100
    gm = np.loadtxt(r'F:\NRSA\Input\GMs\th1.th')
    dt = 0.01
    materials = tuple({'Steel01': (200, 100, 0.02)} for _ in range(100))
    PDtMaterials = tuple({'Steel01': (200, 100, 0.02)} for _ in range(100))
    material = {'Steel01': (200, 100, 0.02)}
    with SDOF_Helper(suppress=False):
        # state, collapse, result = SDOF_batched_solver(100, ls_T, gm, dt, materials, [2]*100)

        # for i in range(100):
        #     state, result = SDOF_solver(T, gm, dt, material, uy=2, fv_duration=0)

        state, collapse, result = PDtSDOF_batched_solver(100, h, ls_T, ls_grav, gm, dt, PDtMaterials, ls_uy=[2]*100, fv_duration=0)
    # print(state)
    # print(result[8][0])
    plt.plot(t, A)
    plt.show()
    np.savetxt(r'F:\NRSA\temp\t.txt', t)
    np.savetxt(r'F:\NRSA\temp\A.txt', A)


