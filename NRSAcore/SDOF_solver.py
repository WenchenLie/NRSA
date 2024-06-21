"""
----------------- 单自由度时程分析求解器 ----------------------
共包含三个进行非线性SDOF体系时程分析的函数，
分别为SDOF_solver、SDOF_batched_solver和PDtSDOF_batched_solver，
其中
SDOF_solver：            普通非线性单自由度体系
SDOF_batched_solver：    可在同一模型空间下同时建立多个SDOF以进行批量分析
PDtSDOF_batched_solver： 可进一步考虑P-Delta效应（同样可批量分析）
注：
批量分析目前仅支持相同地震动，
但各个SDOF的周期、材料、屈服位移（用于计算累积塑性位移）、倒塌判定位移、最大分析位移均可单独指定。
各个函数的输入、输出参数可见对应的文档注释和类型注解。
各个函数计算后返回一个dict，可用的键包括：
* 是否收敛，'converge': bool
* 是否倒塌，'collapse': bool | tuple[bool, ...]
* 最大相对位移：'maxDisp': float | list[float]]
* 最大相对速度：'maxVel': float | list[float]]
* 最大绝对加速度：'maxAccel': float | list[float]]
* 累积弹塑性耗能：'Ec': float | list[float]]
* 累积Rayleigh阻尼耗能：'Ev': float | list[float]]
* 最大基底反力：'maxReaction': float | list[float]]
* 累积位移：'CD': float | list[float]]
* 累积塑性位移：'CPD': float | list[float]]
* 残余位移：'resDisp': float | list[float]]
建模方法：
所有SDOF采用零长度单元建模，采用一致激励对SDOF进行非线性时程分析，节点提取的直接地  
震响应为相对响应，为了计算结构绝对响应，额外建立一个大质量零刚度的特殊SDOF，用于提  
取结构的基底位移、速度和加速度，在此基础上计算其余SDOF的绝对响应。
"""

import sys
import re
from math import pi
from typing import Dict, Tuple, List
from pathlib import Path
import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.utils import SDOF_Error, SDOF_Helper, a2u


__all__ = ['SDOF_solver', 'SDOF_batched_solver', 'PDtSDOF_batched_solver']


if __name__ == "__main__":
    # 一些全局变量用于储存结构响应时程
    TIME = []  # 时间序列
    A_R = []  # 相对加速度
    A_A = []  # 绝对加速度
    V_R = []  # 相对速度
    V_A = []  # 绝对速度
    U_R = []  # 相对位移
    U_A = []  # 绝对位移
    A_BASE = []  # 底部加速度
    V_BASE = []  # 底部速度
    U_BASE = []  # 底部位移
    REACTION = []  # 底部剪力
    REACTION_HYS = []  # 底部剪力(仅材料内力)
    REACTION_RAY = []  # 底部剪力(仅阻尼力)
    E_HYS = []  # 材料累积耗能
    E_RAY = []  # 阻尼累积耗能
    CD_ = []  # 累积变形
    CPD_ = []  # 累积塑性变形
    TEMP = []  # 其他响应


def SDOF_solver(
        T: int | float,
        gm: np.ndarray,
        dt: float,
        materials: Dict[str, tuple],
        uy: float=None,
        fv_duration: float=0,
        SF: float=None,
        zeta: float=0.05,
        m: float=1,
        g: float=9800,
        collapse_disp: float=1e14,
        maxAnalysis_disp: float=1e15,
    ) -> dict[str, bool | float]:
    """SDOF求解函数，每次调用对一个SDOF进行非线性时程分析。
    模型结构为两个具有相同位置的结点，中间采用zeroLength单元连接。

    Args:
        T (int | float): 周期
        gm (np.ndarray): 地震动加速度时程（单位为g）
        dt (float): 时程步长
        materials (Dict[str, tuple]): 材料属性，包括材料名和参数（不包括编号）
        uy (float, optional): 屈服位移，仅用于计算累积塑性应变，默认None即计算值为None
        fv_duration (float, optional): 自由振动时长，默认为0
        SF (float, optional): 地震动放大系数，默认不额外进行缩放
        zeta (float, optional): 阻尼比，默认0.05
        m (float, optional): 质量，默认1
        g (float, optional): 重力加速度，默认9800
        collapse_disp (float, optional): 倒塌位移判定准则，默认1e14
        maxAnalysis_disp (float, optional): 最大分析位移，默认1e15

    Returns: dict[str, bool | float]
        键值对依次包括：
        * 是否收敛，'converge': bool
        * 是否倒塌，'collapse': bool
        * 最大相对位移：'maxDisp': float
        * 最大相对速度：'maxVel': float
        * 最大绝对加速度：'maxAccel': float
        * 累积弹塑性耗能：'Ec': float
        * 累积Rayleigh阻尼耗能：'Ev': float
        * 最大基底反力：'maxReaction': float
        * 累积位移：'CD': float
        * 累积塑性位移：'CPD': float
        * 残余位移：'resDisp': float
    """
    model = _SDOF_solver(T, gm, dt, materials, uy, fv_duration, SF, zeta, m, g, collapse_disp, maxAnalysis_disp)
    results = model.get_results()
    return results


def SDOF_batched_solver(
        N_SDOFs: int,
        ls_T: tuple[float, ...],
        gm: np.ndarray,
        dt: float,
        ls_materials: tuple[Dict[str, tuple], ...],
        ls_uy: tuple[float, ...]=None,
        fv_duration: float=0,
        SF: tuple=(1,)*1000000,
        zeta: tuple[float, ...]=(0.05,)*1000000,
        ls_m: tuple[float, ...]=(1,)*1000000,
        g: float=9800,
        ls_collapse_disp: tuple[float, ...]=(1e14,)*1000000,
        ls_maxAnalysis_disp: tuple[float, ...]=(1e15,)*1000000,
    ) -> dict[str, tuple[bool, ...] | list[float]]:
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
        SF (tuple, optional): 地震动放大系数，默认不额外进行缩放
        zeta (float, optional): 阻尼比，默认0.05
        ls_m (tuple[float, ...], optional): 质量，默认1
        g (float, optional): 重力加速度，默认9800
        ls_collapse_disp (tuple[float, ...], optional): 倒塌位移判定准则，默认1e14
        ls_maxAnalysis_disp (tuple[float, ...], optional): 最大分析位移，默认1e15

    Returns: dict[str, bool | tuple[bool, ...] | list[float]]
        键值对依次包括：
        * 是否收敛，'converge': bool
        * 是否倒塌，'collapse': tuple[bool, ...]
        * 最大相对位移：'maxDisp': list[float]]
        * 最大相对速度：'maxVel': list[float]]
        * 最大绝对加速度：'maxAccel': list[float]]
        * 累积弹塑性耗能：'Ec': list[float]]
        * 累积Rayleigh阻尼耗能：'Ev': list[float]]
        * 最大基底反力：'maxReaction': list[float]]
        * 累积位移：'CD': list[float]]
        * 累积塑性位移：'CPD': list[float]]
        * 残余位移：'resDisp': list[float]]
    """
    model = _SDOF_batched_solver(N_SDOFs, ls_T, gm, dt, ls_materials, ls_uy, fv_duration, SF, zeta, ls_m, g, ls_collapse_disp, ls_maxAnalysis_disp)
    results = model.get_results()
    return results


def PDtSDOF_batched_solver(
        N_SDOFs: int,
        ls_h: tuple[float, ...],
        ls_T: tuple[float, ...],
        ls_grav: tuple[float, ...],
        gm: np.ndarray,
        dt: float,
        ls_materials: tuple[Dict[str, tuple], ...],
        ls_uy: tuple[float, ...]=None,
        fv_duration: float=0,
        ls_SF: tuple=(1,)*1000000,
        ls_zeta: tuple[float, ...]=(0.05,)*1000000,
        ls_m: tuple[float, ...]=(1,)*1000000,
        g: float=9800,
        ls_collapse_disp: tuple[float, ...]=(1e14,)*1000000,
        ls_maxAnalysis_disp: tuple[float, ...]=(1e15,)*1000000,
    ) -> dict[str, tuple[bool, ...] | list[float]]:
    """SDOF求解函数，每次调用可对多个SDOF在同一模型空间下进行非线性时程分析，
    可考虑P-Delta效应

    Args:
        N_SDOFs (int): SDOF体系的数量
        ls_h (tuple[float, ...]): 结构等效高度
        ls_T (tuple[float, ...]): 周期
        ls_grav (tuple[float, ...]): 竖向荷载(需为正值)
        gm (np.ndarray): 地震动加速度时程（单位为g）
        dt (float): 时程步长
        ls_materials (tuple[Dict[str, tuple], ...]): 材料属性(弯矩-转角关系)，包括材料名和参数（不包括编号）
        uy (tuple[float, ...], optional): 屈服转角，仅用于计算累积塑性应变，默认None即计算值为None
        fv_duration (float, optional): 自由振动时长，默认为0
        ls_SF (tuple, optional): 地震动放大系数，默认不额外进行缩放
        ls_zeta (float, optional): 阻尼比，默认0.05
        ls_m (tuple[float, ...], optional): 质量，默认1
        g (float, optional): 重力加速度，默认9800
        ls_collapse_disp (tuple[float, ...], optional): 倒塌转角判定准则，默认1e14
        ls_maxAnalysis_disp (tuple[float, ...], optional): 最大分析转角，默认1e15

    Returns: dict[str, bool | tuple[bool, ...] | list[float]]
        键值对依次包括：
        * 是否收敛，'converge': bool
        * 是否倒塌，'collapse': tuple[bool, ...]
        * 最大相对位移：'maxDisp': list[float]]
        * 最大相对速度：'maxVel': list[float]]
        * 最大绝对加速度：'maxAccel': list[float]]
        * 累积弹塑性耗能：'Ec': list[float]]
        * 累积Rayleigh阻尼耗能：'Ev': list[float]]
        * 最大基底反力：'maxReaction': list[float]]
        * 累积位移：'CD': list[float]]
        * 累积塑性位移：'CPD': list[float]]
        * 残余位移：'resDisp': list[float]]
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
    model = _PDtSDOF_batched_solver(N_SDOFs, ls_h, ls_T, ls_grav, gm, dt, ls_materials, ls_uy, fv_duration, ls_SF, ls_zeta, ls_m, g, ls_collapse_disp, ls_maxAnalysis_disp)
    results = model.get_results()
    return results


# ---------------------------------------------------------------------------
# --------------------------------- 单个SDOF求解 -----------------------------
# ---------------------------------------------------------------------------

class _SDOF_solver:
    def __init__(self,
            T: int | float,
            gm: np.ndarray,
            dt: float,
            materials: Dict[str, tuple],
            uy: float=None,
            fv_duration: float=0,
            SF: float=1,
            zeta: float=0.05,
            m: float=1,
            g: float=9800,
            collapse_disp: float=1e14,
            maxAnalysis_disp: float=1e15,):
        self.T = T
        self.gm = gm
        self.dt = dt
        self.materials = materials
        self.uy = uy
        self.fv_duration = fv_duration
        self.SF = SF
        self.zeta = zeta
        self.m = m
        self.g = g
        self.collapse_disp = collapse_disp
        self.maxAnalysis_disp = maxAnalysis_disp
        self.NPTS = len(gm)
        self.duration = (self.NPTS - 1) * dt + fv_duration
        omega = 2 * pi / T
        self.NPTS = len(gm)
        self.duration = (self.NPTS - 1) * dt + fv_duration
        self.a = 0
        self.b = 2 * zeta / omega
        # self.a = 2 * zeta * omega
        # self.b = 0
        print(f'a = {self.a}, b = {self.b}')
        self.run_model()


    def run_model(self):
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        ops.node(1, 0, 0)
        ops.node(2, 0, 0)
        ops.node(1000, 0, 0)
        ops.node(2000, 0, 0)   # 用于提取绝对加速度
        ops.fix(1, 1, 1, 1)
        ops.fix(2, 0, 1, 1)
        ops.fix(1000, 1, 1, 1)
        ops.fix(2000, 0, 1, 1)
        ops.mass(2, self.m, 0, 0)
        ops.mass(2000, 1e10, 0, 0)
        matTag = 1
        for matType, paras in self.materials.items():
            ops.uniaxialMaterial(matType, matTag, *_update_para(matTag, *paras))
            matTag += 1
        ops.uniaxialMaterial('Parallel', matTag, *range(1, matTag))
        matTag += 1
        ops.uniaxialMaterial('Elastic', matTag, 0)
        ops.element('zeroLength', 1, 1, 2, '-mat', matTag - 1, '-dir', 1, '-doRayleigh', 1)  # 弹塑性
        ops.element('zeroLength', 2, 1000, 2000, '-mat', matTag, '-dir', 1, '-doRayleigh', 0)  # 弹塑性
        ops.region(1, '-ele', 1, '-rayleigh', self.a, 0, self.b, 0)  # Rayleigh阻尼
        ops.timeSeries('Path', 1, '-dt', self.dt, '-values', *self.gm, '-factor', self.g)
        ops.pattern('UniformExcitation', 1, 1, '-accel', 1, '-fact', self.SF)

        # 分析
        converge, collapse, response = self.time_history_analysis()
        results = dict()
        results['converge'] = converge
        results['collapse'] = collapse
        results['maxDisp'] = response[0]
        results['maxVel'] = response[1]
        results['maxAccel'] = response[2]
        results['Ec'] = response[3]
        results['Ev'] = response[4]
        results['maxReaction'] = response[5]
        results['CD'] = response[6]
        results['CPD'] = response[7]
        results['resDisp'] = response[8]
        self.results = results


    def time_history_analysis(self, min_factor: float=1e-6, max_factor: float=1) -> Tuple[bool, bool, tuple]:
        """自适应时程分析，可根据收敛状况自动调整步长和迭代算法

        Args:
            min_factor (float): 自适应步长的最小调整系数
            max_factor (float): 自适应步长的最大调整系数
        
        Return: Tuple[bool, bool, tuple]
            * (1) - 是否收敛
            * (2) - 是否倒塌
            * (3) - 结构响应结果
        """
        result = (0,) * 100  # 用来储存结构响应
        algorithms = [("KrylovNewton",), ("NewtonLineSearch",), ("Newton",), ("SecantNewton",)]
        algorithm_id = 0
        ops.wipeAnalysis()
        ops.constraints("Plain")
        ops.numberer("Plain")
        ops.system("BandGeneral")
        ops.test("EnergyIncr", 1.0e-5, 30)
        ops.algorithm("KrylovNewton")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")
        
        collapse_flag = False
        maxAna_flag = False
        factor = 1
        dt_init = self.dt
        dt = dt_init
        while True:
            ok = ops.analyze(1, dt)
            if ok == 0:
                # 当前步收敛
                result = self.get_responses(result)  # 计算当前步结构响应
                current_collapse_flag, maxAna_flag = self.SDR_tester()  # 判断当前步是否收敛
                collapse_flag = collapse_flag or current_collapse_flag
                if (ops.getTime() >= self.duration or (abs(ops.getTime() - self.duration) < 1e-5)) and not collapse_flag:
                    return True, False, result[: 9]  # 分析成功，结构不倒塌
                if ops.getTime() >= self.duration and collapse_flag:
                    return True, True, result[: 9]  # 分析成功，结构倒塌
                if maxAna_flag:
                    return True, True, result[: 9]  # 分析成功，结构倒塌
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
                        return False, True, result[: 9]
                    if algorithm_id == 4 and not collapse_flag:
                        return False, False, result[: 9]
            dt = dt_init * factor
            if dt + ops.getTime() > self.duration:
                dt = self.duration - ops.getTime()
            ops.algorithm(*algorithms[algorithm_id])


    def SDR_tester(self) -> tuple[bool, bool]:
        """
        return (tuple[bool, bool]): 是否倒塌？是否超过最大计算位移？
        """
        if self.collapse_disp > self.maxAnalysis_disp:
            raise SDOF_Error('`MaxAnalysisDrift`应大于`CollapseDrift`')
        result = (False, False)
        u = abs(ops.nodeDisp(2, 1) - ops.nodeDisp(1, 1))
        if u >= self.collapse_disp:
            result = (True, False)
        if u >= self.maxAnalysis_disp:
            result = (True, True)
        return result


    def get_responses(self, input_result: tuple[float]) -> Tuple:
        """获取分析结果
        """
        maxDisp, maxVel, maxAccel, Ec, Ev, maxReaction, CD, CPD, u_old,\
            F_hys_old, F_ray_old, u_cent, *_ = input_result
        # 最大相对位移
        u = ops.nodeDisp(2, 1)
        du = u - u_old
        maxDisp = max(maxDisp, abs(u))
        # 最大相对速度
        v = ops.nodeVel(2, 1)
        maxVel = max(maxVel, abs(v))
        # 最大绝对加速度
        a_base = -ops.nodeAccel(2000, 1)
        a_a = a_base + ops.nodeAccel(2, 1)
        maxAccel = max(maxAccel, abs(a_a))
        # 最大基底反力
        ops.reactions('-dynamic', '-rayleigh')
        F_total = ops.nodeReaction(1, 1)
        maxReaction = max(maxReaction, abs(F_total))
        # 累积弹塑性耗能
        ops.reactions('-dynamic')
        F_hys = ops.nodeReaction(1, 1)
        Si = 0.5 * (F_hys + F_hys_old) * du
        Ec += Si
        # 累积Rayleigh耗能
        ops.reactions('-rayleigh')
        F_ray = ops.nodeReaction(1, 1)
        Si = -0.5 * (F_ray + F_ray_old) * du
        Ev += Si
        # 累积变形
        CD += abs(du)
        # 累积塑性变形
        if self.uy is None:
            CPD = None
        else:
            if u > u_cent + self.uy:
                # 正向屈服
                CPD += u - (u_cent + self.uy)
                u_cent += u - (u_cent + self.uy)
            elif u < u_cent - self.uy:
                # 负向屈服
                CPD += u_cent - self.uy - u
                u_cent -= u_cent - self.uy - u
            else:
                CPD += 0
        if __name__ == "__main__":
            u_base = -ops.nodeDisp(2000, 1)  # 基底位移
            v_base = -ops.nodeVel(2000, 1)  # 基底速度
            u_a = u_base + u  # 绝对位移
            v_a = v_base + v  # 绝对速度
            a = ops.nodeAccel(2, 1)  # 相对加速度
            TIME.append(ops.getTime())
            A_R.append(a)
            A_A.append(a_a)
            V_R.append(v)
            V_A.append(v_a)
            U_R.append(u)
            U_A.append(u_a)
            A_BASE.append(a_base)
            V_BASE.append(v_base)
            U_BASE.append(u_base)
            REACTION.append(F_total)
            REACTION_HYS.append(F_hys)
            REACTION_RAY.append(F_ray)
            E_HYS.append(Ec)
            E_RAY.append(Ev)
            CD_.append(CD)
            CPD_.append(CPD)
        return maxDisp, maxVel, maxAccel,\
            Ec, Ev, maxReaction,\
            CD, CPD, u,\
            F_hys, F_ray, u_cent
        # 12个参数


    def get_results(self) -> dict[str, bool | float]:
        return self.results


# -------------------------------------------------------------------------------
# --------------------------------- 多个SDOF批量求解 -----------------------------
# -------------------------------------------------------------------------------

class _SDOF_batched_solver:
    def __init__(self,
            N_SDOFs: int,
            ls_T: tuple[float, ...],
            gm: np.ndarray,
            dt: float,
            ls_materials: tuple[Dict[str, tuple], ...],
            ls_uy: tuple[float, ...]=None,
            fv_duration: float=0,
            SF: tuple=(1,)*1000000,
            zeta: tuple[float, ...]=(0.05,)*1000000,
            ls_m: tuple[float, ...]=(1,)*1000000,
            g: float=9800,
            ls_collapse_disp: tuple[float, ...]=(1e14,)*1000000,
            ls_maxAnalysis_disp: tuple[float, ...]=(1e15,)*1000000,):
        if not (N_SDOFs == len(ls_T) == len(ls_materials)):
            raise SDOF_Error(f'SDOF数量、周期数量、材料数量不等！({N_SDOFs}, {len(ls_T)}, {len(ls_materials)})')
        self.N_SDOFs = N_SDOFs
        self.ls_T = ls_T
        self.gm = gm
        self.gm_u = a2u(gm, dt)
        self.dt = dt
        self.ls_materials = ls_materials
        self.ls_uy = ls_uy
        self.fv_duration = fv_duration
        self.SF = SF
        self.zeta = zeta
        self.ls_m = ls_m
        self.g = g
        self.ls_collapse_disp = ls_collapse_disp
        self.ls_maxAnalysis_disp = ls_maxAnalysis_disp
        self.NPTS = len(gm)
        self.duration = (self.NPTS - 1) * dt + fv_duration
        self.NPTS = len(gm)
        self.duration = (self.NPTS - 1) * dt + fv_duration
        self.run_model()


    def run_model(self):
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        nodeTag = 1  # 当前节点编号
        self.baseNodes = []  # 基底节点编号
        self.ctrlNodes = []  # 控制节点的编号
        matTag = 1  # 当前材料编号
        ctrlMats = []  # 控制材料的编号
        eleTag = 1  # 当前单元编号
        self.ctrlEles = []  # 控制单元的编号
        for i in range(self.N_SDOFs):
            # 节点、约束、质量
            ops.node(nodeTag, 0, 0)
            ops.node(nodeTag + 1, 0, 0)
            ops.fix(nodeTag, 1, 1, 1)
            ops.fix(nodeTag + 1, 0, 1, 1)
            ops.mass(nodeTag + 1, self.ls_m[i], 0, 0)
            inode, jnode = nodeTag, nodeTag + 1
            self.baseNodes.append(nodeTag)
            self.ctrlNodes.append(nodeTag + 1)
            nodeTag += 2
            # 材料
            material = self.ls_materials[i]
            matTag_start = matTag
            for matType, paras in material.items():
                ops.uniaxialMaterial(matType, matTag, *_update_para(matTag, *paras))
                matTag += 1
            ops.uniaxialMaterial('Parallel', matTag, *range(matTag_start, matTag))
            ctrlMats.append(matTag)
            matTag += 1
            # 单元
            ops.element('zeroLength', eleTag, inode, jnode, '-mat', matTag - 1, '-dir', 1, '-doRayleigh', 1)  # 弹塑性
            T = self.ls_T[i]
            omega = 2 * pi / T
            b = 2 * self.zeta[i] / omega
            ops.region(i + 1, '-ele', eleTag, '-rayleigh', 0, 0, b, 0)  # Rayleigh阻尼
            self.ctrlEles.append(eleTag)
            eleTag += 1
        # 时程分析
        ops.timeSeries('Path', 1, '-dt', self.dt, '-values', *self.gm_u, '-factor', self.g)
        ops.pattern('MultipleSupport', 1)
        gmTag = 1
        for tag in self.baseNodes:
            ops.groundMotion(gmTag, 'Plain', '-disp', 1, '-fact', self.SF[gmTag - 1])
            ops.imposedMotion(tag, 1, gmTag)
            gmTag += 1
        converge, collapse, response = self.time_history_analysis()
        results = dict()
        results['converge'] = converge
        results['collapse'] = collapse
        results['maxDisp'] = response[0]
        results['maxVel'] = response[1]
        results['maxAccel'] = response[2]
        results['Ec'] = response[3]
        results['Ev'] = response[4]
        results['maxReaction'] = response[5]
        results['CD'] = response[6]
        results['CPD'] = response[7]
        results['resDisp'] = response[8]
        self.results = results
    

    def time_history_analysis(self, min_factor=1e-6, max_factor=1
                ) -> Tuple[bool, Tuple[bool, ...], list[tuple]]:
        """自适应时程分析，可根据收敛状况自动调整步长和迭代算法

        Args:
            min_factor (float): 自适应步长的最小调整系数
            max_factor (float): 自适应步长的最大调整系数
        
        Return: Tuple[bool, Tuple[bool, ...], list[tuple]]
            * (1) - 是否收敛
            * (2) - 是否倒塌
            * (3) - 结构响应结果
        """
        result = tuple([0.0] * self.N_SDOFs for _ in range(100))  # 用来储存结构响应
        algorithms = [("KrylovNewton",), ("NewtonLineSearch",), ("Newton",), ("SecantNewton",)]
        algorithm_id = 0
        ops.wipeAnalysis()
        # ops.constraints("Transformation")
        ops.constraints("Penalty", 1e19, 1e19)
        ops.numberer("Plain")
        ops.system("BandGeneral")
        ops.test("EnergyIncr", 1.0e-5, 30)
        ops.algorithm("KrylovNewton")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")

        collapse_flag = (False,) * self.N_SDOFs
        maxAna_flag = False
        factor = 1
        dt_init = self.dt
        dt = dt_init
        while True:
            ok = ops.analyze(1, dt)
            if ok == 0:
                # 当前步收敛
                result = self.get_responses(result)  # 计算当前步结构响应
                current_collapse_flag, maxAna_flag = self.SDR_tester()  # 判断当前步是否收敛
                collapse_flag = tuple(collapse_flag[i] or current_collapse_flag[i] for i in range(self.N_SDOFs))
                if (ops.getTime() >= self.duration) or maxAna_flag or (abs(ops.getTime() - self.duration) < 1e-5):
                    return [True] * self.N_SDOFs, collapse_flag, result[: 9]
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
                        return [False] * self.N_SDOFs, collapse_flag, result[: 9]
            dt = dt_init * factor
            if dt + ops.getTime() > self.duration:
                dt = self.duration - ops.getTime()
            ops.algorithm(*algorithms[algorithm_id])


    def SDR_tester(self) -> Tuple[tuple[bool], bool]:
        """
        return Tuple[tuple[bool], bool]: 各个SDOF是否倒塌？是否全部超过最大计算位移？
        """
        results_collapse = []
        results_maxAna = []
        for i in range(self.N_SDOFs):
            if self.ls_collapse_disp[i] > self.ls_maxAnalysis_disp[i]:
                raise SDOF_Error('`MaxAnalysisDrift`应大于`CollapseDrift`')
            u = abs(ops.nodeDisp(self.ctrlNodes[i], 1) - ops.nodeDisp(self.baseNodes[i], 1))
            if u >= self.ls_collapse_disp[i]:
                results_collapse.append(True)
            else:
                results_collapse.append(False)  
            if u >= self.ls_maxAnalysis_disp[i]:
                results_maxAna.append(True)
            else:
                results_maxAna.append(False)
        return tuple(results_collapse), all(results_maxAna)


    def get_responses(self, input_result: Tuple[List[float], ...]) -> Tuple[list[float], ...]:
        """
        获取分析结果
        """
        maxDisp, maxVel, maxAccel, Ec, Ev, maxReaction, CD, CPD, ls_u_old,\
            ls_F_Hys_old, ls_F_Ray_old, ls_u_cent, *_ = input_result
        ls_u = []
        ls_F_Hys = []
        ls_F_Ray = []
        for i in range(self.N_SDOFs):
            ctrl_node = self.ctrlNodes[i]
            base_node = self.baseNodes[i]
            eleTag = self.ctrlEles[i]
            u_old = ls_u_old[i]
            F_Hys_old = ls_F_Hys_old[i]
            F_Ray_old = ls_F_Ray_old[i]
            u_cent = ls_u_cent[i]
            # 最大相对位移
            u: float = ops.nodeDisp(ctrl_node, 1) - ops.nodeDisp(base_node, 1)
            ls_u.append(u)
            du = u - u_old
            maxDisp[i] = max(maxDisp[i], abs(u))
            # 最大相对速度
            v: float = ops.nodeVel(ctrl_node, 1) - ops.nodeVel(base_node, 1)
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
            if self.ls_uy is None:
                CPD = None
            else:
                uy = self.ls_uy[i]
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
        if __name__ == "__main__":
            TIME.append(ops.getTime())
            # TODO copy自第一个SDOF求解器
            # u_base = -ops.nodeDisp(2000, 1)  # 基底位移
            # v_base = -ops.nodeVel(2000, 1)  # 基底速度
            # u_a = u_base + u  # 绝对位移
            # v_a = v_base + v  # 绝对速度
            # a = ops.nodeAccel(2, 1)  # 相对加速度
            # TIME.append(ops.getTime())
            # A_R.append(a)
            # A_A.append(a_a)
            # V_R.append(v)
            # V_A.append(v_a)
            # U_R.append(u)
            # U_A.append(u_a)
            # A_BASE.append(a_base)
            # V_BASE.append(v_base)
            # U_BASE.append(u_base)
            # REACTION.append(F_total)
            # REACTION_HYS.append(F_hys)
            # REACTION_RAY.append(F_ray)
            # E_HYS.append(Ec)
            # E_RAY.append(Ev)
            # CD_.append(CD)
            # CPD_.append(CPD)
        return maxDisp, maxVel, maxAccel,\
            Ec, Ev, maxReaction,\
            CD, CPD, ls_u,\
            ls_F_Hys, ls_F_Ray, ls_u_cent
        # 12个参数


    def get_results(self) -> dict[str, bool | tuple[bool, ...] | list[float]]:
        return self.results


# -----------------------------------------------------------------------------------
# --------------------------------- SDOF批量求解 -------------------------------------
# ------------------------------- （可考虑P-Delta） ----------------------------------
# -----------------------------------------------------------------------------------

class _PDtSDOF_batched_solver:
    def __init__(self,
            N_SDOFs: int,
            ls_h: tuple[float, ...],
            ls_T: tuple[float, ...],
            ls_grav: tuple[float, ...],
            gm: np.ndarray,
            dt: float,
            ls_materials: tuple[Dict[str, tuple], ...],
            ls_uy: tuple[float, ...]=None,
            fv_duration: float=0,
            SF: tuple=(1,)*1000000,
            zeta: tuple[float, ...]=(0.05,)*1000000,
            ls_m: tuple[float, ...]=(1,)*1000000,
            g: float=9800,
            ls_collapse_disp: tuple[float, ...]=(1e14,)*1000000,
            ls_maxAnalysis_disp: tuple[float, ...]=(1e15,)*1000000,):
        if not (N_SDOFs == len(ls_T) == len(ls_materials)):
            raise SDOF_Error(f'SDOF数量、周期数量、材料数量不等！({N_SDOFs}, {len(ls_T)}, {len(ls_materials)})')
        self.N_SDOFs = N_SDOFs
        self.ls_h = ls_h
        self.ls_T = ls_T
        self.ls_grav = ls_grav
        self.gm = gm
        self.dt = dt
        self.ls_materials = ls_materials
        self.ls_uy = ls_uy
        self.fv_duration = fv_duration
        self.SF = SF
        self.zeta = zeta
        self.ls_m = ls_m
        self.g = g
        self.ls_collapse_disp = ls_collapse_disp
        self.ls_maxAnalysis_disp = ls_maxAnalysis_disp
        self.NPTS = len(gm)
        self.duration = (self.NPTS - 1) * dt + fv_duration
        self.run_model()


    def run_model(self):
        A_rigid = 1e10
        I_rigid = 1e10
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        nodeTag = 1  # 当前节点编号
        self.baseNodes = []  # 基底节点编号
        self.midNodes = []  # 中间节点（与刚性杆下部连接的节点）编号
        self.ctrlNodes = []  # 控制节点的编号
        matTag = 1  # 当前材料编号
        ctrlMats = []  # 控制材料的编号
        eleTag = 1  # 当前单元编号
        self.ctrlEles = []  # 控制单元的编号
        ops.geomTransf('PDelta', 1)
        for i in range(self.N_SDOFs):
            # 节点、约束、质量
            h = self.ls_h[i]
            inode, jnode, knode = nodeTag, nodeTag + 1, nodeTag + 2
            ops.node(inode, 0, 0)
            ops.node(jnode, 0, h)
            ops.node(knode, 0, h)
            ops.fix(inode, 1, 1, 0)
            # ops.fix(jnode, 0, 1, 1)
            ops.fix(knode, 1, 1, 1)
            # ops.equalDOF(inode, jnode, 1, 2)
            ops.mass(jnode, self.ls_m[i], 0, 0)
            self.baseNodes.append(inode)
            self.ctrlNodes.append(jnode)
            self.midNodes.append(knode)
            nodeTag += 3
            # 材料
            material = self.ls_materials[i]
            matTag_start = matTag
            for matType, paras in material.items():
                ops.uniaxialMaterial(matType, matTag, *_update_para(matTag, *paras))
                matTag += 1
            ops.uniaxialMaterial('Parallel', matTag, *range(matTag_start, matTag))
            ctrlMats.append(matTag)
            matTag += 1
            # 单元
            ops.element('zeroLength', eleTag, jnode, knode, '-mat', matTag - 1, '-dir', 1, '-doRayleigh', 1)  # 零长度弹塑性弹簧
            ops.element('elasticBeamColumn', eleTag + 1, inode, jnode, A_rigid, 206000, I_rigid, 1)  # 刚性梁
            T = self.ls_T[i]
            omega = 2 * pi / T
            b = 2 * self.zeta[i] / omega
            ops.region(i + 1, '-ele', eleTag, '-rayleigh', 0, 0, b, 0)  # Rayleigh阻尼
            self.ctrlEles.append(eleTag)
            eleTag += 2

        # 竖向荷载
        ops.timeSeries('Linear', 11)
        ops.pattern('Plain', 11, 11)
        for i in range(self.N_SDOFs):
            F = self.ls_grav[i]
            if F == 0:
                continue
            ops.load(self.ctrlNodes[i], 0, -F, 0)

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
        ops.timeSeries('Path', 1, '-dt', self.dt, '-values', *self.gm, '-factor', self.g)
        ops.pattern('MultipleSupport', 1)
        gmTag = 1
        for tag1, tag2 in zip(self.baseNodes, self.midNodes):
            ops.groundMotion(gmTag, 'Plain', '-accel', 1, '-fact', self.SF[gmTag - 1])
            ops.imposedMotion(tag1, 1, gmTag)
            ops.imposedMotion(tag2, 1, gmTag)
            gmTag += 1
        converge, collapse, response = self.time_history_analysis()
        results = dict()
        results['converge'] = converge
        results['collapse'] = collapse
        results['maxDisp'] = response[0]
        results['maxVel'] = response[1]
        results['maxAccel'] = response[2]
        results['Ec'] = response[3]
        results['Ev'] = response[4]
        results['maxReaction'] = response[5]
        results['CD'] = response[6]
        results['CPD'] = response[7]
        results['resDisp'] = response[8]
        self.results = results
    

    def time_history_analysis(self, min_factor=1e-6, max_factor=1
            ) -> Tuple[bool, Tuple[bool, ...], list[tuple]]:
        """自适应时程分析，可根据收敛状况自动调整步长和迭代算法

        Args:
            min_factor (float): 自适应步长的最小调整系数
            max_factor (float): 自适应步长的最大调整系数
        
        Return: Tuple[bool, Tuple[bool, ...], list[tuple]]
            * (1) - 是否收敛
            * (2) - 是否倒塌
            * (3) - 结构响应结果
        """
        result = tuple([0.0] * self.N_SDOFs for _ in range(100))  # 用来储存结构响应
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
        dt_init = self.dt
        collapse_flag = (False,) * self.N_SDOFs
        maxAna_flag = False
        factor = 1
        dt = dt_init
        while True:
            ok = ops.analyze(1, dt)
            if ok == 0:
                # 当前步收敛
                result = self.get_responses(result)  # 计算当前步结构响应
                current_collapse_flag, maxAna_flag = self.SDR_tester()  # 判断当前步是否收敛
                collapse_flag = tuple(collapse_flag[i] or current_collapse_flag[i] for i in range(self.N_SDOFs))
                if (ops.getTime() >= self.duration) or maxAna_flag or (abs(ops.getTime() - self.duration) < 1e-5):
                    return [True] * self.N_SDOFs, collapse_flag, result[: 9]
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
                        return [False] * self.N_SDOFs, collapse_flag, result[: 9]
            dt = dt_init * factor
            if dt + ops.getTime() > self.duration:
                dt = self.duration - ops.getTime()
            ops.algorithm(*algorithms[algorithm_id])
    

    def SDR_tester(self):
        """
        return Tuple[tuple[bool], bool]: 各个SDOF是否倒塌？是否全部超过最大计算位移？
        """
        results_collapse = []
        results_maxAna = []
        for i in range(self.N_SDOFs):
            if self.ls_collapse_disp[i] > self.ls_maxAnalysis_disp[i]:
                raise SDOF_Error('`ls_collapse_disp`应大于`ls_maxAnalysis_disp`')
            u = ops.nodeDisp(self.ctrlNodes[i], 1) - ops.nodeDisp(self.midNodes[i], 1)
            if abs(u) >= self.ls_collapse_disp[i]:
                results_collapse.append(True)
            else:
                results_collapse.append(False)  
            if abs(u) >= self.ls_maxAnalysis_disp[i]:
                results_maxAna.append(True)
            else:
                results_maxAna.append(False)
        return tuple(results_collapse), all(results_maxAna)


    def get_responses(self, input_result: Tuple[List[float], ...]) -> Tuple[list[float], ...]:
        """
        获取分析结果
        """
        # t.append(ops.getTime())
        maxDisp, maxVel, maxAccel, Ec, Ev, maxReaction, CD, CPD, ls_u_old,\
            ls_F_Hys_old, ls_F_Ray_old, ls_u_cent, *_ = input_result
        ls_u = []
        ls_F_Hys = []
        ls_F_Ray = []
        for i in range(self.N_SDOFs):
            ctrl_node = self.ctrlNodes[i]
            base_node = self.baseNodes[i]
            mid_node = self.midNodes[i]
            eleTag = self.ctrlEles[i]
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
            # 最大相对速度
            v: float = ops.nodeVel(ctrl_node, 1) - ops.nodeVel(mid_node, 1)
            maxVel[i] = max(maxVel[i], abs(v))
            # if i == 0:
            #     V.append(v)
            # 最大绝对加速度
            a: float = ops.nodeAccel(ctrl_node, 1)
            maxAccel[i] = max(maxAccel[i], abs(a))
            # if i == 0:
            #     A_BASE.append(ops.nodeAccel(base_node, 1))
            #     A.append(a)
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
            if self.ls_uy is None:
                CPD = None
            else:
                uy = self.ls_uy[i]
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
        if __name__ == "__main__":
            TIME.append(ops.getTime())
            # TODO copy自第一个SDOF求解器
            # u_base = -ops.nodeDisp(2000, 1)  # 基底位移
            # v_base = -ops.nodeVel(2000, 1)  # 基底速度
            # u_a = u_base + u  # 绝对位移
            # v_a = v_base + v  # 绝对速度
            # a = ops.nodeAccel(2, 1)  # 相对加速度
            # TIME.append(ops.getTime())
            # A_R.append(a)
            # A_A.append(a_a)
            # V_R.append(v)
            # V_A.append(v_a)
            # U_R.append(u)
            # U_A.append(u_a)
            # A_BASE.append(a_base)
            # V_BASE.append(v_base)
            # U_BASE.append(u_base)
            # REACTION.append(F_total)
            # REACTION_HYS.append(F_hys)
            # REACTION_RAY.append(F_ray)
            # E_HYS.append(Ec)
            # E_RAY.append(Ev)
            # CD_.append(CD)
            # CPD_.append(CPD)
        return maxDisp, maxVel, maxAccel,\
            Ec, Ev, maxReaction,\
            CD, CPD, ls_u,\
            ls_F_Hys, ls_F_Ray, ls_u_cent
        # 12个参数


    def get_results(self) -> dict[str, bool | tuple[bool, ...] | list[float]]:
        return self.results


def _update_para(matTag: int, *paras: float | str):
    """将材料参数中的 ^ 替换为引用的编号，有多少个 ^ 就代表引用前多少个材料的编号

    Args:
        matTag (int): 当前材料的编号
        paras (float | str): 材料参数
    """
    para_new = []
    for para in paras:
        res = re.findall(r'^\^+$', str(para))
        if res:
            refTag = matTag - len(res[0])
            if refTag < 1:
                raise SDOF_Error(f'引用材料编号小于1\n{paras}')
            para_new.append(refTag)
        else:
            para_new.append(para)
    return para_new



if __name__ == "__main__":
    ls_T = (0.005,)
    T = 0.005
    Cy = 10
    alpha = 0
    m = 1
    Fy = m * 9800 * Cy
    k = 4 * pi**2 / T**2 * m
    uy = Fy / k
    print(f'Fy = {Fy}, k = {k}')
    h = 1
    ls_grav = (0,) * 1
    gm = np.loadtxt(Path(__file__).parent.parent/'Input/GMs'/'th1.th')
    print('Maximun of gm:', max(abs(gm)))
    dt = 0.01
    materials = tuple({'Steel01': (Fy, k, alpha)} for _ in range(1))
    # PDtMaterials = tuple({'Steel01': (3924, 986.9604401089356, 0.0)} for _ in range(1))
    PDtMaterials = ({'Steel01': (3924, 986.9604401089356, 0.0), 'Parallel': ('^')},)
    material = {'Steel01': (Fy, k, alpha)}
    with SDOF_Helper(suppress=False):
        # results = SDOF_batched_solver(1, ls_T, gm, dt, materials, [2])

        for i in range(1):
            results = SDOF_solver(T, gm, dt, material, uy=uy, fv_duration=0, SF=1)

        # results = PDtSDOF_batched_solver(1, (h,), ls_T, ls_grav, gm, dt, PDtMaterials, ls_uy=[3.9758432461253355]*1, fv_duration=0, ls_collapse_disp=(105,), ls_SF=(5.984352224424968,))
    print(results)
    # print(state)
    # print(result[8][0])
    resType = E_RAY
    plt.plot(TIME, resType)
    plt.show()
    # np.savetxt(r'F:\NRSA\temp\t.txt', t)
    np.savetxt(Path.cwd()/'temp/t-res.txt', np.column_stack((TIME, resType)))


