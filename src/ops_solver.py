"""
--------------------- 基于OpenSees单自由度时程分析求解器 ----------------------
特性:
(1) 建模：SDOF体系通过两个相同位置的结点和中间的 zeroLength 单元连接。
(2) 地震动施加：根据 D'Alembert 原理将地震动等效为上部结点的荷载施加。
(3) 求解：通过自适应调整时间步长和迭代算法，获得良好的收敛性。
(4) 输出：多种响应的峰值和时程响应(可选)，包括相对位移、相对速度、绝对加速度、
    累积弹塑性耗能、累积黏滞阻尼耗能、底部反力、累积位移、累积塑性位移、残余位移
"""

import re
from math import pi, sqrt
from typing import Dict, Tuple

import numpy as np

from .utils import SDOFError, SDOFHelper
from .utils import is_iterable
from . import opensees as ops


__all__ = ['ops_solver']

def ops_solver(
        T: float | float,
        ag: np.ndarray,
        dt: float,
        materials: Dict[str, tuple],
        uy: float=None,
        fv_duration: float=0,
        sf: float=None,
        P: float=0,
        h: float=1,
        zeta: float=0.05,
        m: float=1,
        g: float=9800,
        collapse_disp: float=1e14,
        maxAnalysis_disp: float=1e15,
        record_res: bool=False,
        **kwargs
    ) -> dict[str, float] | tuple[dict[str, float], tuple[np.ndarray, ...]]:
    """SDOF求解函数，每次调用对一个SDOF进行非线性时程分析。
    模型结构为两个具有相同位置的结点，中间采用zeroLength单元连接。

    Args:
        T (float | float): 周期
        ag (np.ndarray): 地震动加速度时程（单位为g）
        dt (float): 时程步长
        materials (Dict[str, tuple]): 材料属性，包括材料名和参数（不包括编号）
        uy (float, optional): 屈服位移，仅用于计算累积塑性应变，默认None即计算值为None
        fv_duration (float, optional): 自由振动时长，默认为0
        sf (float, optional): 地震动放大系数，默认不额外进行缩放
        P (float, optional): 引起P-Delta效应的竖向载荷，默认None
        h (float, optional): 引起P-Delta效应的SDOF体系高度，默认1
        zeta (float, optional): 阻尼比，默认0.05
        m (float, optional): 质量，默认1
        g (float, optional): 重力加速度，默认9800
        collapse_disp (float, optional): 倒塌位移判定准则，默认1e14
        maxAnalysis_disp (float, optional): 最大分析位移，默认1e15
        record_res (bool, optional): 是否记录时程响应

    Returns: dict[str, float] | tuple[dict[str, float], tuple[np.ndarray, ...]]
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

    Notes:
    -----
    模型结构示意图：

        gravity
           ↓   非线性弹簧(zeroLength)
         2 o-----/\/\/\/\/\/\----o 3 (accel input)
           |
           | ← 刚性梁(ElasticBeamColumn)
           |
           |      边界条件:
           |      inode: U1=U2=1, U3=0
           |      jnode: U1=U2=U3=0
           |      knode: U1=U2=U3=1
           |
           |
         1 o (accel input)
    """
    with SDOFHelper(False, False):
        model = _ops_solver(T, ag, dt, materials, uy, fv_duration, sf, P, h, zeta, m, g, collapse_disp, maxAnalysis_disp, record_res)
        results = model.get_results()
    return results


class _ops_solver:
    def __init__(self,
            T: float | float,
            ag: np.ndarray,
            dt: float,
            materials: Dict[str, tuple],
            uy: float=None,
            fv_duration: float=0,
            sf: float=1,
            P: float=0,
            h: float=1,
            zeta: float=0.05,
            m: float=1,
            g: float=9800,
            collapse_disp: float=1e14,
            maxAnalysis_disp: float=1e15,
            record_res: bool=False):
        self.T = T
        self.ag = ag
        self.dt = dt
        self.materials = materials
        self.uy = uy
        self.fv_duration = fv_duration
        self.sf = sf
        self.P = P
        self.h = h
        self.zeta = zeta
        self.m = m
        self.g = g
        self.collapse_disp = collapse_disp
        self.maxAnalysis_disp = maxAnalysis_disp
        self.record_res = record_res
        self.NPTS = len(ag)
        self.duration = (self.NPTS - 1) * dt + fv_duration
        omega = 2 * pi / T
        self.a = 0
        self.b = 2 * zeta / omega
        # self.a = 2 * zeta * omega
        # self.b = 0
        # self.a = 0
        # self.b = 0
        # print(f'a = {self.a}, b = {self.b}')
        # self.c = 2 * m * zeta * omega
        # print(f'c = {self.c}')
        self.time = []  # 时间序列
        self.ag_scaled = []  # 地面运动
        self.disp_th = []  # 相对位移
        self.vel_th = []  # 相对速度
        self.accel_th = []  # 绝对加速度
        self.Ec_th = []  # 累积弹塑性耗能
        self.Ev_th = []  # 累积Rayleigh阻尼耗能
        self.CD_th = []  # 累积变形
        self.CPD_th = []  # 累积塑性变形
        self.reaction_th = []  # 底部反力
        self.eleForce_th = []  # 材料力
        self.dampingForce_th = []  # 黏滞阻尼力
        self.run_model()


    def run_model(self):
        ops.wipe()
        ops.model('basic', '-ndm', 1, '-ndf', 1)
        ops.node(1, 0)
        ops.node(2, 0)
        ops.fix(1, 1)
        ops.fix(2, 0)
        ops.mass(2, self.m)
        matTag = 1
        for matType, paras in self.materials.items():
            ops.uniaxialMaterial(matType, matTag, *paras)
            matTag += 1
        if self.P != 0:
            k_neg = -self.P / self.h  # P-Delta效应引起的负刚度
            ops.uniaxialMaterial('Elastic', matTag, k_neg)
            matTag += 1
        ops.uniaxialMaterial('Parallel', matTag, *range(1, matTag))
        matTag += 1
        ops.uniaxialMaterial('Elastic', matTag, 0)
        ops.element('zeroLength', 1, 1, 2, '-mat', matTag - 1, '-dir', 1, '-doRayleigh', 1)  # 弹塑性
        ops.region(1, '-ele', 1, '-rayleigh', self.a, 0, self.b, 0)  # Rayleigh阻尼
        ops.timeSeries('Path', 1, '-dt', self.dt, '-values', *self.ag, '-factor', self.g * self.sf)
        ops.pattern('Plain', 1, 1, '-fact', -self.m)
        ops.load(2, 1)
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
        result = (0,) * 20  # 用来储存结构响应
        ops.wipeAnalysis()
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("EnergyIncr", 1.0e-5, 50)
        ops.algorithm("NewtonLineSearch")
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
                if (ops.getTime() >= self.duration or (abs(ops.getTime() - self.duration) < 1e-5)) and collapse_flag:
                    return True, True, result[: 9]  # 分析成功，结构倒塌
                if maxAna_flag:
                    return True, True, result[: 9]  # 分析成功，结构倒塌
                factor *= 2
                factor = min(factor, max_factor)
            else:
                # 当前步不收敛
                if factor == min_factor and collapse_flag:
                    return False, True, result[: 9]  # 分析不收敛，结构倒塌
                if factor == min_factor and not collapse_flag:
                    return False, False, result[: 9]  # 分析不收敛，结构不倒塌
                factor *= 0.5
                if factor < min_factor:
                    factor = min_factor
                    if collapse_flag:
                        return False, True, result[: 9]
                    if not collapse_flag:
                        return False, False, result[: 9]
            dt = dt_init * factor
            if dt + ops.getTime() > self.duration:
                dt = self.duration - ops.getTime()


    def SDR_tester(self) -> tuple[bool, bool]:
        """
        return (tuple[bool, bool]): 是否倒塌？是否超过最大计算位移？
        """
        if self.collapse_disp > self.maxAnalysis_disp:
            raise SDOFError('`MaxAnalysisDrift`应大于`CollapseDrift`')
        result = (False, False)
        u = abs(ops.nodeDisp(2, 1))
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
        u: float= ops.nodeDisp(2, 1)
        du = u - u_old
        maxDisp = max(maxDisp, abs(u))
        # 最大相对速度
        v: float = ops.nodeVel(2, 1)
        maxVel = max(maxVel, abs(v))
        # 最大绝对加速度
        a_base: float = -ops.getLoadFactor(1) / self.m
        a_a: float = a_base + ops.nodeAccel(2, 1)
        maxAccel = max(maxAccel, abs(a_a))
        # 最大基底反力
        ops.reactions('-dynamic')
        F_total: float = -ops.nodeReaction(1, 1)
        ops.reactions('-rayleigh')
        F_ray: float = -ops.nodeReaction(1, 1)
        ops.reactions('-rayleigh', '-dynamic')
        F_hys: float = -ops.nodeReaction(1, 1)  # F_total = F_ray + F_hys
        maxReaction = max(maxReaction, abs(F_total))
        # 累积弹塑性耗能
        Si = 0.5 * (F_hys + F_hys_old) * du
        Ec += Si
        # 累积Rayleigh耗能
        Si = 0.5 * (F_ray + F_ray_old) * du
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
        if self.record_res:
            self.time.append(ops.getTime())
            self.ag_scaled.append(a_base)
            self.disp_th.append(u)
            self.vel_th.append(v)
            self.accel_th.append(a_a)
            self.Ec_th.append(Ec)
            self.Ev_th.append(Ev)
            self.CD_th.append(CD)
            self.CPD_th.append(CPD)
            self.reaction_th.append(F_total)
            self.eleForce_th.append(F_hys)
            self.dampingForce_th.append(F_ray)
        return maxDisp, maxVel, maxAccel,\
            Ec, Ev, maxReaction,\
            CD, CPD, u,\
            F_hys, F_ray, u_cent
        # 12个参数

    def get_results(self) -> dict[str, bool | float]:
        if not self.record_res:
            return self.results
        else:
            return self.results, (self.time, self.ag_scaled, self.disp_th,\
                self.vel_th, self.accel_th, self.Ec_th, self.Ev_th, self.CD_th,\
                self.CPD_th, self.reaction_th, self.eleForce_th,\
                self.dampingForce_th)


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
                raise SDOFError(f'引用材料编号小于1\n{paras}')
            para_new.append(refTag)
        else:
            para_new.append(para)
    return para_new
