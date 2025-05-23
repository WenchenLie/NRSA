"""
----------------- 单自由度时程分析求解器 ----------------------
共包含三个进行非线性SDOF体系时程分析的函数，
分别为SDOF_solver、SDOF_batched_solver和PDtSDOF_batched_solver，
其中
SDOF_solver：            非线性单自由度体系(可考虑简化等效的P-Delta效应)

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

import re
from math import pi, sqrt
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
if __name__ == "__main__":
    from utils import SDOFError, SDOFHelper, a2u
    from utils import is_iterable
else:
    from .utils import SDOFError, SDOFHelper, a2u
    from .utils import is_iterable


__all__ = ['SDOF_solver']


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
        P: float=None,
        h: float=1,
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
        P (float, optional): 引起P-Delta效应的竖向载荷，默认None
        h (float, optional): 引起P-Delta效应的SDOF体系高度，默认1
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
        model = _SDOF_solver(T, gm, dt, materials, uy, fv_duration, SF, P, h, zeta, m, g, collapse_disp, maxAnalysis_disp)
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
            P: float=None,
            h: float=1,
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
        self.P = P
        self.h = h
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
        # self.a = 0
        # self.b = 0
        # print(f'a = {self.a}, b = {self.b}')
        # self.c = 2 * m * zeta * omega
        # print(f'c = {self.c}')
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
            if not is_iterable(paras):
                paras = (paras,)
            ops.uniaxialMaterial(matType, matTag, *_update_para(matTag, *paras))
            matTag += 1
        if self.P is not None:
            k_neg = -self.P / self.h  # P-Delta效应引起的负刚度
            ops.uniaxialMaterial('Elastic', matTag, k_neg)
            matTag += 1
        ops.uniaxialMaterial('Parallel', matTag, *range(1, matTag))
        matTag += 1
        ops.uniaxialMaterial('Elastic', matTag, 0)
        ops.element('zeroLength', 1, 1, 2, '-mat', matTag - 1, '-dir', 1, '-doRayleigh', 1)  # 弹塑性
        ops.region(1, '-ele', 1, '-rayleigh', self.a, 0, self.b, 0)  # Rayleigh阻尼
        ops.timeSeries('Path', 1, '-dt', self.dt, '-values', *self.gm, '-factor', self.g)
        # ops.pattern('UniformExcitation', 1, 1, '-accel', 1, '-fact', self.SF)
        ops.pattern('Plain', 1, 1, '-fact', -self.m * self.SF)
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
                if ops.getTime() >= self.duration and collapse_flag:
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
        a_base: float = -ops.getLoadFactor(1) / self.m * self.SF
        a_a: float = a_base + ops.nodeAccel(2, 1)
        maxAccel = max(maxAccel, abs(a_a))
        # 最大基底反力
        ops.reactions('-dynamic')
        F_total: float = ops.nodeReaction(1, 1)
        ops.reactions('-rayleigh')
        F_ray: float = ops.nodeReaction(1, 1)
        ops.reactions('-rayleigh', '-dynamic')
        F_hys: float = ops.nodeReaction(1, 1)  # F_total = F_ray + F_hys
        maxReaction = max(maxReaction, abs(F_total))
        # 累积弹塑性耗能
        Si = -0.5 * (F_hys + F_hys_old) * du
        Ec += Si
        # 累积Rayleigh耗能
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
            # u_base: float = -ops.nodeDisp(2000, 1)  # 基底位移
            # v_base: float = -ops.nodeVel(2000, 1)  # 基底速度
            # u_a = u_base + u  # 绝对位移
            # v_a = v_base + v  # 绝对速度
            a: float = ops.nodeAccel(2, 1)  # 相对加速度
            TIME.append(ops.getTime())
            A_R.append(a)
            A_A.append(a_a)
            V_R.append(v)
            # V_A.append(v_a)
            U_R.append(u)
            # U_A.append(u_a)
            A_BASE.append(a_base)
            # V_BASE.append(v_base)
            # U_BASE.append(u_base)
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


if __name__ == "__main__":
    from src.spectrum import spectrum

    N_SDOF = 1
    ls_T = (0.005,) * N_SDOF
    # T = 0.005
    paras1 = (1348.597853399964,  98696.04401089359, 0.02)
    paras2 = (1348.5978533999642, 98696.04401089359, 0.02)
    Fy1, E1, alpha1 = paras1
    Fy2, E2, alpha2 = paras2
    uy1 = paras1[0] / paras1[1]
    uy2 = paras2[0] / paras2[1]
    m = 1
    # T = 2 * pi  * sqrt(m / k0)
    T = 0.02
    print('T =', T)
    gm = np.loadtxt(r'F:\Projects\OpenSAS\GMs\th22.th')
    dt = 0.02
    RSA, _, _ = spectrum(gm, dt, np.array([0.02]))
    R = 1
    Fe = RSA * 9800 * m
    print('Fe =', Fe)
    # materials = tuple({'TSSCB': (F1, k0, ugap, F2, k1, k2, beta)} for _ in range(N_SDOF))
    # PDtMaterials = tuple({'Steel01': (3924, 986.9604401089356, 0.0)} for _ in range(1))
    # PDtMaterials = tuple({'Steel01': (Fy, k, alpha)} for _ in range(N_SDOF))
    material1 = {'Steel01': paras1}
    material2 = {'Steel01': paras2}
    with SDOFHelper(suppress=False):
        # with VizTracer():
        # for i in range(N_SDOF):
        #     results = SDOF_solver(T, gm, dt, material, uy=uy, fv_duration=30, SF=1, m=m)

        results1 = SDOF_solver(T, gm, dt, material1, uy=uy1, fv_duration=0, SF=1, m=m, P=0, h=1)
        t1, u1, F_hys1 = TIME, U_R, REACTION_HYS
        TIME, U_R, REACTION_HYS = [], [], []
        results2 = SDOF_solver(T, gm, dt, material2, uy=uy1, fv_duration=0, SF=1, m=m, P=0, h=1)
        t2, u2, F_hys2 = TIME, U_R, REACTION_HYS
    # print('maxDisp', results['maxDisp'])
    # print('miu', results['maxDisp'] / uy)
    # print(state)
    # print(result[8][0])
    # resType = A_A
    # print(t1[:])
    plt.plot(t1, u1, label='1')
    plt.plot(t2, u2, label='2')
    # plt.plot(U_R, REACTION_HYS)
    plt.legend()
    plt.show()
    # np.savetxt(f'temp/data.txt', np.column_stack((U_R, REACTION_HYS)))
    # np.savetxt(f'temp/data1.txt', np.column_stack((u1, F_hys1)))


