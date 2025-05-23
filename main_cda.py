from math import pi, sqrt
import numpy as np
from src.analysis import ConstantDuctilityAnalysis


def material_definition(
    Ti: float,
    m: float,
    R: float,
    Sa: float,
    *args: float
) -> tuple[dict[str, tuple | float], float, float]:
    """给定周期点、质量、强度折减系数、弹性谱加速度，用户自定义opensees材料参数的计算方法  
    用户可修改传入的`args`参数和函数内部的计算过程，但不能修改函数的前四个传入参数  
    返回值中的屈服力和弹性刚度将用于计算延性需求

    Args:
        Ti (float): 周期点
        m (float): 质量
        R (float): 强度折减系数
        Sa (float): 弹性谱加速度(g)
        Args (float): 定义opensees材料所需的相关参数，一般建议取为无量纲系数，并以此计算定义材料所需的直接参数

    Returns:
        tuple[dict[str, tuple | float], float, float]: OpenSees材料定义格式

    Definition
    ----------
    OpenSees材料定义格式为{`材料名`: (参数1, 参数2, ...)}，不包括材料编号。  
    例如：  
    >>> ops_paras = {'Steel01': (Fy, E, b)}

    其中，`Fy`，`E`和`b`应直接幅值或通过`Ti`、`m`、`R`和`Sa`计算得到。  
    当需要使用多个材料进行并联时，可在`ops_paras`中定义多个材料。  
    例如：  
    >>> ops_paras = {'Steel01': (Fy, E1, b), 'Elastic': E2}

    Notes:
    ------
    * 当材料参数只有一个时，可省略括号  
    * 所采用的材料必须包含在OpenSeesPy的材料库中  
    * 材料所预测的力学行为应随参数变化而连续，并受参数微扰动的影响小，否则在等延性分析中可能无法稳定收敛
    """
    # ===========================================================
    # --------------- ↓ 用户只能更改以下代码 ↓ --------------------
    alpha, = args
    Fe = m * Sa * 9800  # 弹性SDOF最大力需求
    E = (2 * pi / Ti) ** 2 * m
    Fy = Fe / R
    ops_paras = {'Steel01': (Fy, E, alpha)}
    yield_strength, initial_stiffness = Fy, E
    # --------------- ↑ 用户只能更改以上代码 ↑ --------------------
    # ===========================================================
    return ops_paras, yield_strength, initial_stiffness


if __name__ == "__main__":
    T = np.arange(0.02, 6, 0.02)
    material_paras: dict[str, float] = {
        'alpha': 0.02
    }  # 材料定义所需参数，键名可自定义，字典长度应与material_definition函数中args参数个数一致
    # 需Python 3.7+从而保证字典的键值对顺序不变
    model = ConstantDuctilityAnalysis('Test')
    model.set_working_directory('H:/NRSA_results')
    model.analysis_settings(T, material_definition, material_paras,
        damping=0.05,
        target_ductility=3,
        R_init=1,
        R_incr=5,
        tol_ductility=0.01,
        tol_R=0.001,
        max_iter=100,
        thetaD=0,
        mass=1
    )
    model.select_ground_motions('./GMs', [f'th{i}' for i in range(1, 45)], suffix='.th')
    model.running_settings(parallel=20, auto_quit=False, hidden_prints=True, show_monitor=True)
    model.run()

