import time
from math import pi, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    miu_ls = [2, 3, 4]
    time_start = time.time()
    T = np.arange(0.02, 6, 0.02)
    for miu in miu_ls:
        print(f'Running with miu={miu}')
        material_paras: dict[str, float] = {
            'alpha': 0.02
        }  # Required parameters for material definition, can be customized by user.
        # The length of the dictionary should be the same as the number of arguments in the `material_definition` function.
        # Requires Python 3.7+ to preserve the order of dictionary items.
        model = ConstantDuctilityAnalysis(f'Test_{miu}', cls_cache=True)
        model.set_working_directory(f'./CDA_results/{miu}', folder_exists='delete')
        model.analysis_settings(T, material_definition, material_paras,
            damping=0.05,  # Damping ratio
            target_ductility=miu,  # Target ductility
            R_init=1,  # Initial strength reduction factor (R)
            R_incr=5,  # Incremental value of R for each iteration
            tol_ductility=0.01,  # Tolerance for target ductility
            tol_R=0.001,  # Tolerance between adjacent R values
            max_iter=100,  # Maximum number of iterations
            thetaD=0  # P-Delta coefficient
        )
        model.select_ground_motions('./data/GMs', ['Northridge', 'Kobe'], suffix='.txt')
        model.running_settings(parallel=2, auto_quit=True, hidden_prints=True, show_monitor=True, solver='auto')
        model.run()
    time_end = time.time()
    print(f'Elapsed time: {time_end - time_start:.2f}')

    # Compare with SeismoSignal results
    g = 9800
    plt.figure(figsize=(12, 8))
    for i, gm in enumerate(['Northridge', 'Kobe']):
        plt.subplot(2, 1, i + 1)
        for j, miu in enumerate(miu_ls):
            res = pd.read_csv(f'./CDA_results/{miu}/results/{gm}.csv')
            T = res['T']
            a = res['maxAccel'] / g
            res_ssm = np.loadtxt(f'./data/SeismiSignal_results/{gm}.txt', skiprows=1)
            T_ssm = res_ssm[:, 0]
            a_ssm = res_ssm[:, 2 + j]
            plt.plot(T, a, label=f'NRSA (miu={miu})', c='black')
            plt.plot(T_ssm, a_ssm, label=f'SeismoSignal (miu={miu})', c='red', ls='--')
        plt.xlabel('Period (s)')
        plt.ylabel('Peak Acceleration (g)')
        plt.xlim(0, 6)
        plt.ylim(0)
        plt.title(f'Acceleration-Period Curves ({gm})')
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot R-μ-T curves
    plt.figure(figsize=(12, 8))
    for i, gm in enumerate(['Northridge', 'Kobe']):
        plt.subplot(2, 1, i + 1)
        for j, miu in enumerate([2, 3, 4]):
            res = pd.read_csv(f'./CDA_results/{miu}/results/{gm}.csv')
            T = res['T']
            R = res['R']
            plt.plot(T, R, label=f'miu={miu}')
        plt.xlabel('Period (s)')
        plt.ylabel('R')
        plt.xlim(0, 6)
        plt.ylim(0)
        plt.title(f'R-μ-T Curves ({gm})')
        plt.legend()
    plt.tight_layout()
    plt.show()
