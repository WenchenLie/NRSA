from math import pi
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from NRSAcore.Task import Task


if __name__ == "__main__":

    g = 9800
    task = Task('TestModel', 'temp')
    
    # 1 定义模型参数
    # (1) 常数型参数
    zeta = 0.05
    P_norm = 0.0
    m = 1
    h = 1
    # (2) 独立参数(通常是无量纲参数)
    T = np.arange(0.2, 2.2, 0.2)
    Cy = np.array([0.4, 0.8, 1.2])
    alpha = [0, 0.05, 0.1]
    # (3) 从属参数(通常直接用于SDOF计算的参数)
    get_Fy = lambda m, Cy: m * g * Cy  # 与m, Cy相关
    get_k = lambda T, m: 4 * pi**2 / T**2 * m  # 与T, m相关
    get_P = lambda P_norm, m: P_norm * m * g  # 与P_norm, m相关
    get_uy = lambda Fy, k: Fy / k  # 与Fy，k相关
    # 设置参数(注意从属参数的先后定义顺序)
    task.add_constant('zeta', zeta)
    task.add_constant('P_norm', P_norm)
    task.add_constant('m', m)
    task.add_constant('h', h)
    task.add_independent_parameter('T', T)
    task.add_independent_parameter('Cy', Cy)
    task.add_independent_parameter('alpha', alpha)
    task.add_dependent_parameter('Fy', get_Fy, 'm', 'Cy')
    task.add_dependent_parameter('k', get_k, 'T', 'm')
    task.add_dependent_parameter('uy', get_uy, 'Fy', 'k')
    task.add_dependent_parameter('P', get_P, 'P_norm', 'm')
    
    # 2 设置模型基本参数
    task.define_basic_parameters(
        period='T',
        mass='m',
        damping='zeta',
        gravity='P',
        yield_disp='uy',
        height='h'
    )

    # 3 定义材料
    # 按照 {材料名: (参数1, 参数2, ...)} 的格式填写
    # 特殊情况：
    # 材料参数如果是引用之前定义的模型参数的，加"$$$"
    # 材料参数如果要使用上一个材料的编号，采用"^"，如果引用前n个材料的编号，则输入n个"^"
    # 填多个材料可自动并联
    material = {
        'Steel01': ('$$$Fy', '$$$k', '$$$alpha')
    }  # 填多个材料可自动并联
    task.set_materials(material)

    # 4 定义地震动
    task.select_ground_motions([f'th{i}'for i in range(1, 8)], '.th')
    task.scale_ground_motions('j', (1, 2, 2), plot=False)

    # 5 导出模型
    task.generate_models()

