from math import pi
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from NRSAcore.Task import Task


if __name__ == "__main__":

    dir_temp = Path('temp')
    
    # 定义模型参数
    g = 9810
    m = 1
    T = np.arange(0.2, 2.2, 0.2)
    Cy = np.array([0.4, 0.8, 1.2])
    Fy = Cy * m * g
    alpha = [0, 0.05, 0.1]
    k = 4 * pi**2 / T**2 * m
    P_norm = 0.8
    P = P_norm * m * g
    zeta = 0.05
    material = {
        'Steel01': ('Fy', 'k', 'alpha')
    }  # 填多个材料可自动并联

    task = Task()
    task.add_parameters('m', m, False)
    task.add_parameters('T', T, False)
    task.add_parameters('Cy', Cy, False)
    task.add_parameters('Fy', Fy, True)
    task.add_parameters('alpha', alpha, True)
    task.add_parameters('k', k, True)
    task.add_parameters('P_norm', P_norm, False)
    task.add_parameters('P', P, False)
    task.add_parameters('zeta', zeta, True)

    task.set_model('T', 'm', 'zeta', 'P')
    task.set_materials(material)
    task.select_ground_motions([f'th{i}'for i in range(1, 8)], '.th')
    task.scale_ground_motions('j', (1, 2, 2), plot=False)
    task.link_dependent_para('T', 'k')
    task.link_dependent_para('m', 'k')
    task.link_dependent_para('Cy', 'Fy')
    task.link_dependent_para('P_norm', 'P')
    task.link_dependent_para('P', 'm')
    task.generate_models(r'C:\Users\Admin\Desktop\NRSA\temp', 'model')
    

    