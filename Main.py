from math import pi
from pathlib import Path
import numpy as np
import NRSAcore


def generate_task():

    g = 9800
    task = NRSAcore.Task('LCF', r'G:\LCFwkd')
    
    # 1 定义模型参数
    # (1) 常数型参数
    m = 1
    # P_norm = 0
    # h = 1000
    # (2) 独立参数(通常是无量纲参数)
    T = np.arange(0.2, 6.2, 0.2)
    Cy = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8])
    alpha = [0, 0.02, 0.05, 0.1, 0.2]
    zeta = [0.02, 0.03, 0.05, 0.1, 0.2]
    # Cy = np.array([0.05, 0.1])
    # alpha = [0, 0.02]
    # zeta = [0.02, 0.05]
    # (3) 从属参数(通常直接用于SDOF计算的参数)
    get_Fy = lambda m, Cy: m * g * Cy  # 与m, Cy相关
    get_k = lambda T, m: 4 * pi**2 / T**2 * m  # 与T, m相关
    get_P = lambda P_norm, m: P_norm * m * g  # 与P_norm, m相关
    get_uy = lambda Fy, k: Fy / k  # 与Fy，k相关
    # 设置参数(注意从属参数的先后定义顺序)
    # task.add_constant('zeta', zeta)
    task.add_constant('m', m)
    # task.add_constant('P_norm', P_norm)
    # task.add_constant('h', h)
    task.add_independent_parameter('T', T)
    task.add_independent_parameter('Cy', Cy)
    task.add_independent_parameter('alpha', alpha)
    task.add_independent_parameter('zeta', zeta)
    task.add_dependent_parameter('Fy', get_Fy, 'm', 'Cy')
    task.add_dependent_parameter('k', get_k, 'T', 'm')
    task.add_dependent_parameter('uy', get_uy, 'Fy', 'k')
    # task.add_dependent_parameter('P', get_P, 'P_norm', 'm')
    
    # 2 设置模型基本参数
    task.define_basic_parameters(
        period='T',
        mass='m',
        damping='zeta',
        # gravity='P',
        yield_disp='uy',
        # height='h'
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
    task.records_from_pickle(r'G:\NRSA_working\3046records.records')
    # files = []
    # for file in Path(r'F:\重要数据\小波库\7Records').iterdir():
    # for file in Path(r'F:\重要数据\小波库\3046Records').iterdir():
    #     if file.suffix == '.json':
    #         continue
    #     files.append(file.stem)
    # task.select_ground_motions(files, '.txt')
    # task.scale_ground_motions('e', None, plot=False, spec_from_h5=r'G:\NGAWest2\Spectra.hdf5')
    # task.select_ground_motions([f'th{i}' for i in range(1, 45)], '.th')
    # task.scale_ground_motions('e', None, plot=False)

    # 5 导出模型
    task.generate_models()


def analysis():

    model = NRSAcore.SDOFmodel(
        r'G:\NRSA_working\3046records.records',
        r'G:\LCFwkd\LCF.json',
        r'G:\LCFwkd\LCF.csv',
        r'G:\LCFwkd'
    )
    model.set_analytical_options(
        'constant_strength',
        PDelta=False,
        batch=20,
        auto_quit=False,
        parallel=20,
        save_interval=1200
    )
    model.run()


if __name__ == "__main__":
    
    # generate_task()
    # analysis()
    model = NRSAcore.SDOFmodel.restart(
        r'G:\NRSA_working\3046records.records',
        r'G:\LCFwkd\LCF.json',
        r'G:\LCFwkd\LCF.csv',
        r'G:\LCFwkd',
        r'G:\LCFwkd\LCF.instance'
    )
    model.parallel = 20
    model.run()
    # pass

