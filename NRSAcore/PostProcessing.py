import sys
import json
from math import isclose
from pathlib import Path
from typing import Callable, Literal
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import utils


VAR = str | tuple[str, Callable, str, str]

class PostProcessing:
    """读取结果文件，进行后处理
    可读取的响应类型包括：
    (1) xxx_overview.json文件中：
    * para_name字段中的变量
    (2) xxx_result.h5文件中：
    * converge: 是否计算收敛(1或0)
    * collapse: SDOF是否倒塌(1或0)
    * maxDisp: 最大相对位移
    * maxVel: 最大绝对速度
    * maxAccel: 最大觉得加速度
    * Ec: 累积弹塑性耗能
    * Ev: 累积Rayleigh阻尼耗能
    * maxReaction: 最大底部剪力
    * CD: 累积位移
    * CPD: 累积塑性位移(需在定义模型时指定屈服位移)
    * resDisp: 残余位移
    (3) 与弹性反应谱相关的变量:
    * Fe: 最大基底剪力
    * ue: 最大位移
    * ae: 最大加速度
    * ve: 最大速度
    * Ee: 弹性应变能
    * PGA: 峰值加速度
    * PGV: 峰值速度
    * PGD: 峰值位移
    """
    g = 9800

    def __init__(self, model_name: str, working_directory: str | Path) -> None:
        """后处理器，初始化时应代入h5结果文件和json模型文件

        Args:
            model_name (str): 模型名称
            working_directory (str | Path): 工作路径文件夹
        """
        self.logger = utils.LOGGER
        self.model_name = model_name
        self.wkd = Path(working_directory)
        utils.check_file_exists(self.wkd / f'{model_name}_overview.json')
        utils.check_file_exists(self.wkd / f'{model_name}_paras.h5')
        utils.check_file_exists(self.wkd / f'{model_name}_spectra.h5')
        utils.check_file_exists(self.wkd / f'{model_name}_results.h5')
        self.available_paras = {
            'model': [],
            'result': [],
            'spec': [],
            'both': []
        }  # 绘制非线性反应谱曲线时可用的参数
        self._read_files()
        self._get_available_paras()
        self.curves: list[utils.Curve] = []  # 生成的曲线


    def _read_files(self):
        """读取4个文件，生成以下四个主要实例属性：
        * model_overview (dict)
        * model_paras (Dataframe)
        * model_spectra (Dataframe)
        * model_result (Dataframe)
        以及下面的实例属性：
        * N_SDOFs (int): SDOF模型的总数量
        * GM_N (int): 地震动数量
        * GM_names (list[str]): 地震动名称
        """
        self.logger.info('正在读取数据')
        with open(self.wkd / f'{self.model_name}_overview.json', 'r') as f:
            self.model_overview: dict = json.load(f)
            self.N_SDOFs = self.model_overview['N_SDOF']  # SDOF模型的数量
        with h5py.File(self.wkd / f'{self.model_name}_paras.h5', 'r') as f:
            columns = utils.decode_list(f['columns'][:])
            paras = f['parameters'][:]
            self.model_paras = pd.DataFrame(paras, columns=columns)
            self.model_paras['ID'] = self.model_paras['ID'].astype(int)
            self.model_paras['ground_motion'] = self.model_paras['ground_motion'].astype(int)
        with h5py.File(self.wkd / f'{self.model_name}_spectra.h5', 'r') as f:
            self.model_spectra = {}
            GM_N = 0
            GM_names = []
            for item in f:
                if item == 'T':
                    self.model_spectra['T'] = f['T'][:]
                else:
                    self.model_spectra[item] = {}
                    self.model_spectra[item]['RSA'] = f[item]['RSA'][:] * self.g
                    self.model_spectra[item]['RSV'] = f[item]['RSV'][:]
                    self.model_spectra[item]['RSD'] = f[item]['RSD'][:]
                    GM_N += 1
                    GM_names.append(item)
            self.GM_N = GM_N
            self.GM_names = GM_names
        with h5py.File(self.wkd / f'{self.model_name}_results.h5', 'r') as f:
            response_type = utils.decode_list(f['response_type'][:])
            data = np.zeros((self.N_SDOFs, len(response_type) + 1))
            data[:, 0] = [i + 1 for i in range(self.N_SDOFs)]
            for i in range(self.N_SDOFs):
                data[i, 1:] = f[str(i + 1)][:]
            self.model_results = pd.DataFrame(data, columns=['ID']+response_type)
            self.model_results['ID'] = self.model_results['ID'].astype(int)
        self.logger.success('读取数据完成')


    def show_files(self):
        """展示已读取的文件的部分内容"""
        print('(1) model_overview')
        print('keys:', list(self.model_overview.keys()))
        print('(2) model_paras')
        print(self.model_paras.head())
        print('(3) model_spectra')
        print(f"keys: ['T', {self.GM_names[0]}, {self.GM_names[1]}, ...]")
        print('(4) model_results')
        print(self.model_results.head())


    def _get_available_paras(self):
        """根据读取的文件，生成可用的参数"""
        # 来自于模型定义
        self.available_paras['model'] = self.model_overview['para_name']
        # 来自于计算结果(即响应类型)
        self.available_paras['result'] = self.model_results.columns.to_list()[1:]
        # 来自于反应谱(即弹性结构响应)
        self.available_paras['spec'] = ['Fe', 'ue', 'ae', 've', 'Ee', 'PGA', 'PGV', 'PGD']
        # 来自于上述所有来源
        self.available_paras['both'] = self.available_paras['model'] + self.available_paras['result'] + self.available_paras['spec']


    def generatte_curve(self,
        name: str,
        var_x: VAR,
        var_y: VAR,
        *conditions: tuple[VAR, float | int],
        ) -> utils.Curve:
        """提取非线性反应谱分析结果

        Args:
            name (str): 任取一个曲线名（跟之前的不能重复）
            var_x (VAR): 横坐标变量名
            var_y (VAR): 纵坐标变量名
            condition (tuple[VAR, float | int]): 约束条件
            两个变量的相对误差值若小于该值则可认为相等

        Returns (utils.Curve): 返回一个Curve对象

        Example:
            >>> extract_result('T', 'maxDisp', ('Cy', 0.5))  # (1)
            >>> miu = lamda x, y: x / y
            >>> extract_result('T', (miu, 'maxDisp', 'uy'), ('Cy', 0.5))  # (2)
            其中:
            (1) 以模型文件中'T'为横坐标，结果文件中'maxDisp'为纵坐标，
            模型文件中'Cy=0.5'为约束条件，计算非线性反应谱曲线。
            (2) 以'T'为横坐标，结果文件中以'maxDisp'和'uy'作为参数输入至`miu`函数，并以返回值作为纵坐标，以'Cy=0.5'作为约束条件绘制非线性反应谱曲线。
            注：变量名应在json模型文件的'para_name'或结果文件的'response_type'中定义
        """
        self._var_isexists(var_x, 'both')
        self._var_isexists(var_y, 'both')
        for condition in conditions:
            self._var_isexists(condition[0], 'both')
        ls_n = [i + 1 for i in range(self.N_SDOFs)]  # 所有SDOF模型的编号
        available_ls_n: list[int] = []  # 符合约束条件的SDOF模型编号
        for n in ls_n:
            for condition in conditions:
                para_name, value = condition
                if not isclose(value, self._get_value(para_name, n, 'both')):
                    break
            else:
                available_ls_n.append(n)
        # 横坐标值
        x_values = np.zeros(len(available_ls_n))
        if isinstance(var_x, str):
            # var_x是一个直接给出的变量
            x_name = var_x
            x_values = self._get_value(x_name, available_ls_n, 'both')
        else:
            # var_x需要通过其他参数计算得到
            x_name, func = var_x[: 2]  # 变量名和函数
            other_vars_names = var_x[2:]  # 其他参数的名称
            other_vars_values: list[np.ndarray] = []
            for var_name in other_vars_names:
                other_vars_values.append(self._get_value(var_name, available_ls_n, 'both'))
            x_values = func(*other_vars_values) 
        # 纵坐标值
        y_values = np.zeros(len(available_ls_n))
        if isinstance(var_y, str):
            # var_y是一个直接给出的变量
            y_name = var_y
            y_values = self._get_value(y_name, available_ls_n, 'both')
        else:
            # var_y需要通过其他参数计算得到
            y_name, func = var_y[: 2]  # 变量名和函数
            other_vars_names = var_y[2:]  # 其他参数的名称
            other_vars_values: list[np.ndarray] = []
            for var_name in other_vars_names:
                other_vars_values.append(self._get_value(var_name, available_ls_n, 'both'))
            y_values = func(*other_vars_values)
        x_values, y_values = self._zip_values(x_values, y_values, available_ls_n, f'{y_name}-{x_name}')
        # 实例化曲线
        label_ls = []
        for condition in conditions:
            label_ls.append(f'{condition[0]}={condition[1]}')
        label = ', '.join(label_ls)
        curve = utils.Curve(name, x_values, y_values, x_name, y_name, label, self.wkd, self.GM_names)
        return curve


    def _zip_values(self,
            x_values: np.ndarray,
            y_values: np.ndarray,
            available_ls_n: list[int],
            name: str=None,
            tol: float=1e-6
        ) -> tuple[np.ndarray, np.ndarray]:
        """压缩横坐标的数值。压缩前横坐标与纵坐标均为一维数组且的数值长度一样，】
        需根据地震动名称，将相同地震动对应的横坐标取平均值。\n
        输入：
            x_values.shape: (n * GM_N,)
            x_values.shape: (n * GM_N,)
        输出：
            x_values.shape: (n,)
            x_values.shape: (GM_N, n) 
        """
        if not len(x_values) == len(y_values) == len(available_ls_n):
            raise utils.SDOF_Error(f'x_values, y_values, available_ls_n的长度不一致({len(x_values)}, {len(y_values)}, {len(available_ls_n)})')
        all_gm_idx = self.model_paras[self.model_paras['ID'].isin(available_ls_n)]['ground_motion'].to_numpy()
        separated_x_values = []  # 按照地震动分离的横坐标值
        separated_y_values = []  # 按照地震动分离的横坐标值
        for gm_idx in list(set(all_gm_idx)):
            separated_x_values.append(x_values[all_gm_idx==gm_idx])
            separated_y_values.append(y_values[all_gm_idx==gm_idx])
        sum_std = sum(np.std(separated_x_values, 0))
        if sum_std > tol:
            self.logger.warning(f'曲线 {name} 不同地震动下横坐标数值不同，将不进行横坐标压缩')
        else:
            x_values = np.mean(separated_x_values, 0)
            y_values = np.array(separated_y_values)
        return x_values, y_values


    def _var_isexists(self, var: VAR, source: Literal['model', 'result', 'spec', 'both'], other_names: list=[]):
        """检查变量是否在模型文件或结果文件中定义"""
        if isinstance(var, str):
            if var in self.available_paras[source]:
                return
            if var in other_names:
                return
        elif isinstance(var, tuple):
            var = var[2:]
            for vari in var:
                self._var_isexists(vari, source)
            else:
                return
        else:
            raise utils.SDOF_Error(f'变量 {var} 应为 str 或 tuple[str, ...]类型')
        raise utils.SDOF_Error(f'无法找到变量：{var}')
        

    def _get_value(self,
            name: str,
            n: int | list[int],
            source: Literal['model', 'result', 'spec', 'both']
        ) -> float | np.ndarray:
        """获取变量的值，如果n是int，则返回一个数，如果n是list，则返回一个ndarray"""
        if not isinstance(n, (int, list)):
            raise utils.SDOF_Error(f'参数 n 应为int或list类型：{type(n)}')
        if source == 'model':
            if isinstance(n, int):
                value = self.model_paras[self.model_paras['ID']==n][name].item()
            elif isinstance(n, list):
                value = np.zeros(len(n))
                for i, n_ in enumerate(n):
                    value[i] = self.model_paras[self.model_paras['ID']==n_][name].item()
        elif source == 'result':
            if isinstance(n, int):
                value = self.model_results[self.model_results['ID']==n][name].item()
            elif isinstance(n, list):
                value = np.zeros(len(n))
                for i, n_ in enumerate(n):
                    value[i] = self.model_results[self.model_results['ID']==n_][name].item()
        elif source == 'spec':
            period_name = self.model_overview['basic_para']['period']
            mass_name = self.model_overview['basic_para']['mass']
            if isinstance(n, int):
                T_value = self._get_value(period_name, n, 'model')  # 模型定义的周期
                mass_value = self._get_value(mass_name, n, 'model')  # 模型定义的质量
                gm_idx: int = self.model_paras[self.model_paras['ID']==n]['ground_motion'].item()
                gm_name: str = self.model_overview['ground_motions']['name_dt_SF'][str(gm_idx)][0]
                RSA_spec = self.model_spectra[gm_name]['RSA']  # 地震动反应谱
                RSV_spec = self.model_spectra[gm_name]['RSV']
                RSD_spec = self.model_spectra[gm_name]['RSD']
                ae = utils.get_y(self.model_spectra['T'], RSA_spec, T_value)
                ve = utils.get_y(self.model_spectra['T'], RSV_spec, T_value)
                ue = utils.get_y(self.model_spectra['T'], RSD_spec, T_value)
                PGA = self.model_overview['ground_motions']['name_PGAVD'][str(gm_idx)][0]
                PGV = self.model_overview['ground_motions']['name_PGAVD'][str(gm_idx)][1]
                PGD = self.model_overview['ground_motions']['name_PGAVD'][str(gm_idx)][2]
                Fe = ae * mass_value
                Ee = 0.5 * ue * Fe
                d_value = {'ae': ae, 've': ve, 'ue': ue, 'Fe': Fe, 'Ee': Ee,
                           'PGA': PGA, 'PGV': PGV, 'PGD': PGD}
                value = d_value[name]
            elif isinstance(n, list):
                ae = np.zeros(len(n))
                ve = np.zeros(len(n))
                ue = np.zeros(len(n))
                Fe = np.zeros(len(n))
                Ee = np.zeros(len(n))
                PGA = np.zeros(len(n))
                PGV = np.zeros(len(n))
                PGD = np.zeros(len(n))
                for i, n_ in enumerate(n):
                    T_value = self._get_value(period_name, n_, 'model')  # 模型定义的周期
                    mass_value = self._get_value(mass_name, n_, 'model')  # 模型定义的质量
                    gm_idx: int = self.model_paras[self.model_paras['ID']==n_]['ground_motion'].item()
                    gm_name: str = self.model_overview['ground_motions']['name_dt_SF'][str(gm_idx)][0]
                    RSA_spec = self.model_spectra[gm_name]['RSA']  # 地震动反应谱
                    RSV_spec = self.model_spectra[gm_name]['RSV']
                    RSD_spec = self.model_spectra[gm_name]['RSD']
                    ae[i] = utils.get_y(self.model_spectra['T'], RSA_spec, T_value)
                    ve[i] = utils.get_y(self.model_spectra['T'], RSV_spec, T_value)
                    ue[i] = utils.get_y(self.model_spectra['T'], RSD_spec, T_value)
                    PGA[i] = self.model_overview['ground_motions']['name_PGAVD'][str(gm_idx)][0]
                    PGV[i] = self.model_overview['ground_motions']['name_PGAVD'][str(gm_idx)][1]
                    PGD[i] = self.model_overview['ground_motions']['name_PGAVD'][str(gm_idx)][2]
                Fe = ae * mass_value
                Ee = 0.5 * ue * Fe
                d_value = {'ae': ae, 've': ve, 'ue': ue, 'Fe': Fe, 'Ee': Ee,
                           'PGA': PGA, 'PGV': PGV, 'PGD': PGD}
                value = d_value[name]
        elif source == 'both':
            for src in ['model', 'result', 'spec']:
                try:
                    value = self._get_value(name, n, src)
                    return value
                except:
                    pass
            raise utils.SDOF_Error(f'无法找到变量 {name} 的值')
        else:
            raise utils.SDOF_Error('Error - 2')
        return value


    @staticmethod
    def _isequal(x: float | int, y: float | int, tol: float):
        """判断两个变量是否相等"""
        return abs(x - y) <= tol
    






if __name__ == "__main__":
    results = PostProcessing('LCF', r'G:\LCFwkd')
    # results.show_files()
    # 应能运行ndarray类型的计算
    miu = lambda maxDisp, uy: maxDisp / uy
    R = lambda Fe, Fy: Fe / Fy
    # T = lambda T, uy: T / uy
    cylce = lambda CD, maxDisp: CD / maxDisp / 4
    E_total = lambda Ec, Ev: Ec + Ev
    # curve = results.generatte_curve('curve_test', 'T', ('miu', miu, 'maxDisp', 'uy'), ('Cy', 0.1), ('alpha', 0.02), ('zeta', 0.05))
    # curve = results.generatte_curve('curve_test', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.05), ('alpha', 0.02), ('zeta', 0.02))
    # curve = results.generatte_curve('curve_test', 'T', ('R', R, 'Fe', 'Fy'), ('Cy', 0.1), ('alpha', 0.02), ('zeta', 0.05))
    # curve = results.generatte_curve('curve_test', 'T', 'Ev', ('Cy', 0.1), ('alpha', 0.02), ('zeta', 0.05))
    # curve = results.generatte_curve('curve_test', 'T', ('E_total', E_total, 'Ec', 'Ev'), ('Cy', 0.1), ('alpha', 0.02), ('zeta', 0.05))
    curve = results.generatte_curve('curve_test', 'T', 'PGV', ('Cy', 0.1), ('alpha', 0.02), ('zeta', 0.05))
    # curve1 = results.generatte_curve('TCycle_Cy0.05', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.05), ('alpha', 0.02), ('zeta', 0.05))
    # curve2 = results.generatte_curve('TCycle_Cy0.1', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.1), ('alpha', 0.02), ('zeta', 0.05))
    # curve3 = results.generatte_curve('TCycle_Cy0.2', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0.02), ('zeta', 0.05))
    # curve4 = results.generatte_curve('TCycle_Cy0.4', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.4), ('alpha', 0.02), ('zeta', 0.05))
    # curve5 = results.generatte_curve('TCycle_Cy0.6', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.6), ('alpha', 0.02), ('zeta', 0.05))
    # curve6 = results.generatte_curve('TCycle_Cy0.8', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.8), ('alpha', 0.02), ('zeta', 0.05))
    # curve7 = results.generatte_curve('TCycle_a0', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0), ('zeta', 0.05))
    # curve8 = results.generatte_curve('TCycle_a0.05', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0.05), ('zeta', 0.05))
    # curve9 = results.generatte_curve('TCycle_a0.1', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0.1), ('zeta', 0.05))
    # curve10 = results.generatte_curve('TCycle_a0.2', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0.2), ('zeta', 0.05))
    # curve11 = results.generatte_curve('TCycle_zeta0.02', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0.02), ('zeta', 0.02))
    # curve12 = results.generatte_curve('TCycle_zeta0.03', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0.02), ('zeta', 0.03))
    # curve13 = results.generatte_curve('TCycle_zeta0.1', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0.02), ('zeta', 0.1))
    # curve14 = results.generatte_curve('TCycle_zeta0.2', 'T', ('cylce', cylce, 'CD', 'maxDisp'), ('Cy', 0.2), ('alpha', 0.02), ('zeta', 0.2))

    curve.show(True, True)
    # curve1.show(True, False)
    # curve2.show(True, False)
    # curve3.show(True, False)
    # curve4.show(True, False)
    # curve5.show(True, False)
    # curve6.show(True, False)
    # curve7.show(True, False)
    # curve8.show(True, False)
    # curve9.show(True, False)
    # curve10.show(True, False)
    # curve11.show(True, False)
    # curve12.show(True, False)
    # curve13.show(True, False)
    # curve14.show(True, False)


    # results.to_csv(x_vars=['T', ('miu', miu, 'maxDisp', 'uy')], y_vars=[('miu', miu, 'maxDisp', 'uy')])

