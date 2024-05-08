import sys
import json
from math import isclose
from pathlib import Path
from typing import Callable, Literal
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils import utils


VAR = str | tuple[str, Callable, str, str]

class PostProcessing:
    """读取结果文件，进行后处理
    可选的响应类型包括：
    (1) h5文件中：
    * converge: 是否计算收敛
    * collapse: SDOF是否倒塌
    * maxDisp: 最大相对位移
    * maxVel: 最大绝对速度
    * maxAccel: 最大觉得加速度
    * Ec: 累积弹塑性耗能
    * Ev: 累积Rayleigh阻尼耗能
    * maxReaction: 最大底部剪力
    * CD: 累积位移
    * CPD: 累积塑性位移(需在定义模型时指定屈服位移)
    * resDisp: 残余位移
    (2) 弹性结构最大响应:
    * Fe: 最大基底剪力
    * ue: 最大位移
    * ae: 最大加速度
    * ve: 最大速度
    * Ee: 弹性应变能
    """
    spectral_response = ['Fe', 'ue', 'ae', 've', 'Ee']

    def __init__(self,
            h5_file: str | Path,
            model_file: str | Path,
            spec_file: str | Path,
            output_dir) -> None:
        """后处理器，初始化时应代入h5结果文件和json模型文件

        Args:
            h5_file (str | Path): .h5结果文件
            model_file (str | Path): .json模型文献
            output_dir (str | Path): 输出文件夹
        """
        self.logger = utils.LOGGER
        h5_file = Path(h5_file)
        spec_file = Path(spec_file)
        h5 = h5py.File(h5_file)
        self.logger.success(f'已导入h5结果文件：{h5_file.name}')
        model_file = Path(model_file)
        with open(model_file, 'r') as f:
            model = json.load(f)
        self.logger.success(f'已导入json模型文件：{h5_file.name}')
        with open(spec_file, 'r') as f:
            spec_data: dict = json.load(f)
        self.logger.success(f'已导入地震动反应谱文件：{spec_file.name}')
        self.output_dir = output_dir
        self.h5 = h5
        self.model = model
        self.spec_data = spec_data
        self.T_spec = np.array(spec_data['T'])
        self.curves: list[utils.Curve] = []
        self.GM_N = len(model['ground_motions']['dt_SF'])
        self.GM_names = list(model['ground_motions']['dt_SF'].keys())


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
        # 挑选符合约束条件的SDOF模型
        available_models: list[str] = list(self.model['SDOF_models'].keys())  # 符合条件的模型序号
        for condition in conditions:
            para_name, value = condition
            for n, paras in self.model['SDOF_models'].items():
                if not isclose(paras[para_name], value) and n in available_models:
                    available_models.remove(n)
        if len(available_models) == 0:
            raise utils.SDOF_Error(f'无法找到符合约束条件的参数值')
        # 横坐标值
        x_values = np.array([])
        if isinstance(var_x, str):
            # var_x是一个直接给出的变量
            x_name = var_x
            for n in available_models:
                x_values = np.append(x_values, self._get_value(x_name, n, 'model'))
        else:
            # var_x需要通过其他参数计算得到
            x_name, func = var_x[: 2]  # 变量名和函数
            for n in available_models:
                other_vars_names = var_x[2:]  # 其他参数的名称
                other_vars_values = []  # 其他参数的值
                for var_name in other_vars_names:
                    other_vars_values.append(self._get_value(var_name, n, 'model'))
                x_values = np.append(x_values, func(*other_vars_values))
        # 纵坐标值
        y_values = np.zeros((self.GM_N, len(x_values)))
        if isinstance(var_y, str):
            # var_y是一个直接给出的变量
            y_name = var_y
            for col, n in enumerate(available_models):
                try:
                    y_values[:, col] = self._get_value(y_name, n, 'both')
                except ValueError:
                    y_values[:, col] = self._get_value(y_name, n, 'both')
        else:
            # var_y需要通过其他参数计算得到
            y_name, func = var_y[: 2]  # 变量名和函数
            for col, n in enumerate(available_models):
                other_vars_names = var_y[2:]
                other_vars_values = []
                for var_name in other_vars_names:
                    try:
                        other_vars_values.append(self._get_value(var_name, n, 'both'))
                    except KeyError:
                        other_vars_values.append(self._get_value(var_name, n, 'both'))
                y_values[:, col] = func(*other_vars_values)
        # 实例化曲线
        label_ls = []
        for condition in conditions:
            label_ls.append(f'{condition[0]}={condition[1]}')
        label = ', '.join(label_ls)
        curve = utils.Curve(name, x_values, y_values, x_name, y_name, label, self.output_dir, self.GM_names)
        return curve


    def export(self):
        pass  # TODO
        self.h5.close()


    def _var_isexists(self, var: VAR, source: Literal['model', 'result', 'spec', 'both']):
        """检查变量是否在模型文件或结果文件中定义"""
        src: dict[str, list] = {
            'model': self.model['para_name'],  # 参数来自.json模型文件
            'result': utils.decode_list(self.h5['response_type']),  # 参数来自.h5结果文件
            'spec': self.spectral_response,  # 参数来自弹性结构反应谱分析结果
            'both': self.model['para_name'] + utils.decode_list(self.h5['response_type']) + self.spectral_response  # 参数来自上述三者
        }
        var_names = src[source]
        if isinstance(var, str):
            if var in var_names:
                return
        elif isinstance(var, tuple):
            var = var[2:]
            for vari in var:
                self._var_isexists(vari, 'both')
            else:
                return
        else:
            raise utils.SDOF_Error(f'变量 {var} 应为 str 或 tuple[str, ...]类型')
        s = {'model': '模型', 'result': '结果', 'both': '模型和结果'}
        raise utils.SDOF_Error(f'无法在{s[source]}文件找到变量：{var}')
        

    def _get_value(self, name: str, n: str, source: Literal['model', 'result', 'spec', 'both'],
            ) -> float | np.ndarray:
        """获取变量的值"""
        if source == 'model':
            value = self.model['SDOF_models'][n][name]
        elif source == 'result':
            value = np.array([])
            idx = utils.decode_list(self.h5['response_type'][:]).index(name)
            for gm_name in self.GM_names:
                value = np.append(value, self.h5[gm_name][n][:][idx])
        elif source == 'spec':
            period_name = self.model['basic_para']['period']
            mass_name = self.model['basic_para']['mass']
            T_value = self._get_value(period_name, n, 'model')  # 模型定义的周期
            mass_value = self._get_value(mass_name, n, 'model')  # 模型定义的质量
            value = []
            for gm_name in self.GM_names:
                RSA_spec = self.spec_data[gm_name]['RSA']  # 地震动反应谱
                RSV_spec = self.spec_data[gm_name]['RSV']
                RSD_spec = self.spec_data[gm_name]['RSD']
                ae = utils.get_y(self.T_spec, RSA_spec, T_value)
                ve = utils.get_y(self.T_spec, RSV_spec, T_value)
                ue = utils.get_y(self.T_spec, RSD_spec, T_value)
                Fe = ae * mass_value
                Ee = 0.5 * ue * Fe
                if name == 'ae':
                    value.append(ae)
                elif name == 've':
                    value.append(ve)
                elif name == 'ue':
                    value.append(ue)
                elif name == 'Fe':
                    value.append(Fe)
                elif name == 'Ee':
                    value.append(Ee)
                else:
                    raise utils.SDOF_Error(f'不支持的输出类型：{name}')
            value = np.array(value)
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
    results = PostProcessing(
        Path(__file__).parent.parent/'Output'/'TestModel.h5',
        Path(__file__).parent.parent/'temp'/'TestModel.json',
        Path(__file__).parent.parent/'temp'/'TestModel_spectra.json',
        Path(__file__).parent.parent/'Output')
    # 应能运行ndarray类型的计算
    miu = lambda maxDisp, uy: maxDisp / uy
    # R = lambda Fe, Fy: Fe / Fy
    # T = lambda T, uy: T / uy
    curve1 = results.generatte_curve('miu-T curve', 'T', ('miu', miu, 'maxDisp', 'uy'), ('Cy', 0.5), ('alpha', 0))
    # curve2 = results.generatte_curve('u-T curve', 'T', 'resDisp', ('Cy', 0.4), ('alpha', 0.05))
    # curve3 = results.generatte_curve('R-T curve', 'T', ('R', R, 'Fe', 'Fy'), ('Cy', 0.4), ('alpha', 0))
    # curve4 = results.generatte_curve('u-T curve', 'T', 'maxDisp', ('Cy', 0.4), ('alpha', 0))
    # curve4 = results.generatte_curve('u-T curve', 'T', 'maxDisp', ('Cy', 0.4), ('alpha', 0))
    curve1.show(True)
    # curve2.show(True)
    # curve3.show()
    # curve4.show(True)
    results.export()

