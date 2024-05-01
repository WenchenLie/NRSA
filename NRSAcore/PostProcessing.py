import sys
import json
from pathlib import Path
from typing import Callable, Literal
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils import utils


VAR = str | tuple[Callable, str, str]

class PostProcessing:
    """读取结果文件，进行后处理"""

    def __init__(self,
            h5_file: str | Path,
            model_file: str | Path) -> None:
        """后处理器，初始化时应代入h5结果文件和json模型文件

        Args:
            h5_file (str | Path): .h5结果文件
            model_file (str | Path): .json模型文献
        """
        self.logger = utils.LOGGER
        h5_file = Path(h5_file)
        h5 = h5py.File(h5_file)
        self.logger.success(f'已导入h5结果文件：{h5_file.name}')
        model_file = Path(model_file)
        with open(model_file, 'r') as f:
            model = json.load(f)
        self.logger.success(f'已导入json模型文件：{h5_file.name}')
        self.h5 = h5
        self.model = model


    def extract_curve(self,
        name: str,
        var_x: VAR,
        var_y: VAR,
        *conditions: tuple[VAR, float | int],
        tol: float=1e-6):
        """提取非线性反应谱分析结果

        Args:
            name (str): 任取一个曲线名（跟之前的不能重复）
            var_x (VAR): 横坐标变量名
            var_y (VAR): 纵坐标变量名
            condition (tuple[VAR, float | int]): 约束条件
            tol (float, optional): 判断约束条件中的等式时的可接受相对误差，
            两个变量的相对误差值若小于该值则可认为相等

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
        self._var_isexists(var_x, 'model')
        self._var_isexists(var_y, 'result')
        for condition in conditions:
            self._var_isexists(condition[0], 'model')
        # 挑选符合约束条件的SDOF模型
        available_moel: list[str] = []  # 符合条件的模型序号
        for condition in conditions:
            para_name = condition[0]
            for n, paras in self.model['SDOF_models']:
                pass  # TODO



    def _var_isexists(self, var: VAR, source: Literal['model', 'result']):
        """检查变量是否在模型文件或结果文件中定义"""
        src = {
            'model': self.model['para_name'],
            'result': utils.decode_list(self.h5['response_type'])
        }
        var_names = src[source]
        if isinstance(var, str):
            if var in var_names:
                return
        elif isinstance(var, tuple):
            var = var[1:]
            for vari in var:
                self._var_isexists(vari)
            else:
                return
        else:
            raise utils.SDOF_Error(f'变量 {var} 应为 str 或 tuple[str, ...]类型')
        raise utils.SDOF_Error(f'无法在模型文件或结果文件中找到变量：{var}')
        

    def _isequal(x: float | int, y: float | int, tol: float):
        """判断两个变量是否相等"""
        return abs(x - y) / abs(x) <= tol



if __name__ == "__main__":
    results = PostProcessing(
        Path(__file__).parent.parent/'Output'/'model.h5',
        Path(__file__).parent.parent/'temp'/'model.json')
    miu = lambda maxDisp, uy: maxDisp / uy



