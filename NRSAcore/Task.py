import sys
import json
import itertools
import string
import random
from pathlib import Path
from typing import Literal, Callable
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

import numpy as np
import pandas as pd
import dill as pickle
from SeismicUtils.Records import Records

from NRSAcore.Records import Records
from utils.utils import Task_Error, LOGGER
from utils import utils


class Task:
    """
    生成SDOF分析任务  
    关键实例属性：  
    (1) self.paras = {参数名: (参数值, 参数类型)}  
    参数名: str  
    参数值：int | float | list  
    参数类型: Literal[1, 2, 3], 1-常数型, 2-独立参数, 3-从属参数  
    (2) self.task_info, 用于生成json文件的记录所有SDOF模型信息的字典  
    (3) self.independent_paras = list[str], 所有独立参数的参数名  
    (4) self.dependent_paras = dict[参数名, list[映射函数, *独立参数名]]  
    (5) self.constant_paras = list[str], 所有常数型参数的参数名  
    """
    def __init__(self, task_name: str, working_directory: str | Path):
        """创建一个分析任务

        Args:
            task_name (str): 任务名称
            working_directory (str | Path): 工作路径文件夹
        """
        working_directory = Path(working_directory)
        self.task_name = task_name
        characters = string.ascii_letters + string.digits  # 随机生成32位校验码
        self.verification_code = ''.join(random.choice(characters) for _ in range(32))
        self.wkd = working_directory
        self.logger = LOGGER
        self.logger.success('欢迎使用非线性反应谱分析程序')
        utils.creat_folder(working_directory, 'overwrite')
        # 所有用到的模型参数（1-常数，2-独立参数，3-从属参数）
        self.paras: dict[str, tuple[int | float | list, Literal[1, 2, 3]]] = {}
        self.GM_N = 0
        self.GM_names = []  # 地震动名
        self.GM_dts = []  # 地震动步长
        self.GM_SF = []  # 缩放系数
        self.GM_PGA = []  # 原始地震动的PGA
        self.GM_PGV = []
        self.GM_PGD = []
        self.independent_paras = []  # 独立参数
        self.dependent_paras: dict[str, list[Callable, str]] = {}  # 从属参数
        self.constant_paras = []  # 常数型参数
        self.task_info = {
            'model_name': self.task_name,
            'verification_code': self.verification_code,  # 32位校验码
            'para_name': [],  # 所有参数的名称
            'para_values': {},  # 所有参数的值 {参数名: 参数值}
            'constant': [],  # 常数型参数名
            'independent_para': [],  # 独立参数名
            'dependent_para': [],  # 从属参数名
            'material_format': {},  # 材料格式
            'basic_para': {
                'period': None,  # 周期
                'damping': None,  # 阻尼
                'mass': None,  # 质量
                'gravity': None,  # 竖向荷载（可选）
                'height': None,  # 等效SDOF高度（可选）
                'yield_disp': None,  # 屈服位移（可选）
                'material_paras': [],  # 定义材料所需的参数
            },  # 定义SDOF模型所需的直接参数名
            'ground_motions': {},
            'N_SDOF': None,  # 所有参数所有组合情况，依次为独立参数，常数型参数，从属参数
            'total_calculation': None
        }
    
    def _get_values(self, name: str) -> float | int | np.ndarray | list:
        """根据参数名获取参数值"""
        return self.paras[name][0]

    def _get_type(self, name: str) -> Literal[1, 2, 3]:
        """根据参数名获取参数类型"""
        return self.paras[name][1]

    def add_constant(self, name: str, value: int | float):
        """设置常数型参数（如阻尼比等）

        Args:
            name (str): 参数名
            value (int | float): 参数值
        """
        if not isinstance(name, str):
            raise Task_Error(f'参数名的类型只能为str（{type(name)}）')
        if not isinstance(value, (int, float)):
            raise Task_Error(f'常数型参数值只能为int, float类型之一（{type(value)}）')
        if name in self.paras:
            self.logger.warning(f'参数 {name} 已存在，将覆盖')
        self.paras[name] = (value, 1)
        self.constant_paras.append(name)
        self.logger.info(f'添加常数型参数 {name} = {value}')

    def add_independent_parameter(self, name: str, value: list | np.ndarray):
        """设置独立参数（如周期等）

        Args:
            name (str): 参数名
            value (list | np.ndarray): 参数值
        """
        if not isinstance(name, str):
            raise Task_Error(f'参数名的类型只能为str（{type(name)}）')
        if not isinstance(value, (list, np.ndarray)):
            raise Task_Error(f'独立参数值只能为list | np.ndarray类型之一（{type(value)}）')
        if name in self.paras:
            self.logger.warning(f'参数 {name} 已存在，将覆盖')
        self.paras[name] = (value, 2)
        self.independent_paras.append(name)
        self.logger.info(f'添加独立参数 {name} = {value}')
    
    def add_dependent_parameter(self, name: str, func: Callable, *independent_paras: str):
        """设置从属参数

        Args:
            name (str): 参数名
            func (Callable): 映射函数(输入独立参数，输出该从属参数)
            independent_paras (str): 该参数所从属的独立参数名，可为多个，不可为空
        """
        if not isinstance(name, str):
            raise Task_Error(f'参数名的类型只能为str（{type(name)}）')
        if len(independent_paras) == 0:
            raise Task_Error(f'参数 independent_paras 未指定')
        for item in independent_paras:
            if not isinstance(item, str):
                raise Task_Error(f'参数 independent_paras 中所有元素的类型都只能为str（{type(item)}）')
        if name in self.paras:
            self.logger.warning(f'参数 {name} 已存在，将覆盖')
        self.paras[name] = (None, 3)
        self.dependent_paras[name] = [func, *independent_paras]
        self.logger.info(f'添加从属参数 {name} = func{independent_paras}')

    def define_basic_parameters(self,
            period: str,
            mass: str,
            damping: str,
            gravity: str=None,
            height: str=None,
            yield_disp: str=None,
            yield_strength: str=None,
            collapse_disp: str=None,
            maxAnaDisp: str=None,
        ):
        """定义一些SDOF模型的基本参数，即运行SDOF需要的直接参数

        Args:
            period (str): 周期
            mass (str): 质量
            damping (str): 阻尼比
            gravity (str, optional): 竖向荷载
            height (str, optional): 等效SDOF的高度（用于考虑P-Delta）
            yield_disp (str, optional): 屈服位移（用于计算累积塑性位移）
            yield_strength (str, optional): 屈服强度（用于不断调整以计算等延性谱）
            collapse_disp (float, optional): 倒塌判定位移
            maxAnaDisp (float, optional): 最大分析位移
        """
        for item in [period, mass, damping, gravity, height, yield_disp]:
            if (item is not None) and (item not in self.paras):
                raise Task_Error(f'参数 {item} 未定义！')
            if (item is not None) and (not isinstance(item, str)):
                raise Task_Error(f'参数 {item} 应为str类型（{type(item)}）')
        self.period = period
        self.mass = mass
        self.damping = damping
        self.gravity = gravity
        self.height = height
        self.yield_disp = yield_disp
        self.yield_strength = yield_strength
        self.collapse_disp = collapse_disp
        self.maxAnaDisp = maxAnaDisp
        self.logger.success(f'已定义结构周期，共 {len(self._get_values(period))} 种')

    def set_materials(self, materials: dict[str, tuple[str | int, float, list, np.ndarray]]):
        """设置模型材料

        Args:
            materials (dict[str, tuple[str | int, float, list, np.ndarray]]): 材料定义格式（不包括材料编号）

        Examples:
            >>> mat = {
                'Steel01': (Fy, k, alpha),
                'Elastic': E
            }
            >>> set_materials(mat)
            其中`Fy`、`k`、`alpha`、`E`均可为ModelParameter对象或float、str类型。
            当设有多种材料时，将自动并联
        """
        identified_paras = set()
        for key, val in materials.items():
            for para in val:
                res = self.identify_para(para)
                if isinstance(para, str) and res:
                    identified_paras.add(res)
                    if res not in self.paras:
                        raise Task_Error(f'参数 {res} 未定义')
        self.logger.success(f'已识别参数: {identified_paras}')
        self.materials = materials
        self.material_paras = list(identified_paras)  # 定义SDOF材料需要用到的参数

    @staticmethod
    def identify_para(para: str) -> str | None:
        """识别参数是否存在引用"""
        if not isinstance(para, str):
            return None
        if len(para) <= 3:
            return None
        if para[: 3] == '$$$':
            return para[3:]

    @staticmethod
    def Sa(T: np.ndarray, S: np.ndarray, T0: float, withIdx=False) -> float:
        for i in range(len(T) - 1):
            if T[i] <= T0 <= T[i+1]:
                k = (S[i+1] - S[i]) / (T[i+1] - T[i])
                S0 = S[i] + k * (T0 - T[i])
                if withIdx:
                    return S0, i
                else:
                    return S0
        else:
            raise ValueError(f'无法找到周期点{T0}对应的加速度谱值！')

    @staticmethod
    def RMSE(a: np.ndarray, b: np.ndarray) -> float:
        # 均方根误差
        return np.sqrt(np.mean((a - b) ** 2))
    
    @staticmethod
    def geometric_mean(data):  # 计算几何平均数
        total = 1
        n = len(data)
        for i in data:
            total *= pow(i, 1 / n)
        return total

    def records_from_pickle(self,
        pkl_file: Path | str,
        ) -> None:
        """从pickle导入并缩放地震动，其中pickle文件从GroungMotons项目获得

        Args:
            pkl_file (Path | str): pickle文件路径
        """
        if not Path(pkl_file).exists():
            raise FileNotFoundError(f'无法找到文件：{str(Path(pkl_file).absolute())}')
        with open(pkl_file, 'rb') as f:
            sys.path.append(Path(__file__).parent.as_posix())
            records: Records = pickle.load(f)
            del sys.path[-1]
        self.GM_N = records.N_gm
        self.GM_names = []
        self.task_info['ground_motions']['number'] = self.GM_N
        self.logger.success(f'已从{Path(pkl_file).name}导入{self.GM_N}条地震动')

    def generate_models(self) -> dict:
        """生成所有SDOF模型的参数，共生成2个文件，分别为：
        * {model_name}.json: 记录了模型概括、参数取值概括、地震动步长等信息
        * {model_name}.csv: 记录每个SDOF模型所包含的所有参数的详细取值
        """
        self._set_task_info()  # 写入task_info
        self.logger.info(f'正在写入：{self.task_name}.json')
        with open(self.wkd / f'{self.task_name}.json', 'w') as f:
            json.dump(self.task_info, f, indent=4)
        self.logger.success('已生成: ' + str((self.wkd / f'{self.task_name}.json').absolute()))
        self.all_values.to_csv(self.wkd / f'{self.task_name}.csv', index=False)
        self.logger.success(f'共生成 {self.N_SDOF} 个SDOF模型')
        self.logger.success(f'共需进行 {self.N_calc} 次计算')

    def _set_task_info(self):
        """定义self.task_info"""
        for name, (value, type_) in self.paras.items():
            if isinstance(value, np.ndarray):
                value = list(value)
            self.task_info['para_name'].append(name)  # 所有参数名
            self.task_info['para_values'][name] = value  # 所有参数的取值
            if type_ == 1:
                self.task_info['constant'].append(name)  # 常数型参数
            elif type_ == 2:
                self.task_info['independent_para'].append(name)  # 独立参数
            elif type_ == 3:
                self.task_info['dependent_para'].append(name)  # 从属参数
            else:
                raise Task_Error(f'未知type: {type_}')
        for matType, paras in self.materials.items():
            self.task_info['material_format'][matType] = list(paras)  # 材料格式
        # 基本模型参数
        self.task_info['basic_para']['period'] = self.period
        self.task_info['basic_para']['damping'] = self.damping
        self.task_info['basic_para']['mass'] = self.mass
        self.task_info['basic_para']['gravity'] = self.gravity
        self.task_info['basic_para']['height'] = self.height
        self.task_info['basic_para']['yield_disp'] = self.yield_disp
        self.task_info['basic_para']['yield_strength'] = self.yield_strength
        self.task_info['basic_para']['material_paras'] = self.material_paras
        self.task_info['basic_para']['collapse_disp'] = self.collapse_disp
        self.task_info['basic_para']['maxAnaDisp'] = self.maxAnaDisp
        # 生成参数组合
        comb_values: list[float] = []  # 组合前每种参数的取值
        comb_names: list[str] = ['ID']  # 组合前各个参数的名称
        N_SDOF = 1
        N_calc = 1
        # 1 地震动
        ls_gmidx = [i + 1 for i in range(self.GM_N)]
        N_calc *= len(ls_gmidx)
        comb_values.append(ls_gmidx)
        comb_names.append('ground_motion')
        # 2 独立参数
        for name in self.independent_paras:
            N_SDOF *= len(self._get_values(name))
            N_calc *= len(self._get_values(name))
            comb_values.append(self._get_values(name))
            comb_names.append(name)
        # 将地震动和独立参数进行组合
        values = np.zeros((N_calc, len(comb_names)))
        res = itertools.product(*comb_values)
        for i, line in enumerate(res):
            values[i] = np.array([[0] + list(line)])
        values[:, 0] = [i + 1 for i in range(len(values))]
        values = pd.DataFrame(values, columns=comb_names)
        values['ground_motion'] = values['ground_motion'].astype(int)
        values['ID'] = values['ID'].astype(int)
        # 3 常数参数
        for name in self.constant_paras:
            values[name] = self._get_values(name)
        # 4 从属参数
        for name, (func, *idpd_names) in self.dependent_paras.items():
            for idpd_name in idpd_names:
                if not idpd_name in values.columns:
                    raise Task_Error(f'未找到从属参数 {name} 所依赖的参数 {idpd_name} 的值')
            idpd_values = [values[idpd_name] for idpd_name in idpd_names]  # 独立参数的值 list[Series]
            # 计算从属参数的值
            dpd_values = func(*idpd_values)
            values[name] = dpd_values
        # 完成
        self.N_SDOF = N_SDOF
        self.N_calc = N_calc
        self.task_info['N_SDOF'] = self.N_SDOF
        self.task_info['total_calculation'] = self.N_calc
        self.all_values = values

