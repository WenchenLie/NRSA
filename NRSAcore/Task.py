import sys
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
from typing import Literal, Callable
from loguru import logger
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
from NRSAcore.ModelParameter import ModelParameter
from NRSAcore.Spectrum import Spectrum
from utils.utils import Task_Error
from utils import utils


logger.remove()
logger.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <red>|</red> <level>{level}</level> <red>|</red> <level>{message}</level>",
    level="DEBUG"
)


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
    dir_main = Path(__file__).parent.parent
    dir_temp = dir_main / 'temp'
    dir_input = dir_main / 'Input'
    dir_gm = dir_input / 'GMs'


    def __init__(self):
        self.logger = logger
        self.logger.success('欢迎使用非线性反应谱分析程序')
        utils.creat_folder(self.dir_temp, 'overwrite')
        # 所有用到的模型参数（1-常数，2-独立参数，3-从属参数）
        self.paras: dict[str, tuple[int | float | list, Literal[1, 2, 3]]] = {}
        self.GM_N = 0
        self.GM_names = []  # 地震动名
        self.GM_dts = []  # 地震动步长
        self.GM_SF = []  # 缩放系数
        self.independent_paras = []  # 独立参数
        self.dependent_paras: dict[str, list[Callable, str]] = {}  # 从属参数
        self.constant_paras = []  # 常数型参数
        self.task_info = {
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
                'heigth': None,  # 等效SDOF高度（可选）
                'yield_disp': None,  # 屈服位移（可选）
                'material_paras': [],  # 定义材料所需的参数
            },  # 定义SDOF模型所需的直接参数名
            'ground_motions': {
                'suffix': None,  # 地震动后缀
                'dt_SF': {}  # 地震动步长及缩放系数
            },
            'SDOF_models': {}  # 所有参数所有组合情况，依次为独立参数，常数型参数，从属参数
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
        

    def define_model(self,
            period: str,
            mass: str,
            damping: str,
            gravity: str,
            height: str=None,
            yield_disp: str=None,
            ):
        """设置运行SDOF需要的直接参数

        Args:
            period (str): 周期
            mass (str): 质量
            damping (str): 阻尼比
            gravity (str): 竖向荷载
            height (str, optional): 等效SDOF的高度（用于考虑P-Delta）
            yield_disp (str, optional): 屈服位移（用于计算累积塑性位移）
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
    def identify_para(para: str):
        """识别参数是否存在引用"""
        if not isinstance(para, str):
            return
        if len(para) <= 3:
            return None
        if para[: 3] == '$$$':
            return para[3:]


    def select_ground_motions(self, GMs: list[str], suffix: str='.txt'):
        """选择地震动文件

        Args:
            GMs (list[str]): 一个包含所有地震动文件名(不包括后缀)的列表  
            suffix (str, optional): 地震动文件后缀，默认为.txt

        Example:
            >>> select_ground_motions(GMs=['GM1', 'GM2'], suffix='.txt')
        """
        self.suffix = suffix
        self.task_info['ground_motions']['suffix'] = suffix
        self.GM_names = GMs
        with open(self.dir_gm / 'GM_info.json', 'r') as f:
            dt_dict = json.loads(f.read())
        for gm_name in self.GM_names:
            self.GM_dts.append(dt_dict[gm_name])
        self.GM_N = len(self.GM_names)
        self.logger.success(f'已导入 {self.GM_N} 条地震动')


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


    def scale_ground_motions(self,
            method: str, para, path_spec_code: Path=None, SF_code: float=1.0, save_SF=False,
            plot=True, save_unscaled_spec=False, save_scaled_spec=False):
        """缩放地震动

        Args:
            method (str): 地震动的缩放方法，为'a'-'g'：  
            * [a] 按Sa(T=0)匹配反应谱, pare=None  
            * [b] 按Sa(T=Ta)匹配反应谱, para=Ta  
            * [c] 按Sa(Ta) ~ Sa(Tb)匹配反应谱, para=(Ta, Tb)  
            * [d] 指定PGA, para=PGA  
            * [e] 不缩放  
            * [f] 指定相同缩放系数, para=SF  
            * [g] 按文件指定, para=path: str (如'temp/GM_SFs.txt')，文件包含一列n行个数据  
            * [h] 按Sa,avg(T1, T2)匹配反应谱，即T1~T2间的加速度谱值的几何平均数，para=(T1, T2)  
            * [i] 指定Sa(Ta), para=(Ta, Sa)  
            * [j] 指定Sa,avg(Ta~Tb), para=(Ta, Tb, Sa,avg)\n
            分别代表n条地震动的缩放系数  
            para: 地震动缩放所需参数，与`method`的取值有关  
            path_spec_code (Path): 目标谱的文件路径，文件应包含两列数据，为周期和加速度谱值  
            SF_code (float, optional): 读取目标谱时将目标谱乘以一个缩放系数，默认为1  
            save (bool, optional): 是否保存缩放后的缩放系数(将保存至temp文件夹，
            可以作为`method`取'g'时`para`参数对应的文件路径，默认为False  
            plot (bool, optional): 是否绘制缩放后地震动反应谱与目标谱的对比图，默认为True  
            save_unscaled_spec (bool, optional): 是否保存未缩放地震动反应谱，默认False  
            save_scaled_spec (bool, optional): 是否保存缩放后地震动反应谱，默认False
        """
        self.method = method
        self.th_para = para
        if path_spec_code:
            data = np.loadtxt(path_spec_code)
            T = data[:, 0]
            Sa_code = data[:, 1] * SF_code
            Sv_code = Sa_code * T / (2 * pi)
            Sd_code = Sa_code * (T / (2 * pi)) ** 2
        else:
            T = np.arange(0, 6.02, 0.01)
            Sa_code = None
            Sv_code = None
            Sd_code = None
        self.T = T
        self.scaled_GM_RSA = np.zeros((self.GM_N, len(T)))
        self.scaled_GM_RSV = np.zeros((self.GM_N, len(T)))
        self.scaled_GM_RSD = np.zeros((self.GM_N, len(T)))
        if method == 'g':
            SF_path = para
            SFs = np.loadtxt(SF_path)
        self.GM_RSA = np.zeros((self.GM_N, len(T)))
        self.GM_RSV = np.zeros((self.GM_N, len(T)))
        self.GM_RSD = np.zeros((self.GM_N, len(T)))
        is_print = True
        for idx, gm_name in enumerate(self.GM_names):
            print(f'正在缩放地震动...({idx+1}/{self.GM_N})     \r', end='')
            th = np.loadtxt(self.dir_gm / f'{gm_name}{self.suffix}')
            RSA, RSV, RSD = Spectrum(ag=th, dt=self.GM_dts[idx], T=T)  # 计算地震动反应谱
            th = np.loadtxt(self.dir_gm / f'{gm_name}{self.suffix}')
            RSA, RSV, RSD = Spectrum(ag=th, dt=self.GM_dts[idx], T=T)  # 计算地震动反应谱
            self.GM_RSA[idx] = RSA
            self.GM_RSV[idx] = RSV
            self.GM_RSD[idx] = RSD    
            if method == 'a':
                T0 = 0
                SF = self.Sa(T, Sa_code, T0) / self.Sa(T, RSA, T0)
                self.GM_SF.append(SF)
            elif method == 'b':
                T0 = para
                if is_print:
                    self.logger.info(f'Sa(T1) = {self.Sa(T, RSA, T0)}')
                    is_print = False
                SF = self.Sa(T, Sa_code, T0) / self.Sa(T, RSA, T0)
                self.GM_SF.append(SF)
            elif method == 'c':
                T1, T2 = para
                idx1, idx2 = self.Sa(T, RSA, T1, True)[1], self.Sa(T, RSA, T2, True)[1]
                init_SF = 1.0  # 初始缩放系数
                learning_rate = 0.01  # 学习率
                num_iterations = 40000  # 迭代次数
                init_SF = np.mean(Sa_code[idx1: idx2]) / np.mean(RSA[idx1: idx2])
                SF = utils.gradient_descent(RSA[idx1: idx2], Sa_code[idx1: idx2], init_SF, learning_rate, num_iterations)
            elif method == 'd':
                PGA = para
                SF = PGA / max(abs(th))
            elif method == 'e':
                SF = 1
            elif method == 'f':
                SF = para
            elif method == 'g':
                SF = SFs[idx] 
            elif method == 'h':
                Sa_i_code = []
                Sa_i = []
                T1, T2 = para
                for i in range(len(T)):
                    Ti = T[i]
                    if T1 <= Ti <= T2:
                        Sa_i_code.append(Sa_code[i])
                        Sa_i.append(RSA[i])
                Sa_avg_code = self.geometric_mean(Sa_i_code)
                Sa_avg = self.geometric_mean(Sa_i)
                SF = Sa_avg_code / Sa_avg
                if is_print:
                    self.logger.info(f'Sa,avg = {Sa_avg_code}')
                    is_print = False
            elif method == 'i':
                Ta, Sa_target = para
                Sa_gm = self.Sa(T, RSA, Ta)
                SF = Sa_target / Sa_gm
            elif method == 'j':
                Ta, Tb, Sa_target = para
                Sa_gm_avg = self.geometric_mean(RSA[(Ta <= T) & (T <= Tb)])
                SF = Sa_target / Sa_gm_avg
            else:
                self.logger.error('"method"参数错误！')
                raise Task_Error('"method"参数错误！')
            self.scaled_GM_RSA[idx] = RSA * SF
            self.scaled_GM_RSV[idx] = RSV * SF
            self.scaled_GM_RSD[idx] = RSD * SF
            self.GM_SF.append(SF)
            if save_SF:
                np.savetxt(self.dir_temp / 'GM_SFs.txt', self.GM_SF)  # 保存缩放系数
                np.savetxt(self.dir_temp / 'GM_SFs.txt', self.GM_SF)  # 保存缩放系数
        if save_unscaled_spec:
            data_RSA = np.zeros((len(T), self.GM_N + 1))
            data_RSV = np.zeros((len(T), self.GM_N + 1))
            data_RSD = np.zeros((len(T), self.GM_N + 1))
            data_RSA[:, 0] = T
            data_RSV[:, 0] = T
            data_RSD[:, 0] = T
            data_RSA[:, 1:] = self.GM_RSA.T
            data_RSV[:, 1:] = self.GM_RSV.T
            data_RSD[:, 1:] = self.GM_RSD.T
            pct_A, pct_V, pct_D = np.zeros((len(T), 3)), np.zeros((len(T), 3)), np.zeros((len(T), 3))
            for i in range(len(T)):
                pct_A[i, 0] = np.percentile(data_RSA[i, 1:], 16)
                pct_A[i, 1] = np.percentile(data_RSA[i, 1:], 50)
                pct_A[i, 2] = np.percentile(data_RSA[i, 1:], 84)
                pct_V[i, 0] = np.percentile(data_RSV[i, 1:], 16)
                pct_V[i, 1] = np.percentile(data_RSV[i, 1:], 50)
                pct_V[i, 2] = np.percentile(data_RSV[i, 1:], 84)
                pct_D[i, 0] = np.percentile(data_RSD[i, 1:], 16)
                pct_D[i, 1] = np.percentile(data_RSD[i, 1:], 50)
                pct_D[i, 2] = np.percentile(data_RSD[i, 1:], 84)
            np.savetxt(self.dir_temp / 'Unscaled_RSA.txt', data_RSA, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSV.txt', data_RSV, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSD.txt', data_RSD, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSA_pct.txt', pct_A, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSV_pct.txt', pct_V, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSD_pct.txt', pct_D, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSA.txt', data_RSA, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSV.txt', data_RSV, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSD.txt', data_RSD, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSA_pct.txt', pct_A, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSV_pct.txt', pct_V, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Unscaled_RSD_pct.txt', pct_D, fmt='%.5f')
            self.logger.info(f'已保存未缩放反应谱至temp文件夹')
        if save_scaled_spec:
            data_RSA = np.zeros((len(T), self.GM_N + 1))
            data_RSV = np.zeros((len(T), self.GM_N + 1))
            data_RSD = np.zeros((len(T), self.GM_N + 1))
            data_RSA[:, 0] = T
            data_RSV[:, 0] = T
            data_RSD[:, 0] = T
            data_RSA[:, 1:] = self.scaled_GM_RSA.T
            data_RSV[:, 1:] = self.scaled_GM_RSV.T
            data_RSD[:, 1:] = self.scaled_GM_RSD.T
            pct_A, pct_V, pct_D = np.zeros((len(T), 3)), np.zeros((len(T), 3)), np.zeros((len(T), 3))
            for i in range(len(T)):
                pct_A[i, 0] = np.percentile(data_RSA[i, 1:], 16)
                pct_A[i, 1] = np.percentile(data_RSA[i, 1:], 50)
                pct_A[i, 2] = np.percentile(data_RSA[i, 1:], 84)
                pct_V[i, 0] = np.percentile(data_RSV[i, 1:], 16)
                pct_V[i, 1] = np.percentile(data_RSV[i, 1:], 50)
                pct_V[i, 2] = np.percentile(data_RSV[i, 1:], 84)
                pct_D[i, 0] = np.percentile(data_RSD[i, 1:], 16)
                pct_D[i, 1] = np.percentile(data_RSD[i, 1:], 50)
                pct_D[i, 2] = np.percentile(data_RSD[i, 1:], 84)
            np.savetxt(self.dir_temp / 'Scaled_RSA.txt', data_RSA, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSV.txt', data_RSV, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSD.txt', data_RSD, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSA_pct.txt', pct_A, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSV_pct.txt', pct_V, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSD_pct.txt', pct_D, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSA.txt', data_RSA, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSV.txt', data_RSV, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSD.txt', data_RSD, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSA_pct.txt', pct_A, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSV_pct.txt', pct_V, fmt='%.5f')
            np.savetxt(self.dir_temp / 'Scaled_RSD_pct.txt', pct_D, fmt='%.5f')
            self.logger.info(f'已保存缩放反应谱至temp文件夹')
        plt.subplot(131)
        if method == 'a':
            plt.scatter(0, Sa_code[0], color='blue', zorder=99999)
        elif method == 'b':
            plt.scatter(T0, self.Sa(T, Sa_code, T0), color='blue', zorder=99999)
        elif method == 'c':
            plt.scatter(para, [self.Sa(T, Sa_code, para[0]), self.Sa(T, Sa_code, para[1])], color='blue', zorder=99999)
        elif method == 'd':
            plt.scatter(0, PGA, color='blue', zorder=99999)
        for i in range(self.GM_N):
            plt.subplot(131)
            plt.plot(T, self.scaled_GM_RSA[i], color='grey')
            plt.subplot(132)
            plt.plot(T, self.scaled_GM_RSV[i], color='grey')   
            plt.subplot(133)
            plt.plot(T, self.scaled_GM_RSD[i], color='grey')    
        plt.subplot(131)
        if Sa_code is not None:
            plt.plot(T, Sa_code, label='Code', color='red')
            plt.legend()
        plt.xlabel('T [s]')
        plt.ylabel('RSA [g]')
        plt.subplot(132)
        if Sv_code is not None:
            plt.plot(T, Sv_code, label='Code', color='red')
            plt.legend()
        plt.xlabel('T [s]')
        plt.ylabel('RSV [mm/s]')
        plt.subplot(133)
        if Sd_code is not None:
            plt.plot(T, Sd_code, label='Code', color='red')
            plt.legend()
        plt.xlabel('T [s]')
        plt.ylabel('RSD [mm]')
        if plot:
            plt.show()
        else:
            plt.close()
        self.scaling_finished = True


    def generate_models(self, dir_path: Path | str=None, file_name: str=None) -> dict:
        """生成所有SDOF模型的参数，并将SDOF计算任务导出为json或返回一个字典

        Args:
            dir_path (Path | str, optional): json文件的导出路径文件夹，若为None给则不导出
            file_name (str, optional): json文件名

        Returns:
            dict: 保护计算任务信息的字典
        """
        self._set_task_info()  # 写入task_info
        if dir_path and file_name:
            dir_path = Path(dir_path)
            with open(dir_path / f'{file_name}.json', 'w') as f:
                f.write(json.dumps(self.task_info, indent=4))
        self.logger.success(f'共生成 {self.N_SDOF} 个SDOF模型')
        return self.task_info


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
        self.task_info['basic_para']['material_paras'] = self.material_paras
        for gm_name, dt, SF in zip(self.GM_names, self.GM_dts, self.GM_SF):
            self.task_info['ground_motions']['dt_SF'][gm_name] = (dt, SF)
        # 生成参数组合
        # 1 写入独立参数组合
        independent_para_values = []
        for name in self.independent_paras:
            independent_para_values.append(self._get_values(name))
        comb = list(itertools.product(*independent_para_values))
        for i, paras in enumerate(comb):
            self.task_info['SDOF_models'][i + 1] = {}
            for j, name in enumerate(self.independent_paras):
                self.task_info['SDOF_models'][i + 1][name] = paras[j]
        N_SDOF = i + 1  # SDOF的数量
        # 2 写入常数参数
        for i in range(1, N_SDOF + 1):
            for name in self.constant_paras:
                self.task_info['SDOF_models'][i][name] = self._get_values(name)
        # 3 写入从属参数
        for i in range(1, N_SDOF + 1):
            for name, (func, *idpd_names) in self.dependent_paras.items():
                # 获取独立参数的值
                for j in range(len(idpd_names)):
                    if not idpd_names[j] in self.task_info['SDOF_models'][i]:
                        raise Task_Error(f'未找到从属参数 {name} 所依赖的参数 {idpd_names[j]} 的值，请注意从属参数的添加顺序')
                idpd_values = [self.task_info['SDOF_models'][i][idpd_names[j]] for j in range(len(idpd_names))]
                # 计算从属参数的值
                dpd_value = func(*idpd_values)
                self.task_info['SDOF_models'][i][name] = dpd_value
        self.N_SDOF = N_SDOF

