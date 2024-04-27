import sys
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
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
    """生成SDOF分析任务
    """
    dir_main = Path(__file__).parent.parent
    dir_temp = dir_main / 'temp'
    dir_input = dir_main / 'Input'
    dir_gm = dir_input / 'GMs'


    def __init__(self):
        self.logger = logger
        utils.creat_folder(self.dir_temp, 'overwrite')
        self.paras: dict[str, tuple[list, bool]] = {}  # 所有用到的模型参数
        self.GM_N = 0
        self.GM_names = []  # 地震动名
        self.GM_dts = []  # 地震动步长
        self.GM_SF = []  # 缩放系数
        self.task_info = {
            'para_name': [],  # 参数名称
            'independent_para': [],  # 独立参数
            'dependent_para': {},  # 从属参数
            'para_values': {},  # 参数范围
            'material_format': {},  # 材料格式
            'ground_motions': {
                'suffix': None,  # 地震动后缀
                'dt_SF': {}  # 地震动步长及缩放系数
            },
            'SDOF_models': {}  # 参数组合情况(顺序与独立参数名称对应)
        }  # 任务信息，记录了所有SDOF模型的建模信息


    def add_parameters(self, name: str, value: int | float | list | np.ndarray, is_independent: bool):
        """设置模型参数

        Args:
            name (str): 模型参数名
            value (int | float | list | np.ndarray): 参数值
            is_independent (bool): 是否是独立参数
        """
        if not isinstance(value, (int, float, list, np.ndarray)):
            raise Task_Error(f'参数值只能为int, float, list, np.ndarray类型之一（{type(value)}）')
        if name in self.paras:
            self.logger.warning(f'参数 {name} 已存在，将覆盖')
        if isinstance(value, (int, float)):
            value = [value]
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        self.paras[name] = (value, is_independent)
        self.task_info['para_name'].append(name)
        self.task_info['para_values'][name] = value
        if is_independent:
            self.task_info['independent_para'].append(name)
            

    def set_model(self,
            T: str,
            m: str,
            zeta: str,
            P: str):
        """设置模型基本参数，包括周期、质量、阻尼比。
        注：周期和质量不是互为独立变量，当周期和质量的数量大于1时，二者数量应相同

        Args:
            T (str): 周期
            m (str): 质量
            zeta (str): 阻尼比
            P (str): 结构重力
        """
        self.T = self.paras[T][0]
        self.m = self.paras[T][0]
        self.zeta = self.paras[zeta][0]
        self.logger.success(f'已定义结构周期与质量，共 {len(T)} 种')


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
        for matType, paras in materials.items():
            paras_ = []
            for para in paras:
                if para in self.paras.keys():
                    # 该参数属于模型参数
                    paras_.append('$$$' + para)
                else:
                    paras_.append(para)
            self.task_info['material_format'][matType] = paras_


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
        for gm_name, dt, SF in zip(self.GM_names, self.GM_dts, self.GM_SF):
            self.task_info['ground_motions']['dt_SF'][gm_name] = (dt, SF)
        self.scaling_finished = True


    def link_dependent_para(self, dependent: str, independent: str):
        """设置从属参数关系（每个从属参数都依赖于一个独立参数）

        Args:
            dependent (str): 从属参数
            independent (str): 从属参数所依赖的独立参数
        """
        if not dependent in self.paras.keys():
            raise Task_Error(f'参数 {dependent} 未定义')
        if not independent in self.paras.keys():
            raise Task_Error(f'参数 {independent} 未定义')
        self.task_info['dependent_para'][dependent] = independent


    def generate_models(self, dir_path: Path | str=None, file_name: str=None) -> dict:
        """生成所有SDOF模型的参数，并将SDOF计算任务导出为json或返回一个字典

        Args:
            dir_path (Path | str, optional): json文件的导出路径文件夹，若为None给则不导出
            file_name (str, optional): json文件名

        Returns:
            dict: 保护计算任务信息的字典
        """
        para_unkonw_independent = set(self.task_info['para_name']) - set(self.task_info['independent_para']) - set(self.task_info['dependent_para'].keys())
        if para_unkonw_independent != set():
            raise Task_Error(f'以下参数未指派是否为独立参数：{para_unkonw_independent}')
        ls = []
        for ind_para in self.task_info['independent_para']:
            ls.append(self.task_info['para_values'][ind_para])
        res = list(itertools.product(*ls))
        for i, paras in enumerate(res):
            self.task_info['SDOF_models'][i + 1] = paras
        if dir_path and file_name:
            dir_path = Path(dir_path)
            with open(dir_path / f'{file_name}.json', 'w') as f:
                f.write(json.dumps(self.task_info, indent=4))
        self.logger.success(f'共生成 {i + 1} 个SDOF模型')
        return self.task_info




if __name__ == "__main__":

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
    

