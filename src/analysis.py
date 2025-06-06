import os, sys
import json
from math import pi
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import LOGGER, PERIOD
from .NRSA import NRSA


class ConstantDuctilityAnalysis(NRSA):
    def __init__(self, job_name: str, cache_dir: Path | str='cache'):
        super().__init__(job_name, cache_dir, analysis_type='CDA')

    def analysis_settings(self,
            period: np.ndarray | float,
            material_function: Callable[[float, float, float, float], tuple[str, list, float, float]],
            material_paras: dict[str, tuple | float],
            damping: float,
            target_ductility: float,
            R_init: float,
            R_incr: float,
            tol_ductility: float,
            tol_R: float,
            max_iter: int,
            thetaD: float=0,
            mass: float=1,
            height: float=1,
            fv_duration: float=0.0,
        ):
        """设置分析参数

        Args:
            period (np.ndarray | float): 等延性谱周期序列
            material_function (Callable[[float, float, float, float], tuple[str, list, float, float]]): 获取opensees材料格式的函数
            material_paras (dict[str, float]): 材料定义所需参数
            damping (float): 阻尼比
            target_ductility (float): 目标延性
            R_init (float): 初始强度折减系数(R)
            R_incr (float): 强度折减系数(R)递增值
            tol_ductility (float): 延性(μ)收敛容差
            tol_R (float): 相邻强度折减系数(R)收敛容差
            max_iter (int): 最大迭代次数
            thetaD (float): P-Delta系数
            mass (float): 质量，默认1
            height (float, optional): 高度，默认1
            fv_duration (float, optional): 自由振动持续时间，默认0.0
        
        Converge criteria for constant ductility analysis:
        --------------------------------------------------
        `abs(μ - μ_target) / μ_target < tol_ductility`  
        `abs(R1 - R2) / R2 < tol_R`  
        where R1 and R2 are the adjacent R values.

        Note:
        -----
        * `mass`不会影响`R`和具有长度和时间量纲的响应，但会影响具有力量纲的响应，
        力量纲响应与`mass`成正比
        * 等延性分析中，延性容差`tol_ductility`建议不低于0.01，强度折减系数`R`容差
        建议不低于0.001
        """
        Ti = 0
        super().analysis_settings(period, Ti, material_function, material_paras,
                                  damping, target_ductility, R_init, R_incr,
                                  tol_ductility, tol_R, max_iter, thetaD, mass,
                                  height, fv_duration)

class CSA_THA(NRSA):
    def __init__(self, job_name: str, cache_dir: Path | str, analysis_type: str):
        super().__init__(job_name, cache_dir, analysis_type=analysis_type)

    def scale_ground_motions(self,
            method: str,
            para: None | float | tuple | str,
            target_spectrum: str | None=None,
            sf_targspec: float=1.0,
            save_sf=False,
            save_scaled_spec=False,
            plot=True,
            save_fig=False
        ):
        """缩放地震动，仅运行时程分析前需要调用，
        如果运行IDA或者Pushover，可以不调用。  

        Args:
            method (str): 地震动的缩放方法，为'a'-'g'：  
            * [a] 按Sa(T=0)匹配反应谱, pare=None  
            * [b] 按Sa(T=Ta)匹配反应谱, para=Ta  
            * [c] 按Sa(Ta) ~ Sa(Tb)匹配反应谱, para=(Ta, Tb)  
            * [d] 指定PGA, para=PGA  
            * [e] 不缩放  
            * [f] 指定相同缩放系数, para=sf  
            * [g] 按文件指定, para=path: str (如'temp/GM_sfs.txt')，文件包含一列n行个数据  
            * [h] 按Sa,avg(T1, T2)匹配反应谱，即T1~T2间的加速度谱值的几何平均数，para=(T1, T2)  
            * [i] 指定Sa(Ta), para=(Ta, Sa)  
            * [j] 指定Sa,avg(Ta~Tb), para=(Ta, Tb, Sa,avg)\n
            分别代表n条地震动的缩放系数  
            para: 地震动缩放所需参数，与`method`的取值有关  
            target_spectrum (np.ndarray, optional): 目标谱数据，应包含两列数据，为周期和加速度谱值，默认None  
            sf_targspec (float, optional): 目标谱的缩放系数，默认为1  
            save_sf (bool, optional): 是否保存缩放后的缩放系数(将保存至temp文件夹，
            可以作为`method`取'g'时`para`参数对应的文件路径，默认为False  
            save_unscaled_spec (bool, optional): 是否保存未缩放地震动反应谱，默认False  
            save_scaled_spec (bool, optional): 是否保存缩放后地震动反应谱，默认False
            plot (bool, optional): 是否绘制缩放后地震动反应谱与目标谱的对比图，默认为True
            save_fig (bool, optional): 是否保存缩放后地震动反应谱与目标谱的对比图，默认为False
        
        Note:
        -----
        缩放系数都是基于5%阻尼比反应谱来确定的
        """
        if self.analysis_type == 'CSA':
            period = self.period
        else:
            period = PERIOD
        method = method
        if target_spectrum is not None:
            T_code, Sa_code = target_spectrum[:, 0], target_spectrum[:, 1]
            Sa_code *= sf_targspec
            Sv_code = Sa_code * T_code / (2 * pi)
            Sd_code = Sa_code * (T_code / (2 * pi)) ** 2
        else:
            T_code = period
            Sa_code = None
            Sv_code = None
            Sd_code = None
        scaled_GM_RSA = np.zeros((len(period), self.GM_N))
        scaled_GM_RSV = np.zeros((len(period), self.GM_N))
        scaled_GM_RSD = np.zeros((len(period), self.GM_N))
        if method == 'g':
            sf_path = para
            sfs = np.loadtxt(sf_path)
        for idx, gm_name in enumerate(self.GM_names):
            print(f'  Calculating scaling factors... ({idx+1}/{self.GM_N})     \r', end='')
            th = np.loadtxt(self.GM_folder / f'{gm_name}{self.suffix}')
            RSA, RSV, RSD = self.unscaled_RSA_5pct[:, idx], self.unscaled_RSV_5pct[:, idx], self.unscaled_RSD_5pct[:, idx]  # 无缩放5%阻尼比反应谱
            if method == 'a':
                if target_spectrum is None:
                    raise ValueError('Argument `target_spectrum` should be given')
                T0 = 0
                sf = self.get_y(T_code, Sa_code, T0) / self.get_y(period, RSA, T0)
            elif method == 'b':
                if target_spectrum is None:
                    raise ValueError('Argument `target_spectrum` should be given')
                T0 = para
                sf = self.get_y(T_code, Sa_code, T0) / self.get_y(period, RSA, T0)
            elif method == 'c':
                if target_spectrum is None:
                    raise ValueError('Argument `target_spectrum` should be given')
                T1, T2 = para
                idx1, idx2 = self.get_y(T_code, RSA, T1, True)[1], self.get_y(period, RSA, T2, True)[1]
                init_sf = 1.0  # 初始缩放系数
                learning_rate = 0.01  # 学习率
                num_iterations = 40000  # 迭代次数
                init_sf = np.mean(Sa_code[idx1: idx2]) / np.mean(RSA[idx1: idx2])
                sf = self.gradient_descent(RSA[idx1: idx2], Sa_code[idx1: idx2], init_sf, learning_rate, num_iterations)
            elif method == 'd':
                PGA = para
                sf = PGA / max(abs(th))
            elif method == 'e':
                sf = 1
            elif method == 'f':
                sf = para
            elif method == 'g':
                sf = sfs[idx] 
            elif method == 'h':
                if target_spectrum is None:
                    raise ValueError('Argument `target_spectrum` should be given')
                Sa_i_code = []
                Sa_i = []
                T1, T2 = para
                for i in range(len(T_code)):
                    Ti = T_code[i]
                    if T1 <= Ti <= T2:
                        Sa_i_code.append(Sa_code[i])
                for i in range(len(period)):
                    Ti = period[i]
                    if T1 <= Ti <= T2:
                        Sa_i.append(RSA[i])
                Sa_avg_code = self.geometric_mean(Sa_i_code)
                Sa_avg = self.geometric_mean(Sa_i)
                sf = Sa_avg_code / Sa_avg
                if is_print:
                    LOGGER.info(f'Sa,avg = {Sa_avg_code}')
                    is_print = False
            elif method == 'i':
                Ta, Sa_target = para
                Sa_gm = self.get_y(period, RSA, Ta)
                sf = Sa_target / Sa_gm
            elif method == 'j':
                Ta, Tb, Sa_target = para
                Sa_gm_avg = self.geometric_mean(RSA[(Ta <= period) & (period <= Tb)])
                sf = Sa_target / Sa_gm_avg
            else:
                LOGGER.error('The `method` parameter is incorrect!')
                raise ValueError('The `method` parameter is incorrect!')
            scaled_GM_RSA[:, idx] = RSA * sf
            scaled_GM_RSV[:, idx] = RSV * sf
            scaled_GM_RSD[:, idx] = RSD * sf
            self.GM_indiv_sf[idx] = sf
        if save_sf:
            sf_dict = {'global scaling': self.GM_global_sf,
                       '//': 'Global scaling factor is used when record file is read.',
                       '//': 'Total scaling factor is equal to global scaling factor multiplied by individual scaling factors.'}
            for gm_name, sf in zip(self.GM_names, self.GM_indiv_sf):
                sf_dict[gm_name] = sf
            file_path = (self.wkdir / 'GM_scaling_factors.json').as_posix()
            json.dump(sf_dict, open(file_path, 'w'), indent=4)
            LOGGER.info(f'Scaling factors have been saved to {file_path}')
        if save_scaled_spec:
            data_RSA = np.zeros((len(period), self.GM_N + 1))
            data_RSV = np.zeros((len(period), self.GM_N + 1))
            data_RSD = np.zeros((len(period), self.GM_N + 1))
            data_RSA[:, 0] = period
            data_RSV[:, 0] = period
            data_RSD[:, 0] = period
            data_RSA[:, 1:] = scaled_GM_RSA
            data_RSV[:, 1:] = scaled_GM_RSV
            data_RSD[:, 1:] = scaled_GM_RSD
            stat_A, stat_V, stat_D = np.zeros((len(period), 6)), np.zeros((len(period), 6)), np.zeros((len(period), 6))
            stat_A[:, 0] = period
            stat_V[:, 0] = period
            stat_D[:, 0] = period
            stat_A[:, 1] = np.percentile(data_RSA[:, 1:], 16, axis=1)
            stat_A[:, 2] = np.percentile(data_RSA[:, 1:], 50, axis=1)
            stat_A[:, 3] = np.percentile(data_RSA[:, 1:], 84, axis=1)
            stat_A[:, 4] = np.mean(data_RSA[:, 1:], axis=1)
            stat_A[:, 5] = np.std(data_RSA[:, 1:], axis=1)
            stat_V[:, 1] = np.percentile(data_RSV[:, 1:], 16, axis=1)
            stat_V[:, 2] = np.percentile(data_RSV[:, 1:], 50, axis=1)
            stat_V[:, 3] = np.percentile(data_RSV[:, 1:], 84, axis=1)
            stat_V[:, 4] = np.mean(data_RSV[:, 1:], axis=1)
            stat_V[:, 5] = np.std(data_RSV[:, 1:], axis=1)
            stat_D[:, 1] = np.percentile(data_RSD[:, 1:], 16, axis=1)
            stat_D[:, 2] = np.percentile(data_RSD[:, 1:], 50, axis=1)
            stat_D[:, 3] = np.percentile(data_RSD[:, 1:], 84, axis=1)
            stat_D[:, 4] = np.mean(data_RSD[:, 1:], axis=1)
            stat_D[:, 5] = np.std(data_RSD[:, 1:], axis=1)
            if not (self.wkdir / 'Scaled_spectra').exists():
                os.makedirs(self.wkdir / 'Scaled_spectra')
            folder = self.wkdir / 'Scaled_spectra'
            data_RSA = pd.DataFrame(data_RSA, columns=['T']+self.GM_names)
            data_RSV = pd.DataFrame(data_RSV, columns=['T']+self.GM_names)
            data_RSD = pd.DataFrame(data_RSD, columns=['T']+self.GM_names)
            stat_A = pd.DataFrame(stat_A, columns=['T', '16%', '50%', '84%', 'Mean', 'Std'])
            stat_V = pd.DataFrame(stat_V, columns=['T', '16%', '50%', '84%', 'Mean', 'Std'])
            stat_D = pd.DataFrame(stat_D, columns=['T', '16%', '50%', '84%', 'Mean', 'Std'])
            data_RSA.to_csv(folder / 'Scaled_RSA.csv', index=False)
            data_RSV.to_csv(folder / 'Scaled_RSV.csv', index=False)
            data_RSD.to_csv(folder / 'Scaled_RSD.csv', index=False)
            stat_A.to_csv(folder / 'Scaled_RSA_stat.csv', index=False)
            stat_V.to_csv(folder / 'Scaled_RSV_stat.csv', index=False)
            stat_D.to_csv(folder / 'Scaled_RSD_stat.csv', index=False)
            LOGGER.info(f'Scaled 5%-damping spectra have been saved to {folder.as_posix()}')
        plt.figure(figsize=(15, 4))
        plt.subplot(131)
        if method == 'a':
            plt.scatter(0, Sa_code[0], color='blue', zorder=99999)
        elif method == 'b':
            plt.scatter(T0, self.get_y(T_code, Sa_code, T0), color='blue', zorder=99999)
        elif method == 'c':
            plt.scatter(para, [self.get_y(T_code, Sa_code, para[0]), self.get_y(T_code, Sa_code, para[1])], color='blue', zorder=99999)
        elif method == 'd':
            plt.scatter(0, PGA, color='blue', zorder=99999)
        elif method == 'h':
            plt.scatter(para, [self.get_y(T_code, Sa_code, para[0]), self.get_y(T_code, Sa_code, para[1])], color='blue', zorder=99999)
        elif method == 'i':
            plt.scatter(para[0], para[1], color='blue', zorder=99999)
        for i in range(self.GM_N):
            plt.subplot(131)
            plt.plot(period, scaled_GM_RSA[:, i], color='grey')
            plt.subplot(132)
            plt.plot(period, scaled_GM_RSV[:, i], color='grey')   
            plt.subplot(133)
            plt.plot(period, scaled_GM_RSD[:, i], color='grey')    
        plt.subplot(131)
        if Sa_code is not None:
            plt.plot(T_code, Sa_code, label='Code', color='red')
            plt.legend()
        plt.xlabel('T [s]')
        plt.ylabel('Acceleration [g]')
        plt.subplot(132)
        if Sv_code is not None:
            plt.plot(T_code, Sv_code, label='Code', color='red')
            plt.legend()
        plt.xlabel('T [s]')
        plt.ylabel('Velocity [mm/s]')
        plt.subplot(133)
        if Sd_code is not None:
            plt.plot(T_code, Sd_code, label='Code', color='red')
            plt.legend()
        plt.xlabel('T [s]')
        plt.ylabel('Displacement [mm]')
        plt.tight_layout()
        if save_fig:
            plt.savefig(self.wkdir / f'Spectra.png', dpi=300)
        if plot:
            plt.show()
        else:
            plt.close()
        self.scaling_finished = True
    
    def run(self):
        if not self.scaling_finished:
            LOGGER.warning('Please run `scale_ground_motions` before running analysis!')
        super().run()

    @staticmethod
    def get_y(T: np.ndarray, S: np.ndarray, T0: float, withIdx=False) -> float:
        for i in range(len(T) - 1):
            if T[i] <= T0 <= T[i+1]:
                k = (S[i+1] - S[i]) / (T[i+1] - T[i])
                S0 = S[i] + k * (T0 - T[i])
                if withIdx:
                    return S0, i
                else:
                    return S0
        else:
            raise ValueError(f'Cannot find the spectral acceleration at period {T0}！')

    @staticmethod
    def RMSE(a: np.ndarray, b: np.ndarray) -> float:
        """计算均方根误差"""
        return np.sqrt(np.mean((a - b) ** 2))

    @staticmethod
    def gradient_descent(a, b, init_SF, learning_rate, num_iterations):
        """梯度下降法"""
        f = init_SF
        for _ in range(num_iterations):
            error = a * f - b
            gradient = 2 * np.dot(error, a) / len(a)
            f -= learning_rate * gradient
        return f

    @staticmethod
    def geometric_mean(data):
        """计算几何平均数"""
        total = 1
        n = len(data)
        for i in data:
            total *= pow(i, 1 / n)
        return total


class ConstantStrengthAnalysis(CSA_THA):
    def __init__(self, job_name, cache_dir='cache'):
        super().__init__(job_name, cache_dir, analysis_type='CSA')

    def analysis_settings(self,
            period: np.ndarray | float,
            material_function: Callable[[float, float], tuple[str, list, float, float]],
            material_paras: dict[str, tuple | float],
            damping: float,
            thetaD: float=0,
            mass: float=1,
            height: float=1,
            fv_duration: float=0.0,                  
        ):
        """设置分析参数

        Args:
            period (np.ndarray | float): 等延性谱周期序列
            material_function (Callable[[float, float, float, float], tuple[str, list, float, float]]): 获取opensees材料格式的函数
            material_paras (dict[str, float]): 材料定义所需参数
            damping (float): 阻尼比
            thetaD (float): P-Delta系数
            mass (float): 质量，默认1
            height (float, optional): 高度，默认1
            fv_duration (float, optional): 自由振动持续时间，默认0.0
        """
        Ti = 0
        super().analysis_settings(period, Ti, material_function, material_paras, damping,
            10, 1, 1, 0.01, 0.01, 100,
            thetaD, mass, height, fv_duration)


class TimeHistoryAnalysis(CSA_THA):
    def __init__(self, job_name, cache_dir='cache'):
        super().__init__(job_name, cache_dir, analysis_type='THA')

    def analysis_settings(self,
            Ti: float | None,
            material_function: Callable[[float, float], tuple[str, list, float, float]],
            material_paras: dict[str, tuple | float],
            damping: float,
            thetaD: float=0,
            mass: float=1,
            height: float=1,
            fv_duration: float=0.0,                  
        ):
        """设置分析参数

        Args:
            Ti (float | None): 周期点
            material_function (Callable[[float, float, float, float], tuple[str, list, float, float]]): 获取opensees材料格式的函数
            material_paras (dict[str, float]): 材料定义所需参数
            damping (float): 阻尼比
            thetaD (float): P-Delta系数
            mass (float): 质量，默认1
            height (float, optional): 高度，默认1
            fv_duration (float, optional): 自由振动持续时间，默认0.0
        """
        super().analysis_settings(None, Ti, material_function, material_paras, damping,
            10, 1, 1, 0.01, 0.01, 100,
            thetaD, mass, height, fv_duration)
        
    def get_results(self, plot: bool=True) -> np.ndarray:
        """获取分析结果

        Args:
            plot (bool, optional): 是否绘制结果图
        
        Returns:
            np.ndarray: 返回ndarray，各列依次为：时间序列，地面加速度，相对位移，
              相对速度，绝对加速度，累积塑性能量，累积黏滞阻尼耗能，
              累积位移，累积塑性位移，底部反应力
        """
        plt.figure(figsize=(17, 12))
        titles = ['Groung motions', 'Relative Disp.', 'Relative Vel.',
                  'Absolute Accel.', 'Cum. Plas. Energy Diss.', 'Cum. Visc. Energy Diss.',
                  'Cum. Disp.', 'Cum. Plas. Disp.', 'Base Reaction', 'Material Hyst. Response',
                  'Visc. Damp Hyst. Response', 'Total Hyst. Response']
        ylabels = ['Acceleration [g]', 'Displacement', 'Velocity',
                   'Acceleration', 'Energy', 'Energy', 'Displacement',
                   'Displacement', 'Reaction', 'Force', 'Force', 'Force']
        for gm_name in self.GM_names:
            results: np.ndarray = np.load(self.wkdir / f'results/{gm_name}.npy')
            time_, ag_scaled, disp_th, vel_th, accel_th, Ec_th, Ev_th, CD_th, CPD_th, reaction_th, eleForce_th, dampingForce_th = results.T
            plt.subplot(341)
            plt.title(titles[0])
            plt.plot(time_, ag_scaled / 9800, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[0])
            plt.legend()
            plt.subplot(342)
            plt.title(titles[1])
            plt.plot(time_, disp_th, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[1])
            plt.legend()
            plt.subplot(343)
            plt.title(titles[2])
            plt.plot(time_, vel_th, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[2])
            plt.legend()
            plt.subplot(344)
            plt.title(titles[3])
            plt.plot(time_, accel_th, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[3])
            plt.legend()
            plt.subplot(345)
            plt.title(titles[4])
            plt.plot(time_, Ec_th, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[4])
            plt.legend()
            plt.subplot(346)
            plt.title(titles[5])
            plt.plot(time_, Ev_th, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[5])
            plt.legend()
            plt.subplot(347)
            plt.title(titles[6])
            plt.plot(time_, CD_th, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[6])
            plt.legend()
            plt.subplot(348)
            plt.title(titles[7])
            plt.plot(time_, CPD_th, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[7])
            plt.legend()
            plt.subplot(349)
            plt.title(titles[8])
            plt.plot(time_, reaction_th, label=gm_name)
            plt.xlabel('Time [s]')
            plt.ylabel(ylabels[8])
            plt.legend()
            plt.subplot(3, 4, 10)
            plt.title(titles[9])
            plt.plot(disp_th, eleForce_th, label=gm_name)
            plt.xlabel('Displement')
            plt.ylabel(ylabels[9])
            plt.legend()
            plt.subplot(3, 4, 11)
            plt.title(titles[10])
            plt.plot(disp_th, dampingForce_th, label=gm_name)
            plt.xlabel('Displement')
            plt.legend()
            plt.subplot(3, 4, 12)
            plt.title(titles[11])
            total_force = eleForce_th + dampingForce_th
            plt.plot(disp_th, total_force, label=gm_name)
            plt.xlabel('Displement')
            plt.ylabel(ylabels[11])
            plt.legend()
        plt.tight_layout()
        plt.savefig(self.wkdir / f'Results.png', dpi=600)
        if plot:
            plt.show()
        plt.close()
        return results
