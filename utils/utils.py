import os
import sys
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
sys.path.append(str(Path(__file__).parent.parent))
from loguru import logger
from PyQt5.QtWidgets import QMessageBox
from NRSAcore.ModelParameter import ModelParameter
from scipy.integrate import cumulative_trapezoid


logger.remove()
logger.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <red>|</red> <level>{level}</level> <red>|</red> <level>{message}</level>",
    level="DEBUG"
)
LOGGER = logger


def check_file_exists(path: str | Path):
    """检查文件是否存在，如果不存在则抛出异常

    Args:
        path (str | Path): 文件路径
    """
    path = Path(path)
    if not path.exists():
        raise SDOF_Error(f'路径不存在：{path.absolute()}')


def creat_folder(
        path: Path | str,
        exists: Literal['overwrite', 'delete', 'ask']='ask',
        logger=None
        ) -> bool:
    """创建文件夹

    Args:
        path (Path): 路径
        exists (str, optional): 如果文件夹存在，如何处理('overwrite', 'delete', or 'ask')
        logger (logger, optional): 日志

    Returns:
        bool: True - Go on, False - Quit.
    """
    path = Path(path)
    if not path.exists():
        os.makedirs(path)
    else:
        if exists == 'overwrite':
            pass
        elif exists == 'delete':
            shutil.rmtree(path=path)
            os.makedirs(path)
        elif exists == 'ask':
            res1 = QMessageBox.question(None, '警告', f'{path}已存在，是否删除？')
            if res1 == QMessageBox.Yes:
                shutil.rmtree(path)
                os.makedirs(path)
                if logger:
                    logger.warning(f'已删除并新建{path}')
                return True
            else:
                res2 = QMessageBox.question(None, '警告', f'是否覆盖数据？')
                if res2 == QMessageBox.Yes:
                    return True
                else:
                    if logger:
                        logger.warning('已退出分析')
                    return False
    return True


def generate_period_series(Ta: float, Tb: float, dT: float=None, num: int=None) -> ModelParameter:
    """生成周期序列

    Args:
        Ta (float): 起始周期
        Tb (float): 结束周期
        dT (float, optional): 周期间隔
        num (int, optional): 周期点数量

    Returns (ModelParameter): 一个ModelParameter类型的实例
    """
    if not any([dT, num]):
        raise ValueError('必须定义`dT`或`num`的其中一个')
    if all([dT, num]):
        raise ValueError('`dT`和`num`不能同时指定')
    if dT:
        T = np.arange(Ta, Tb, dT)
    else:
        T = np.linspace(Ta, Tb, num)
    T = T.tolist()
    T = ModelParameter('T', T)
    return T


class SDOF_Error(Exception):
    def __init__(self, message="SDOF_Error"):
        self.message = message
        super().__init__(self.message)


class Task_Error(Exception):
    def __init__(self, message="Task_Error"):
        self.message = message
        super().__init__(self.message)


class ModelParameter_Error(Exception):
    def __init__(self, message="ModelParameter_Error"):
        self.message = message
        super().__init__(self.message)


class SDOF_Helper:
    def __init__(self, getTime=True, suppress=True):
        """运行SDOF分析时的上下文管理器

        Args:
            getTime (bool, optional): 是否统计运行时间，默认是
            suppress (bool, optional): 是否屏蔽输出，默认是
        """
        self.suppress = suppress
        self.getTime = getTime
        self._original_stdout = None
        self._original_stderr = None

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        if self.getTime:
            self.t_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
        if self.getTime:
            self.t_end = time.time()
            elapsed_time = self.t_end - self.t_start
            print(f'Run time: {elapsed_time:.4f} s')


def gradient_descent(a, b, init_SF, learning_rate, num_iterations):
    """梯度下降法
    """
    f = init_SF
    for _ in range(num_iterations):
        error = a * f - b
        gradient = 2 * np.dot(error, a) / len(a)
        f -= learning_rate * gradient
    return f        


def decode_list(ls: list[bytes]):
    """将从hdf5文件读取的字节串解码为字符串"""
    ls_new = [s.decode('utf-8') for s in ls]
    return ls_new


class Curve:
    def __init__(self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_label: str,
        y_label: str,
        label: str,
        wkd: Path,
        gm_names: list[str]
    ):
        if x_values.ndim == y_values.ndim == 1:
            self.is_zipped = False
        else:
            self.is_zipped = True
        if self.is_zipped:
            if not len(x_values) == y_values.shape[1]:
                raise SDOF_Error(f'曲线 {name} 横纵坐标数据量不等（{len(x_values)}, {len(y_values)}）')
            if not x_values.ndim == 1:
                raise SDOF_Error(f'曲线 {name} 的横坐标应为一维')
            if not y_values.ndim == 2:
                raise SDOF_Error(f'曲线 {name} 的总坐标应为二维')
        self.name = name
        self.N = len(y_values)  # 地震动数量
        self.x_values = x_values
        self.y_values = y_values
        self.x_label = x_label
        self.y_label = y_label
        if not label:
            label = y_label
        self.label = label
        self.wkd = wkd
        self.gm_names = gm_names
        self._statistics()
    
    def _statistics(self):
        """计算统计特征"""
        if self.is_zipped:
            # 2、16、50、84、98分位线
            self.pct_2 = np.percentile(self.y_values, 2, axis=0)
            self.pct_16 = np.percentile(self.y_values, 16, axis=0)
            self.pct_50 = np.percentile(self.y_values, 50, axis=0)
            self.pct_84 = np.percentile(self.y_values, 84, axis=0)
            self.pct_98 = np.percentile(self.y_values, 98, axis=0)
            self.mean = np.mean(self.y_values, axis=0)  # 均值
            self.std = np.std(self.y_values, axis=0)  # 标准差
        else:
            self.pcc = np.corrcoef(self.x_values, self.y_values)[0, 1]

    def show(self, save_result: bool=False, plotfig=True, plot_scatter=True):
        """展示曲线

        Args:
            save_result (bool, optional): 是否保持曲线到输出文件夹，默认False
            plotfig (bool, optional): 是否绘制曲线图，默认True
            plot_scatter (bool, optional): 是否绘制散点(若数据量大可不绘制)
        """
        label = self.label
        if not self.is_zipped:
            plt.scatter(self.x_values, self.y_values, c='grey', label=label)
        else:
            if plot_scatter:
                for i in range(self.N):
                    plt.scatter(self.x_values, self.y_values[i], c='grey', label=label)
                    label = None
            plt.plot(self.x_values, self.pct_2, c='orange', label='pct2')
            plt.plot(self.x_values, self.pct_16, c='green', label='pct16')
            plt.plot(self.x_values, self.pct_50, c='blue', label='pct50')
            plt.plot(self.x_values, self.pct_84, c='green', label='pct84')
            plt.plot(self.x_values, self.pct_98, c='orange', label='pct98')
            plt.plot(self.x_values, self.mean, c='red', label='Meam')
            plt.plot(self.x_values, self.std, c='brown', label='STD')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        if save_result:
            plt.savefig(self.wkd / f'{self.name}.png', dpi=600)
            df = pd.DataFrame()
            if self.is_zipped:
                df[self.x_label] = self.x_values
                for i, gm_name in enumerate(self.gm_names):
                    df[gm_name] = self.y_values[i]  # 输出所有散点
                df_2 = pd.DataFrame(pd.Series(self.pct_2), columns=['2%'])
                df_16 = pd.DataFrame(pd.Series(self.pct_16), columns=['16%'])
                df_50 = pd.DataFrame(pd.Series(self.pct_50), columns=['50%'])
                df_84 = pd.DataFrame(pd.Series(self.pct_84), columns=['84%'])
                df_98 = pd.DataFrame(pd.Series(self.pct_98), columns=['98%'])
                df_mean = pd.DataFrame(pd.Series(self.mean), columns=['Mean'])
                df_std = pd.DataFrame(pd.Series(self.std), columns=['STD'])
                df = pd.concat([df, df_2, df_16, df_50, df_84, df_98, df_mean, df_std], axis=1)
            else:
                df[self.x_label] = self.x_values
                df[self.y_label] = self.y_values
                df['PCC'] = self.pcc
            df.to_csv(self.wkd / f'{self.name}.csv', index=None)
        if plotfig:
            plt.show()
        plt.close()


def get_y(x_line: np.ndarray, y_line: np.ndarray, x: float | int) -> float:
    """获取曲线在某横坐标值处的纵坐标"""
    if x < min(x_line) or x > max(x_line):
        raise SDOF_Error(f'横坐标值{x}超出曲线横坐标范围({min(x_line)}, {max(x_line)})')
    for i in range(1, len(x_line)):
        if x_line[i - 1] <= x <= x_line[i]:
            k = (y_line[i] - y_line[i - 1]) / (x_line[i] - x_line[i - 1])
            y = y_line[i - 1] + k * (x - x_line[i - 1])
            break
    return y


def integral(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """计算y对x的不定积分"""
    if not len(x) == len(y):
        raise SDOF_Error(f'数组x和y的长度不相等({len(x)}, {len(y)})')
    int_y = [0]
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        int_y.append((y[i] + y[i - 1]) * dx / 2 + int_y[i - 1])
    return np.array(int_y)


def a2u(a: np.ndarray, dx: float) -> np.ndarray:
    """将加速度积分为位移，并进行基线修正

    Args:
        a (np.ndarray): 加速度序列
        dx (float): 加速度步长

    Returns:
        np.ndarray: 位移序列
    """
    v = cumulative_trapezoid(a, dx=dx, initial=0)
    u = cumulative_trapezoid(v, dx=dx, initial=0)
    baseline = np.linspace(0, u[-1], len(u))
    u -= baseline
    return u