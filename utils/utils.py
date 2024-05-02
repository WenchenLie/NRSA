import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
sys.path.append(str(Path(__file__).parent.parent))
from loguru import logger
from PyQt5.QtWidgets import QMessageBox
from NRSAcore.ModelParameter import ModelParameter


logger.remove()
logger.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <red>|</red> <level>{level}</level> <red>|</red> <level>{message}</level>",
    level="DEBUG"
)
LOGGER = logger


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
        output_dir: Path
    ):
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
        self.label = label
        self.output_dir = output_dir
        self._statistics()
    
    def _statistics(self):
        """计算统计特征"""
        # 16、50、84分位线
        self.pct_16 = np.percentile(self.y_values, 16, axis=0)
        self.pct_50 = np.percentile(self.y_values, 50, axis=0)
        self.pct_84 = np.percentile(self.y_values, 84, axis=0)
        self.mean = np.mean(self.y_values, axis=0)  # 均值
        self.std = np.std(self.y_values, axis=0)

    def show(self, savefig: bool=False, plotfig=True):
        """展示曲线

        Args:
            savefig (bool, optional): 是否保持曲线到输出文件夹，默认False
            plotfig (bool, optional): 是否绘制曲线图，默认True
        """
        label = self.label
        for i in range(self.N):
            plt.scatter(self.x_values, self.y_values[i], c='grey', label=label)
            label = None
        plt.plot(self.x_values, self.pct_16, c='green', label='pct16')
        plt.plot(self.x_values, self.pct_50, c='blue', label='pct50')
        plt.plot(self.x_values, self.pct_84, c='green', label='pct84')
        plt.plot(self.x_values, self.mean, c='red', label='Meam')
        # plt.plot(self.x_values, self.std, c='yellow', label='STD')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        if savefig:
            plt.savefig(self.output_dir / f'{self.name}.png', dpi=600)
        if plotfig:
            plt.show()
        plt.close()



