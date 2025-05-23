import os
import sys
import time
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parent.parent))
from loguru import logger
from PyQt5.QtWidgets import QMessageBox
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
        raise FileExistsError(f'路径不存在：{path.absolute()}')


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


class SDOFError(Exception):
    def __init__(self, message="SDOF_Error"):
        self.message = message
        super().__init__(self.message)


class TaskError(Exception):
    def __init__(self, message="Task_Error"):
        self.message = message
        super().__init__(self.message)


class CurveError(Exception):
    def __init__(self, message="CurveError"):
        self.message = message
        super().__init__(self.message)


class SDOFHelper:
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



def get_y(x_line: np.ndarray, y_line: np.ndarray, x: float | int) -> float:
    """获取曲线在某横坐标值处的纵坐标"""
    if x < min(x_line) or x > max(x_line):
        raise ValueError(f'横坐标值{x}超出曲线横坐标范围({min(x_line)}, {max(x_line)})')
    for i in range(1, len(x_line)):
        if x_line[i - 1] <= x <= x_line[i]:
            k = (y_line[i] - y_line[i - 1]) / (x_line[i] - x_line[i - 1])
            y = y_line[i - 1] + k * (x - x_line[i - 1])
            break
    return y


def integral(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """计算y对x的不定积分"""
    if not len(x) == len(y):
        raise ValueError(f'数组x和y的长度不相等({len(x)}, {len(y)})')
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

def is_iterable(obj):
    """判断对象是否可迭代"""
    try:
        iter(obj)
        return True
    except:
        return False