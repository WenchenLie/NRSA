import os
import sys
import shutil
import time
import numpy as np
from pathlib import Path
from typing import Literal
sys.path.append(str(Path(__file__).parent.parent))
from NRSAcore.ModelParameter import ModelParameter


def creat_folder(
        path: Path | str,
        exists: Literal['overwrite', 'delete', 'ask']='ask'
        ) -> bool:
    """创建文件夹

    Args:
        path (Path): 路径
        exists (str, optional): 如果文件夹存在，如何处理('overwrite', 'delete', or 'ask')

    Returns:
        bool: True — Go on, False - Quit.
    """
    path = Path(path)
    print(path.absolute())
    if not path.exists():
        os.makedirs(path)
    else:
        if exists == 'overwrite':
            pass
        elif exists == 'delete':
            shutil.rmtree(path=path)
            os.makedirs(path)
        elif exists == 'ask':
            res = input(f'文件夹{path}存在！\n[Enter] - 删除\n[o] - 覆盖\n[q] - 退出\n请选择：')
            if res == '':
                shutil.rmtree(path=path)
                os.makedirs(path)
            elif res == 'o':
                pass
            elif res == 'q':
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


if __name__ == "__main__":
    T = generate_period_series(0.1, 1, 0.02)
    print(T.name)
    ModelParameter.print_var()