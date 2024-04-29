import sys
import json
import numpy as np
from typing import Literal
from pathlib import Path
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from NRSAcore._Win import _Win
from utils.utils import SDOF_Error
from utils import utils


class SDOFmodel:

    dir_main = Path(__file__).parent.parent
    dir_temp = dir_main / 'temp'
    dir_input = dir_main / 'Input'
    dir_gm = dir_input / 'GMs'

    def __init__(self, json_file: Path | str=None, task: dict=None):
        """导入模型

        Args:
            json_file (Path | str, optional): 从json文件导入，默认None
            task (dict, optional): 从Task类生成的字典导入
        """
        if json_file is None and task is None:
            raise SDOF_Error('参数 json_file 与 task 至少应指定一个')
        if json_file:
            with open(json_file, 'r') as f:
                self.task = json.loads(f.read())
        if task:
            self.task = task
        self._get_task_info()


    def _get_task_info(self):
        """获取分析任务信息"""
        self.GM_N = len(self.task['ground_motions']['dt_SF'])  # 地震动数量
        self.N_SDOF = len(self.task['SDOF_models'])


    def set_analytical_options(self,
            analysis_type: Literal['constant_ductility', 'constant_strength'],
            PDelta: bool=False,
            batch: int=1,
            parallel: int=0,
            ductility_tol: float=0.01):
        """设置分析参数

        Args:
            analysis_type (Literal['constant_ductility', 'constant_strength']): 分析类型，等延性或等屈服强度
            PDelta (bool, optional): 是否考虑P-Delta效应，默认False
            batch (int, optional): 在同一模型空间下建立的SDOF数量，默认1
            parallel (int, optional): 是否开启多进程并行计算，默认0，即不开启，不为0时为开启的进程数量，
            每个子进程处理一条地震波
            ductility_tol (float, optional): 等延性分析时目标延性的收敛容差，默认0.01
        """
        if analysis_type not in ['constant_ductility', 'constant_strength']:
            raise SDOF_Error(f'未知分析类型：{analysis_type}')
        if not isinstance(batch, int):
            raise SDOF_Error('参数 batch 应为整数')
        if batch < 1:
            raise SDOF_Error('参数 batch 应大于等于1')
        self.analysis_type = analysis_type
        self.PDelta = PDelta
        self.batch = batch
        self.parallel = parallel
        self.ductility_tol = ductility_tol


    def run(self):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        app = QApplication(sys.argv)
        win = _Win(self)
        win.show()
        app.exec_() 


if __name__ == "__main__":
    model = SDOFmodel(json_file=r'C:\Users\Admin\Desktop\NRSA\temp\model.json')
    model.set_analytical_options('constant_strength')
    model.run()









