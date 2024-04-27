import sys
import json
import numpy as np
from typing import Literal
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
from NRSAcore.Win import Win
from utils.utils import SDOF_Error
from utils import utils


class SDOFmodel:
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


    def set_analysis_options(self,
            analysis_type: Literal['constant_ductlity', 'constant_strength'],
            PDelta: bool=False,
            batch: int=1,
            parallel: int=0):
        """设置分析参数

        Args:
            analysis_type (Literal['constant_ductlity', 'constant_strength']): 分析类型，等延性或等屈服强度
            PDelta (bool, optional): 是否考虑P-Delta效应，默认False
            batch (int, optional): 在同一模型空间下建立的SDOF数量，默认1
            parallel (int, optional): 是否开启多进程并行计算，默认0，即不开启，不为0时为开启的进程数量
        """
        self.analysis_type = analysis_type
        self.PDelta = PDelta
        self.batch = batch
        self.parallel = parallel


    def run(self):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        app = QApplication(sys.argv)
        win = Win(self)
        win.show()
        app.exec_() 


if __name__ == "__main__":
    model = SDOFmodel(json_file=r'C:\Users\Admin\Desktop\NRSA\temp\model.json')
    model.set_analysis_options('constant_strength')
    model.run()









