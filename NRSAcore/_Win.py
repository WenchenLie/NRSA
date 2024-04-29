from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Analysis import SDOFmodel
import os
import sys
import time
import datetime
import multiprocessing
from typing import Literal
from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QWidget
from ui.Win import Ui_Win
from NRSAcore.SDOF_solver import *


class _Win(QDialog):

    def __init__(self, task: SDOFmodel) -> None:
        """监控窗口

        Args:
            task (SDOFmodel): SDOFmodel类的实例
        """
        super().__init__()
        self.ui = Ui_Win()
        self.ui.setupUi(self)
        self.task = task
        self.init_ui()


    def init_ui(self):
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint)
        self.ui.pushButton.clicked.connect(self.kill)
        time_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.ui.label.setText(f'开始时间：{time_start}')
        if self.task.analysis_type == 'constant_ductility':
            self.ui.label_2.setText('分析类型：等延性')
        elif self.task.analysis_type == 'constant_strength':
            self.ui.label_2.setText('分析类型：性能需求')
        self.ui.label_4.setText(f'地震动数量：{self.task.GM_N}')
        if self.task.PDelta:
            self.ui.label_5.setText('P-Delta效应：考虑')
        else:
            self.ui.label_5.setText('P-Delta效应：不考虑')
        self.ui.label_3.setText(f'SDOF数量：{self.task.N_SDOF}')

        
    def kill(self):
        """点击中断按钮"""
        pass  # TODO

    def run(self):
        self.worker = Worker(self.task, self)
        self.worker.start()



class Worker(QThread):
    """处理计算任务的子线程"""

    def __init__(self, task: SDOFmodel, win: _Win) -> None:
        super().__init__()
        self.task = task
        self.win = win

    def kill(self):
        """中断计算"""
        pass  # TODO

    def run(self):
        """开始运行子线程"""
        if self.task.analysis_type == 'constant_ductility':
            self.run_constant_ductility()
        elif self.task.analysis_type == 'constant_strength':
            self.run_constant_strength()

    def run_constant_ductility(self):
        """等延性分析"""
        pass  # TODO

    def run_constant_strength(self):
        """等强度分析"""
        pass  # TODO
    