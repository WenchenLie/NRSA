import sys
import json
import shutil
import numpy as np
from typing import Literal
from pathlib import Path
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

from loguru import logger
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from NRSAcore._Win import _Win
from utils.utils import SDOF_Error
from utils import utils


class SDOFmodel:

    dir_main = Path(__file__).parent.parent
    dir_temp = dir_main / 'temp'
    dir_input = dir_main / 'Input'
    dir_output = dir_main / 'Output'
    dir_gm = dir_input / 'GMs'

    def __init__(self, json_file: Path | str=None, task: dict=None):
        """导入模型

        Args:
            json_file (Path | str, optional): 从json文件导入，默认None
            task (dict, optional): 从Task类生成的字典导入
        """
        self.logger = logger
        if json_file is None and task is None:
            raise SDOF_Error('参数 json_file 与 task 至少应指定一个')
        if json_file:
            with open(json_file, 'r') as f:
                self.task = json.loads(f.read())
                self.logger.success(f'已导入：{json_file.name}')
        if task:
            self.task = task
        self.json_file = json_file
        self.model_name = json_file.stem
        self.construct_QApp()
        self._get_task_info()


    def construct_QApp(self):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        self.app = QApplication(sys.argv)


    def _get_task_info(self):
        """获取分析任务信息"""
        self.GM_N = len(self.task['ground_motions']['dt_SF'])  # 地震动数量
        self.N_SDOF = len(self.task['SDOF_models'])


    def set_analytical_options(self,
            output_dir: Path | str,
            analysis_type: Literal['constant_ductility', 'constant_strength'],
            fv_duration: float=0,
            PDelta: bool=False,
            batch: int=1,
            parallel: int=1,
            ductility_tol: float=0.01,
            auto_quit: bool=False,
            g: float=9800,
            solver: Literal['SDOF_solver', 'SDOF_batched_solver', 'PDtSDOF_batched_solver']=None):
        """设置分析参数

        Args:
            output_dir (Path | str): 用于储存计算结果的文件夹
            analysis_type (Literal['constant_ductility', 'constant_strength']): 分析类型，等延性或等屈服强度
            fv_duration (float, optional): 自由振动时长
            PDelta (bool, optional): 是否考虑P-Delta效应，默认False
            batch (int, optional): 在同一模型空间下建立的SDOF数量，默认1(该值不影响计算结果，但可能影响计算效率)
            parallel (int, optional): 是否开启多进程并行计算，默认0，即不开启，不为0时为开启的进程数量，
            每个子进程处理一条地震波
            ductility_tol (float, optional): 等延性分析时目标延性的收敛容差，默认0.01
            auto_quit (bool, optional): 分析完成后是否自动关闭监控窗口，默认否
            g (float, optional)：重力加速度(mm/s^2)，默认9800
            solver (str, optional): 指定SDOF求解器，通常会自动选择，也可手动指定
        """
        if analysis_type not in ['constant_ductility', 'constant_strength']:
            raise SDOF_Error(f'未知分析类型：{analysis_type}')
        if not isinstance(batch, int):
            raise SDOF_Error('参数 batch 应为整数')
        if batch < 1:
            raise SDOF_Error('参数 batch 应大于等于1')
        if not isinstance(parallel, int):
            raise SDOF_Error('参数 parallel 应为整数')
        if parallel < 1:
            raise SDOF_Error('参数 parallel 应大于等于1')
        output_dir = Path(output_dir)
        self.output_dir = output_dir
        if batch == 1 and not PDelta:
            func_type = 1
        elif batch > 1 and not PDelta:
            func_type = 2
        elif PDelta:
            func_type = 3
        else:
            raise SDOF_Error('Error - 1')
        if solver:
            if solver == 'SDOF_solver':
                func_type = 1
                batch = 1
                PDelta = False
            elif solver == 'SDOF_batched_solver':
                func_type = 2
                PDelta = False
            elif solver == 'PDtSDOF_batched_solver':
                func_type = 3
                PDelta = True
            else:
                raise SDOF_Error(f'未知求解器类型：{solver}')
        from _Win import FUNC
        solver_name = FUNC[func_type].__name__
        self.logger.info(f'SDOF求解器：{solver_name}')
        self.func_type = func_type
        self.analysis_type = analysis_type
        self.fv_duration = fv_duration
        self.PDelta = PDelta
        self.batch = batch
        self.parallel = parallel
        self.ductility_tol = ductility_tol
        self.auto_quit = auto_quit
        self.g = g


    def run(self):
        """开始运行分析"""
        if not utils.creat_folder(self.output_dir):
            self.logger.warning('已退出分析')
            return
        win = _Win(self, self.logger)
        win.show()
        self.app.exec_() 


if __name__ == "__main__":
    model = SDOFmodel(json_file=Path(__file__).parent.parent/'temp'/'model.json')
    model.set_analytical_options(
        Path(__file__).parent.parent / 'Output',
        'constant_strength',
        PDelta=True,
        batch=10,
        auto_quit=False,
        parallel=10
    )
    model.run()









