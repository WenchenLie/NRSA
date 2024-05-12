import sys
import json
from typing import Literal
from pathlib import Path
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

import h5py
import pandas as pd 
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from NRSAcore._Win import _Win
from utils.utils import SDOF_Error, LOGGER, check_file_exists
from utils import utils


class SDOFmodel:

    dir_main = Path(__file__).parent.parent
    dir_input = dir_main / 'Input'
    dir_gm = dir_input / 'GMs'

    def __init__(self, model_name: str, working_directory: str | Path):
        """导入模型，形成以下三个实例属性
        * model_overview
        * model_paras
        * model_spectra

        Args:
            model_name (str): 模型名称
            working_directory (str | Path): 工作路径文件夹
        """
        self.logger = LOGGER
        self.model_name = model_name
        self.wkd = Path(working_directory)
        check_file_exists(self.wkd / f'{model_name}_overview.json')
        check_file_exists(self.wkd / f'{model_name}_paras.h5')
        check_file_exists(self.wkd / f'{model_name}_spectra.h5')
        self._read_files()
        self._construct_QApp()
        self._get_task_info()


    def _read_files(self):
        """打开三个文件"""
        with open(self.wkd / f'{self.model_name}_overview.json', 'r') as f:
            self.model_overview: dict = json.load(f)
        with h5py.File(self.wkd / f'{self.model_name}_paras.h5', 'r') as f:
            columns = utils.decode_list(f['columns'][:])
            paras = f['parameters'][:]
            self.model_paras = pd.DataFrame(paras, columns=columns)
            self.model_paras['ID'] = self.model_paras['ID'].astype(int)
            self.model_paras['ground_motion'] = self.model_paras['ground_motion'].astype(int)
        with h5py.File(self.wkd / f'{self.model_name}_spectra.h5', 'r') as f:
            self.model_spectra = {}
            for item in f:
                if item == 'T':
                    self.model_spectra['T'] = f['T'][:]
                else:
                    self.model_spectra[item] = {}
                    self.model_spectra[item]['RSA'] = f[item]['RSA'][:]
                    self.model_spectra[item]['RSV'] = f[item]['RSV'][:]
                    self.model_spectra[item]['RSD'] = f[item]['RSD'][:]


    def _construct_QApp(self):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        self.app = QApplication(sys.argv)


    def _get_task_info(self):
        """获取分析任务信息"""
        self.GM_N = len(self.model_overview['ground_motions']['name_dt_SF'])  # 地震动数量
        self.N_SDOF = self.model_overview['N_SDOF']  # 单自由度总数量


    def set_analytical_options(self,
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
        from NRSAcore._Win import FUNC
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
        win = _Win(self, self.logger)
        win.show()
        self.app.exec_() 


if __name__ == "__main__":
    # SDOFmodel.dir_gm = Path(r'F:\重要数据\小波库\7Records')
    # _Win.dir_gm = Path(r'F:\重要数据\小波库\7Records')
    model = SDOFmodel('LCF', r'G:\LCFwkd')
    model.set_analytical_options(
        'constant_strength',
        PDelta=False,
        batch=10,
        auto_quit=False,
        parallel=1
    )
    model.run()









