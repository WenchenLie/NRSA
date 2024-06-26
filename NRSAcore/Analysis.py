import sys
import json
from typing import Literal
from pathlib import Path
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

import dill as pickle
import pandas as pd
from SeismicUtils.Records import Records
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from NRSAcore._Win import _Win
from utils.utils import SDOF_Error, LOGGER
from utils import utils


class SDOFmodel:
    g = 9800
    N_response_types = 11  # SDOF响应类型的数量

    def __init__(self,
            records_file: Path | str, 
            overview_file: Path | str,
            SDOFmodel_file: Path | str,
            output_dir: Path | str
        ):
        """导入地震动文件、模型概览、SDOF模型参数，设置输出文件夹路径

        Args:
            records_file (Path | str): 地震动文件(.pkl)
            overview_file (Path | str): 模型概览文件(.json)
            SDOFmodel_file (Path | str): SDOF模型参数(.csv)
            output_dir (Path | str): 输出文件夹路径
        """
        self.logger = LOGGER
        utils.check_file_exists(records_file)
        utils.check_file_exists(overview_file)
        utils.check_file_exists(SDOFmodel_file)
        utils.creat_folder(output_dir, 'overwrite')
        self.output_dir = Path(output_dir)
        self._read_files(records_file, overview_file, SDOFmodel_file)
        self._construct_QApp()
        self._get_task_info()

    def _read_files(self, records_file, overview_file, SDOFmodel_file):
        """打开三个文件，获得以下实例属性：
        * self.records (Records)
        * self.model_overview (dict)
        * self.model_paras (DataFrame)
        """
        # 读取地震动
        with open(records_file, 'rb') as f:
            self.records: Records = pickle.load(f)
        # 读取模型概览
        with open(overview_file, 'r') as f:
            self.model_overview: dict = json.load(f)
        # 读取模型参数
        self.model_paras = pd.read_csv(SDOFmodel_file)
        # with h5py.File(self.wkd / f'{self.model_name}_paras.h5', 'r') as f:
        #     columns = utils.decode_list(f['columns'][:])
        #     paras = f['parameters'][:]
        #     self.model_paras = pd.DataFrame(paras, columns=columns)
        #     self.model_paras['ID'] = self.model_paras['ID'].astype(int)
        #     self.model_paras['ground_motion'] = self.model_paras['ground_motion'].astype(int)
        # with h5py.File(self.wkd / f'{self.model_name}_spectra.h5', 'r') as f:
        #     self.model_spectra = {}
        #     for item in f:
        #         if item == 'T':
        #             self.model_spectra['T'] = f['T'][:]
        #         else:
        #             self.model_spectra[item] = {}
        #             self.model_spectra[item]['RSA'] = f[item]['RSA'][:]
        #             self.model_spectra[item]['RSV'] = f[item]['RSV'][:]
        #             self.model_spectra[item]['RSD'] = f[item]['RSD'][:]

    def _construct_QApp(self):
        app = QApplication.instance()
        if not app:
            QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
            self.app = QApplication(sys.argv)
        else:
            self.app = app

    def _get_task_info(self):
        """获取分析任务信息"""
        self.N_GM: int = self.model_overview['ground_motions']['number']  # 地震动数量
        self.N_SDOF: int = self.model_overview['N_SDOF']  # 单自由度总数量
        self.N_calc: int = self.model_overview['total_calculation']  # 所需总计算次数


    def set_analytical_options(self,
            analysis_type: Literal['constant_ductility', 'constant_strength'],
            fv_duration: float=0,
            PDelta: bool=False,
            batch: int=1,
            parallel: int=1,
            ductility_tol: float=0.01,
            auto_quit: bool=False,
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


    def run(self):
        """开始运行分析"""
        win = _Win(self, self.logger)
        win.show()
        self.app.exec_() 


if __name__ == "__main__":

    model = SDOFmodel(
        r'G:\NRSA_working\3046records.pkl',
        r'G:\NRSA_working\LCF_overview.json',
        r'G:\NRSA_working\LCF_SDOFmodels.csv',
        r'G:\NRSA_working'
    )
    model.set_analytical_options(
        'constant_strength',
        PDelta=False,
        batch=20,
        auto_quit=False,
        parallel=20
    )
    # model.run()









