import sys
import json
from typing import Literal
from pathlib import Path

import dill as pickle
import pandas as pd
from SeismicUtils.Records import Records
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from NRSAcore._Win import _Win
from utils.utils import SDOFError, LOGGER
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
        self.is_restart = False  # 是否属于重启动
        self.finished_id: list[int] = []  # 已完成的SDOF编号
        self.finished_gm: list[str] = []  # 已完成的地震动
        utils.check_file_exists(records_file)
        utils.check_file_exists(overview_file)
        utils.check_file_exists(SDOFmodel_file)
        utils.creat_folder(output_dir, 'overwrite')
        self.output_dir = Path(output_dir)
        self._read_files(records_file, overview_file, SDOFmodel_file)
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
            solver: Literal['SDOF_solver', 'SDOF_batched_solver', 'PDtSDOF_batched_solver']=None,
            save_interval: float=None):
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
            save_interval (float, optional): 保存间隔(s)，若不指定则不定时保存
        """
        if analysis_type not in ['constant_ductility', 'constant_strength']:
            raise SDOFError(f'未知分析类型：{analysis_type}')
        if not isinstance(batch, int):
            raise SDOFError('参数 batch 应为整数')
        if batch < 1:
            raise SDOFError('参数 batch 应大于等于1')
        if not isinstance(parallel, int):
            raise SDOFError('参数 parallel 应为整数')
        if parallel < 1:
            raise SDOFError('参数 parallel 应大于等于1')
        if batch == 1 and not PDelta:
            func_type = 1
        elif batch > 1 and not PDelta:
            func_type = 2
        elif PDelta:
            func_type = 3
        else:
            raise SDOFError('Error - 1')
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
                raise SDOFError(f'未知求解器类型：{solver}')
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
        if save_interval is None:
            save_interval = 1e10
        else:
            self.logger.info(f'结果将每隔 {save_interval}s 保存一次')
        self.solver = solver
        self.save_interval = save_interval
    
    def run(self):
        """开始运行分析"""
        try:
            import tables
        except ImportError:
            raise ImportError('请先使用pip或conda安装tables库')  # 否则无法利用pandas读写.h5文件
        self._get_analysis_options()
        if not self.is_restart:
            model_name = self.model_overview['model_name']
            with open(self.output_dir/f'{model_name}.instance', 'wb') as f:
                pickle.dump(self.analysis_options, f)
        self._construct_QApp()
        win = _Win(self, self.logger)
        win.show()
        self.app.exec_()

    def _get_analysis_options(self):
        """获取set_analytical_options方法调用时传入的参数以及已经完成的id和gm"""
        self.analysis_options = {}
        self.analysis_options['analysis_type'] = self.analysis_type
        self.analysis_options['fv_duration'] = self.fv_duration
        self.analysis_options['PDelta'] = self.PDelta
        self.analysis_options['batch'] = self.batch
        self.analysis_options['parallel'] = self.parallel
        self.analysis_options['ductility_tol'] = self.ductility_tol
        self.analysis_options['auto_quit'] = self.auto_quit
        self.analysis_options['solver'] = self.solver
        self.analysis_options['save_interval'] = self.save_interval
        self.analysis_options['finished_id'] = []
        self.analysis_options['finished_gm'] = []
        self.analysis_options['verification_code'] = self.model_overview['verification_code']
    
    @classmethod
    def restart(cls,
            records_file: Path | str, 
            overview_file: Path | str,
            SDOFmodel_file: Path | str,
            output_dir: Path | str,
            pkl_file: Path | str
        ):
        """重启动分析

        Args:
            records_file (Path | str): 地震动文件(.pkl)
            overview_file (Path | str): 模型概览文件(.json)
            SDOFmodel_file (Path | str): SDOF模型参数(.csv)
            output_dir (Path | str): 输出文件夹路径
            pkl_file (Path | str): 上一次计算时生成的instance文件(.instance)
        """
        pkl_file = Path(pkl_file)
        utils.check_file_exists(pkl_file)
        with open(pkl_file, 'rb') as f:
            analysis_options: dict = pickle.load(f)
        instance = cls(records_file, overview_file, SDOFmodel_file, output_dir)
        # 检查校验码
        code1 = instance.model_overview['verification_code']
        code2 = analysis_options['verification_code']
        if not code1 == code2:
            raise SDOFError(f'{overview_file.name}与{pkl_file.name}的校验码不符')
        instance.set_analytical_options(
            analysis_options['analysis_type'],
            analysis_options['fv_duration'],
            analysis_options['PDelta'],
            analysis_options['batch'],
            analysis_options['parallel'],
            analysis_options['ductility_tol'],
            analysis_options['auto_quit'],
            analysis_options['solver'],
            analysis_options['save_interval'],
        )
        instance.finished_id = analysis_options['finished_id']
        instance.finished_gm = analysis_options['finished_gm']
        instance.is_restart = True
        return instance
        


if __name__ == "__main__":

    sys.path.append(str(Path(__file__).parent.parent.absolute()))
    model = SDOFmodel(
        r'G:\NRSA_working\3046records.pkl',
        r'G:\NRSA_working\LCF_overview.json',
        r'G:\NRSA_working\LCFSDOFmodels.csv',
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









