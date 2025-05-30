import sys, os
import re
import time
import json
import multiprocessing
from math import isclose
from pathlib import Path
from typing import Literal, Callable

import shutil
import numpy as np
from .config import LOGGER, ANA_TYPE_NAME, AVAILABLE_SOLVERS, SOLVER_TYPING
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from .Win import Win
from .get_spectrum import get_spectrum


SKIP = False

class NRSA:
    cwd = Path().cwd()
    # T = np.arange(0.01, 6, 0.01)
    has_GM_data = False

    def __init__(self,
                 job_name: str,
                 cache_dir: Path | str='cache',
                 *,
                 analysis_type: Literal['CDA', 'CSA']=None):
        """非线性反应谱分析

        Args:
            job_name (str): 分析工况名
            cache_dir (bool, optional): 是否将地震动反应谱缓存，下次实例化时将自
              动读取缓存，默认为"cache"
            analysis_type (Literal['CDA', 'CSA']): CDA - Constant ductility analysis, CSA - Constant strength analysis
        """
        LOGGER.success(f'=============== {ANA_TYPE_NAME[analysis_type]} ===============')
        self.start_time = time.time()
        self.job_name = job_name
        self.analysis_type = analysis_type
        self.cache_dir = cache_dir
        self._init_variables()
        self._init_QApp()
        if not Path('logs').exists():
            os.makedirs('logs')
        LOGGER.success(f'Job name defined: {self.job_name}')

    def _init_QApp(self):
        app = QApplication.instance()
        if not app:
            QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
            self.app = QApplication(sys.argv)
        else:
            self.app = app

    def _init_variables(self):
        """初始化实例变量"""
        self.wkdir: Path = None
        self.period: np.ndarray = None
        self.material_function: Callable[[float, float, float, float], tuple[str, list, float, float]] = None
        self.material_paras: dict[str, float] = None
        self.damping = None
        self.target_ductility = None
        self.thetaD = 0
        self.mass = 1
        self.height = 1
        self.fv_duration: float = 0.0
        self.suffix: str
        self.GM_names: list[str]
        self.GM_folder: Path
        self.GM_N: int
        self.GM_global_sf: float
        self.GM_indiv_sf: list[float]
        self.scaling_finished = False
        self.GM_dts: list
        self.GM_NPTS: list[int]
        self.GM_durations: list[float]  # 不包括自由振动持续时间
        self.unscaled_RSA_5pct: np.ndarray  # 5%阻尼，无缩放反应谱
        self.unscaled_RSV_5pct: np.ndarray
        self.unscaled_RSD_5pct: np.ndarray
        self.parallel = 1
        self.auto_quit = False
        self.damping_equal_5pct = False
        self.unscaled_spectra_folder_5pct: Path  # 5%阻尼比反应谱路径
        self.unscaled_spectra_folder_spc: Path  # 指定阻尼比反应谱路径
        self.kwargs = {}  # 用于输入到求解器的参数

    def set_working_directory(self, wkdir: str | Path, folder_exists: Literal['ask', 'overwrite', 'delete']='ask'):
        """设置工作路径

        Args:
            wkdir (str | Path): 工作路径
            folder_exists (Literal['ask', 'overwrite', 'delete'], optional): 如果工作路径文件夹已存在，是否询问、覆盖、删除，默认询问
        """
        self.wkdir = Path(wkdir).absolute()
        if not self._check_path_name(self.wkdir):
            raise Exception('Analysis has been terminated')
        if not self._check_folder(self.wkdir, folder_exists):
            global SKIP
            SKIP = True
            return
        if not Path(self.wkdir / 'results').exists():
            os.makedirs(self.wkdir / 'results')
        LOGGER.success(f'Working directory has been set: {self.wkdir.as_posix()}')

    def analysis_settings(self,
            period: np.ndarray,
            material_function: Callable[[float, float, float, float], tuple[str, list, float, float]],
            material_paras: dict[str, tuple | float],
            damping: float,
            target_ductility: float,
            R_init: float,
            R_incr: float,
            tol_ductility: float,
            tol_R: float,
            max_iter: int,
            thetaD: float=0,
            mass: float=1,
            height: float=1,
            fv_duration: float=0.0,
        ):
        """设置分析参数

        Args:
            period (np.ndarray): 等延性谱周期序列
            material_function (Callable[[float, float, float, float], tuple[str, list, float, float]]): 获取opensees材料格式的函数
            material_paras (dict[str, float]): 材料定义所需参数
            damping (float): 阻尼比
            target_ductility (float): 目标延性
            R_init (float): 初始强度折减系数(R)
            R_incr (float): 强度折减系数(R)递增值
            tol_ductility (float): 延性(μ)收敛容差
            tol_R (float): 相邻强度折减系数(R)收敛容差
            max_iter (int): 最大迭代次数
            thetaD (float): P-Delta系数
            mass (float): 质量，默认1
            height (float, optional): 高度，默认1
            fv_duration (float, optional): 自由振动持续时间，默认0.0
        
        Converge criteria for constant ductility analysis:
        --------------------------------------------------
        `abs(μ - μ_target) / μ_target < tol_ductility`  
        `abs(R1 - R2) / R2 < tol_R`  
        where R1 and R2 are the adjacent R values.

        Note:
        -----
        * `mass`不会影响`R`和具有长度和时间量纲的响应，但会影响具有力量纲的响应，
        力量纲响应与`mass`成正比
        * 等延性分析中，延性容差`tol_ductility`建议不低于0.01，强度折减系数`R`容差
        建议不低于0.001
        """
        # 只做相关检查，不做实际的操作
        if period[0] == 0:
            LOGGER.error('The first period cannot be 0')
            raise Exception('Analysis has been terminated')
        for key in material_paras.keys():
            if not isinstance(key, str):
                LOGGER.error(f'The material parameter key should be a string: {key}')
                raise Exception('Analysis has been terminated')
        if not isinstance(damping, (int, float)) and damping < 0:
            LOGGER.error(f'The damping ratio should be a non-negative number: {damping}')
            raise Exception('Analysis has been terminated')
        if not isinstance(target_ductility, (int, float)) and target_ductility <= 0:
            LOGGER.error(f'The target ductility should be a positive number: {target_ductility}')
            raise Exception('Analysis has been terminated')
        if not isinstance(max_iter, int) and max_iter <= 1:
            LOGGER.error(f'The maximum iteration should be a positive integer greater than 1: {max_iter}')
            raise Exception('Analysis has been terminated')
        if not isinstance(thetaD, (int, float)) and thetaD < 0:
            LOGGER.error(f'The P-Delta coefficient should be a non-negative number: {thetaD}')
            raise Exception('Analysis has been terminated')
        if not isinstance(mass, (int, float)) and mass <= 0:
            LOGGER.error(f'The mass should be a positive number: {mass}')
            raise Exception('Analysis has been terminated')
        if not isinstance(height, (int, float)) and height <= 0:
            LOGGER.error(f'The height should be a positive number: {height}')
            raise Exception('Analysis has been terminated')
        if not isinstance(fv_duration, (int, float)) and fv_duration < 0:
            LOGGER.error(f'The free vibration duration should be a non-negative number: {fv_duration}')
            raise Exception('Analysis has been terminated')
        if isclose(damping, 0.05):
            self.damping_equal_5pct = True
        self.period = np.array(period)
        self.material_function = material_function
        self.material_paras = material_paras
        self.mass = mass
        self.damping = damping
        self.thetaD = thetaD
        self.target_ductility = target_ductility
        self.R_init = R_init
        self.R_incr = R_incr
        self.tol_ductility = tol_ductility
        self.tol_R = tol_R
        self.max_iter = max_iter
        self.height = height
        self.fv_duration = fv_duration
        LOGGER.success('Analysis settings have been set')

    def select_ground_motions(self,
            GM_folder: str | Path,
            GMs: list[str],
            suffix: str='.txt',
            th_scaling: float=1.0,
        ):
        """选择地震动加速度时程文件，所有地震地震动应放在一个文件夹中，并在该文件夹中有一个GM_info.json文件，该文件应包含地震动文件名及其时间间隔  

        Args:
            GM_folder (str | Path): 地震动文件所在文件夹
            GMs (list[str]): 一个包含所有地震动文件名(不包括后缀)的列表  
            suffix (str, optional): 地震动文件后缀，默认为.txt
            th_scaling (float, optional): 地震动时程缩放系数(在读取时程数据时使用)

        Example:
            >>> select_ground_motions('.data/GMs', GMs=['GM1', 'GM2'], suffix='.txt')
        
        Note:
        -----
        运行等延性分析时，`th_scaling`不会影响`R`值和其他无量纲响应的结果，但会影响有量纲响应的结果。
        """
        # 只统计地震动的数据位置、数量、时间间隔等信息，不存储地震动时程数据
        GM_folder = Path(GM_folder)
        if not self._check_path_name(GM_folder):
            raise Exception('Analysis has been terminated')
        self.suffix = suffix
        self.GM_names = GMs
        with open(GM_folder / 'GM_info.json', 'r') as f:
            dt_dict = json.loads(f.read())
        self.GM_dts, self.GM_NPTS, self.GM_durations = [], [], []
        for name in self.GM_names:
            self.GM_dts.append(dt_dict[name])
            th = np.loadtxt(GM_folder / f'{name}{suffix}')
            self.GM_NPTS.append(len(th))
            self.GM_durations.append(round((len(th) - 1) * dt_dict[name], 6))
        self.GM_N = len(self.GM_names)
        self.GM_folder = GM_folder
        self.GM_global_sf = th_scaling
        self.GM_indiv_sf = [1] * self.GM_N
        spectra_folder = self.wkdir / 'Unscaled 5%-damping spectra'
        if not Path(spectra_folder).exists():
            os.makedirs(spectra_folder)
        self.unscaled_RSA_5pct = np.zeros((len(self.period), self.GM_N))
        self.unscaled_RSV_5pct = np.zeros((len(self.period), self.GM_N))
        self.unscaled_RSD_5pct = np.zeros((len(self.period), self.GM_N))
        self.unscaled_RSA_spc = np.zeros((len(self.period), self.GM_N))  # 特定阻尼比反应谱
        for i in range(self.GM_N):
            print(f'  Calculating unscaled 5%-damping response spectra ({i+1}/{self.GM_N})   \r', end='')
            th = np.loadtxt(self.GM_folder / f'{self.GM_names[i]}{self.suffix}') * self.GM_global_sf
            RSA, RSV, RSD = get_spectrum(ag=th, dt=self.GM_dts[i], T=self.period, zeta=0.05, cache_dir=self.cache_dir)
            np.savetxt(spectra_folder / f'RSA_{self.GM_names[i]}.txt', RSA)
            self.unscaled_RSA_5pct[:, i] = RSA
            self.unscaled_RSA_spc[:, i] = RSA  # 先用5%阻尼比计算反应谱，如果分析用阻尼比不等于5%再进行计算
            self.unscaled_RSV_5pct[:, i] = RSV
            self.unscaled_RSD_5pct[:, i] = RSD
        print('', end='')
        np.savetxt(spectra_folder / 'Periods.txt', self.period)
        LOGGER.success(f'{self.GM_N} ground motion records have been selected')

    def running_settings(self,
            parallel: int=1,
            auto_quit: bool=False,
            hidden_prints: bool=True,
            show_monitor: bool=True,
            solver: SOLVER_TYPING='auto',
            **kwargs
        ):
        """运行设置

        Args:
            parallel (int, optional): 并行计算数，0-取CPU核心数，1-单进程，N-N进程
            auto_quit (bool, optional): 运行结束后是否自动关闭监控器
            hidden_prints (bool, optional): 是否隐藏求解过程中的输出，默认为True
            show_monitor (bool, optional): 是否显示监控器，默认为True
            solver (SOLVER_TYPING, optional): 求解器类型，默认为'auto'，
              即按照'Newmark-Newton'->'OPS'的顺序选择，不收敛则向后切换
            **kwargs: 用于输入到求解器的参数
        """
        if not isinstance(parallel, int) and parallel < 0:
            LOGGER.error(f'The parallel parameter should be a non-negative integer: {parallel}')
            raise Exception('Analysis has been terminated')
        if parallel == 0:
            parallel = multiprocessing.cpu_count()
        if solver not in AVAILABLE_SOLVERS:
            raise ValueError(f'Solver should be one of {AVAILABLE_SOLVERS}')
        self.parallel = parallel
        self.gm_batch_size = 1  # 每个进程处理的地震动数量，暂时不支持
        self.auto_quit = auto_quit
        self.solver = solver
        self.hidden_prints = hidden_prints
        self.show_monitor = show_monitor
        self.kwargs = kwargs

    def run(self):
        if (not self.damping_equal_5pct) and (self.analysis_type == 'CDA'):
            self._write_unscaled_spectra_with_specific_damping()
        job = {}
        job['Job name'] = self.job_name
        job['Time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        job['Working directory'] = self.wkdir.as_posix()
        job['Material function'] = self.material_function.__name__
        job['Material parameters'] = self.material_paras
        job['Damping ratio'] = self.damping
        job['Target ductility'] = self.target_ductility
        job['P-Delta coefficient'] = self.thetaD
        job['Mass'] = self.mass
        job['Height'] = self.height
        job['Free vibration duration'] = self.fv_duration
        job['Ground motion suffix'] = self.suffix
        job['Parallel'] = self.parallel
        job['Groung motion folder'] = self.GM_folder.as_posix()
        job['Number of ground motions'] = self.GM_N
        job['Ground motion scaling factor'] = self.GM_global_sf
        job['Periods'] = self.period.tolist()
        job['Ground motion names'] = self.GM_names
        job['Ground motion time intervals'] = self.GM_dts
        job['Ground motion lengths'] = self.GM_NPTS
        job['Ground motion durations'] = self.GM_durations
        with open(self.wkdir / 'job.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(job, indent=4, ensure_ascii=False))
        if not self.show_monitor:
            self.auto_quit = True
        win = Win(self)
        if self.show_monitor:
            win.show()
        self.app.exec_()

    def _write_unscaled_spectra_with_specific_damping(self):
        """如果分析阻尼比不等于，则重新计算特定阻尼比反应谱并幅值路径"""
        spectra_folder = self.wkdir / f'Unscaled {self.damping:.2%}-damping spectra'
        if not Path(spectra_folder).exists():
            os.makedirs(spectra_folder)
        for i in range(self.GM_N):
            print(f'  Calculating unscaled {self.damping:.2%}-damping response spectra ({i+1}/{self.GM_N})   \r', end='')
            th = np.loadtxt(self.GM_folder / f'{self.GM_names[i]}{self.suffix}')
            RSA, _, _ = get_spectrum(ag=th, dt=self.GM_dts[i], T=self.period, zeta=self.damping, cache_dir=self.cache_dir)
            self.unscaled_RSA_spc[:, i] = RSA
            np.savetxt(spectra_folder / f'RSA_{self.GM_names[i]}.txt', RSA)
        print('', end='')
        np.savetxt(spectra_folder / 'Periods.txt', self.period)
        self.unscaled_spectra_folder_spc = spectra_folder

    @staticmethod
    def _check_path_name(*paths: Path):
        """检查路径名是否包含中文字符"""
        for path in paths:
            if re.search('[！@#￥%……&*（）—【】：；“‘”’《》，。？、\u4e00-\u9fff]', path.as_posix()):
                LOGGER.error(f'Chineses or full-width characters are not allowed in path names: {path.as_posix()}')
                return False
        return True

    @staticmethod
    def _check_folder(folder, folder_exists: Literal['ask', 'overwrite', 'delete']):
        if folder_exists not in ['ask', 'overwrite', 'delete']:
            raise ValueError(f'`folder_exists` should be "ask", "overwrite", or "delete"：{folder_exists}')
        folder = Path(folder)
        # 判断输出文件夹是否存在
        if os.path.exists(folder):
            if folder_exists == 'ask':
                res1 = QMessageBox.question(None, 'Warning', f'"{folder}" already exists, do you want to delete it?')
                if res1 == QMessageBox.Yes:
                    shutil.rmtree(folder)
                    os.makedirs(folder)
                    return True
                else:
                    res2 = QMessageBox.question(None, 'Warning', 'Do you want to overwrite the data?')
                    if res2 == QMessageBox.Yes:
                        return True
                    else:
                        LOGGER.warning('Analysis has been terminated')
                        return False
            elif folder_exists == 'overwrite':
                return True
            elif folder_exists == 'delete':
                shutil.rmtree(folder)
                LOGGER.warning(f'"{folder.as_posix()}" exists, has been deleted')
                os.makedirs(folder)
                return True
        else:
            os.makedirs(folder)
            return True

    def __getattribute__(self, name):
        if SKIP == True:
            return lambda *args, **kwargs: None  # 跳过分析
        else:
            return super().__getattribute__(name)
