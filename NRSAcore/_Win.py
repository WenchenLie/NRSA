from __future__ import annotations
import multiprocessing.managers
import multiprocessing.queues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Analysis import SDOFmodel
import time
import traceback
import multiprocessing
from pathlib import Path

import h5py
import loguru
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QMessageBox, QDialog

from NRSAcore.Task import Task
from NRSAcore.SDOF_solver import *
from ui.Win import Ui_Win
from utils.utils import SDOF_Error


FUNC = {
    1: SDOF_solver,  # 单个SDOF
    2: SDOF_batched_solver,  # 批量SDOF
    3: PDtSDOF_batched_solver,  # 批量SDOD且考虑PDelta
}


class _Win(QDialog):
    def __init__(self, task: SDOFmodel, logger: loguru.Logger) -> None:
        """监控窗口

        Args:
            task (SDOFmodel): SDOFmodel类的实例
            logger (loguru.Logger): 日志
        """
        super().__init__()
        self.ui = Ui_Win()
        self.ui.setupUi(self)
        self.logger = logger
        self.records = task.records
        self.model_overview = task.model_overview
        self.model_paras = task.model_paras
        self.output_dir = task.output_dir
        self.N_GM = task.N_GM
        self.N_SDOF = task.N_SDOF
        self.N_calc = task.N_calc
        self.func_type = task.func_type
        self.analysis_type = task.analysis_type
        self.fv_duration = task.fv_duration
        self.PDelta = task.PDelta
        self.batch = task.batch
        self.parallel = task.parallel
        self.ductility_tol = task.ductility_tol
        self.auto_quit = task.auto_quit
        self.g = task.g
        self.N_response_types = task.N_response_types
        self.init_ui()
        self.run()

    def init_ui(self):
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint)
        self.ui.pushButton.clicked.connect(self.kill)
        time_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.ui.label.setText(f'开始时间：{time_start}')
        if self.analysis_type == 'constant_ductility':
            self.ui.label_2.setText('分析类型：等延性')
        elif self.analysis_type == 'constant_strength':
            self.ui.label_2.setText('分析类型：性能需求')
        self.ui.label_4.setText(f'地震动数量：{self.N_GM}')
        if self.PDelta:
            self.ui.label_5.setText('P-Delta效应：考虑')
        else:
            self.ui.label_5.setText('P-Delta效应：不考虑')
        model_name = self.model_overview['model_name']
        self.ui.label_10.setText(f'任务名：{model_name}')
        self.ui.label_3.setText(f'SDOF数量：{self.N_SDOF}')
        self.ui.label_9.setText(f'总计算数：{self.N_calc}')
        self.ui.label_8.setText(f'SDOF求解器：{FUNC[self.func_type].__name__}')
        self.ui.pushButton_3.clicked.connect(self.pause_resume)

    def kill(self):
        """点击中断按钮"""
        if QMessageBox.question(self, '警告', '是否中断计算？') == QMessageBox.Yes:
            self.logger.error('已中断计算')
            self.worker.kill()

    def pause_resume(self):
        """暂停或恢复计算"""
        if self.ui.pushButton_3.text() == '暂停':
            self.ui.pushButton_3.setText('继续')
            self.ui.groupBox_2.setTitle('计算信息（已暂停）')
            self.logger.warning('已暂停计算')
        else:
            self.ui.pushButton_3.setText('暂停')
            self.ui.groupBox_2.setTitle('计算信息')
            self.logger.warning('已继续计算')
        self.worker.pause_resume()

    def run(self):
        self.worker = _Worker(self)
        self.worker.signal_set_progressBar.connect(self.set_progressBar)
        self.worker.signal_set_finished_SDOF.connect(self.set_finished_SDOF)
        self.worker.signal_finish_all.connect(self.finish_all)
        self.worker.start()
    
    def set_progressBar(self, tuple_):
        """设置进度条(进度值, 文本)"""
        val, text = tuple_
        self.ui.label_6.setText(text)
        self.ui.progressBar.setValue(val)
    
    def set_finished_SDOF(self, int_):
        """设置显示已完成SDOF的数量"""
        self.ui.label_7.setText(f'已计算次数：{int_}')

    def finish_all(self):
        self.ui.pushButton_2.setEnabled(True)
        if self.auto_quit:
            self.ui.pushButton_2.click()


class _Worker(QThread):
    """处理计算任务的子线程"""
    signal_set_progressBar = pyqtSignal(tuple)
    signal_set_finished_SDOF = pyqtSignal(int)
    signal_finish_all = pyqtSignal()

    def __init__(self, win: _Win) -> None:
        super().__init__()
        self.win = win
        self.model_overview = win.model_overview
        self.model_paras = win.model_paras
        self.records = win.records
        self.output_dir = win.output_dir
        self.N_GM = win.N_GM
        self.N_SDOF = win.N_SDOF
        self.N_calc = win.N_calc
        self.func_type = win.func_type
        self.analysis_type = win.analysis_type
        self.fv_duration = win.fv_duration
        self.PDelta = win.PDelta
        self.batch = win.batch
        self.parallel = win.parallel
        self.ductility_tol = win.ductility_tol
        self.auto_quit = win.auto_quit
        self.g = win.g
        self.N_response_types = win.N_response_types
        self.logger = self.win.logger
        # 新定义的实例属性
        self.reuslt_column = ['id', 'converge', 'collapse', 'maxDisp', 'maxVel',
            'maxAccel', 'Ec', 'Ev', 'maxReaction', 'CD', 'CPD', 'resDisp']
        self.results_L1 = np.zeros((self.N_calc, len(self.reuslt_column)))
        self.results_L1[:, 0] = np.arange(1, len(self.results_L1) + 1, 1)
        self.model_name = self.model_overview['model_name']
        self.queue = multiprocessing.Manager().Queue()  # 进程通信
        self.stop_event = multiprocessing.Manager().Event()  # 中断事件
        self.pause_event = multiprocessing.Manager().Event()  # 暂停事件
        self.pause_event.set()
        self.lock = multiprocessing.Manager().Lock()  # 进程锁

    # def creat_shared_var(self):
    #     """创建多进程共享变量"""
    #     self.results_L1 = multiprocessing.Array('f', np.prod(self.N_calc, len(self.reuslt_column)), lock=False)
    #     self.results_L1[:, 0] = np.arange(1, len(self.results_L1) + 1, 1)
    #     self.N_response_types = multiprocessing.Value('H', self.N_response_types, lock=False)
    #     self.test_var = '----test----'  # HACK:

    def kill(self):
        """中断计算"""
        self.stop_event.set()

    def pause_resume(self):
        """暂定或恢复计算"""
        if self.pause_event.is_set():
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def run(self):
        """开始运行子线程"""
        if self.analysis_type == 'constant_ductility':
            self.run_constant_ductility()
        elif self.analysis_type == 'constant_strength':
            self.run_constant_strength()
 
    def run_constant_ductility(self):
        """等延性分析"""
        s = '（多进程）' if self.parallel > 1 else ''
        self.logger.success(f'开始进行：等延性谱分析{s}')
        ...  # TODO: 等延性分析

    def run_constant_strength(self):
        """等强度分析，使用records.pkl的地震动(无缩放)"""
        s = '（多进程）' if self.parallel > 1 else ''
        self.logger.success(f'开始进行：性能需求谱分析{s}')
        ls_paras: list[tuple] = []
        queue = self.queue
        stop_event = self.stop_event
        pause_event = self.pause_event
        lock = self.lock
        batch = self.batch
        SF = 1
        for gm_idx, (gm, dt) in enumerate(self.records.get_unscaled_records()):
            gm_name = self.records.get_record_name()[gm_idx]
            df_paras = self.model_paras[self.model_paras['ground_motion']==int(gm_idx+1)]
            ls_SDOF = df_paras['ID'].to_list()
            args = (self.N_response_types, queue, stop_event, pause_event, lock, self.func_type,
                    self.model_overview, self.model_paras, ls_SDOF, batch, gm, dt, self.fv_duration,
                    SF, self.g, gm_name)
            ls_paras.append(args)
        with multiprocessing.Pool(self.parallel) as pool:
            for i in range(self.N_GM):
                # 一个进程处理一条地震动的所有计算
                pool.apply_async(_run_constant_strength, ls_paras[i])  # 设置进程池
            self.get_queue(queue)
            pool.close()
            pool.join()

    def get_queue(self, queue: multiprocessing.Queue):
        """进程通讯"""
        finished_GM = 0  # 已计算完成的地震动
        finished_SDOF = 0  # 已计算完成的SDOF
        while True:
            if not queue.empty():
                for key, value in queue.get().items():
                    flag, contents = key, value
                if flag == 'a':  # 地震动完成
                    # other: 该条地震动计算的SDOF数量
                    finished_GM += 1
                    finished_SDOF += contents
                    self.signal_set_progressBar.emit((int(finished_GM/self.N_GM*100), f'已计算地震动：{finished_GM}'))
                    self.signal_set_finished_SDOF.emit(finished_SDOF)
                elif flag == 'b':  # 传递该条地震动得到的结果
                    # 在地震动完成计算时收到
                    results_L2, ls_SDOF = contents
                    self.L2_to_L1(results_L2, ls_SDOF)
                elif flag == 'h':  # 中断计算
                    # other = '中断计算'
                    self.signal_finish_all.emit()
                    break
                elif flag == 'i':  # 异常
                    # other: 异常
                    print(contents[1])
                    raise contents[0]
                if finished_GM == self.N_GM:
                    # 所有计算完成
                    self.logger.success('计算完成')
                    self.logger.success(f'生成结果文件：{self.model_results}')
                    self.signal_finish_all.emit()
                    break

    def L2_to_L1(self,
            results_L2: np.ndarray,
            ls_SDOF: list[int],
        ):
        """将SDOF计算结果写入cls.results

        Args:
            results_L2 (np.ndarray): 某地震动下计算得到的L2结果
            ls_SDOF (list[int]): 结果对应的id
        """
        self.lock.acquire()
        try:
            if not len(results_L2) == len(ls_SDOF):
                # L1的行数与ls_SDOF的长度必须相等，保证结果与id一一对应
                self.logger.error(f'results_L2的长度({len(results_L2)})与ls_SDOF({len(ls_SDOF)})不一致')
                raise SDOF_Error('_Win.py, _write_results, Error - 1')
            for id_ in ls_SDOF:
                idx = id_ - 1
                self.results_L1[idx, 1:] = results_L2[idx]  # L2写入L1
        except Exception as error:
            self.lock.release()
            return error
        else:
            self.lock.release()
            return

def _run_constant_strength(*args, **kwargs):
    """SDOF计算函数，每次调用求解一条地震动

    Args (16):
        queue (multiprocessing.Queue): 进程通信
        stop_event (multiprocessing.Event): 进程终止事件
        pause_event (multiprocessing.Event): 进程暂停事件
        lock (multiprocessing.Lock): 进程锁
        func_type (int): SDOF求解器类型
        * 1 - 单个SDOF求解
        * 2 - 批量SDOF求解
        * 3 - 批量SDOF求解，同时可考虑P-Delta\n
        model_overview (dict): model_overview字典
        model_paras (Dataframe): model_paras表格
        ls_SDOF (list[int]): 该地震动所拥有的SDOF模型编号
        batch (int): 同一模型空间下所建立的SDOF模型数量

        后续参数见SDOF求解器
    """
    # results_L1: np.ndarray
    N_response_types: int
    queue: multiprocessing.Queue
    stop_event: multiprocessing.Event
    pause_event: multiprocessing.Event
    lock: multiprocessing.Lock
    func_type: int
    model_overview: dict
    model_paras: pd.DataFrame
    ls_SDOF: list[int]  # 待计算的模型编号
    batch: int  # 批量计算的数量
    gm: np.ndarray
    dt: float
    fv_duration: float
    SF: float
    g: float
    gm_name: str
    try:
        # results_L1, N_response_types,\
        N_response_types,\
        queue, stop_event, pause_event, lock, func_type, model_overview, model_paras,\
        ls_SDOF, batch, gm, dt, fv_duration, SF, g, gm_name = args
        func = FUNC[func_type]
        T_name: str = model_overview['basic_para']['period']
        zeta_name: str = model_overview['basic_para']['damping']
        m_name: str = model_overview['basic_para']['mass']
        P_name: str = model_overview['basic_para']['gravity']
        h_name: str = model_overview['basic_para']['height']
        uy_name: str = model_overview['basic_para']['yield_disp']
        collapseDisp_name: str = model_overview['basic_para']['collapse_disp']
        maxAnaDisp_name: str = model_overview['basic_para']['maxAnaDisp']
        mat_paras = model_overview['basic_para']['material_paras']
        materials = model_overview['material_format']
        finished_SDOF = 0  # 该地震动下已经计算完成的SDOF的数量
        results_L2 = np.zeros((len(ls_SDOF), N_response_types))  # 用于存储临时计算结果
        if func_type == 1:
            for idx, id_ in enumerate(ls_SDOF):
                id_: int  # 模型序号
                if stop_event.is_set():
                    queue.put({'h': '中断计算'})
                    return
                pause_event.wait()
                line = model_paras[model_paras['ID']==id_]
                T = line[T_name].item()
                zeta = line[zeta_name].item()
                m = line[m_name].item()
                uy = line[uy_name].item()
                materials = _parse_material(model_overview, model_paras, id_)
                if collapseDisp_name:
                    collapseDisp = line[collapseDisp_name].item()
                else:
                    collapseDisp = 1e10
                if maxAnaDisp_name:
                    maxAnaDisp = line[maxAnaDisp_name].item()
                else:
                    maxAnaDisp = 2e10
                results_L3 = SDOF_solver(T, gm, dt, materials, uy, fv_duration, SF, zeta, m, g,
                            collapseDisp, maxAnaDisp)
                results_L2[idx] = list(results_L3.values())
                finished_SDOF += 1
        elif func_type == 2:
            ls_batches = _split_batch(ls_SDOF, batch)
            line_idx = 0  # 从多少行开始写入results_L2
            for ls_batch in ls_batches:
                if stop_event.is_set():
                    queue.put({'h': '中断计算'})
                    return
                pause_event.wait()
                N_SDOFs = len(ls_batch)
                df = model_paras[model_paras['ID'].isin(ls_batch)]
                ls_T = df[T_name].to_list()
                ls_materials = tuple(_parse_material(model_overview, model_paras, id_) for id_ in ls_batch)
                ls_uy = df[uy_name].to_list()
                ls_SF = tuple(SF for _ in ls_batch)
                ls_zeta = df[zeta_name].to_list()
                ls_m = df[m_name].to_list()
                if collapseDisp_name:
                    ls_collapseDisp = df[collapseDisp_name].to_list()
                else:
                    ls_collapseDisp = tuple(1e10 for _ in ls_batch)
                if maxAnaDisp_name:
                    ls_maxAnaDisp = df[maxAnaDisp_name].to_list()
                else:
                    ls_maxAnaDisp = tuple(2e10 for _ in ls_batch)
                results_L3 = SDOF_batched_solver(N_SDOFs, ls_T, gm, dt, ls_materials, ls_uy, fv_duration, ls_SF,
                    ls_zeta, ls_m, g, ls_collapseDisp, ls_maxAnaDisp)
                # 将该batch的结果写入results_L2
                results_L2[line_idx: line_idx + N_SDOFs] = np.array(list(results_L3.values())).T
                line_idx += N_SDOFs
                finished_SDOF += N_SDOFs
        elif func_type == 3:
            ls_batches = _split_batch(ls_SDOF, batch)
            line_idx = 0  # 从多少行开始写入results_L2
            for ls_batch in ls_batches:
                if stop_event.is_set():
                    queue.put({'h': '中断计算'})
                    return
                pause_event.wait()
                N_SDOFs = len(ls_batch)
                df = model_paras[model_paras['ID'].isin(ls_batch)]
                ls_h = df[h_name].to_list()
                ls_T = df[T_name].to_list()
                ls_grav = df[P_name].to_list()
                ls_materials = tuple(_parse_material(model_overview, model_paras, id_) for id_ in ls_batch)
                ls_uy = df[uy_name].to_list()
                ls_SF = tuple(SF for _ in ls_batch)
                ls_zeta = df[zeta_name].to_list()
                ls_m = df[m_name].to_list()
                if collapseDisp_name:
                    ls_collapseDisp = df[collapseDisp_name].to_list()
                else:
                    ls_collapseDisp = tuple(1e10 for _ in ls_batch)
                if maxAnaDisp_name:
                    ls_maxAnaDisp = df[maxAnaDisp_name].to_list()
                else:
                    ls_maxAnaDisp = tuple(2e10 for _ in ls_batch)
                results_L3 = PDtSDOF_batched_solver(N_SDOFs, ls_h, ls_T, ls_grav, gm, dt, ls_materials, ls_uy, fv_duration, ls_SF,
                    ls_zeta, ls_m, g, ls_collapseDisp, ls_maxAnaDisp)
                # 将该batch的结果写入results_L2
                results_L2[line_idx: line_idx + N_SDOFs] = np.array(list(results_L3.values())).T
                line_idx += N_SDOFs
                finished_SDOF += N_SDOFs
        # error = _L2_to_L1(results_L1, results_L2, ls_SDOF, lock)
        # if error is not None:
        #     raise error
        queue.put({'b': (results_L2, ls_SDOF)})
        queue.put({'a': finished_SDOF})
    except Exception as e:
        tb = traceback.format_exc()
        queue.put({'i': (e, tb)})
        return

def _split_batch(ls_SDOF: list[int], batch: int) -> list[list[int]]:
    """拆分模型"""
    ls_SDOF_ = ls_SDOF.copy()
    ls_batch = []
    while True:
        ls_batch.append(ls_SDOF_[: batch])
        del ls_SDOF_[: batch]
        if not ls_SDOF_:
            break
    return ls_batch

def _parse_material(model_overview: dict, model_paras: pd.DataFrame, id_: int) -> dict:
    """解析材料字典"""
    old_materials: dict = model_overview['material_format']
    materials = {}
    for matType, old_paras in old_materials.items():
        paras = []
        for old_para in old_paras:
            para = Task.identify_para(old_para)
            if para:
                paras.append(model_paras[model_paras['ID']==id_][para].item())
            else:
                paras.append(old_para)
        materials[matType] = paras
    return materials

def _L2_to_L1(
        results_L1: np.ndarray,
        results_L2: np.ndarray,
        ls_SDOF: list[int],
        lock: multiprocessing.Lock
    ):
    """将SDOF计算结果写入cls.results

    Args:
        results_L1 (np.ndarray): 所有分析结果
        results_L2 (np.ndarray): 某地震动下计算得到的L2结果
        ls_SDOF (list[int]): 结果对应的id
        lock (multiprocessing.Lock): 进程锁
    """
    lock.acquire()
    try:
        if not len(results_L2) == len(ls_SDOF):
            # L1的行数与ls_SDOF的长度必须相等，保证结果与id一一对应
            lock.release()
            raise SDOF_Error('_Win.py, _write_results, Error - 1')
        for id_ in ls_SDOF:
            idx = id_ - 1
            results_L1[idx, 1:] = results_L2[idx]  # L2写入L1
    except Exception as error:
        lock.release()
        return error
    else:
        lock.release()
        return

# TODO:
def _load_result_file(h5_file: Path | str):
    """加载结果文件(用于重启动计算)

    Args:
        h5_file (Path | str): .h5文件路径
    """
    pass


# TODO:
"""
计算结果保存策略
首先定义1300_0000x10的L1变量，第一列改成id，每条地震动计算后通过queue
传递L2，在主进程将L2写入L1，当到达保存数据的节点时，将其转换为DadaFrame，
然后用下面代码追加写入.h5(file)文件：
>>> with pd.HDFStore(hdf5_file, 'a') as store: 
>>>     df = pd.DataFrame(arr, columns=['id']+['other_responses']*N_response_types)
>>>     df['id'] = df['id'].astype(int)
>>>     store.append('results', df, index=False, append=False,
                     complib='blosc:zstd', complevel=2)   
"""