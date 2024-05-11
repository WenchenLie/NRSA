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
    3: PDtSDOF_batched_solver,  # 批量SDOD考虑PDelta
}


class _Win(QDialog):

    dir_main = Path(__file__).parent.parent
    dir_input = dir_main / 'Input'
    dir_gm = dir_input / 'GMs'

    def __init__(self, task: SDOFmodel, logger: loguru.Logger) -> None:
        """监控窗口

        Args:
            task (SDOFmodel): SDOFmodel类的实例
        """
        super().__init__()
        self.ui = Ui_Win()
        self.ui.setupUi(self)
        self.task = task
        self.model_overview = task.model_overview
        self.model_paras = task.model_paras
        self.model_spectra = task.model_spectra
        self.logger = logger
        self.init_ui()
        self.run()


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
        self.ui.label_8.setText(f'SDOF求解器：{FUNC[self.task.func_type].__name__}')
        self.ui.pushButton_3.clicked.connect(self.pause_resume)
        
        
    def kill(self):
        """点击中断按钮"""
        if QMessageBox.question(self, '警告', '是否中断计算？') == QMessageBox.Yes:
            self.worker.kill()


    def pause_resume(self):
        """暂停或恢复计算"""
        if self.ui.pushButton_3.text() == '暂停':
            self.ui.pushButton_3.setText('继续')
            self.ui.groupBox_2.setTitle('计算信息（已暂停）')
        else:
            self.ui.pushButton_3.setText('暂停')
            self.ui.groupBox_2.setTitle('计算信息')
        self.worker.pause_resume()


    def run(self):
        self.worker = Worker(self.task, self)
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
        self.ui.label_7.setText(f'已计算SDOF：{int_}')

    def finish_all(self):
        self.ui.pushButton_2.setEnabled(True)
        if self.task.auto_quit:
            self.ui.pushButton_2.click()



class Worker(QThread):
    """处理计算任务的子线程"""
    signal_set_progressBar = pyqtSignal(tuple)
    signal_set_finished_SDOF = pyqtSignal(int)
    signal_finish_all = pyqtSignal()

    def __init__(self, task: SDOFmodel, win: _Win) -> None:
        super().__init__()
        self.task = task
        self.win = win
        self.model_overview = task.model_overview
        self.model_paras = task.model_paras
        self.model_spectra = task.model_spectra
        self.logger = self.win.logger
        self.queue = multiprocessing.Manager().Queue()  # 进程通信
        self.stop_event = multiprocessing.Manager().Event()  # 中断事件
        self.pause_event = multiprocessing.Manager().Event()  # 暂停事件
        self.pause_event.set()
        self.lock = multiprocessing.Manager().Lock()  # 进程锁
        
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
        # self.divide_tasks()
        if self.task.analysis_type == 'constant_ductility':
            self.run_constant_ductility()
        elif self.task.analysis_type == 'constant_strength':
            self.run_constant_strength()

    # def divide_tasks(self):
    #     """划分计算任务"""
    #     ls_SDOF = [i + 1 for i in range(self.task.N_SDOF)]  # 所有SDOF模型的序号(1~NSDOF)
    #     ls_batch = []  # 同一模型空间下SDOF的序号, list[list[int]]
    #     if self.task.batch > 1:
    #         while True:
    #             ls_batch.append(ls_SDOF[: self.task.batch])
    #             del ls_SDOF[: self.task.batch]
    #             if not ls_SDOF:
    #                 break
    #     self.ls_SDOF = ls_SDOF
    #     self.ls_batch = ls_batch


    def run_constant_ductility(self):
        """等延性分析"""
        s = '（多进程）' if self.task.parallel > 1 else ''
        self.logger.success(f'开始进行：等延性谱分析{s}')
        ...  # TODO


    def run_constant_strength(self):
        """等强度分析"""
        s = '（多进程）' if self.task.parallel > 1 else ''
        self.logger.success(f'开始进行：性能需求谱分析{s}')
        ls_paras: list[tuple] = []
        queue = self.queue
        stop_event = self.stop_event
        pause_event = self.pause_event
        lock = self.lock
        self.model_results = self.task.wkd / f'{self.task.model_name}_results.h5'  # 生成的结果文件
        suffix = self.model_overview['ground_motions']['suffix']
        batch = self.task.batch
        for gm_idx, (gm_name, dt, SF) in self.model_overview['ground_motions']['name_dt_SF'].items():
            df_paras = self.model_paras[self.model_paras['ground_motion']==int(gm_idx)]
            ls_SDOF = df_paras['ID'].to_list()
            gm = np.loadtxt(_Win.dir_gm / f'{gm_name}{suffix}')
            args = (queue, stop_event, pause_event, lock, self.model_results,
                    self.task.func_type, self.model_overview, self.model_paras, ls_SDOF, batch,
                    gm, dt, self.task.fv_duration, SF, self.task.g, gm_name)
            ls_paras.append(args)
        with multiprocessing.Pool(self.task.parallel) as pool:
            for i in range(self.task.GM_N):
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
                flag, *other = queue.get()
                if flag == 'a':  # 地震动完成
                    # other: 该条地震动计算的SDOF数量
                    finished_GM += 1
                    finished_SDOF += other[0]
                    self.signal_set_progressBar.emit((int(finished_GM/self.task.GM_N*100), f'已计算地震动：{finished_GM}'))
                    self.signal_set_finished_SDOF.emit(finished_SDOF)
                elif flag == 'h':  # 中断计算
                    # other = '中断计算'
                    self.signal_finish_all.emit()
                    break
                elif flag == 'i':  # 异常
                    # other: 异常
                    print(other[1])
                    raise other[0]
                if finished_GM == self.task.GM_N:
                    # 所有计算完成
                    self.logger.success('计算完成')
                    self.logger.success(f'生成结果文件：{self.model_results}')
                    self.signal_finish_all.emit()
                    break



def _run_constant_strength(*args, **kwargs):
    """SDOF计算函数，每次调用求解一条地震动

    Args (16):
        queue (multiprocessing.Queue): 进程通信
        stop_event (multiprocessing.Event): 进程终止事件
        pause_event (multiprocessing.Event): 进程暂停事件
        lock (multiprocessing.Lock): 进程锁
        model_results (Path): 待写入的计算结果文件
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
    queue: multiprocessing.Queue
    stop_event: multiprocessing.Event
    pause_event: multiprocessing.Event
    lock: multiprocessing.Lock
    model_results: Path
    func_type: int
    model_overview: dict
    model_paras: pd.DataFrame
    ls_SDOF: list[int]
    batch: int
    gm: np.ndarray
    dt: float
    fv_duration: float
    SF: float
    g: float
    gm_name: str
    try:
        queue, stop_event, pause_event, lock, model_results, func_type, model_overview, model_paras,\
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
        # (n: SDOF模型的序号)
        if func_type == 1:
            for n in ls_SDOF:
                n: int
                if stop_event.is_set():
                    queue.put(('h', '中断计算'))
                    return
                pause_event.wait()
                line = model_paras[model_paras['ID']==n]
                T = line[T_name].item()
                zeta = line[zeta_name].item()
                m = line[m_name].item()
                uy = line[uy_name].item()
                materials = _parse_material(model_overview, model_paras, n)
                if collapseDisp_name:
                    collapseDisp = line[collapseDisp_name].item()
                else:
                    collapseDisp = 1e10
                if maxAnaDisp_name:
                    maxAnaDisp = line[maxAnaDisp_name].item()
                else:
                    maxAnaDisp = 2e10
                result = SDOF_solver(T, gm, dt, materials, uy, fv_duration, SF, zeta, m, g,
                            collapseDisp, maxAnaDisp)
                res = _write_result(model_results, result, gm_name, n, lock)
                if res:
                    raise res
                finished_SDOF += 1
        elif func_type == 2:
            ls_batches = _split_batch(ls_SDOF, batch)
            for ls_batch in ls_batches:
                if stop_event.is_set():
                    queue.put(('h', '中断计算'))
                    return
                pause_event.wait()
                N_SDOFs = len(batch)
                ls_T = tuple(task_info['SDOF_models'][str(n)][T_name] for n in batch)
                ls_materials = tuple(_parse_material(task_info, str(n)) for n in batch)
                ls_uy = tuple(task_info['SDOF_models'][str(n)][uy_name] for n in batch)
                ls_SF = tuple(SF for _ in batch)
                ls_zeta = tuple(task_info['SDOF_models'][str(n)][zeta_name] for n in batch)
                ls_m = tuple(task_info['SDOF_models'][str(n)][m_name] for n in batch)
                if collapseDisp_name:
                    ls_collapseDisp = tuple(task_info['SDOF_models'][str(n)][collapseDisp_name] for n in batch)
                else:
                    ls_collapseDisp = tuple(1e10 for _ in batch)
                if maxAnaDisp_name:
                    ls_maxAnaDisp = tuple(task_info['SDOF_models'][str(n)][maxAnaDisp_name] for n in batch)
                else:
                    ls_maxAnaDisp = tuple(2e10 for _ in batch)
                result = SDOF_batched_solver(N_SDOFs, ls_T, gm, dt, ls_materials, ls_uy, fv_duration, ls_SF,
                    ls_zeta, ls_m, g, ls_collapseDisp, ls_maxAnaDisp)
                error = _write_result(model_results, result, gm_name, batch, lock)
                if error:
                    raise error
                finished_SDOF += N_SDOFs
        elif func_type == 3:
            for batch in ls_batch:
                if stop_event.is_set():
                    queue.put(('h', '中断计算'))
                    return
                pause_event.wait()
                N_SDOFs = len(batch)
                ls_h = tuple(task_info['SDOF_models'][str(n)][h_name] for n in batch)
                ls_T = tuple(task_info['SDOF_models'][str(n)][T_name] for n in batch)
                ls_grav = tuple(task_info['SDOF_models'][str(n)][P_name] for n in batch)
                ls_materials = tuple(_parse_material(task_info, str(n)) for n in batch)
                ls_uy = tuple(task_info['SDOF_models'][str(n)][uy_name] for n in batch)
                ls_SF = tuple(SF for _ in batch)
                ls_zeta = tuple(task_info['SDOF_models'][str(n)][zeta_name] for n in batch)
                ls_m = tuple(task_info['SDOF_models'][str(n)][m_name] for n in batch)
                if collapseDisp_name:
                    ls_collapseDisp = tuple(task_info['SDOF_models'][str(n)][collapseDisp_name] for n in batch)
                else:
                    ls_collapseDisp = tuple(1e10 for _ in batch)
                if maxAnaDisp_name:
                    ls_maxAnaDisp = tuple(task_info['SDOF_models'][str(n)][maxAnaDisp_name] for n in batch)
                else:
                    ls_maxAnaDisp = tuple(2e10 for _ in batch)
                result = PDtSDOF_batched_solver(N_SDOFs, ls_h, ls_T, ls_grav, gm, dt, ls_materials, ls_uy, fv_duration, ls_SF,
                    ls_zeta, ls_m, g, ls_collapseDisp, ls_maxAnaDisp)
                error = _write_result(model_results, result, gm_name, batch, lock)
                if error:
                    raise error
                finished_SDOF += N_SDOFs
        queue.put(('a', finished_SDOF))
    except Exception as e:
        tb = traceback.format_exc()
        queue.put(('i', e, tb))
        return


def _split_batch(ls_SDOF: list[int], batch: int) -> list[list[int]]:
    """拆分模型"""
    ls_batch = []
    while True:
        ls_batch.append(ls_SDOF[: batch])
        del ls_SDOF[: batch]
        if not ls_SDOF:
            break
    return ls_batch


def _parse_material(model_overview: dict, model_paras: pd.DataFrame, n: int) -> dict:
    """解析材料字典"""
    old_materials: dict = model_overview['material_format']
    materials = {}
    for matType, old_paras in old_materials.items():
        paras = []
        for old_para in old_paras:
            para = Task.identify_para(old_para)
            if para:
                paras.append(model_paras[model_paras['ID']==n][para].item())
            else:
                paras.append(old_para)
        materials[matType] = paras
    return materials
        

def _write_result(
    model_results: Path | str,
    results: dict,
    gm_name: str,
    n: int | list[int],
    lock: multiprocessing.Lock
    ):
    """将SDOF计算结果写入json文件，每次调用会打开generated_file并进行写入

    Args:
        model_results (Path | str): 输出文件夹中的json文件
        results (dict): SDOF求解器返回的结果
        batched (bool): 是否设置批量计算
        lock (multiprocessing.Lock): 进程锁
    """
    model_results = Path(model_results)
    lock.acquire()
    try:
        if not model_results.exists():
            f = h5py.File(model_results, 'w')
        else:
            f = h5py.File(model_results, 'a')
        # 写入响应类型（只会写入一次）
        if not 'response_type' in f:
            f.create_dataset('response_type', data=list(results.keys()))
        # 写入响应数据
        if isinstance(n, int):
            # results: dict[str, bool | float]
            f.create_dataset(str(n), data=list(results.values()))
        elif isinstance(n, list):
            # results: dict[str, bool | tuple[bool, ...] | list[float]]
            for i, response in enumerate(list(zip(*(results.values())))):  
                # *将响应结果转置
                f.create_dataset(str(n[i]), data=response)
        else:
            raise SDOF_Error(f'不支持的参数 n 类型：{type(n)}')
        f.close()
    except Exception as error:
        lock.release()
        f.close()
        return error
    else:
        lock.release()
        return