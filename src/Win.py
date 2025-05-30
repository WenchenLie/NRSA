from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .NRSA import NRSA
import os, time, json
import multiprocessing

import numpy as np
from scipy.interpolate import interp1d
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QMessageBox, QDialog

from ui.Win import Ui_Win
from .config import LOGGER, ANA_TYPE_NAME
from .constant_ductility_iteration import constant_ductility_iteration
from .constant_strength_analysis import constant_strength_analysis


class Win(QDialog):
    def __init__(self, nrsa: NRSA):
        """监控窗口

        Args:
            nrsa (NRSA): NRSA类的实例
        """
        super().__init__()
        self.ui = Ui_Win()
        self.ui.setupUi(self)
        self.init_var(nrsa)
        self.init_ui()
        self.start_worker()

    def init_var(self, nrsa: NRSA):
        """实例属性"""
        # 将ConstantDuctilityAnalysis实例的变量作为_Win的实例属性
        self.period = nrsa.period
        self.analysis_type = nrsa.analysis_type
        self.start_time = nrsa.start_time
        self.solver = nrsa.solver
        self.hidden_prints = nrsa.hidden_prints
        self.job_name = nrsa.job_name
        self.wkdir = nrsa.wkdir
        self.period = nrsa.period
        self.material_function = nrsa.material_function
        self.material_paras = nrsa.material_paras
        self.damping = nrsa.damping
        self.target_ductility = nrsa.target_ductility
        self.R_init = nrsa.R_init
        self.R_incr = nrsa.R_incr
        self.tol_ductility = nrsa.tol_ductility
        self.tol_R = nrsa.tol_R
        self.max_iter = nrsa.max_iter
        self.solver = nrsa.solver
        self.thetaD = nrsa.thetaD
        self.mass = nrsa.mass
        self.he = nrsa.height
        self.fv_duration = nrsa.fv_duration
        self.suffix = nrsa.suffix
        self.GM_names = nrsa.GM_names
        self.GM_folder = nrsa.GM_folder
        self.GM_N = nrsa.GM_N
        self.GM_global_sf = nrsa.GM_global_sf
        self.GM_dts = nrsa.GM_dts
        self.GM_NPTS = nrsa.GM_NPTS
        self.GM_durations = nrsa.GM_durations
        self.GM_indiv_sf = nrsa.GM_indiv_sf
        self.unscaled_RSA_5pct = nrsa.unscaled_RSA_5pct
        self.unscaled_RSA_spc = nrsa.unscaled_RSA_spc
        self.parallel = nrsa.parallel
        self.gm_batch_size = nrsa.gm_batch_size
        self.auto_quit = nrsa.auto_quit
        self.show_monitor = nrsa.show_monitor
        self.kwargs = nrsa.kwargs
        # 新实例属性
        self.N_period = len(self.period)  # 周期点数量
        self.finished_gm = 0  # 已完成的地震动数量
        self.finished_num = 0  # 已计算次数

    def init_ui(self):
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint)
        time_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.ui.label.setText(f'Start time: {time_start}')
        self.ui.label_10.setText(f'Job name: {self.job_name}')
        self.ui.label_2.setText(f'Analysis type: {ANA_TYPE_NAME[self.analysis_type]}')
        self.ui.label_8.setText(f'Period number: {self.N_period}')
        self.ui.label_4.setText(f'Ground motion number: {self.GM_N}')
        self.ui.label_3.setText(f'Parallel: {self.parallel}')
        self.ui.pushButton_4.clicked.connect(self.ui_open_wkdir)
        self.ui.pushButton.clicked.connect(self.ui_kill)
        self.ui.pushButton_3.clicked.connect(self.ui_pause_resume)
        self.ui_update_progressBar((int(self.finished_gm / self.GM_N * 100), f'Finished ground motions: {self.finished_gm}'))
        self.ui_update_finished_gm(0)
        self.ui_update_finished_ana(0)
        self.ui_update_average_iteration(0)

    def ui_open_wkdir(self):
        """打开工作目录"""
        os.startfile(self.wkdir)

    def ui_kill(self):
        """点击中断按钮"""
        if QMessageBox.question(self, 'Warning', 'Are you sure to interrupt the analysis?') == QMessageBox.Yes:
            LOGGER.warning('Interrupting analysis')
            self.worker.kill()

    def ui_pause_resume(self):
        """点击暂停/继续按钮"""
        self.worker.pause_resume()
        if self.worker.pause_event.is_set():
            self.ui.pushButton_3.setText('Pause')
            self.ui.label_12.setText('Status: Running')
        else:
            self.ui.pushButton_3.setText('Resume')
            self.ui.label_12.setText('Status: Paused')

    def ui_update_progressBar(self, tuple_):
        """设置进度条(进度值, 文本)"""
        val, text = tuple_
        self.ui.label_6.setText(text)
        self.ui.progressBar.setValue(val)

    def ui_update_finished_gm(self, n: int):
        """设置已完成的地震动数量"""
        self.ui.label_6.setText(f'Finished ground motions: {n}')

    def ui_update_finished_ana(self, n: int):
        """设置已计算次数"""
        self.ui.label_7.setText(f'Finished analyses: {n}')

    def ui_update_finished_sdof(self, n: int):
        """设置已完成的SDOF数量"""
        pass

    def ui_update_uncorverged_ana(self, n: int):
        """设置未收敛的分析数量"""
        self.ui.label_9.setText(f'Unconverged analyses: {n}')

    def ui_updata_uncorverged_iter(self, n: int):
        """设置未收敛的迭代次数"""
        self.ui.label_11.setText(f'Unconverged iterations: {n}')

    def ui_update_average_iteration(self, n: float):
        """设置平均迭代次数"""
        self.ui.label_5.setText(f'Average iterations: {n:.2f}')

    def ana_termination(self):
        """分析结束"""
        self.ui.pushButton_2.setEnabled(True)
        self.ui.pushButton_3.setEnabled(False)
        self.ui.pushButton.setEnabled(False)
        self.ui.label_12.setText('Status: Completed')
        if self.auto_quit:
            self.ui.pushButton_2.click()

    def ana_suscess(self):
        """计算成功"""
        LOGGER.success('Analysis completed')
    
    def ana_failed(self, tuple_):
        """计算失败"""
        LOGGER.success('Analysis completed')
        LOGGER.warning(f'Unconverged iterations: {tuple_[1]}, Unconverged analyses: {tuple_[0]}')

    def start_worker(self):
        self.worker = _Worker(self)
        self.worker.signal_update_progressBar.connect(self.ui_update_progressBar)
        self.worker.signal_update_finished_gm.connect(self.ui_update_finished_gm)
        self.worker.signal_update_finished_ana.connect(self.ui_update_finished_ana)
        self.worker.signal_update_finished_sdof.connect(self.ui_update_finished_sdof)
        self.worker.signal_update_uncorverged_ana.connect(self.ui_update_uncorverged_ana)
        self.worker.signal_updata_uncorverged_iter.connect(self.ui_updata_uncorverged_iter)
        self.worker.signal_update_average_iteration.connect(self.ui_update_average_iteration)
        self.worker.signal_ana_termination.connect(self.ana_termination)
        self.worker.signal_ana_suscess.connect(self.ana_suscess)
        self.worker.signal_ana_failed.connect(self.ana_failed)
        self.worker.start()

class _Worker(QThread):
    """处理计算任务的子线程"""
    signal_update_progressBar = pyqtSignal(tuple)  # 更新进度条
    signal_update_finished_gm = pyqtSignal(int)  # 更新已完成地震动数量
    signal_update_finished_ana = pyqtSignal(int)  # 更新已完成分析次数
    signal_update_finished_sdof = pyqtSignal(int)  # 更新已完成SDOF数量
    signal_update_uncorverged_ana = pyqtSignal(int)  # 更新未收敛的地震动数量
    signal_updata_uncorverged_iter = pyqtSignal(int)  # 更新未收敛的迭代次数
    signal_update_average_iteration = pyqtSignal(float)  # 更新平均迭代次数
    signal_ana_termination = pyqtSignal()  # 计算结束(可能是中断或完成)
    signal_ana_suscess = pyqtSignal()  # 计算成功
    signal_ana_failed = pyqtSignal(tuple)  # 计算失败

    def __init__(self, win: Win) -> None:
        super().__init__()
        self.win = win
        self.analysis_type = win.analysis_type
        self.start_time = win.start_time
        self.job_name = win.job_name
        self.wkdir = win.wkdir
        self.period = win.period
        self.material_function = win.material_function
        self.material_paras = win.material_paras
        self.damping = win.damping
        self.target_ductility = win.target_ductility
        self.R_init = win.R_init
        self.R_incr = win.R_incr
        self.tol_ductility = win.tol_ductility
        self.tol_R = win.tol_R
        self.max_iter = win.max_iter
        self.solver = win.solver
        self.hidden_prints = win.hidden_prints
        self.thetaD = win.thetaD
        self.mass = win.mass
        self.he = win.he
        self.fv_duration = win.fv_duration
        self.suffix = win.suffix
        self.GM_names = win.GM_names
        self.GM_folder = win.GM_folder
        self.GM_N = win.GM_N
        self.GM_dts = win.GM_dts
        self.GM_NPTS = win.GM_NPTS
        self.GM_durations = win.GM_durations
        self.GM_global_sf = win.GM_global_sf
        self.GM_indiv_sf = win.GM_indiv_sf
        self.unscaled_RSA_5pct = win.unscaled_RSA_5pct
        self.unscaled_RSA_spc = win.unscaled_RSA_spc
        self.parallel = win.parallel
        self.gm_batch_size = win.gm_batch_size
        self.auto_quit = win.auto_quit
        self.N_period = win.N_period
        self.finished_gm = win.finished_gm
        self.finished_num = win.finished_num
        self.show_monitor = win.show_monitor
        self.kwargs = win.kwargs
        # 一些控制线程的属性
        self.reuslt_column = ['id', 'converge', 'collapse', 'maxDisp', 'maxVel',
            'maxAccel', 'Ec', 'Ev', 'maxReaction', 'CD', 'CPD', 'resDisp']
        self.queue = multiprocessing.Manager().Queue()  # 进程通信
        self.stop_event = multiprocessing.Manager().Event()  # 中断事件
        self.pause_event = multiprocessing.Manager().Event()  # 暂停事件
        self.pause_event.set()
        self.lock = multiprocessing.Manager().Lock()  # 进程锁
        self.init_log()

    def init_log(self):
        """初始化日志"""
        self.log = {
            '//': '------------- Running log -------------',
            'Job name': self.job_name,
            'Analysis type': self.analysis_type,
            'Start time': time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(self.start_time)),
            'Parallel': self.parallel,
            'Ground motion folder': self.GM_folder.as_posix(),
            'Ground motions': {}
        }
        for gm_name in self.GM_names:
            self.log['Ground motions'][gm_name] = {}

    def get_queue(self, queue: multiprocessing.Queue):
        """进程通讯"""
        finished_gm = 0  # 已计算完成的地震动
        finished_ana = 0  # 已进行的分析次数
        finished_sdof = 0  # 已完成的SDOF数量
        uncorverged_ana = 0  # 未收敛的地震动数量
        uncorverged_iter = 0  # 未收敛的迭代数量
        start_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(self.start_time))
        while True:
            if not queue.empty():
                for key, value in queue.get().items():
                    flag, contents = key, value
                if flag == 'a':
                    # 该条地震动完成
                    GM_name, num_ana, num_period, iter_converge, solving_converge, start_time, end_time, mean_ana = contents
                    finished_gm += 1
                    finished_ana += num_ana
                    finished_sdof += num_period
                    self.signal_update_progressBar.emit((int(finished_gm / self.GM_N * 100), f'Finished ground motions: {finished_gm}'))
                    self.signal_update_finished_ana.emit(finished_ana)
                    self.signal_update_finished_sdof.emit(finished_sdof)
                    self.signal_update_average_iteration.emit(finished_ana / finished_sdof)
                    self.log['Ground motions'][GM_name]['Start time'] = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(start_time))
                    self.log['Ground motions'][GM_name]['End time'] = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(end_time))
                    self.log['Ground motions'][GM_name]['Elapsed time'] = round(end_time - start_time, 3)
                    self.log['Ground motions'][GM_name]['Analysis number'] = num_ana
                    self.log['Ground motions'][GM_name]['Analysis converge'] = bool(solving_converge)
                    if self.analysis_type == 'CDA':
                        self.log['Ground motions'][GM_name]['Iteration converge'] = bool(iter_converge)
                        self.log['Ground motions'][GM_name]['Average analyses'] = mean_ana
                elif flag == 'b':
                    # 计算不收敛
                    GM_name, Ti, solver_paras = contents
                    uncorverged_ana += 1
                    self.signal_update_uncorverged_ana.emit(uncorverged_ana)
                elif flag == 'c':
                    # 迭代不收敛
                    GM_name, Ti, current_tol, current_tol_R = contents
                    uncorverged_iter += 1
                    self.signal_updata_uncorverged_iter.emit(uncorverged_iter)
                elif flag == 'd':
                    # 未知异常
                    error, tb = contents
                    print(tb)
                    raise error
                elif flag == 'e':
                    # 中断计算
                    break
                elif flag == 'f':
                    # 该地震动计算开始
                    ...
                if finished_gm == self.GM_N:
                    # 所有计算完成
                    if uncorverged_ana == 0 and uncorverged_iter == 0:
                        self.signal_ana_suscess.emit()
                    else:
                        self.signal_ana_failed.emit((uncorverged_ana, uncorverged_iter))
                    break
        end_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        elaspsed_time = round(time.time() - self.start_time, 3)
        self.log['End time'] = end_time
        self.log['Elapsed time'] = elaspsed_time
        self.log['Total analyses'] = finished_ana
        if finished_sdof == 0:
            ave_ana = 0
        else:
            ave_ana = finished_ana / finished_sdof
        self.log['Average analyses'] = round(ave_ana, 2)
        self.log['Total SDOFs'] = finished_sdof
        self.log['Unconverged analyses'] = uncorverged_ana
        self.log['Unconverged iterations'] = uncorverged_iter
        self.lock.acquire()
        json.dump(self.log, open(self.wkdir / f'Log_{start_date}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        json.dump(self.log, open(f'logs/Log_{start_date}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        self.lock.release()
        self.signal_ana_termination.emit()

    def kill(self):
        """中断计算"""
        self.stop_event.set()
        if not self.pause_event.is_set():
            self.queue.put({'e': '中断计算'})

    def pause_resume(self):
        """暂停或恢复计算"""
        if self.pause_event.is_set():
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def run(self):
        """开始运行子线程，进行等延性分析"""
        s = '(Multi-processing)' if self.parallel > 1 else ''
        LOGGER.success(f'Running started: {ANA_TYPE_NAME[self.analysis_type]} {s}')
        ls_paras: list[tuple] = []
        queue = self.queue
        stop_event = self.stop_event
        pause_event = self.pause_event
        lock = self.lock
        for gm_idx in range(self.GM_N):
            wkdir = self.wkdir
            periods = self.period
            material_function = self.material_function
            material_paras = self.material_paras
            damping = self.damping
            target_ductility = self.target_ductility
            thetaD = self.thetaD
            mass = self.mass
            he = self.he
            gm_name = self.GM_names[gm_idx]
            gm_th = np.loadtxt(self.GM_folder / f'{gm_name}{self.suffix}')
            scaling_factor = self.GM_global_sf
            dt = self.GM_dts[gm_idx]
            fv_duration = self.fv_duration
            R_init = self.R_init
            R_incr = self.R_incr
            if self.analysis_type == 'CDA':
                # 等延性分析中，Sa为采用分析阻尼比的无缩放谱加速度
                if isinstance(self.unscaled_RSA_spc, dict):
                    Sa_ls = self.unscaled_RSA_spc[self.damping][:, gm_idx]
                else:
                    Sa_ls = self.unscaled_RSA_spc[:, gm_idx]
            elif self.analysis_type == 'CSA':
                # 等强度分析中，Sa为采用5%阻尼比的缩放后的谱加速度
                Sa_ls = self.unscaled_RSA_5pct[:, gm_idx] * self.GM_indiv_sf[gm_idx]
            solver = self.solver
            tol_ductility = self.tol_ductility
            tol_R = self.tol_R
            max_iter = self.max_iter
            hidden_prints = self.hidden_prints
            kwargs = self.kwargs
            if self.analysis_type == 'CDA':
                args = (wkdir, periods, material_function, material_paras, damping, target_ductility,\
                    thetaD, mass, he, gm_name, gm_th, scaling_factor, dt, fv_duration, R_init, R_incr,\
                    Sa_ls, solver, tol_ductility, tol_R, max_iter, hidden_prints, queue, stop_event, pause_event,\
                    lock)
                func = constant_ductility_iteration
            elif self.analysis_type == 'CSA':
                scaling_factor *= self.GM_indiv_sf[gm_idx]
                args = (wkdir, periods, material_function, material_paras, damping, thetaD, mass, he,\
                    gm_name, gm_th, scaling_factor, dt, fv_duration, Sa_ls, solver, hidden_prints,\
                    queue, stop_event, pause_event, lock)
                func = constant_strength_analysis
            else:
                assert False, f'Unknow running type: {self.analysis_type}'
            ls_paras.append(args)
        self.set_button_enabled()
        with multiprocessing.Pool(self.parallel) as pool:
            for idx_gm in range(self.GM_N):
                pool.apply_async(func, args=ls_paras[idx_gm], kwds=kwargs)
            LOGGER.info('Analysis started')
            self.get_queue(queue)
            pool.close()
            pool.join()

    def set_button_enabled(self):
        """将`保存`, `中断`, `暂停`三个按钮激活"""
        self.win.ui.pushButton.setEnabled(True)
        self.win.ui.pushButton_3.setEnabled(True)
