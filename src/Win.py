from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .NRSA import NRSA
import os, time, json
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import QMessageBox, QDialog

from ui.Win import Ui_Win
from .config import LOGGER, ANA_TYPE_NAME, PERIOD
from .constant_ductility_iteration import constant_ductility_iteration
from .constant_strength_analysis import constant_strength_analysis
from .time_history_analysis import time_history_analysis


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
        self.analysis_type = nrsa.analysis_type
        self.start_time = nrsa.start_time
        self.solver = nrsa.solver
        self.hidden_prints = nrsa.hidden_prints
        self.job_name = nrsa.job_name
        self.wkdir = nrsa.wkdir
        self.skip_existed_res = nrsa.skip_existed_res
        if nrsa.period is not None:
            self.period = nrsa.period
        else:
            self.period = PERIOD
        self.Ti = nrsa.Ti
        self.material_function = nrsa.material_function
        self.material_paras = nrsa.material_paras
        self.para_groups = nrsa.para_groups
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
        if self.analysis_type == 'THA':
            self.N_period = 1  # 时程只有一个周期点
        self.finished_gm = 0  # 已完成的地震动数量
        self.finished_num = 0  # 已计算次数
        self.finished_sdof = 0  # 已完成的SDOF数量
        self.required_sdof = self.N_period * self.GM_N * len(self.para_groups)  # 所需计算的SDOF数量
        self.finished_analyses = 0  # 已完成的分析数量
        self.progress: list[tuple[float, int, int]] = []  # 进度记录, 包括时间戳，分析数量，SDOF数量，每1秒更新一次

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
        self.ui_update_progressBar(0)
        self.ui_update_finished_gm(0)
        self.ui_update_finished_ana(0)
        self.ui_update_average_iteration(0)

    def ui_open_wkdir(self):
        """打开工作目录"""
        os.startfile(self.wkdir)

    def ui_kill(self):
        """点击中断按钮"""
        if QMessageBox.question(self, 'Warning',
                'Are you sure to interrupt the analysis?') == QMessageBox.Yes:
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

    def ui_update_progressBar(self, val: int):
        """设置进度条(进度值, 文本)"""
        self.ui.progressBar.setValue(val)

    def ui_update_finished_gm(self, n: int):
        """设置已完成的地震动数量"""
        pass

    def ui_update_finished_ana(self, n: int):
        """设置已计算次数"""
        self.ui.label_7.setText(f'Finished analyses: {n}')
        self.finished_analyses = n

    def ui_update_finished_sdof(self, n: int):
        """设置已完成的SDOF数量"""
        self.ui.label_6.setText(f'Finished SDOF tasks: {n}')
        self.finished_sdof = n

    def ui_update_uncorverged_ana(self, n: int):
        """设置未收敛的分析数量"""
        self.ui.label_9.setText(f'Unconverged analyses: {n}')

    def ui_updata_uncorverged_iter(self, n: int):
        """设置未收敛的迭代次数"""
        self.ui.label_11.setText(f'Unconverged iterations: {n}')

    def ui_update_average_iteration(self, n: float):
        """设置平均迭代次数"""
        self.ui.label_5.setText(f'Average iterations: {n:.2f}')
        
    def update_progress(self):
        """更新计算进度"""
        crt_time = time.time()  # 当前时间
        self.progress.append((crt_time, self.finished_analyses, self.finished_sdof))
        if len(self.progress) > 30:
            last_time, last_ana, last_sdof = self.progress.pop(0)
        elif len(self.progress) > 1:
            last_time, last_ana, last_sdof = self.progress[0]
        elif len(self.progress) == 1:
            return
        speed_ana = (self.finished_analyses - last_ana) / (crt_time - last_time)  # 分析速度 (analyses/s)
        speed_sdof = (self.finished_sdof - last_sdof) / (crt_time - last_time)  # SDOF速度 (tasks/s)
        if speed_ana == 0:
            remaining_time = ''
        else:
            remaining_time = (self.required_sdof - self.finished_sdof) / speed_sdof  # 剩余时间 (s)
            remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))  # 剩余时间
        self.ui_update_speed(speed_ana, speed_sdof)
        self.ui_update_remaining_time(remaining_time)
   
    def ui_update_speed(self, speed_ana: float, speed_sdof: float):
        """更新analysis速度和SDOF速度"""
        self.ui.label_14.setText(f'Speed: {speed_sdof:.1f} tasks/s, {speed_ana:.1f} analyses/s')
        
    def ui_update_remaining_time(self, remaining_time: str):
        """更新剩余时间"""
        self.ui.label_13.setText(f'Remaining time: {remaining_time}')

    def ana_termination(self):
        """分析结束"""
        self.ui.pushButton_2.setEnabled(True)
        self.ui.pushButton_3.setEnabled(False)
        self.ui.pushButton.setEnabled(False)
        self.ui.label_12.setText('Status: Completed')
        if self.auto_quit:
            self.ui.pushButton_2.click()
        self.timer.stop()

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
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(1000)
        self.worker.start()

class _Worker(QThread):
    """处理计算任务的子线程"""
    signal_update_progressBar = pyqtSignal(int)  # 更新进度条
    signal_update_finished_gm = pyqtSignal(int)  # 更新已完成地震动数量
    signal_update_finished_ana = pyqtSignal(int)  # 更新已完成分析次数
    signal_update_finished_sdof = pyqtSignal(int)  # 更新已完成SDOF数量
    signal_update_uncorverged_ana = pyqtSignal(int)  # 更新未收敛的地震动数量
    signal_updata_uncorverged_iter = pyqtSignal(int)  # 更新未收敛的迭代次数
    signal_update_average_iteration = pyqtSignal(float)  # 更新平均迭代次数
    signal_ana_termination = pyqtSignal()  # 计算结束(可能是中断或完成)
    signal_ana_suscess = pyqtSignal()  # 计算成功
    signal_ana_failed = pyqtSignal(tuple)  # 计算失败
    # signal_send_results = pyqtSignal(list)  # 发送结果

    def __init__(self, win: Win) -> None:
        super().__init__()
        self.win = win
        self.analysis_type = win.analysis_type
        self.start_time = win.start_time
        self.job_name = win.job_name
        self.wkdir = win.wkdir
        self.skip_existed_res = win.skip_existed_res
        self.period = win.period
        self.Ti = win.Ti
        self.material_function = win.material_function
        self.material_paras = win.material_paras
        self.para_groups = win.para_groups
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
        }

    def get_queue(self, queue: multiprocessing.Queue):
        """进程通讯"""
        finished_gm = 0  # 已计算完成的地震动
        finished_ana = 0  # 已进行的分析次数
        finished_sdof = 0  # 已完成的SDOF数量
        uncorverged_ana = 0  # 未收敛的地震动数量
        uncorverged_iter = 0  # 未收敛的迭代数量
        counter_finished = 0  # 已完成的任务数
        start_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(self.start_time))
        while True:
            if not queue.empty():
                for key, value in queue.get().items():
                    flag, contents = key, value
                if flag == 'a':
                    # 该条地震动完成
                    counter_finished += 1
                    GM_name, num_ana, num_period, iter_converge, solving_converge,\
                        start_time, end_time, mean_ana = contents
                    finished_gm += 1
                    finished_ana += num_ana
                    finished_sdof += num_period
                    self.signal_update_progressBar.emit(int(counter_finished / self.counter_requried * 100))
                    self.signal_update_finished_ana.emit(finished_ana)
                    self.signal_update_finished_sdof.emit(finished_sdof)
                    if finished_sdof == 0:
                        ave_iter = 0
                    else:
                        ave_iter = finished_ana / finished_sdof
                    self.signal_update_average_iteration.emit(ave_iter)
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
                if counter_finished == self.counter_requried:
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
        queue = self.queue
        stop_event = self.stop_event
        pause_event = self.pause_event
        lock = self.lock
        para_groups = self.para_groups
        num_paras = len(para_groups)  # 材料参数的组合数
        subfolder = 'results'
        self.set_button_enabled()
        self.counter_requried = 0  # SDOF任务计数器
        th_shm_ls: list[SharedMemory] = []
        Sa_shm_ls: list[SharedMemory] = []
        NPTS_ls: list[int] = []
        for gm_idx in range(self.GM_N):
            # 将地震动数据存储到共享内存中
            gm_name = self.GM_names[gm_idx]
            gm_th = np.loadtxt(self.GM_folder / f'{gm_name}{self.suffix}')
            NPTS_ls.append(gm_th.shape[0])
            shm = SharedMemory(create=True, size=gm_th.nbytes)
            shm_array = np.ndarray(gm_th.shape, dtype=gm_th.dtype, buffer=shm.buf)
            shm_array[:] = gm_th[:]
            th_shm_ls.append(shm)
            if self.analysis_type == 'CDA':
                # 等延性分析中，Sa为采用分析阻尼比的无缩放谱加速度
                if isinstance(self.unscaled_RSA_spc, dict):
                    Sa_ls = self.unscaled_RSA_spc[self.damping][:, gm_idx]
                else:
                    Sa_ls = self.unscaled_RSA_spc[:, gm_idx]
            elif self.analysis_type in ['CSA', 'THA']:
                # 等强度分析或时程分析中，Sa为采用5%阻尼比的缩放后的谱加速度
                Sa_ls = self.unscaled_RSA_5pct[:, gm_idx] * self.GM_indiv_sf[gm_idx]
            # 将Sa存储到共享内存中
            Sa_shm = SharedMemory(create=True, size=Sa_ls.nbytes)
            Sa_shm_array = np.ndarray(Sa_ls.shape, dtype=Sa_ls.dtype, buffer=Sa_shm.buf)
            Sa_shm_array[:] = Sa_ls[:]
            Sa_shm_ls.append(Sa_shm)
        N_period = len(self.period)
        period_shm = SharedMemory(create=True, size=self.period.nbytes)
        period_shm_array = np.ndarray(self.period.shape, dtype=self.period.dtype, buffer=period_shm.buf)
        period_shm_array[:] = self.period[:]
        N_PERIOD = len(PERIOD)
        PERIOD_shm = SharedMemory(create=True, size=PERIOD.nbytes)  # 全局周期，仅用于计算地震动反应谱获得Sa(Ti)
        PERIOD_shm_array = np.ndarray(PERIOD.shape, dtype=PERIOD.dtype, buffer=PERIOD_shm.buf)
        PERIOD_shm_array[:] = PERIOD[:]
        with multiprocessing.Pool(self.parallel) as pool:
            for para_group in para_groups:
                if num_paras > 1:
                    subfolder = 'results_' + '_'.join([str(term) for term in para_group])
                for gm_idx in range(self.GM_N):
                    self.counter_requried += 1
                    wkdir = self.wkdir
                    skip = self.skip_existed_res
                    Ti = self.Ti
                    material_function = self.material_function
                    damping = self.damping
                    target_ductility = self.target_ductility
                    thetaD = self.thetaD
                    mass = self.mass
                    he = self.he
                    gm_name = self.GM_names[gm_idx]
                    gm_shm = th_shm_ls[gm_idx]
                    Sa_shm = Sa_shm_ls[gm_idx]
                    NPTS = NPTS_ls[gm_idx]
                    scaling_factor = self.GM_global_sf
                    dt = self.GM_dts[gm_idx]
                    fv_duration = self.fv_duration
                    R_init = self.R_init
                    R_incr = self.R_incr
                    solver = self.solver
                    tol_ductility = self.tol_ductility
                    tol_R = self.tol_R
                    max_iter = self.max_iter
                    hidden_prints = self.hidden_prints
                    kwargs = self.kwargs
                    if skip and (wkdir / subfolder / f'{gm_name}.csv').exists():
                        if self.analysis_type in ['CDA', 'CSA']:
                            finished_sdof = N_period
                        elif self.analysis_type == 'THA':
                            finished_sdof = 1
                        queue.put({'a': (gm_name, 0, finished_sdof, None, 1, 0, 0, None)})
                        continue
                    if self.analysis_type == 'CDA':
                        args = (wkdir, subfolder, period_shm.name, N_period, material_function, para_group, damping,\
                                target_ductility, thetaD, mass, he, gm_name, gm_shm.name, NPTS, scaling_factor, dt,\
                                fv_duration, R_init, R_incr, Sa_shm.name, solver, tol_ductility, tol_R, max_iter,\
                                hidden_prints, queue, stop_event, pause_event, lock)
                        func = constant_ductility_iteration
                    elif self.analysis_type == 'CSA':
                        scaling_factor *= self.GM_indiv_sf[gm_idx]
                        args = (wkdir, subfolder, period_shm.name, N_period, material_function, para_group, damping,\
                                thetaD, mass, he, gm_name, gm_shm.name, NPTS, scaling_factor, dt, fv_duration, Sa_shm.name,\
                                solver, hidden_prints, queue, stop_event, pause_event, lock)
                        func = constant_strength_analysis
                    elif self.analysis_type == 'THA':
                        scaling_factor *= self.GM_indiv_sf[gm_idx]
                        args = (wkdir, subfolder, Ti, material_function, para_group, damping, thetaD, mass, he,\
                                gm_name, gm_shm.name, NPTS, scaling_factor, dt, fv_duration, PERIOD_shm.name, N_PERIOD,\
                                Sa_shm.name, solver, hidden_prints, queue, stop_event, pause_event, lock)
                        func = time_history_analysis
                    else:
                        assert False, f'Unknow running type: {self.analysis_type}'
                    pool.apply_async(func, args=args, kwds=kwargs)
            LOGGER.info('Analysis started')
            self.get_queue(queue)
        for shm in th_shm_ls:
            shm.close()
            shm.unlink()
        period_shm.close()
        period_shm.unlink()
        PERIOD_shm.close()
        PERIOD_shm.unlink()
        Sa_shm.close()
        Sa_shm.unlink()
 
    def set_button_enabled(self):
        """将`保存`, `中断`, `暂停`三个按钮激活"""
        self.win.ui.pushButton.setEnabled(True)
        self.win.ui.pushButton_3.setEnabled(True)
