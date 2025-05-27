# newmark_cy.pyx
# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fabs, fmax
import re
import openseespy.opensees as ops

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t INT_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _update_para(int matTag, tuple paras):
    cdef:
        list para_new = []
        str para_str
        list res
    
    for para in paras:
        if isinstance(para, (int, float)):
            para_new.append(para)
        else:
            para_str = str(para)
            res = re.findall(r'^\^+$', para_str)
            if res:
                refTag = matTag - len(res[0])
                if refTag < 1:
                    raise ValueError(f"Invalid material reference: {paras}")
                para_new.append(refTag)
            else:
                para_new.append(para)
    return para_new

@cython.boundscheck(False)
@cython.wraparound(False)
def newmark_solver(
    double T,
    cnp.ndarray[DTYPE_t, ndim=1] ag,
    double dt,
    dict materials,
    double uy=0.0,
    double fv_duration=0.0,
    double sf=1.0,
    double P=0.0,
    double h=1,
    double zeta=0.05,
    double m=1,
    double g=9800,
    double collapse_disp=1e14,
    double maxAnalysis_disp=1e15,
    double tol=1e-5,
    int max_iter=100,
    double beta=0.25,
    double gamma=0.5,
    bint display_info=True,
):
    # 初始化参数
    cdef:
        double omega = 2 * np.pi / T
        double c = 2 * m * zeta * omega
        int NPTS = ag.shape[0]
        double duration = (NPTS - 1) * dt + fv_duration
        cnp.ndarray[DTYPE_t, ndim=1] ag_scaled = ag * sf * g
        
        # 状态变量
        double current_time = 0.0
        double u_prev = 0.0, v_prev = 0.0, a_prev = 0.0
        double u_pred, v_pred, u_next, error, residual, k_t, delta
        double a_next, v_next, du, a_abs
        double maxDisp = 0.0, maxVel = 0.0, maxAccel = 0.0
        double Ec = 0.0, Ev = 0.0, CD = 0.0, CPD = 0.0
        double maxReaction = 0.0
        double F_ray, F_hys, F_total
        double u_old = 0.0, F_hys_old = 0.0, F_ray_old = 0.0, u_cent = 0.0
        int iter_count, i = 0
        bint converge = True
        bint collapse_flag = False
        int matTag = 1
    
    # 初始化材料模型
    ops.wipe()
    for matType, paras in materials.items():
        paras_processed = _update_para(matTag, paras if isinstance(paras, tuple) else (paras,))
        ops.uniaxialMaterial(matType, matTag, *paras_processed)
        matTag += 1
    
    if P != 0.0:
        ops.uniaxialMaterial('Elastic', matTag, -P / h)
        matTag += 1
    
    ops.uniaxialMaterial('Parallel', matTag, *range(1, matTag))
    ops.testUniaxialMaterial(matTag)
    
    while True:
        i += 1

        # 预测步
        u_pred = u_prev + dt * v_prev + (0.5 - beta) * dt**2 * a_prev
        v_pred = v_prev + (1 - gamma) * dt * a_prev
        
        # 牛顿迭代
        u_next = u_pred
        error = 1.0
        iter_count = 0
        
        while error > tol:
            # Newton-Raphson iteration
            ops.setTrialStrain(u_next)
            F_hys = ops.getStress()
            mat_tangent = ops.getTangent()
            
            # 计算残差和刚度
            residual = m * (u_next - u_pred) / (beta * dt ** 2) + \
                      c * (v_pred + gamma * (u_next - u_pred) / (beta*dt)) + \
                      F_hys + m * ag_scaled[i]
            k_t = m / (beta * dt ** 2) + c * gamma / (beta * dt) + mat_tangent
            delta = residual / k_t
            u_next -= delta
            error = fabs(delta)
            iter_count += 1
            if iter_count > max_iter:
                if display_info:
                    print(f'Newton-Raphson iteration failed after {max_iter} iterations ({error} > {tol}).')
                converge = False
                break
        
        # 更新状态
        a_next = (u_next - u_pred) / (beta * dt ** 2)
        v_next = v_pred + gamma * dt * a_next
        u_prev, v_prev, a_prev = u_next, v_next, a_next
        ops.commitState()
        current_time += dt
        
        # 更新结果
        du = u_next - u_old
        maxDisp = fmax(maxDisp, fabs(u_next))
        maxVel = fmax(maxVel, fabs(v_next))
        a_abs = ag_scaled[i] + a_next
        maxAccel = fmax(maxAccel, fabs(a_abs))
        F_ray = c * v_next
        F_total = F_hys + F_ray
        maxReaction = fmax(maxReaction, fabs(F_total))  # 最大应力
        Ec += 0.5 * (F_hys + F_hys_old) * du
        Ev += 0.5 * (F_ray + F_ray_old) * du
        CD += fabs(du)
        if uy == 0.0:
            CPD = 0.0
        else:
            if u_next > u_cent + uy:
                # 正向屈服
                CPD += u_next - (u_cent + uy)
                u_cent += u_next - (u_cent + uy)
            elif u_next < u_cent - uy:
                # 负向屈服
                CPD += u_cent - uy - u_next
                u_cent -= u_cent - uy - u_next
            else:
                CPD += 0
        # 更新状态变量
        u_old = u_next
        F_hys_old, F_ray_old = F_hys, F_ray
    
        if (current_time >= duration or (fabs(current_time - duration) < 1e-5)):
            return {
                'converge': converge,
                'collapse': collapse_flag,
                'maxDisp': maxDisp,
                'maxVel': maxVel,
                'maxAccel': maxAccel,
                'Ec': Ec,
                'Ev': Ev,
                'maxReaction': maxReaction,
                'CD': CD,
                'CPD': CPD,
                'resDisp': u_next
            }
        if dt + current_time > duration:
            dt = duration - current_time
