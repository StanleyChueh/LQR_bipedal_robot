import time
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import argparse
import pinocchio
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
import socket
import json
import threading
import matplotlib.pyplot as plt

NUM_MOTOR = 6

_pinocchio_model = None
_pinocchio_data = None
       
def quat_to_pitch(qw, qx, qy, qz):
    num = 2.0 * (qw*qy + qx*qz)
    den = 1.0 - 2.0 * (qy*qy + qz*qz)
    return np.arctan2(num, den)

# leg PD controller
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def init_pinocchio_model():
    """只在啟動時調用一次"""
    global _pinocchio_model, _pinocchio_data
    urdf_path = "/home/stanley/NTUT_master/crazydog_lqr_control/urdf/crazydog_urdf.urdf"
    _pinocchio_model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())
    
    # 移除輪子質量
    for link in ["L_wheel", "R_wheel"]:
        fid = _pinocchio_model.getFrameId(link)
        jid = _pinocchio_model.frames[fid].parentJoint
        inertia = _pinocchio_model.inertias[jid]
        _pinocchio_model.inertias[jid] = pinocchio.Inertia(0.0, inertia.lever, inertia.inertia)
    
    _pinocchio_data = _pinocchio_model.createData()

# ======================================================
# ----------  Pinocchio：自動計算質心 → 輪軸距離 l ----------
# ======================================================
def compute_COM_and_l(initial_angles):
    """使用預載入的模型"""
    global _pinocchio_model, _pinocchio_data

    nq = _pinocchio_model.nq
    q = np.zeros(nq)
    
    # base 固定在 0 0 0 + quaternion 1 0 0 0
    q[0:7] = np.array([0,0,0, 1,0,0,0])

    # joints = your initial_angles (6 motors)
    q[7:7+len(initial_angles)] = initial_angles

    # ---------- Forward kinematics ----------
    pinocchio.forwardKinematics(_pinocchio_model, _pinocchio_data, q)
    pinocchio.updateFramePlacements(_pinocchio_model, _pinocchio_data)

    # ---------- COM ----------
    com = pinocchio.centerOfMass(_pinocchio_model, _pinocchio_data, q)

    # ---------- 取輪軸中點 ----------
    pL = _pinocchio_data.oMf[_pinocchio_model.getFrameId("L_wheel")].translation
    pR = _pinocchio_data.oMf[_pinocchio_model.getFrameId("R_wheel")].translation
    p_axle = 0.5 * (pL + pR)

    # ---------- 計算 COM 與輪軸距離 ----------
    delta = com - p_axle
    l = np.linalg.norm(delta[[0, 2]])  # 在 sagittal plane

    print("\n===== Pinocchio COM / axle info =====")
    print("angles =", initial_angles)
    print("COM =", com)
    print("Axle center =", p_axle)
    print("Computed l =", l)
    print("====================================\n")

    return l

def build_l_lookup_table(l_min=0.15, l_max=0.25, num_samples=30):
    """預計算 l 值對應的關節角度查找表"""
    print("\n===== 建立高度查找表 =====")
    print(f"範圍: {l_min:.3f} - {l_max:.3f} m")
    print(f"樣本數: {num_samples}")
    print("正在計算...")
    
    l_values = np.linspace(l_min, l_max, num_samples)
    angle_table = []
    
    for i, l_target in enumerate(l_values):
        angles = find_angles_for_target_l(l_target)
        angle_table.append(angles)
        if (i + 1) % 5 == 0:
            print(f"  進度: {i+1}/{num_samples} ({(i+1)/num_samples*100:.1f}%)")
    
    angle_table = np.array(angle_table)
    print("查找表建立完成！")
    print("========================\n")
    return l_values, angle_table

def interpolate_angles_from_l(l_target, l_values, angle_table):
    """從查找表插值獲取關節角度（極快，<1ms）"""
    # 對每個關節角度單獨插值
    interpolated_angles = np.zeros(6)
    for i in range(6):
        interpolated_angles[i] = np.interp(l_target, l_values, angle_table[:, i])
    return interpolated_angles

def find_angles_for_target_l(target_l, initial_guess=None):
    """使用優化方法尋找能達到目標 l 的關節角度"""

    if initial_guess is None:
        initial_guess = np.array([1.27, -2.127, 0, 1.27 , -2.127, 0])
    # 你認為「自然」的 hip / knee 角度
    hip_nat  = 1.27    # 可以自己調
    knee_nat = -2.127   # 可以自己調

    def objective(angles):
        try:
            l_current = compute_COM_and_l_silent(angles)

            hip_L, knee_L, wheel_L, hip_R, knee_R, wheel_R = angles

            # 1) 主要目標： l 接近 target_l
            loss_l = (l_current - target_l)**2

            # 2) 對稱懲罰
            symmetry_penalty = (
                (hip_L - hip_R)**2 +
                (knee_L - knee_R)**2 +
                (wheel_L - wheel_R)**2
            )

            # 3) 自然站姿懲罰：希望接近 (hip_nat, knee_nat)
            # 左右兩邊都拉向同一個自然角
            prior_penalty = (
                (hip_L  - hip_nat )**2 +
                (knee_L - knee_nat)**2 +
                (hip_R  - hip_nat )**2 +
                (knee_R - knee_nat)**2
            )

            # 權重自己調：越大越「黏著」自然站姿
            w_sym   = 0.1
            w_prior = 0.0001

            return loss_l + w_sym * symmetry_penalty + w_prior * prior_penalty
        except:
            return 1e6
    
    # 關節限制（根據你的機器人規格調整）
    bounds = [
        (0, 1.5),      # hip joints
        (-2.61, 0),     # knee joints  
        (-0.5, 0.5),    # wheel joints
        (0, 1.5), 
        (-2.61, 0),
        (-0.5, 0.5)
    ]
    
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        optimized_angles = result.x
        achieved_l = compute_COM_and_l_silent(optimized_angles)
        # print(f"優化結果: 目標 l={target_l:.4f}, 達成 l={achieved_l:.4f}")
        # print(f"優化角度: {optimized_angles}")
        return optimized_angles
    else:
        print(f"優化失敗，使用初始角度")
        return initial_guess

def compute_COM_and_l_silent(angles):
    """不輸出信息的版本，用於優化"""
    global _pinocchio_model, _pinocchio_data
    
    nq = _pinocchio_model.nq
    q = np.zeros(nq)
    q[0:7] = np.array([0,0,0, 1,0,0,0])
    q[7:7+len(angles)] = angles

    pinocchio.forwardKinematics(_pinocchio_model, _pinocchio_data, q)
    pinocchio.updateFramePlacements(_pinocchio_model, _pinocchio_data)

    com = pinocchio.centerOfMass(_pinocchio_model, _pinocchio_data, q)
    pL = _pinocchio_data.oMf[_pinocchio_model.getFrameId("L_wheel")].translation
    pR = _pinocchio_data.oMf[_pinocchio_model.getFrameId("R_wheel")].translation
    p_axle = 0.5 * (pL + pR)

    delta = com - p_axle
    l = np.linalg.norm(delta[[0, 2]])
    
    return l

def build_LQR_6x6(l):
    # === 參數 ===
    w_radius = 0.07046       # wheel radius
    D_distance  = 0.36       # wheel distance(0.32910)
    m_wheel = 0.2805 # 0.28
    M_body  = 5.769   # 6.441
    g = 9.8

    I_wheel = 0.5 * m_wheel * w_radius**2
    Jp      = (1/3) * M_body * l**2

    J_delta = (1/12) * m_wheel * D_distance**2

    # === 原本直立動態 ===
    Qeq = Jp*M_body + (Jp + M_body*l*l) * (2*m_wheel + 2*I_wheel/w_radius**2)

    A23 = -(M_body**2 * l**2 * g) / Qeq
    A43 = M_body*l*g*(M_body + 2*m_wheel + 2*I_wheel/w_radius**2) / Qeq

    B21 = (Jp + M_body*l**2 + M_body*l*w_radius) / (Qeq * w_radius)
    B41 = -((M_body*l/w_radius) + M_body + 2*m_wheel + 2*I_wheel/w_radius**2) / Qeq

    denom = w_radius * (m_wheel * D_distance + I_wheel * D_distance / (w_radius**2) + 2.0 * J_delta / D_distance)

    B61 =  1.0 / denom
    B62 = -1.0 / denom

    A4 = np.array([
        [0, 1,    0,   0],
        [0, 0,  A23,   0],
        [0, 0,    0,   1],
        [0, 0,  A43,   0]
    ])

    B_fwd = np.array([[0.0],
                      [B21],
                      [0.0],
                      [B41]])   # 4×1


    B_turn = np.array([
        [0.0],            # delta_dot does NOT directly affect delta
        [B62]    # delta_ddot = B * u_turn
    ])

    # yaw A:
    A_yaw = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ])

    # === combine to 6×6 ===
    A6 = np.zeros((6,6))
    A6[:4,:4] = A4
    A6[4:,4:] = A_yaw

    B6 = np.zeros((6,2))
    B6[:4,[0]] = B_fwd          # u_fwd
    B6[4:,[1]] = B_turn         # u_turn

    # === LQR weights ===
    Q = np.diag([5, 200, 50, 10,   # x, x_dot, theta, theta_dot
                 80, 10])          # delta, delta_dot

    R = np.diag([0.5, 1.0])        # cost on u_fwd, u_turn

    # solve CARE
    P = solve_continuous_are(A6, B6, Q, R)
    K = np.linalg.inv(R) @ B6.T @ P   # (2×6)

    # print("===== LQR 6x6 Gain K (u_fwd,u_turn) =====")
    # print(K)
    # print(K.shape)
    # print("=========================================\n")

    return K

def lqr_6x6_full_step(m, d, dt, kps, kds, target_dof_pos, K, v_ref=0.0, yaw_rate_ref=0.0, delta_est=0.0):
    x, y, z = d.sensordata[28:31]
    qw, qx, qy, qz = d.sensordata[18:22]
    gx, gy, gz = d.sensordata[22:25]
    vx, vy, vz = d.sensordata[31:34]

    theta = quat_to_pitch(qw,qx,qy,qz)
    theta_dot = gy
    x_dot = vx
    yaw_rate = gz

     # --- yaw 角估測 ---
    delta_est += yaw_rate * dt

    # 6-state error (考慮參考值)
    x_state = np.array([
        0,                      # x position error (always 0 for velocity control)
        x_dot - v_ref,          # velocity error
        theta,                  # pitch angle error (目標為 0)
        theta_dot,              # pitch rate error (目標為 0)
        delta_est,              # yaw angle (integrated from yaw_rate)
        yaw_rate - yaw_rate_ref # yaw rate error
    ])

    # print(f"vx={vx:.3f} (ref={v_ref:.3f}), gz={gz:.3f} (ref={yaw_rate_ref:.3f})")

    # control law
    u_vec = -K @ x_state
    u_fwd, u_turn = float(u_vec[0]), float(u_vec[1])

    # convert to wheel torque
    T_L = 0.5*(u_fwd + u_turn)
    T_R = 0.5*(u_fwd - u_turn)

    T_L = np.clip(T_L, -15, 15)
    T_R = np.clip(T_R, -15, 15)

    # PD control for legs
    tau_leg = pd_control(
        target_dof_pos, d.sensordata[:6], kps,
        np.zeros(6), d.sensordata[6:12], kds
    )
    d.ctrl[0] = tau_leg[0]
    d.ctrl[1] = tau_leg[1]
    d.ctrl[2] = T_R
    d.ctrl[3] = tau_leg[3]
    d.ctrl[4] = tau_leg[4]
    d.ctrl[5] = T_L

    return theta, theta_dot, u_vec, x_dot, delta_est


# ======================================================
# 主程式
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        initial_angles = np.array(config["initial_angles"], dtype=np.float32)

    target_dof_pos = initial_angles.copy()

    init_pinocchio_model()

    # === Pinocchio 計算質心距離 l ===
    l_calculated = compute_COM_and_l(initial_angles)

    # === 建立 LQR 控制器 ===
    K = build_LQR_6x6(l_calculated)

    l_min = max(0.15, l_calculated - 0.10)
    l_max = min(0.35, l_calculated + 0.10)
    l_lookup_values, angle_lookup_table = build_l_lookup_table(l_min, l_max, num_samples=30)

    delta_est = 0.0
    l_previous = l_calculated  # 記錄上一次的 l 值
    
    print("\n===== 控制說明 =====")
    print("使用遙控器程序控制機器人：")
    print("  python test.py config/crazydog.yaml")
    print("==================\n")

    # === MuJoCo 模型 ===
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    theta_list, theta_dot_list, u_fwd_list, u_turn_list = [], [], [], []

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and (time.time() - start < simulation_duration):
            step_start = time.time()

            v_ref = 0.0          
            yaw_rate_ref = 0.0   
            l_target = None      
            
            theta, theta_dot, u_vec, x_dot, delta_est = lqr_6x6_full_step(
                m, d, m.opt.timestep,
                kps, kds, target_dof_pos,
                K,
                v_ref=v_ref,
                yaw_rate_ref=yaw_rate_ref,
                delta_est=delta_est
            )

            # 記錄數據
            theta_list.append(theta)
            theta_dot_list.append(theta_dot)
            u_fwd_list.append(u_vec[0])
            u_turn_list.append(u_vec[1])

            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(max(0, m.opt.timestep - (time.time() - step_start)))

    # ———— 繪圖（美化版） ————
    t = np.arange(len(theta_list)) * m.opt.timestep

    plt.figure(figsize=(12, 8))

    # === Pitch Angle ===
    plt.subplot(3, 1, 1)
    plt.plot(t, theta_list, color='tab:blue', linewidth=2, alpha=1)
    plt.ylabel("Theta (rad)", fontsize=11)
    plt.title("Pitch Angle", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)

    # === Pitch Angular Velocity ===
    plt.subplot(3, 1, 2)
    plt.plot(t, theta_dot_list, color='tab:green', linewidth=2, alpha=1)
    plt.ylabel("Theta dot (rad/s)", fontsize=11)
    plt.title("Pitch Angular Velocity", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)

    # === Control Inputs ===
    plt.subplot(3, 1, 3)
    plt.plot(t, u_fwd_list, color='tab:orange', linewidth=2, alpha=1, label='u_fwd')
    plt.plot(t, u_turn_list, color='tab:purple', linewidth=2, alpha=1, label='u_turn')
    plt.ylabel("Control Input", fontsize=11)
    plt.xlabel("Time (s)", fontsize=11)
    plt.title("Control Inputs", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.4)

    # === 全域調整 ===
    plt.suptitle("Pitch Dynamics and Control Signals", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


