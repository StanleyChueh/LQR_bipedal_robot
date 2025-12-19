import mujoco
import mujoco.viewer
import pinocchio

import time
import numpy as np
import yaml
import argparse

from scipy.linalg import solve_continuous_are

# drawing
import matplotlib.pyplot as plt

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
    global _pinocchio_model, _pinocchio_data
    urdf_path = "/home/stanley/NTUT_master/crazydog_lqr_control/urdf/crazydog_urdf.urdf"
    _pinocchio_model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())
    
    for link in ["L_wheel", "R_wheel"]:
        fid = _pinocchio_model.getFrameId(link)
        jid = _pinocchio_model.frames[fid].parentJoint
        inertia = _pinocchio_model.inertias[jid]
        _pinocchio_model.inertias[jid] = pinocchio.Inertia(0.0, inertia.lever, inertia.inertia)
    
    _pinocchio_data = _pinocchio_model.createData()

# compute l from given joint angles
def compute_COM_and_l(initial_angles):
    # load model
    global _pinocchio_model, _pinocchio_data

    nq = _pinocchio_model.nq
    q = np.zeros(nq)
    
    q[0:7] = np.array([0,0,0, 1,0,0,0])

    # joints = your initial_angles (6 motors)
    q[7:7+len(initial_angles)] = initial_angles

    # ---------- Forward kinematics ----------
    pinocchio.forwardKinematics(_pinocchio_model, _pinocchio_data, q)
    pinocchio.updateFramePlacements(_pinocchio_model, _pinocchio_data)

    # ---------- COM ----------
    com = pinocchio.centerOfMass(_pinocchio_model, _pinocchio_data, q)

    pL = _pinocchio_data.oMf[_pinocchio_model.getFrameId("L_wheel")].translation
    pR = _pinocchio_data.oMf[_pinocchio_model.getFrameId("R_wheel")].translation
    p_axle = 0.5 * (pL + pR)

    # compute l = ||COM - axle||
    delta = com - p_axle
    l = np.linalg.norm(delta[[0, 2]])  # 在 sagittal plane

    print("\n===== Pinocchio COM / axle info =====")
    print("angles =", initial_angles)
    print("COM =", com)
    print("Axle center =", p_axle)
    print("Computed l =", l)
    print("====================================\n")

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

    l_calculated = compute_COM_and_l(initial_angles)

    # === LQR controller ===
    K = build_LQR_6x6(l_calculated)

    delta_est = 0.0

    # === MuJoCo model ===
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

    plt.suptitle("Pitch Dynamics and Control Signals", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


