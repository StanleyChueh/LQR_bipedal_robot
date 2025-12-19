# Crazydog LQR Control

Simulate the two-wheeled Crazydog robot in MuJoCo and stabilize it with a continuous-time LQR (Linear Quadratic Regulator) that handles forward velocity and yaw simultaneously. The project combines:

- MuJoCo scenes (`scene.xml`, `crazydog.xml`, and the meshes in `meshes/`)
- Pinocchio-based estimation of the COM-to-axle distance `l`
- A 6×6 augmented-state LQR controller
- A Pygame keyboard HUD for tele-operation

---

## Highlights
- **Online COM estimation** – Pinocchio loads the URDF, removes wheel masses, and computes the COM-to-axle distance `l`. A lookup table of joint angles vs. `l` can be pre-built on startup.
- **Continuous-time LQR** – The state `[x, ẋ, θ, θ̇, δ, δ̇]` is controlled by an LQR that outputs forward/turn torques, which are converted to independent wheel torques.
- **Leg PD stabilization** – Six joints are stabilized with configurable PD gains to hold either the nominal stance or the optimized pose.
- **Keyboard HUD** – `W/S` command velocity, `A/D` command yaw, `Q/E` tweak the desired `l`, and the HUD mirrors the commands in real time.
- **Data logging & plots** – Pitch, pitch rate, and control commands are recorded and visualized with Matplotlib after the simulation ends.

---

## Requirements
| Item | Description |
| --- | --- |
| OS | Ubuntu 20.04+/WSL or any desktop environment that can run MuJoCo |
| Python | 3.9+ (verified on 3.10) |
| System deps | MuJoCo requires OpenGL/EGL; Pinocchio can be installed via `pip install pin` or distro packages |
| Python packages | `mujoco`, `numpy`, `scipy`, `pinocchio`, `pygame`, `PyYAML`, `matplotlib` |
| Display | Both the Pygame HUD and MuJoCo viewer need an available display (use X11 forwarding or `MUJOCO_GL=egl` if remote) |

```bash
pip install mujoco numpy scipy pin PyYAML pygame matplotlib
```

> **Pinocchio URDF path**  
> `crazydog_lqr_balance.py` expects `/home/ray/crazydog_mujoco/crazydog_urdf/urdf/crazydog_urdf.urdf`. Update the path or create a symlink if yours differs.

---

## Project Layout
```
crazydog_lqr_control/
├── config/                 # YAML controller config
├── crazydog_lqr_balance.py # Main simulation + LQR logic
├── keyboard_controller.py  # Pygame controller and HUD
├── crazydog.xml            # Robot MuJoCo model
├── scene.xml               # Scene setup (ground, lighting, etc.)
├── meshes/                 # STL geometry
├── urdf/crazydog_urdf.urdf # URDF consumed by Pinocchio
└── mujoco_viewer.py        # Utility to list bodies/joints/sensors
```

---

## Quick Start
1. **Install MuJoCo** – Use MuJoCo 3.x and export `LD_LIBRARY_PATH` / `MUJOCO_GL` as needed.  
2. **Install Python deps** – Run the `pip install` command above inside a virtual environment.  
3. **Verify the URDF path** – Adjust the absolute path inside `crazydog_lqr_balance.py` if necessary.  
4. **Launch the controller**:
   ```bash
   python crazydog_lqr_balance.py config/crazydog.yaml
   ```

At startup the script:
1. Loads `config/crazydog.yaml` for PD gains and initial joint targets.
2. Computes `l` via Pinocchio and optionally optimizes to match `--l_override`.
3. Builds the LQR gain matrix `K`.
4. Launches the Pygame keyboard controller and MuJoCo viewer.
5. Runs the control loop, logs telemetry, and plots the data on exit.

---

## Keyboard Controls
| Key | Action |
| --- | --- |
| `W` / `S` | Increase / decrease forward velocity target (m/s) |
| `A` / `D` | Increase / decrease yaw-rate target (rad/s) |
| `Q` / `E` | Nudge `l` target within `[0.15, 0.35]` m |
| `Space`   | Zero out both velocity and yaw commands |

The HUD mirrors every command plus the current `l`. If the window does not appear, double-check your desktop/X11 forwarding setup.

---

## Config Reference (`config/crazydog.yaml`)
| Field | Description |
| --- | --- |
| `xml_path` | MuJoCo scene file, defaults to `scene.xml` (which includes `crazydog.xml`) |
| `simulation_duration` | Viewer runtime in seconds |
| `simulation_dt` | Simulation timestep, also copied to `model.opt.timestep` |
| `kps`, `kds` | Six-axis leg PD gains ordered as `[R_hip, R_knee, R_wheel, L_hip, L_knee, L_wheel]` |
| `initial_angles` | Initial joint targets, also used for COM estimation |

Edit the file and restart the script to apply changes.

---

## Controller Pipeline
1. **COM & `l` estimation** – `compute_COM_and_l` runs Pinocchio forward kinematics and COM computation while nulling wheel masses for a cleaner measurement.
2. **Height lookup table** – `build_l_lookup_table` scans `[0.15, 0.35]` m using `find_angles_for_target_l`, so future `l` requests can be satisfied by interpolation in milliseconds.
3. **LQR synthesis** – `build_LQR_6x6` derives the 6×6 dynamics matrices from `l`, using `Q = diag([5, 200, 50, 10, 80, 10])` and `R = diag([0.5, 1.0])`.
4. **Control application** – `lqr_6x6_full_step` outputs `(u_fwd, u_turn)`, maps them to left/right wheel torques (clamped to ±15 Nm), and applies PD torques to the leg joints.
5. **Logging** – Pitch, pitch rate, and control signals are accumulated and plotted with Matplotlib at the end of the run.

---

## FAQ
- **Pinocchio cannot find the URDF** – Update `urdf_path` inside `compute_COM_and_l`, or provide an environment variable/symlink that matches the expected location.
- **MuJoCo viewer fails to start** – Ensure a working OpenGL context; on headless servers set `export MUJOCO_GL=egl` and run under `Xvfb`/`vglrun`.
- **Pygame cannot open a window** – Verify `$DISPLAY` and that you are running inside a GUI session or forwarding X11 properly.
- **Controller diverges** – Recalculate `l` (possibly via `--l_override`), tweak `kps/kds`, or adjust the LQR weighting matrices.
- **Need quick model introspection** – Run `python mujoco_viewer.py` to print every body/joint/actuator/sensor index.

---

## Future Work
- Persist the `build_l_lookup_table` output as `.npy` so startup time shrinks.
- Replace keyboard commands with UDP or ROS topics for remote/real-robot testing.
- Extend `crazydog_lqr_balance.py` with logging to disk and automated tests.
