import numpy as np
import genesis as gs

gs.init(backend=gs.metal)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, -2.0, 1.5),
        camera_lookat=(0.6, 0.0, 0.2),
        camera_fov=35,
    ),
    show_viewer=False,
)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
)
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

hand = franka.get_link("hand")
down = np.array([0, 1, 0, 0])

init_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
franka.set_qpos(init_qpos)
franka.control_dofs_position(init_qpos)
for _ in range(100):
    scene.step()

def move_to(pos, steps=150):
    qpos = franka.inverse_kinematics(link=hand, pos=np.array(pos), quat=down)
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=steps)
    for wp in path:
        franka.control_dofs_position(wp[:7], motors_dof)
        scene.step()
    for _ in range(30):
        scene.step()

def run_trial(x, y):
    scene.reset()
    cube.set_pos(np.array([x, y, 0.02]))
    franka.set_qpos(init_qpos)
    franka.control_dofs_position(init_qpos)
    for _ in range(100):
        scene.step()

    franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
    for _ in range(50):
        scene.step()

    move_to([x, y, 0.25])
    move_to([x, y, 0.12])
    grasp_q = to_np(franka.get_dofs_position())[:7].copy()

    franka.control_dofs_position(franka.get_dofs_position()[:7], motors_dof)
    franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)

    for _ in range(100):
        scene.step()

    move_to([0.65, 0.0, 0.35])

    for _ in range(200):
        scene.step()

    cube_z = float(cube.get_pos()[2])
    return cube_z > 0.15, grasp_q

import csv

def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

N = 300
rng = np.random.default_rng(100)

xs, qs, oks = [], [], []
for i in range(N):
    r = rng.uniform(0.30, 0.80)
    th = rng.uniform(-np.deg2rad(75), np.deg2rad(75))
    x, y = r * np.cos(th), r * np.sin(th)
    try:
        ok, q = run_trial(x, y)
        ok = bool(ok)
    except Exception as e:
        print("エラー:", e)
        continue
    xs.append([x, y])
    qs.append(q)
    oks.append(int(ok))
    print(f"{i + 1}/{N}: ({x:.2f}, {y:.2f}) -> {'成功' if ok else '失敗'}")

np.savez("dataset.npz", xy=np.array(xs), q=np.array(qs), ok=np.array(oks))
print("保存しました:", len(xs), "件のうち成功", sum(oks), "件")



