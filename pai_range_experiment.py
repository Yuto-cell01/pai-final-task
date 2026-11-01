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

    franka.control_dofs_position(franka.get_dofs_position()[:7], motors_dof)
    franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)

    for _ in range(100):
        scene.step()

    move_to([0.65, 0.0, 0.35])

    for _ in range(200):
        scene.step()

    cube_z = float(cube.get_pos()[2])
    return cube_z > 0.15

import csv

conditions = [
    ("A", (0.40, 0.60), np.deg2rad(30)),
    ("B", (0.40, 0.70), np.deg2rad(45)),
    ("C", (0.35, 0.75), np.deg2rad(60)),
    ("D", (0.30, 0.80), np.deg2rad(75))
]
N = 20
SEEDS = [0, 1, 2]

rows =[]
for name, (r_min, r_max), th_max in conditions:
    rates = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n_ok = 0
        for i in range(N):
            r = rng.uniform(r_min, r_max)
            th = rng.uniform(-th_max, th_max)
            x, y = r * np.cos(th), r * np.sin(th)
            try:
                ok = bool(run_trial(x, y))
            except Exception as e:
                print(" エラー:", e)
                ok = False
            n_ok += ok
            rows.append([name, round(x, 3), round(y, 3), int(ok)])
        rates.append(n_ok / N)
        print(f"条件{name} seed{seed}: {n_ok}/{N}")
    print(f"== 条件{name}: 平均 {np.mean(rates):.2f} (最小 {min(rates):.2f} / 最大{max(rates):.2f}) ==")

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["condition", "seed", "x", "y", "success"])
    writer.writerows(rows)


