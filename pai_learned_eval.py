import numpy as np
import genesis as gs
import torch
import torch.nn as nn
import csv

MODE = "eval"
POLICY_PATH = "policy_v2.pt"

HOVER_Z = 0.25
GRASP_Z = 0.12
LIFT_Z = 0.35
FORCE = -1.0

gs.init(backend=gs.metal)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, -2.0, 1.5),
        camera_lookat=(0.6, 0.0, 0.2),
        camera_fov=35,
    ),
    show_viewer=(MODE == "eval"),
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

def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

def step(n):
    for _ in range(n):
        scene.step()

def move_to(pos, steps=150):
    qpos = franka.inverse_kinematics(link=hand, pos=np.array(pos), quat=down)
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=steps)
    for wp in path:
        franka.control_dofs_position(wp[:7], motors_dof)
        scene.step()
    step(30)

def load_policy(path):
        ck  = torch.load(path)
        net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 7),
        )
        net.load_state_dict(ck["model"])
        net.eval()
        polar = ck.get("features") == "polar"
        xm, xs = ck["x_mean"].numpy(), ck["x_std"].numpy()
        ym, ys = ck["y_mean"].numpy(), ck["y_std"].numpy()

        def predict(x, y):
            feat = [np.hypot(x, y), np.arctan2(y, x)] if polar else [x,y]
            inp = (np.array(feat, dtype=np.float32) -  xm) / xs
            with torch.no_grad():
                out = net(torch.tensor(inp)[None]).numpy()[0]
            return out * ys + ym
        
        return predict

def run_trial_learned(x, y, predict):
    scene.reset()
    cube.set_pos(np.array([x, y, 0.02]))
    franka.set_qpos(init_qpos)
    franka.control_dofs_position(init_qpos)
    step(100)

    franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
    step(50)

    move_to([x, y, HOVER_Z])

    q_goal = predict(x, y)
    q_start = to_np(franka.get_dofs_position())[:7]
    for t in np.linspace(0, 1, 100):
        franka.control_dofs_position(q_start + t * (q_goal - q_start), motors_dof)
        scene.step()
    step(30)

    franka.control_dofs_position(to_np(franka.get_dofs_position())[:7], motors_dof)
    franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
    step(100)

    move_to([x, y, LIFT_Z])
    step(50)

    z = float(to_np(cube.get_pos())[2])
    return z > 0.15, z

predict = load_policy(POLICY_PATH)

if MODE == "CHECK":
    bad = lambda x, y: predict(x, y) + 0.5
    for name, p in [("学習モデル", predict), ("わざと悪いモデル", bad)]:
        for deg in (-45, -20, 0, 20, 45):
            th = np.deg2rad(deg)
            x, y = 0.55 * np.cos(th), 0.55 * np.sin(th)
            ok, z = run_trial_learned(x, y, p)
            print(f"{name} 向き{deg:+d}度: 最終の高さ {z:.3f} m -> {'成功' if ok else '失敗'}")


else:
    conditions = [
        ("A", (0.40, 0.60), np.deg2rad(30)),
        ("B", (0.40, 0.70), np.deg2rad(45)),
        ("C", (0.35, 0.75), np.deg2rad(60)),
        ("D", (0.30, 0.80), np.deg2rad(75))
    ]
    N = 10
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
                    ok, z = run_trial_learned(x, y, predict)
                except Exception as e:
                    print(" エラー:", e)
                    ok = False
                n_ok += ok
                rows.append([name, seed, round(x, 3), round(y, 3), int(ok)])
            rates.append(n_ok / N)
            print(f"条件{name} seed{seed}: {n_ok}/{N}")
        print(f"== 条件{name}: 平均 {np.mean(rates):.2f} (最小 {min(rates):.2f} / 最大{max(rates):.2f}) ==")

    with open("results_policy_v2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "x", "y", "success"])
        writer.writerows(rows)

