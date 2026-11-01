import csv
from collections import defaultdict

import numpy as np

rows = []
count = defaultdict(int)
with open("results.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if not row:
            continue
        name, x, y, ok = row[0], row[-3], row[-2], row[-1]
        seed = row[1] if len(row) == 5 else str(count[name] // 20)
        count[name] += 1
        rows.append({"condition": name, "seed": seed, "x": x, "y": y, "success": ok})

print("読み込んだ行数:", len(rows))
print("条件ごとの行数:", dict(count))
print("最初の3行:", rows[:3])

rates = defaultdict(lambda: defaultdict(list))
for r in rows:
    rates[r["condition"]][r["seed"]].append(int(r["success"]))

print("--- 条件ごとの成功率 ---")
for name in sorted(rates):
    per_seed = [np.mean(v) for v in rates[name].values()]
    print(f"条件{name}: 平均 {np.mean(per_seed):.2f} "
          f"(最小 {min(per_seed):.2f} / 最大 {max(per_seed):.2f})")

points = []
for r in rows:
    px, py = float(r["x"]), float(r["y"])
    points.append({
        "ok": int(r["success"]),
        "r": float(np.hypot(px, py)),
        "theta": float(np.degrees(abs(np.arctan2(py, px)))),
    })


def table(label, key, edges):
    print(f"--- {label}ごとの成功率（全条件まとめ）---")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [p for p in points if lo <= p[key] < hi]
        if sub:
            ok = sum(p["ok"] for p in sub)
            print(f"{lo}〜{hi}: {ok}/{len(sub)} ({ok / len(sub):.0%})")


table("根元からの距離[m]", "r", [0.2, 0.4, 0.5, 0.6, 0.7, 0.9])
table("左右の角度[度](絶対値)", "theta", [0, 15, 30, 45, 60, 75, 90]) 