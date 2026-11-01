import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

d = np.load("dataset.npz")
mask = d["ok"] == 1
XY = d["xy"][mask].astype(np.float64)
Y = d["q"][mask].astype(np.float32)

r = np.hypot(XY[:, 0], XY[:, 1])
th = np.arctan2(XY[:, 1], XY[:, 0])

dev = np.abs(np.degrees(Y[:, 0] -th))
for t in (15, 30, 45):
    print(f"|関節1 - θ| < {t}度: {(dev <t ).sum()} / {len(dev)} 件")

THRESH = 30
keep = dev < THRESH
print(f"外れ値として除く: {(~keep).sum()} 件")

X = np.stack([r, th], 1).astype(np.float32)[keep]
Y = Y[keep]
print("学習に使うデータ:", len(X), "件")

idx = np.random.permutation(len(X))
n_val = max(1, int(len(X) * 0.2))
val_idx, tr_idx = idx[:n_val], idx[n_val:]

x_mean, x_std = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-8
y_mean, y_std = Y[tr_idx].mean(0), Y[tr_idx].std(0) + 1e-8

x_tr = torch.tensor((X[tr_idx] - x_mean) / x_std)
y_tr = torch.tensor((Y[tr_idx] - y_mean) / y_std)
x_val = torch.tensor((X[val_idx] - x_mean) / x_std)

model = nn.Sequential(
    nn.Linear(2, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 7),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(500):
    opt.zero_grad()
    loss = loss_fn(model(x_tr), y_tr)
    loss.backward()
    opt.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d} loss {loss.item():.5f}")

with torch.no_grad():
    pred = model(x_val).numpy() * y_std + y_mean
err = np.abs(pred - Y[val_idx])
baseline = np.abs(Y[val_idx] - y_mean)
print(f"検証データ {len(val_idx)}件")
print(f"予測の平均誤差: {np.degrees(err.mean()):.2f} 度")
print(f"予測の中央値誤差: {np.degrees(np.median(err)):.2f} 度")
print(f"予測の最大誤差: {np.degrees(err.max()):.2f} 度")
print(f" (参考) 平均値を答えた場合の誤差: {np.degrees(baseline.mean()):.2f} 度")

torch.save({
    "model": model.state_dict(),
    "features": "polar",
    "x_mean": torch.tensor(x_mean), "x_std": torch.tensor(x_std),
    "y_mean": torch.tensor(y_mean), "y_std": torch.tensor(y_std),   
}, "policy_v2.pt")
print("policy_v2.pt に保存しました")