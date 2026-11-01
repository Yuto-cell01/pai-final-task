# PAI最終課題：模倣学習による把持姿勢の予測

Genesis上でFranka Pandaアームがブロックを掴んで持ち上げるタスクです。
IKで計算していた掴む姿勢を、模倣学習したMLPに置き換えて比較しました。

## ファイル
- pai_range_experiment.py : IK版で、ブロック位置の範囲を広げたときの成功率を測る実験
- pai_collect_data.py : 位置と掴む姿勢のデータ収集（dataset.npz）
- pai_train_policy_v2.py : MLPの学習（policy_v2.pt）
- pai_learned_eval.py : 学習モデルの評価（MODE = "check" / "eval"）
- analyze_v2.py : 結果の集計

## 実行環境
Python 3.11、genesis-world、PyTorch、NumPy（macOS・Apple SiliconのMetalで動作確認）

## 実行の流れ
1. python pai_collect_data.py
2. python pai_train_policy_v2.py
3. python pai_learned_eval.py

## 結果
ブロックを置く範囲を4段階（条件A〜D）に広げて、成功率を測りました。
成功は、持ち上げた後のブロックの高さが0.15 mを超えた場合です。

| 条件 | 距離 [m] | 向き | IK版（60回） | 学習モデル版（30回） |
|---|---|---|---|---|
| A | 0.40〜0.60 | ±30度 | 0.98 | 1.00 (30/30) |
| B | 0.40〜0.70 | ±45度 | 0.93 | 0.93 (28/30) |
| C | 0.35〜0.75 | ±60度 | 0.90 | 0.83 (25/30) |
| D | 0.30〜0.80 | ±75度 | 0.85 | 0.70 (21/30) |

- 学習モデル版は、条件A・BではIK版と同程度でしたが、範囲が広い条件C・Dでは低くなりました。学習によって成功率が上がった、という結果ではありません。
- 学習モデル版の試行数は各条件30回と少なく、IK版との差が偶然かどうかは判断できません。
- IK版の数字は、実行時の出力から転記した値です（元のCSVは、のちの実行で上書きされて失われました）。学習モデル版の結果は `results_policy_v2.csv` にあります。
- 詳しい手法、考察、限界は、提出したレポートを参照してください。
