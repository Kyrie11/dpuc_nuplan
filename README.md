# DPUC on nuPlan DB (PyTorch)

这是一个将论文方法落到 **nuPlan SQLite -> preprocess -> train -> offline planner evaluation** 的 PyTorch 工程。

当前版本已经补齐了下面四条关键运行链：

1. `interface_best.pt` 在 planner runtime 中真正参与 slot/interface 推理。
2. `support_best.pt` 在 retained-support 选择中真正参与 uplift / support scoring。
3. `dbi_best.pt` 在 selective individualization 中真正参与 agent ranking。
4. `offline_eval` 支持论文 Experiments 一节对应的 **offline 实验套件**、预算曲线、ablation、可靠性统计和 README 中的一键命令。

## 1. 重要说明

这版仓库现在支持的是：

- **完整 train**
- **完整 offline test / offline experiments**
- **论文 Experiments 一节的所有实验的 offline 版本**

这版仓库**不包含官方 nuPlan closed-loop simulator 集成**。因此论文中的 Q1 `Main Closed-Loop Results` 在本仓库里被实现为：

- `main_closed_loop_offline_proxy`
- 使用相同 planner 输出构造 `Score / Collision / Progress / Comfort / RouteSucc / IntScore` 的 **offline proxy**

也就是说：

- 如果你现在只有 `.db`、没有官方 simulator / map runtime，这个仓库已经可以把整篇论文的 experiments 按 **offline protocol** 跑全。
- 如果你后续接入官方 nuPlan simulation runner，可以把 `main_closed_loop_offline_proxy` 替换为真实 closed-loop benchmark。

## 2. 安装

```bash
cd dpuc_nuplan
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# 或者
pip install -r requirements.txt
```

## 3. 数据目录

默认读取：

```bash
/dataset/nuplan/data/cache/
  public_set_val/
  train_boston/
  train_pittsburgh/
  train_singapore/
  train_vegas_2/
```

如果你的路径不同，改 `dpuc/configs/default.yaml` 里的 `data.raw_root`。

## 4. 预处理

### 4.1 检查单个 DB schema

```bash
python scripts/inspect_nuplan_db.py \
  --db /dataset/nuplan/data/cache/public_set_val/<your_file>.db
```

### 4.2 全量预处理

```bash
python scripts/preprocess_nuplan.py \
  --config dpuc/configs/default.yaml
```

### 4.3 小规模调试

```bash
python scripts/preprocess_nuplan.py \
  --config dpuc/configs/default.yaml \
  --limit-db 2
```

输出目录：

```bash
data/processed/
  train/*.pkl
  val/*.pkl
  train_manifest.pkl
  val_manifest.pkl
```

## 5. 训练

### 5.1 全部训练

```bash
python scripts/train_interface.py \
  --config dpuc/configs/default.yaml \
  --stage all
```

### 5.2 单独训练 interface

```bash
python scripts/train_interface.py \
  --config dpuc/configs/default.yaml \
  --stage interface
```

### 5.3 单独训练 support utility head

```bash
python scripts/train_interface.py \
  --config dpuc/configs/default.yaml \
  --stage support
```

### 5.4 单独训练 DBI head

```bash
python scripts/train_interface.py \
  --config dpuc/configs/default.yaml \
  --stage dbi
```

checkpoint 输出：

```bash
outputs/default/checkpoints/
  interface_best.pt
  support_best.pt
  dbi_best.pt
```

## 6. 基础 offline 评测

```bash
python scripts/eval_offline.py \
  --config dpuc/configs/default.yaml \
  --split val
```

输出：

```bash
outputs/default/eval/offline_val_metrics.json
```

这个文件会包含：

- `GapMAE`
- `PairAcc`
- `Top1`
- `DIR`
- `GapPres@K`
- `MassRec@K`
- `BoundaryRec@K`
- `SlotNLL`
- `SlotECE`
- `MinESS`
- `RelVar`
- `FlipRate`
- `SRCC`
- `TopBRecall`
- `GapGain@B`
- `AURC`
- `Worst5DIR`
- `Coverage`
- `FallbackRate`
- `LatencyMs`
- `Score / Collision / Progress / Comfort / RouteSucc / IntScore`（offline proxy）

## 7. 论文 Experiments 一节：完整 offline 测试命令

下面所有命令都只依赖：

- 预处理后的 `data/processed`
- 训练好的三个 checkpoint

### 7.1 一次性跑完整 experiments offline 套件

```bash
python scripts/run_experiments_offline.py \
  --config dpuc/configs/default.yaml \
  --split val
```

输出：

```bash
outputs/default/eval/experiments_val.json
```

这个 JSON 会包含：

- `main_closed_loop_offline_proxy`
- `planner_facing_interface_fidelity`
- `decision_critical_support`
- `frozen_support_bridge`
- `selective_individualization`
- `budget_curves`
- `ablations`
- `reliability`

---

## 8. 按论文实验逐项运行

### Q1: Main Closed-Loop Results（offline proxy）

> 本仓库里这一项是 **offline proxy**，不是官方 closed-loop simulator。

直接运行完整 experiments 套件即可；看：

```bash
outputs/default/eval/experiments_val.json
```

其中：

```json
main_closed_loop_offline_proxy -> ours
```

会给出：

- `Score`
- `Collision`
- `Progress`
- `Comfort`
- `RouteSucc`
- `IntScore`
- `LatencyMs`

---

### Q2: Planner-Facing Interface Fidelity

运行：

```bash
python scripts/run_experiments_offline.py \
  --config dpuc/configs/default.yaml \
  --split val
```

查看：

```json
planner_facing_interface_fidelity
```

其中会自动评测这些 interface baselines：

- `public_only`
- `agnostic`
- `no_switch`
- `single_latent`
- `query_only`
- `full_future_head`
- `ours`

指标：

- `SlotNLL`
- `SlotECE`
- `GapMAE`
- `PairAcc`
- `Top1`
- `DIR`

---

### Q3: Decision-Critical Support under Fixed Budget

运行：

```bash
python scripts/run_experiments_offline.py \
  --config dpuc/configs/default.yaml \
  --split val
```

查看：

```json
decision_critical_support
```

支持这些 retained-support baselines：

- `masstopk`
- `diversetopk`
- `structtopk`
- `uncunion`
- `ours`

关键指标：

- `MassRec@K`
- `BoundaryRec@K`
- `GapPres@K`
- `DIR`
- `LatencyMs`

#### 支持预算曲线 K sweep

已自动包含在：

```json
budget_curves -> support_budget
```

默认 sweep：

- `K in {2,4,6,8,10,12,16}`

---

### Q4: Stability of Frozen-Support Bridge Evaluation

运行：

```bash
python scripts/run_experiments_offline.py \
  --config dpuc/configs/default.yaml \
  --split val
```

查看：

```json
frozen_support_bridge
```

支持 evaluator baselines：

- `perifacemc`
- `directis`
- `frozen_nobridge`
- `ours`

指标：

- `MinESS`
- `RelVar`
- `FlipRate`
- `VOI-MAE`
- `LatencyMs`

---

### Q5: Selective Individualization under Agent Budget

运行：

```bash
python scripts/run_experiments_offline.py \
  --config dpuc/configs/default.yaml \
  --split val
```

查看：

```json
selective_individualization
```

支持 baselines：

- `public_only`
- `nearest-b`
- `ttc-b`
- `entropy-b`
- `boundarysens-b`
- `resamplevoi`
- `random-b`
- `ours`
- `allind`

指标：

- `SRCC`
- `TopBRecall`
- `GapGain@B`
- `IntScore`
- `Collision`
- `LatencyMs`

#### Agent budget 曲线 B sweep

已自动包含在：

```json
budget_curves -> agent_budget
```

默认 sweep：

- `B in {0,1,2,3,4}`

---

### Q6: Budget Curves and Error Decomposition

运行完整 experiments 套件后，直接查看：

```json
budget_curves
```

其中包含：

- `support_budget`
- `agent_budget`

建议你后处理画图：

- `DIR` vs `K`
- `GapPres@K` vs `K`
- `BoundaryRec@K` vs `K`
- `IntScore` vs `B`
- `Collision` vs `B`
- `LatencyMs` vs `B`

---

## 9. Ablation

运行：

```bash
python scripts/run_ablations.py \
  --config dpuc/configs/default.yaml \
  --split val
```

输出：

```bash
outputs/default/eval/ablations_val.json
```

默认 ablation：

- `ours`
- `w_o_action_conditioning`
- `w_o_rescue_support`
- `w_o_uplift_term`
- `w_o_bridge_bank`
- `w_o_exact_dbi`
- `w_o_local_closure_refresh`
- `w_o_correction_fallback`

---

## 10. 可靠性 / fallback / deployment behavior

运行完整 experiments 套件后查看：

```json
reliability
```

包含：

- `NoDiag`
- `DiagOnly`
- `CorrOnly`
- `Ours (Corr+Fallback)`

指标：

- `AURC`
- `Worst5DIR`
- `FlaggedCollision`
- `Coverage`
- `FallbackRate`

---

## 11. 推荐完整复现实验顺序

```bash
# 1) preprocess
python scripts/preprocess_nuplan.py --config dpuc/configs/default.yaml

# 2) train all
python scripts/train_interface.py --config dpuc/configs/default.yaml --stage all

# 3) basic offline summary
python scripts/eval_offline.py --config dpuc/configs/default.yaml --split val

# 4) full experiments suite
python scripts/run_experiments_offline.py --config dpuc/configs/default.yaml --split val

# 5) ablations
python scripts/run_ablations.py --config dpuc/configs/default.yaml --split val
```

---

## 12. 输出文件说明

```bash
outputs/default/
  checkpoints/
    interface_best.pt
    support_best.pt
    dbi_best.pt
  eval/
    offline_val_metrics.json
    experiments_val.json
    ablations_val.json
```

## 13. 当前实现和论文的对应关系

已经对齐的核心点：

- action-conditioned planner-facing interface
- retained-support selection with learned uplift
- frozen-support comparison with bridge baseline variants
- selective individualization with learned DBI runtime
- interface/support/bridge/VOI/ablation/reliability 的 offline experiments

仍需你后续自己接官方 nuPlan runtime 的部分：

- 官方 closed-loop simulator
- map API / route front-end
- 官方 benchmark score 的真实闭环实现

也就是说，这个仓库现在已经足够把论文 **Experiments 一节全部按 offline protocol 跑通**；如果你要投稿最终闭环主表，再把 planner 对接官方 sim runner 即可。
