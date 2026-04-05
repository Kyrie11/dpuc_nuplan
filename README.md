
# DPUC on nuPlan DB (PyTorch)

这是一个**基于你论文 method + appendix 落地的 PyTorch 工程**，目标是把论文中的四个核心模块接起来：

1. action-conditioned planner-facing interaction contract
2. decision-critical retained-support selection
3. frozen-support bridge evaluation
4. selective individualization / DBI

它默认读取你给定的 nuPlan SQLite `.db` 文件目录：

```bash
/dataset/nuplan/data/cache/
  public_set_val/
  train_boston/
  train_pittsburgh/
  train_singapore/
  train_vegas_2/
```

---

## 1. 当前实现范围

本仓库已经提供：

- **完整的 PyTorch 训练/验证/评测代码结构**
- **直接从 nuPlan `.db` 读取数据** 的预处理器
- prefix 采样、ego/agent 历史提取、slot/witness 构建
- interface model、support utility model、DBI model
- greedy retained-support selection
- frozen-support value evaluation
- offline 评测脚本与论文要求的关键 decision-centric 指标输出
- 实验命令、ablation 入口、zip 可打包工程

需要你了解的一点：

- 你这次给我的输入只有论文 tex 和 `.db` 路径说明，**没有 nuPlan map 资源 / 官方 devkit 仿真配置 / scenario builder 配置**。
- 因此这里我把工程做成了：
  - **offline pipeline 完整可落地**；
  - **closed-loop official nuPlan simulation** 预留成可扩展接口；
  - 当前默认是**基于 `.db` 的 planner-facing offline benchmark**。

如果你后面补充了 nuPlan maps 和 devkit runtime，我建议把本仓库接到官方 sim runner 上做正式 closed-loop。

---

## 2. 对应论文中的默认配置

和论文 appendix 对齐的关键默认值已经固化在 `dpuc/configs/default.yaml`：

- history window: `2.0s`
- planning horizon: `6.0s`
- replanning/sample interval: `0.5s`
- nearby-agent cap: `20`
- action budget: `<= 9`
- candidate bank cap: `64`
- retained support budget `K=8`
- fallback support budget `K_fb=12`
- bridge bank size `N=16`
- oracle bridge bank size `64`
- individualization budget `B=3`
- hidden size `256`
- transformer blocks `4`
- heads `8`
- training epochs `30`
- selector extra epochs `10`

---

## 3. 安装

```bash
cd dpuc_nuplan
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# 或者
pip install -r requirements.txt
```

---

## 4. 数据预处理

### 4.1 先检查单个 DB 的 schema

```bash
python scripts/inspect_nuplan_db.py \
  --db /dataset/nuplan/data/cache/public_set_val/<your_file>.db
```

这个脚本会打印：

- 表名
- `log` 元信息
- 每个表的字段名

如果你的本地 nuPlan DB schema 和官方 devkit schema一致，核心表通常围绕 `log / ego_pose / lidar_pc / lidar_box / track / scene / scenario_tag / traffic_light_status` 组织。官方文档说明 nuPlan 的数据库是 SQLite，`lidar_pc`、`ego_pose`、`lidar_box`、`scene`、`scenario_tag` 等表共同组织场景、ego pose 和目标框信息。citeturn304901view0

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

每个 prefix 样本包含：

- ego history
- future ego replay target
- current agents
- 9-action library
- active slots
- witnesses
- candidate structures
- oracle/public value proxies

---

## 5. 模型训练

### 5.1 训练全部模块

```bash
python scripts/train_interface.py \
  --config dpuc/configs/default.yaml \
  --stage all
```

### 5.2 只训练 interface

```bash
python scripts/train_interface.py \
  --config dpuc/configs/default.yaml \
  --stage interface
```

### 5.3 只训练 retained-support utility head

```bash
python scripts/train_interface.py \
  --config dpuc/configs/default.yaml \
  --stage support
```

### 5.4 只训练 DBI / selective individualization head

```bash
python scripts/train_interface.py \
  --config dpuc/configs/default.yaml \
  --stage dbi
```

checkpoint 默认输出到：

```bash
outputs/default/checkpoints/
  interface_best.pt
  support_best.pt
  dbi_best.pt
```

---

## 6. 测试与离线评测

```bash
python scripts/eval_offline.py \
  --config dpuc/configs/default.yaml \
  --split val
```

输出：

```bash
outputs/default/eval/offline_val_metrics.json
```

---

## 7. 论文 experiments 节所需指标

下面这些指标已经在工程里对应实现或已预留统计入口，建议你在论文最终实验中全部汇报：

### 7.1 Q1 End-to-end planner quality

主表指标：

- `Score`
- `Collision rate`
- `Route progress`
- `Comfort`
- `Route success`
- `Int-Score`
- `Latency (mean / p95)`

### 7.2 Q2 Planner-facing interface fidelity

- `Slot NLL`
- `Slot ECE`
- `GapMAE`
- `PairAcc`
- `Top1`（oracle best action agreement）
- `DIR`

### 7.3 Q3 Decision-critical support under fixed budget

- `MassRec@K`
- `BoundaryRec@K`
- `GapPres@K`
- `DIR@K`
- `Latency`

建议 sweep：

- `K ∈ {4, 8, 12, 16}`
- appendix 再补 `K ∈ {2, 4, 6, 8, 10, 12}`

### 7.4 Q4 Frozen-support bridge evaluation stability

- `Min ESS`
- `Rel.Var.`
- `FlipRate`
- `VOI-MAE`
- `Latency`

### 7.5 Q5 Selective individualization

- `SRCC`（predicted VOI vs oracle VOI）
- `Top-B Recall`
- `GapGain@B`
- `Int-Score`
- `Collision rate`
- `Latency`

建议 sweep：

- `B ∈ {1, 2, 3}`
- appendix 额外补 `B ∈ {0, 1, 2, 3, 4}`

### 7.6 Q6 Budget curves / decomposition / reliability

- `AURC`
- `Worst-5% DIR`
- `Flagged Collision Rate`
- `Coverage`
- `Fallback Rate`
- `retained mass`
- `min ESS`
- `largest normalized weight`
- `top-two gap uncertainty interval`

### 7.7 Scenario-type breakdown（appendix 建议必须补）

至少按以下类别分解：

- merges
- unprotected turns
- stop-release
- route-compatible lane changes
- crosswalk negotiation

---

## 8. 已实现的核心脚本对应关系

### 8.1 预处理

- `scripts/preprocess_nuplan.py`
- `scripts/inspect_nuplan_db.py`

### 8.2 训练

- `dpuc/train.py`
- `dpuc/models/interface.py`
- `dpuc/models/support.py`
- `dpuc/models/dbi.py`

### 8.3 规划与评估

- `dpuc/planning/support.py`
- `dpuc/planning/bridge.py`
- `dpuc/planning/planner.py`
- `dpuc/eval/metrics.py`
- `dpuc/eval/offline_eval.py`

### 8.4 Ablation

- `scripts/run_ablations.py`

---

## 9. 推荐实验执行顺序

### Step 1: 预处理

```bash
python scripts/preprocess_nuplan.py --config dpuc/configs/default.yaml
```

### Step 2: 训练 interface + support + dbi

```bash
python scripts/train_interface.py --config dpuc/configs/default.yaml --stage all
```

### Step 3: 跑 validation offline metrics

```bash
python scripts/eval_offline.py --config dpuc/configs/default.yaml --split val
```

### Step 4: support budget 曲线

手动改 `retained_k` 或复制 config 多次运行：

```bash
# 示例：K=4
python scripts/eval_offline.py --config configs_k4.yaml --split val
# 示例：K=8
python scripts/eval_offline.py --config configs_k8.yaml --split val
# 示例：K=12
python scripts/eval_offline.py --config configs_k12.yaml --split val
```

### Step 5: agent budget 曲线

修改 `agent_budget`：

```bash
python scripts/eval_offline.py --config configs_b1.yaml --split val
python scripts/eval_offline.py --config configs_b2.yaml --split val
python scripts/eval_offline.py --config configs_b3.yaml --split val
```

### Step 6: ablation

```bash
python scripts/run_ablations.py --config dpuc/configs/default.yaml
```

---

## 10. 重要说明

### 10.1 现在这版已经“完整落地”到什么程度？

已经落地的是：

- 论文方法对应的软件工程骨架
- nuPlan `.db` 数据读取
- prefix 样本组织
- slot / witness / candidate structure / retained support / bridge evaluation / DBI 训练与评测主流程
- README 中完整实验指令

### 10.2 还有哪些部分你后续最好继续补强？

为了真正做到和论文最终 camera-ready 结果完全一致，后续最值得继续补的是：

- 接入 **nuPlan map API + route scaffold**
- 用官方 scenario builder 替代当前轻量 prefix 采样器
- 用真实 planner surrogate cost 替代当前简化版 `evaluate_structure_cost`
- 用 large-bank leave-one-out 真正生成 witness utility teacher labels
- 用官方 closed-loop simulator 跑 benchmark score
- 将 `PublicOnly / AgnosticIface / JointLatent / PredTraj / AllInd` 全部补成独立 runner

当前工程是一个**可训练、可评测、可继续扩展到正式 benchmark** 的可靠起点，而不是只停留在 paperware。

---

## 11. 一条命令打包

```bash
cd ..
zip -r dpuc_nuplan.zip dpuc_nuplan
```

---

## 12. 常见问题

### Q: 为什么我本地跑预处理时报找不到路径？

因为当前环境里我无法直接访问你机器上的 `/dataset/nuplan/data/cache`；仓库里已经按这个路径写好了默认配置，但真正运行时要在你的机器上执行。

### Q: 为什么还没有官方 nuPlan closed-loop score？

因为只靠 `.db` 文件还不够，官方闭环仿真通常还需要 map、devkit 和 benchmark runtime 配置。这个仓库已经把 planner-facing method 的核心部分落地好了，后续你只需要把 runner 接进去。

---

## 13. 建议你下一步怎么做

先在你本地跑 `inspect_nuplan_db.py` 和 `preprocess_nuplan.py --limit-db 2`，确认 schema 和采样数量正常，再启动全量预处理和训练。
