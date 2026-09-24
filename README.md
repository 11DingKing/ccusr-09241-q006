# 跨区域算网作业安置服务

省级算力调度中心的批处理作业安置后端。将**站点算力时段、路径带宽、租户额度、
数据位置与驻留限制、园区能耗档位**统一纳入安置决策，支持：

- 带依赖的作业组一次提交（DAG 校验，依赖门控）；
- 预留后确认（两阶段），预留超时自动释放；
- 执行前取消；
- 站点失效后的受控迁移（自动重安置，原已确认作业无缝接续）；
- 拆分作业的汇合屏障：子任务只有满足汇合条件才可进入完成态；
- 决策可解释：每次安置记录被哪些硬约束排除、哪些软目标参与排序；
- 命令行推进虚拟时间，确定性复现额度竞争、预留过期、故障迁移、部分子任务失败。

## 架构

```
compute_network_scheduler/
├── domain/            # 领域层：模型、硬约束/软目标、安置引擎（纯函数，无 IO）
│   ├── models.py      #   实体、状态机枚举、全量状态序列化
│   ├── constraints.py #   用量推导（一切占用由预留记录推导）与硬约束评估
│   └── engine.py      #   候选评估、软目标加权排序、决策报告
├── application/       # 应用层：用例编排与可替换端口
│   ├── ports.py       #   Clock / IdGenerator / EventSink 协议
│   └── services.py    #   提交、安置、确认、取消、tick、故障迁移、解释
├── infrastructure/    # 适配层：JSON 状态存储、手动时钟、顺序标识、事件出口
└── interface/         # 接口层：命令行与四个复现场景
```

关键设计：

- **单一事实来源**：算力时段、园区能耗、链路并发、租户并发额度与传输预算的占用
  全部从预留（Reservation）记录推导，没有独立计数器。任何重试（重复确认、重复
  安置、故障重放、迁移重试）都只是复用或替换预留记录，**不会重复扣减额度**。
- **虚拟时间**：时钟是可替换端口，所有时限、预留 TTL、传输窗口、执行窗口都以
  虚拟秒/时段计算，`tick` 命令确定性推进并触发到期事件。
- **汇合屏障**：拆分作业的子任务执行完毕后先进入 `JOIN_WAIT`，只有汇合条件
  （all 或 quorum k）满足时才与作业一起进入完成态。

## 状态机

作业：`PENDING → READY → RESERVED → SCHEDULED → RUNNING → SUCCEEDED / FAILED`，
另有 `CANCELLED`（执行前取消）与 `BLOCKED`（依赖失败或无可行迁移目标）。

子任务：`PENDING → RESERVED → SCHEDULED → RUNNING → JOIN_WAIT → SUCCEEDED`，
另有 `FAILED / CANCELLED`。

预留：`HELD → CONFIRMED → COMPLETED`，另有 `EXPIRED / CANCELLED / SUPERSEDED`。

## 硬约束与软目标

硬约束（排除候选站点，逐条记录在决策报告中）：

| 代码 | 含义 |
| --- | --- |
| `SITE_DOWN` | 站点失效 |
| `DATA_RESIDENCY` | 数据驻留限制不允许在该站点处理 |
| `NO_PATH` | 数据位置到站点无可用路径 |
| `TRANSFER_BUDGET` | 租户网络传输预算不足 |
| `TENANT_QUOTA` | 租户并发算力额度不足（按时段） |
| `COMPUTE_CAPACITY` | 站点算力时段容量不足 |
| `ENERGY_CAP` | 园区能耗上限超限 |
| `LINK_BANDWIDTH` | 链路并发传输数超限 |
| `DEADLINE` | 传输+执行无法在完成时限前结束 |

软目标（对可行候选加权排序，权重见 `Settings.soft_weights`）：
`data_locality`（数据局部性）、`energy_efficiency`（能耗档位）、
`completion_time`（完成时间）、`capacity_headroom`（容量余量）。

## 快速开始

```bash
export PYTHONPATH=$PWD
DB=/tmp/scheduler-demo/state.json

# 初始化演示拓扑（3 站点 / 2 园区 / 2 租户 / 2 数据集）
python3 -m compute_network_scheduler --db $DB init-demo

# 提交带依赖的作业组
cat > /tmp/jobs.json <<'EOF'
{"tenant": "tenant-a", "jobs": [
  {"key": "extract", "dataset_id": "ds-east", "compute_units": 4, "duration_slots": 2, "deadline_slot": 10},
  {"key": "train", "dataset_id": "ds-east", "compute_units": 6, "duration_slots": 2,
   "deadline_slot": 14, "depends_on": ["extract"], "splits": 2, "join": {"kind": "quorum", "k": 2}}
]}
EOF
python3 -m compute_network_scheduler --db $DB submit --file /tmp/jobs.json

# 安置 -> 确认 -> 推进虚拟时间
python3 -m compute_network_scheduler --db $DB place
python3 -m compute_network_scheduler --db $DB confirm
python3 -m compute_network_scheduler --db $DB tick --seconds 600

# 查看决策解释（硬约束排除原因 + 软目标得分）
python3 -m compute_network_scheduler --db $DB explain --job job-0001

# 总览 / 作业 / 事件
python3 -m compute_network_scheduler --db $DB status
python3 -m compute_network_scheduler --db $DB jobs
python3 -m compute_network_scheduler --db $DB events
```

运行数据默认保存在 `~/.compute_network_scheduler/state.json`（可用 `--db` 或
环境变量 `CNS_DB` 覆盖），不写入源码目录。

## 复现场景

四个场景在内存态运行（不污染 `--db`），输出逐步叙述：

```bash
python3 -m compute_network_scheduler scenario quota-contention        # 额度竞争
python3 -m compute_network_scheduler scenario reservation-expiry      # 预留过期
python3 -m compute_network_scheduler scenario fault-migration         # 故障迁移
python3 -m compute_network_scheduler scenario partial-subtask-failure # 部分子任务失败
```

手工复现（以故障迁移为例）：

```bash
python3 -m compute_network_scheduler --db $DB place && \
python3 -m compute_network_scheduler --db $DB confirm && \
python3 -m compute_network_scheduler --db $DB tick --seconds 300 && \
python3 -m compute_network_scheduler --db $DB fail-site --site site-east-1 && \
python3 -m compute_network_scheduler --db $DB tick --seconds 900 && \
python3 -m compute_network_scheduler --db $DB jobs
```

## 命令一览

| 命令 | 说明 |
| --- | --- |
| `init-demo [--force]` | 初始化演示拓扑 |
| `add-park / add-site / add-link / add-tenant / add-dataset` | 拓扑与租户管理 |
| `submit --file jobs.json [--tenant T]` | 一次提交带依赖的作业组 |
| `place [--job ID]` | 对 READY 作业执行安置（生成决策报告） |
| `confirm [--job ID]` | 确认预留（缺省确认全部；幂等） |
| `cancel --job ID` | 执行前取消 |
| `tick --seconds N` | 推进虚拟时间（触发过期、启停、汇合判定） |
| `fail-site --site ID` / `recover-site --site ID` | 站点失效（自动受控迁移）/ 恢复 |
| `fail-task --task ID` | 注入子任务执行失败 |
| `explain --job ID / --decision ID` | 决策解释：硬约束排除与软目标排序 |
| `status` / `jobs [--group ID]` / `events [--since N]` | 查询 |
| `scenario NAME` | 运行复现场景 |

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试守护的不变量（`tests/helpers.py: assert_invariants`，在任意操作序列后调用）：

- 任一站点任一时段算力占用 ≤ 容量；任一园区任一时段能耗 ≤ 上限；
- 任一链路任一时段并发传输 ≤ 上限；任一租户任一时段并发额度、累计传输预算不被突破；
- 每个子任务至多一条持有中的预留（重试不重复扣减的结构保证）；
- 子任务进入完成态 ⇒ 所属作业已进入完成态且汇合条件满足；
- 终态作业不持有任何资源。

## 编译检查

```bash
python3 -m compileall -q compute_network_scheduler tests
```
