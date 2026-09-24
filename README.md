# 跨区域算网作业安置服务

省级算力调度中心的批处理安置后端：把**站点算力时段、路径带宽、租户额度、
数据驻留位置、园区能耗档位**统一纳入一次安置决策，支持带依赖作业组的
「规划 → 预留 → 确认」两段式安置、预留 TTL 超时自动释放、执行前取消、
站点失效后的受控迁移，以及子任务屏障汇合。

时间是只能由命令行/接口显式推进的**虚拟时钟**（整数槽，1 槽 = 1 小时），
因此额度竞争、预留过期、故障迁移、部分失败都可以确定性复现。

## 领域边界

| 模块 | 职责 |
| --- | --- |
| `clock.py` | 虚拟时钟，只进不退，显式推进 |
| `models.py` | 站点 / 链路 / 数据集（多副本）/ 租户 / 作业组 / 子任务 / 预留 / 安置 / 传输计划 |
| `topology.py` | 资源注册与最宽路径（widest-path）选择，故障站点自动剔除 |
| `ledgers.py` | 站点 GPU×槽 与链路带宽×槽 容量账本；租户额度账本（**幂等键扣减/退还**） |
| `planner.py` | 硬约束筛查 + 传输调度 + 软目标 min-max 归一加权排序 |
| `explanation.py` | 决策解释：每个候选被哪条硬约束排除、可行候选的软目标分项 |
| `service.py` | 应用服务：生命周期、TTL、取消、故障迁移、执行模拟、屏障汇合 |
| `repository.py` | 持久化端口 + 原子写 JSON 适配器（运行数据不进源码目录） |
| `api.py` | 标准库 HTTP 接口（无第三方依赖） |
| `cli.py` | 命令行：虚拟时间推进与四个复现场景 |

## 硬约束与软目标

规划器对「子任务 × 在线候选站点」逐条施加硬约束，任一不满足即排除该候选，
并把原因（代码 + 明细）写入决策解释：

- `SITE_FAILED` 站点失效/隔离
- `DATA_RESIDENCY` 数据驻留区域白名单
- `DATA_UNREACHABLE` 全部副本均无可达网络路径
- `SITE_GPU_CAPACITY` 标称 GPU 不足
- `SITE_POWER_CAPACITY` 园区功耗上限折算等效 GPU 不足（功耗档位硬约束）
- `WINDOW_NO_FIT` / `DEADLINE` 时限窗口内放不下 GPU 与传输
- `TENANT_QUOTA` 租户剩余额度不足（候选边际口径 + 整组兜底）
- `TRANSFER_BUDGET` 作业组跨网传输总量超预算

全部通过的候选进入软目标排序：**能耗档位、跨网传输量、额度成本、时限
松弛度**，min-max 归一后按权重（默认 0.35/0.20/0.30/0.15）加权取最小。

## 关键不变量（由测试守护）

1. **重试不重复扣额度**：扣减键为 `组:子任务:尝试序号`，`reserve` 重试
   幂等返回同一预留；规划/预留失败不产生任何账本残留；TTL 超时与取消
   全额退还；迁移用 `attempt+1` 的新键，旧键标记已退。
2. **容量不超发**：GPU（功耗折算后）与逐跳链路带宽在每个槽都不超过上限。
3. **屏障汇合**：子任务执行成功只进入 `SUCCEEDED`，仅当其全部依赖
   `COMPLETED` 时才进入 `COMPLETED`；任一子任务永久失败则整组 `FAILED`，
   没有任何子任务能到达完成态。
4. **迁移受控**：失效站点作为硬约束永不被再次选中；迁移只重安置中止任务
   及其未完成下游；attempt 单调递增，`migrated_from` 可追溯。
5. **预留语义**：未确认预留遇站点失效整体释放；确认后只能执行/迁移，
   不能执行前取消；TTL 到期自动释放并退还。

## 快速开始

```bash
# 编译检查
python3 -m compileall -q compute_network_scheduler tests

# 全部测试
python3 -m unittest discover -s tests -v
```

### 命令行复现四个场景

```bash
python3 -m compute_network_scheduler.cli demo quota-race       # 额度竞争 + 重试幂等
python3 -m compute_network_scheduler.cli demo ttl-expiry       # 预留超时自动释放
python3 -m compute_network_scheduler.cli demo failover         # 站点失效受控迁移
python3 -m compute_network_scheduler.cli demo partial-failure  # 部分子任务失败，屏障阻断
```

手动推进与查询（状态默认落在 `.runtime/state.json`，可用 `--state` 覆盖，
位置不限，可放在子命令前或后）：

```bash
python3 -m compute_network_scheduler.cli reset
python3 -m compute_network_scheduler.cli groups
python3 -m compute_network_scheduler.cli reserve batch-A
python3 -m compute_network_scheduler.cli confirm batch-A
python3 -m compute_network_scheduler.cli advance 3
python3 -m compute_network_scheduler.cli explain batch-A       # 硬约束排除 + 软目标排序
python3 -m compute_network_scheduler.cli fail-site park-east-1
python3 -m compute_network_scheduler.cli migrate batch-FO
python3 -m compute_network_scheduler.cli quota bank
python3 -m compute_network_scheduler.cli capacity
```

### HTTP 接口

```bash
python3 -m compute_network_scheduler.cli serve --port 8080 --state .runtime/state.json
```

```
POST /admin/sites|links|datasets|tenants      资源注册
POST /groups                                  提交带依赖作业组（立即规划，返回安置）
POST /groups/{id}/reserve|confirm|cancel
POST /groups/{id}/migrate                     受控迁移
POST /sites/{id}/fail|recover                 站点失效/恢复
POST /tasks/{gid}/{tid}/fail                  子任务永久故障注入
POST /clock/advance  {"slots": n}             推进虚拟时间
GET  /groups/{id}/explain                     最近决策解释
GET  /decisions/{id}                          指定决策
GET  /quota/{tenant}   /capacity   /groups
```

不可行决策返回 `409`，响应体含完整 `exclusions`（每个候选站点的硬约束
原因清单）；决策同时持久化，可随时复查。

## 演示拓扑

`reset` / `demo` 播种固定拓扑以便复现：三个园区（`park-east-1` 低能耗档、
`park-east-2` 中能耗档、`park-west-1` 高能耗档），两条链路
（园区内 20Gbps、跨区 8Gbps），两个数据集（全区驻留的 `ds-ledger` 三副本、
仅东区驻留的 `ds-east-only` 两副本），租户 `bank` 额度 50。
