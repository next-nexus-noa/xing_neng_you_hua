# Adaptive Micro-batching 与 SCOM 实验规划及当前进度

## 1. 文档目的

本文档统一记录当前系统的实验目标、对比方法、正式实验矩阵、
已完成工作、尚未完成工作和推荐执行顺序。

当前整体研究围绕三个问题展开：

1. **切多少**：每个 pipeline step 应选择多少个 micro-batch，即
   动态选择 \(M\in\{1,2,4\}\)。
2. **怎么切**：在 NPU shape bucket 约束下，如何确定每个
   micro-batch 的 token 容量。
3. **每份有什么**：如何按照请求、prefill/decode 状态和上下文长度，
   将 token 分配到不同 micro-batch，使各阶段计算和通信成本更均衡。

目前对应实现为：

- **Adaptive M selector**：解决“切多少”。
- **SCOM**：以 compute-aware composition 为保底，并进一步搜索
  bucket-safe 容量和两阶段 Broadcast 流水线顺序，解决“怎么切”和
  “每份有什么”。
- **Adaptive + SCOM**：联合决定 M 与 micro-batch 内部组成。

---

## 2. 正式实验统一条件

### 2.1 模型

- Qwen2.5-3B
- Qwen2.5-7B
- Qwen2.5-14B

### 2.2 数据集与输入长度

保留以下 9 个数据档：

| 数据集族 | 输入长度档 |
|---|---|
| GovReport-GAO | 1K、2K、3K、3.5K、3.8K |
| LongBench-QMSum | 1K、3K、3.8K |
| STELLA-Corpus | 1K |

删除：

- STELLA-Corpus-3K
- STELLA-Corpus-3.8K

删除原因是其实际 token 数存在超过 `max_model_len=6144` 的风险，
容易引入截断、失败请求和不一致样本，且已有两个数据集族覆盖长输入。

### 2.3 静态负载

每个“数据档 × 模型”设置两档负载：

- medium：\(1.0C\)
- high：\(2.0C\)

其中 \(C\) 是该模型和数据档在固定 M=1 baseline 下测得的 capacity
QPS。

因此静态场景总数为：

```text
9个数据档 × 3个模型 × 2个负载 = 54个场景
```

### 2.4 服务端统一配置

所有正式对比保持：

```bash
export EXTRA_SERVER_ARGS="--no-enable-prefix-caching"
export TRACE_ENABLED=0
```

并保持一致的：

- `max_model_len=6144`
- PP=2
- TP=1
- `dtype=float16`
- `--enforce-eager`
- `--no-async-scheduling`
- 相同型号的两张 NPU
- 相同驱动、CANN、PyTorch、vLLM 和 vLLM Ascend 版本

PP trace 只用于少量功能验证和机制分析，不用于正式性能数据。

### 2.5 客户端统一配置

论文正式实验建议统一：

```bash
export EXTRA_BENCH_ARGS="--temperature 0 --ignore-eos"
export WARMUP_RUNS=1
export POWER_METRICS_ENABLED=1
export POWER_METRICS_INTERVAL_MS=500
```

作用：

- `--temperature 0`：使用确定性 greedy decoding，减少随机生成差异。
- `--ignore-eos`：每个请求固定生成计划中的 128 个输出 token，
  避免不同方法因提前 EOS 造成输出长度不一致。
- `WARMUP_RUNS=1`：保持所有方法相同的预热条件。

一旦采用这组正式参数，不能把此前
`temperature=None, ignore_eos=False` 的结果直接混入同一张正式结果表。

### 2.6 Capacity QPS 的处理

更改为 `--temperature 0 --ignore-eos` 后，单请求生成成本可能发生变化，
因此论文正式实验应重新执行一次 capacity probe。

新的 probe 数量为：

```text
9个数据档 × 3个模型 = 27组 capacity probe
```

要求：

1. 使用固定 M=1 baseline 测量。
2. 使用与正式实验相同的 `temperature=0 --ignore-eos`。
3. 将 27 个结果发布到共享 `capacity-exchange` 目录。
4. Baseline、Adaptive、SCOM、Adaptive+SCOM 和 Q 全部读取同一份 QPS。
5. 不允许各方法分别测量自己的 \(C\)，否则负载强度不可比。

旧的 capacity QPS 可以继续用于功能验证，但不应作为新工作负载定义下的
最终论文基准。

---

## 3. 静态实验方法

### 3.1 Baseline：固定 M + uniform

配置：

```bash
export MICROBATCH_GROUPING=uniform
```

每个静态场景分别运行：

- M=1
- M=2
- M=4

实验数量：

```text
54 × 3 = 162组
```

用途：

- 给出各固定 M 的真实表现。
- 构造 per-scenario best-fixed-M oracle。
- 作为 Adaptive、SCOM 和 Q 的共同基础对照。

best-fixed-M 必须按场景选择，不能只选一个全局 M。
论文中需要明确选择目标，例如综合得分、SLO 下吞吐或 Pareto 规则。

### 3.2 Adaptive：动态 M + uniform

配置：

```bash
micro_batch=adaptive
export MICROBATCH_GROUPING=uniform
```

实验数量：

```text
54组
```

用途：

- 单独验证 M selector。
- 对比 best-fixed-M 和 Q。
- 分析不同模型、输入长度、负载下的 M 选择分布。

### 3.3 SCOM：固定 M + SCOM

配置：

```bash
export MICROBATCH_GROUPING=scom
```

每个场景运行：

- 固定 M=2
- 固定 M=4

M=1 不存在内部切分，等价于 baseline M=1，无需重复运行。

实验数量：

```text
54 × 2 = 108组
```

用途：

- 隔离验证 SCOM，不受动态 M selector 干扰。
- 分别与相同 M 的 uniform baseline 对比。
- 分析 compute-aware guardrail、shape-safe 容量搜索和缓存命中。

### 3.4 Adaptive + SCOM

配置：

```bash
micro_batch=adaptive
export MICROBATCH_GROUPING=scom
```

实验数量：

```text
54组
```

执行逻辑：

1. Adaptive 为当前 step 选择 M=1/2/4。
2. M=1 时不进行内部切分。
3. M=2/4 时由 SCOM 生成 composition/capacity/order 计划。
4. 无有效收益或不支持的 step 安全回退到 uniform。

这是最终完整方案，也是静态实验的主要方法。

### 3.5 Q：Queue-aware M selector

每个场景运行一次 Queue-aware 模式。

实验数量：

```text
54组
```

用途：

- 作为低复杂度在线启发式 selector 对照。
- 判断 Adaptive 的模型、校准和风险控制是否优于只看队列长度。

### 3.6 静态主实验总量

| 方法 | 组数 |
|---|---:|
| Baseline M=1/2/4 | 162 |
| Adaptive | 54 |
| SCOM M=2/4 | 108 |
| Adaptive + SCOM | 54 |
| Q | 54 |
| 合计 | **432** |

加上 27 组共享 capacity probe：

```text
一次完整静态流程 = 27 + 432 = 459组
```

---

## 4. 四个代表性功能验证场景

在完整矩阵前，先使用以下四个场景进行 trace-enabled 验证：

| 场景 | 目的 |
|---|---|
| GovReport-GAO-2K / 7B / high | 中模型、中长输入、高负载 |
| GovReport-GAO-3K / 3B / high | 小模型、长输入、高负载 |
| GovReport-GAO-3K / 14B / medium | 大模型、长输入、中负载 |
| LongBench-QMSum-3K / 7B / high | 跨数据集泛化 |

诊断阶段：

```bash
export TRACE_ENABLED=1
```

需要确认：

- Adaptive 实际选择的 M=1/2/4 分布。
- SCOM `planned/applied` 数量和应用率。
- `selection_source`：
  - `uniform`
  - `compute_aware_guardrail`
  - `shape_safe_pipeline_search`
- `cache_hit` 数量。
- `decision_overhead_us`。
- `selected_capacities` 与 shape bucket。
- 回退原因。
- 没有错误请求、shape 错误、KV 因果顺序错误或结果错位。

四场景确认功能后，需要关闭 trace 重新运行，才能作为正式性能结果。

---

## 5. 消融实验

### 5.1 最小必要消融

主表中的以下四项本身构成整体框架的核心消融：

1. Baseline：固定 M + uniform。
2. Adaptive：动态 M + uniform。
3. SCOM：固定 M + SCOM。
4. Adaptive + SCOM：完整方案。

它们分别回答：

- Adaptive 单独贡献多少？
- SCOM 单独贡献多少？
- 两者组合后是否互补？
- 组合后是否出现相互干扰？

### 5.2 SCOM 内部消融

只需在代表性场景上执行，不需要覆盖完整 54 场景。

建议对比：

| 模式 | 配置 | 目的 |
|---|---|---|
| uniform | `MICROBATCH_GROUPING=uniform` | 原始等分 |
| 旧 compute-aware | `MICROBATCH_GROUPING=compute_aware` | 只做 composition |
| SCOM guardrail | `scom` + `SCOM_OPTIMIZE_CAPACITIES=0` | 等容量 composition |
| 完整 SCOM | `scom` + `SCOM_OPTIMIZE_CAPACITIES=1` | composition + shape-safe capacity |

重点报告：

- 实际应用率。
- 决策开销。
- 缓存命中率。
- TTFT、TPOT、吞吐、能耗。
- 非均匀容量真正被选择的比例。

如果完整 SCOM 长期只选择 guardrail，也应如实作为实验结论：
当前 NPU shape cliff 下，composition 优化比非均匀 token 容量更重要。

### 5.3 Adaptive 内部消融

考虑到 Adaptive 只是整体框架的一部分，不需要做过大的全矩阵消融。
在四个代表性场景上对比：

- `analytical_only`
- `calibrated`
- `calibrated_risk_aware`

可选附加项：

- 关闭 exploration。
- 关闭或缩短 cooldown/hysteresis。
- 固定安全 M。

如果篇幅有限，优先保留三种 selector mode 的对比。

---

## 6. 动态负载实验

静态实验只能证明不同固定负载点下的表现。Adaptive 的核心价值还需要通过
同一次运行中的负载变化来体现。

### 6.1 动态负载类型

建议设置三种 trace：

1. **阶跃负载**

   ```text
   0.5C → 1.0C → 2.0C → 1.0C
   ```

2. **周期负载**

   ```text
   1.0C ↔ 2.0C 周期切换
   ```

3. **突发负载**

   ```text
   1.0C 稳态 + 短时间 2.0C burst
   ```

动态实验中的 `0.5C` 只是受控阶段，不等同于此前耗时很长的完整 low 静态档。

### 6.2 动态对比方法

至少包括：

- best-fixed-M。
- Q。
- Adaptive。
- SCOM best fixed M。
- Adaptive + SCOM。

### 6.3 动态代表场景

优先使用前述四个代表性场景，不必一开始覆盖完整 54 场景。

若采用：

- 4 个代表场景；
- 3 种动态 trace；
- 5 种方法；
- 3 次重复；

则动态实验为：

```text
4 × 3 × 5 × 3 = 180组
```

可先每项运行一次进行筛选，即 60 组；机制和效果稳定后再补足重复。

### 6.4 动态实验额外指标

除静态指标外，还需要：

- 每个时间窗口的吞吐、TTFT、TPOT。
- M 随时间变化曲线。
- 队列长度随时间变化曲线。
- 负载变化到 M 切换的响应时间。
- 错误切换次数。
- M 抖动次数。
- 每次切换后的 cooldown。
- 突发阶段的排队峰值。
- 恢复到稳态所需时间。
- 相对固定 M 的累计能耗和尾延迟。

---

## 7. 重复实验与统计

### 7.1 推荐方案

考虑完整矩阵成本，采用两级重复策略：

1. 完整 54 场景矩阵先运行一次。
2. 四个或八个代表性场景至少运行 3 次。
3. 动态负载核心结果至少运行 3 次。
4. 对异常或接近噪声水平的结果追加重复。

资源允许时，最严格方案是所有正式实验运行 3 次：

```text
432 × 3 = 1296组静态正式实验
```

### 7.2 报告方式

代表性重复实验报告：

- 均值。
- 标准差。
- 95% 置信区间。
- 相对提升的置信区间。

对于完整矩阵，除平均提升外还应报告：

- 中位数提升。
- P25/P75。
- 最差场景。
- 获胜/持平/退化的场景数量。
- 超过 3% 和 5% 提升的场景比例。

---

## 8. 指标与比较规则

### 8.1 性能指标

- Request throughput。
- Output-token throughput。
- Total-token throughput。
- Mean/P50/P99 TTFT。
- Mean/P50/P99 TPOT。
- Mean/P99 ITL。
- Benchmark duration。

### 8.2 功耗指标

- Average power。
- Peak power。
- Total energy。
- Energy per request。
- Energy per output token。
- Tokens per joule。

### 8.3 Pipeline 指标

诊断场景报告：

- PP makespan。
- Bubble ratio。
- Compute overlap。
- Rank0/Rank1 compute utilization。
- Broadcast/recv/send 时间。
- 分组决策时间。

### 8.4 主要比较

必须包含：

```text
Adaptive            vs best-fixed-M
SCOM M=2            vs uniform M=2
SCOM M=4            vs uniform M=4
Adaptive + SCOM      vs Adaptive
Adaptive + SCOM      vs SCOM best fixed M
Adaptive + SCOM      vs best-fixed-M
Adaptive + SCOM      vs Q
```

### 8.5 Pareto 与综合结果

需要分别构造：

- 吞吐—TTFT Pareto。
- 吞吐—TPOT Pareto。
- 吞吐—能耗 Pareto。
- 时延—能耗 Pareto。

如果使用综合分数，必须在实验前固定归一化和权重规则，不能根据结果临时调整。
同时保留原始指标，避免只报告综合分数。

---

## 9. 数据质量检查

每组实验必须检查：

- `completed=200`。
- `failed=0`。
- 输出 token 数符合 `--ignore-eos` 预期。
- 实际 request rate 与计划一致。
- server/client 的 run_id 一致。
- 数据集路径和模型正确。
- 没有 OOM、shape、通信或 KV cache 异常。
- 正式实验 `TRACE_ENABLED=0`。
- prefix caching 已关闭。
- 功耗采样覆盖正式 benchmark，而不是只覆盖启动阶段。

以下结果不能直接进入最终统计：

- 只有 warmup、没有正式 benchmark。
- 请求失败。
- 输出长度条件不同。
- trace 开关不同。
- QPS 来源不同。
- 超过 `max_model_len` 后被截断或跳过。
- 服务端参数与计划不一致。

---

## 10. 当前进度

| 项目 | 当前状态 | 是否可直接作为最终论文数据 |
|---|---|---|
| 原始 Baseline 完整矩阵 | 已完成 | 否，生成参数尚未统一为 temperature=0 + ignore-eos |
| 原始 Adaptive 完整矩阵 | 已完成 | 否，同上 |
| Q 完整矩阵 | 已完成 | 否，同上 |
| 固定 M compute-aware 代表实验 | 已完成 | 机制与趋势证据，可作为开发依据 |
| Adaptive + compute-aware 四场景 | 已完成 | 机制与趋势证据 |
| SCOM v1 第一场景 | 已完成 | 否；0% applied，暴露规划器问题 |
| SCOM v1 问题定位 | 已完成 | 已确认“无应用但有规划开销” |
| SCOM v2 代码修改 | 已完成 | 尚需容器测试和 NPU 验证 |
| SCOM v2 单元测试代码 | 已补充 | 尚未在目标容器执行 |
| SCOM v2 四场景验证 | 未完成 | 否 |
| 新统一参数的 27 个 capacity probe | 未开始 | 否 |
| 新统一参数的 432 个静态实验 | 未开始 | 否 |
| 静态消融 | 部分完成 | 需按统一参数补关键场景 |
| 动态负载客户端/计划 | 未完成 | 否 |
| 动态负载实验 | 未开始 | 否 |
| 最终 Pareto 与统计分析 | 未完成 | 否 |

现有完整 Baseline、Adaptive 和 Q 数据不是无用数据。它们已经用于：

- 判断方法趋势。
- 选择代表性场景。
- 定位 Adaptive 和分组器问题。
- 估计运行时间。
- 证明原始框架能够跑通。

但采用新的生成参数后，应把它们视为开发/预实验数据，而不是最终主表数据。

---

## 11. 接下来执行计划

### 阶段 A：验证 SCOM v2

1. 在容器中运行：

   ```bash
   cd /vllm-workspace/vllm
   python -m pytest -q \
     tests/v1/worker/test_scom_ubatch.py \
     tests/v1/worker/test_compute_aware_ubatch.py
   ```

2. 运行 Ascend 侧测试：

   ```bash
   cd /vllm-workspace/vllm-ascend
   python -m pytest -q \
     tests/ut/test_envs.py \
     tests/ut/worker/test_model_runner_v1.py
   ```

3. 使用四个代表场景、`TRACE_ENABLED=1` 运行 Adaptive + SCOM。
4. 检查 applied rate、guardrail、cache、开销和正确性。
5. 如果第二个场景仍接近 0% applied 或明显退化，暂停完整矩阵并继续修正。
6. 四场景通过后，关闭 trace 重跑正式性能版本。

### 阶段 B：冻结正式实验协议

冻结：

- 数据集和模型列表。
- 54 场景 CSV。
- `temperature=0 --ignore-eos`。
- prefix caching 关闭。
- warmup 次数。
- 功耗采样参数。
- Adaptive 参数。
- SCOM 参数。
- 软件与硬件版本。

协议冻结后不再边跑完整矩阵边改参数或代码。

### 阶段 C：重新生成正式 capacity QPS

1. 生成 27 组 capacity probe。
2. 使用固定 M=1 baseline。
3. 使用新的客户端生成参数。
4. 发布到新的 `CAPACITY_SESSION`。
5. 检查 27 个 JSON/CSV 条目齐全。

建议使用新的 session 名，避免与旧 QPS 混淆。

### 阶段 D：静态完整矩阵

推荐顺序：

1. Baseline M=1/2/4。
2. Adaptive。
3. Q。
4. SCOM M=2/4。
5. Adaptive + SCOM。

也可以并行运行在不同 NPU 上，但必须共享同一份 capacity QPS，
并保证硬件和软件环境一致。

### 阶段 E：静态分析和补测

1. 自动检查失败和缺失实验。
2. 生成 per-scenario best-fixed-M。
3. 计算所有相对提升。
4. 生成 Pareto。
5. 挑选异常和边界场景。
6. 对代表场景完成至少 3 次重复。
7. 完成最小必要消融。

### 阶段 F：动态负载

1. 实现或确认分阶段 QPS/trace replay 客户端。
2. 先跑 4 场景 × 3 traces × 5 方法，共 60 组筛选实验。
3. 检查 M 响应曲线、抖动和恢复时间。
4. 机制成立后补足 3 次重复，共 180 组。

### 阶段 G：最终论文材料

至少准备：

- 总体架构图。
- Adaptive M selector 流程。
- SCOM composition/capacity/order 流程。
- 静态性能主表。
- 功耗主表。
- Pareto 图。
- 不同模型、输入长度和负载的分组柱状图。
- 动态负载时间序列图。
- 消融表。
- 决策开销和缓存命中表。
- 失败案例与局限性分析。

---

## 12. 完成判据

### 12.1 静态部分完成

满足以下条件后，静态部分可视为完成：

- SCOM v2 四场景功能验证通过。
- 正式协议冻结。
- 27 个新 capacity probe 齐全。
- 432 个静态实验齐全。
- 所有正式结果生成条件一致。
- 失败与异常组完成补跑。
- 代表性场景完成重复实验。
- 最小必要消融完成。
- 静态性能、功耗和 Pareto 分析完成。

### 12.2 整个优化部分完成

在静态部分之外，还需要：

- 动态负载实验完成。
- Adaptive 响应和稳定性得到验证。
- 完整方案相对 best-fixed-M、Q、Adaptive-only 和 SCOM-only 的结论明确。
- 代码、参数、实验计划和分析脚本可复现。
- 论文图表与局限性分析完成。

只有静态矩阵跑完，能够证明方案在多个静态工作点上的效果；
动态负载实验完成后，才能充分支撑“在线自适应”的核心主张。
