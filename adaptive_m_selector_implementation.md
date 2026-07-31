# `vllm-ascend_adaptive` 中 Adaptive M 的实现

## 1. 版本范围

本文只解释以下两个配套目录中的实现：

- Ascend 执行层：`vllm-ascend_adaptive`
- vLLM 控制器层：`vllm_adaptive`

不把后来 `vllm-ascend_adapt_const` 中新增的 compute-aware、SCOM、异步反馈归约和更强的安全保护混入本文。

该版本的核心目标是：对每一个 scheduler step 在线选择 PP micro-batch 数量：

```text
M ∈ {1, 2, 4}
```

这里选择的是“当前 step 切成几份”，不是为整个实验只选择一次固定 M。

## 2. 总体数据流

```text
Scheduler 输出每个请求本 step 的 scheduled tokens
                         │
                         ▼
              提取 workload features
                         │
                         ▼
       生成 M=1/2/4 的合法候选集合
                         │
                         ▼
    解析成本先验 + 分 bucket 在线 EWMA 校准
                         │
                         ▼
       不确定性惩罚、冷却、探索、切换保护
                         │
                         ▼
                 PP0 选出当前 M
                         │
                         ▼
           broadcast_object 广播给所有 PP rank
                         │
                         ▼
          按相同 M 做连续、均匀 token 切分
                         │
                         ▼
              两个 PP stage 流水执行
                         │
                         ▼
       收集实际 critical-path 时间并 MAX 归约
                         │
                         ▼
           更新对应 workload bucket / M 状态
```

## 3. adaptive 在哪里开启

服务端脚本：

`vllm-ascend_adaptive/scripts/run_adaptive_gradient_server.sh`

在 adaptive 正式行中向 vLLM server 传入：

```bash
--enable-adaptive-ubatch
--adaptive-ubatch-max-size ...
--adaptive-ubatch-mode ...
...
```

`ParallelConfig.enable_adaptive_ubatch=True` 后，Ascend runner 的 `_pp_microbatch_enabled()` 返回 true，并进入 adaptive 路径。

capacity probe 行不会开 selector，而是固定：

```bash
VLLM_ASCEND_PP_MICROBATCH=1
VLLM_ASCEND_PP_MICROBATCH_NUM=1
```

因此探针测得的是固定 M=1 capacity，不是 adaptive capacity。

## 4. 每个 step 使用了哪些输入特征

controller 接收的是：

```python
num_scheduled_tokens
```

它表示当前 scheduler step 中，每个请求本次要计算多少 token。

由此构造：

| 特征 | 含义 |
|---|---|
| `total_tokens` | 当前 step 所有请求的 scheduled token 总数 |
| `num_reqs` | 当前 step 的请求数 |
| `max_query_len` | 单请求本 step 最大 scheduled token 数 |
| `prefill_tokens` | 对每个请求按 `max(tokens-1, 0)` 估算的 prefill token |
| `decode_tokens` | 总 token 减去 prefill token |
| `prefill_ratio` | prefill token / total token |
| `avg_tokens_per_req` | 平均每请求 scheduled token |
| `model_billions` | 从模型配置或名称提取的参数规模 |
| `hidden_size` | 模型 hidden size |

注意：该版本没有把外部 QPS、waiting queue 长度、请求等待时间或 TTFT 直接输入 selector。负载变化只能通过“当前 step 中排进来了多少请求、多少 token、prefill/decode 组成”间接反映。

## 5. workload bucket

在线校准不是所有 step 共用一份状态，而是先离散成三维 bucket：

### 5.1 模型规模

```text
small  : model_b < 3
medium : 3 <= model_b <= 7
large  : model_b > 7
```

按当前 3B、7B、14B 实验，3B 和 7B 都落入 `medium`，14B 落入 `large`。

### 5.2 阶段组成

```text
prefill : prefill_ratio >= 0.70
mixed   : 0.10 <= prefill_ratio < 0.70
decode  : prefill_ratio < 0.10
```

### 5.3 token 规模

```text
small  : total_tokens < 256
medium : 256 <= total_tokens < 1024
large  : total_tokens >= 1024
```

最终 bucket key 为：

```text
(model_bucket, phase_bucket, token_bucket)
```

每个 `(bucket, M)` 分别维护校准状态。

## 6. 合法 M 候选是怎么筛选的

M=1 永远是候选。

M=2 需要：

```text
max_size >= 2
total_tokens >= adaptive_ubatch_min_tokens_m2
prefill_ratio >= adaptive_ubatch_prefill_threshold
```

M=4 还需要：

```text
max_size >= 4
total_tokens >= adaptive_ubatch_min_tokens_m4
prefill_ratio >= adaptive_ubatch_min_prefill_ratio_m4
未被 large-model M4 开关禁用
```

脚本默认值为：

```text
M2 最少 token       = 128
M4 最少 token       = 512
M4 最低 prefill 比例 = 0.85
总 prefill 门槛      = 85%
```

因此 decode-heavy、token 很少的 step 会直接只保留 M=1。

## 7. 冷启动解析成本模型

在没有足够运行时样本时，controller 使用针对 310P、PP=2 拟合的解析模型。

对候选 M，先估计：

```text
compute =
    α × T × (1 + Mβ / (T + Mγ))

comm =
    Mδ + ε × T × hidden_size

sample =
    σ × decode_tokens

prior_cost(M) =
    (compute + comm + sample) / effective_overlap(M)
```

其中：

- `T` 是当前 step 的 total tokens；
- `α、β、γ、σ` 按模型规模取不同拟合参数；
- `δ` 表示每份 micro-batch 的固定通信/调度成本；
- `ε × T × hidden_size` 近似 token 相关通信；
- `effective_overlap(M)` 表示 PP 流水可隐藏的成本。

解析模型还做了经验保护：

- prefill 比例低时限制 M=2/4 的可实现 overlap；
- 14B 的 M=4 overlap 被保守截断，避免 trace 中看似有 overlap、端到端却因拆分和 KV 压力退化。

## 8. 三种决策模式

### 8.1 `analytical_only`

只使用上面的解析成本模型。

```text
选择 prior_cost 最低的 M
若相对 M=1 的预测收益低于 min_gain，则回到 M=1
```

它不会根据当前运行的真实反馈修正模型，适合做消融。

### 8.2 `calibrated`

在解析先验上加入运行时校准。

对每个 `(bucket, M)` 维护实际时间与 prior 的对数比值 EWMA：

```text
scale(bucket, M) ≈ EWMA(actual / prior)
calibrated_cost = prior_cost × scale
```

这样同一类 workload 下，解析模型持续高估或低估某个 M 时，后续决策会被纠正。

### 8.3 `calibrated_risk_aware`

在 calibrated cost 上继续加入不确定性：

```text
robust_cost =
    calibrated_cost + risk_kappa × uncertainty
```

不确定性来自：

- 运行时误差对数比的波动；
- 样本不足时的 cold-start penalty。

最终比较的是 `robust_cost`，不是裸的解析预测。

这是脚本默认模式。

## 9. 稳定性和安全机制

### 9.1 最低收益门槛

新 M 相对当前 M 的 robust cost 收益必须超过：

```text
adaptive_ubatch_switch_threshold_pct
```

否则保持当前 M。

### 9.2 最短保持时间

切换后至少保持：

```text
adaptive_ubatch_min_hold_steps
```

避免在相邻 step 中来回跳变。

### 9.3 连续确认

新候选需要连续获胜：

```text
adaptive_ubatch_switch_confirmations
```

次后才真正切换。

### 9.4 不确定候选拒绝

候选的：

```text
uncertainty / calibrated_cost
```

超过阈值后，可被标记为 `uncertainty_too_high`。

当前 M 和安全 M 不会仅因此被立即拒绝，以免没有可执行候选。

### 9.5 cooldown

如果实际执行明显比预测差，非安全 M 会进入 cooldown；运行异常则进入更长的 failure cooldown。

### 9.6 安全回退

默认：

```text
safe_m = 1
```

不支持的执行形态、非法观测或运行异常都会回到安全 M。

### 9.7 受控探索

为了获得 M=2/4 的真实样本，risk-aware 模式可以周期性试探相邻 M，但要求：

- 当前 bucket 已稳定若干 step；
- 距上次探索达到最小间隔；
- 当前没有 bad streak；
- 候选的预测 regret 不超过上限。

探索不是随机按百分比抖动，而是间隔和风险门槛控制的确定性探索。

## 10. PP rank 如何保证使用相同 M

只有 PP0 执行 controller 选择：

```python
local_decision = controller.select(...)
```

然后将完整 decision payload 通过：

```python
pp.broadcast_object(payload, src=0)
```

广播到所有 PP rank。

所以两个 stage 对同一个 step 使用相同 M，不会出现 PP0 按 M=2 发送、PP1 按 M=4 接收的协议错误。

这次 `broadcast_object` 是控制面同步点，也是该版本每 step 的额外开销之一。

## 11. M 选定后如何执行

在 `vllm-ascend_adaptive` 中，adaptive 只决定“切几份”，没有 compute-aware 分组。

选定 M 后：

1. `maybe_create_ubatch_slices(...)` 按原始扁平 token 顺序产生连续切片；
2. 每份 token 数尽量均匀；
3. 每个 micro-batch 分别构造 attention metadata；
4. PP0 执行 micro-batch forward；
5. 中间张量异步 broadcast 给 PP1；
6. PP1 按相同顺序执行；
7. 最后一个 PP rank 合并各 micro-batch 输出。

所以该版本解决的是：

```text
切成多少份
```

没有解决：

```text
每份里放哪些请求/token
```

## 12. 在线反馈是怎么形成闭环的

controller 不能只依赖解析模型，因此会采样实际执行时间。

原版流程为：

1. 判断当前 step 是否需要反馈测量；
2. 测量前同步 NPU；
3. 记录 Host 起始时间；
4. 执行 forward；
5. 测量后同步 NPU；
6. 保存本 PP rank 的 elapsed time；
7. 下一 step 决策前，在 PP CPU group 做 `MAX all_reduce`；
8. PP0 使用两个 rank 中较慢者作为实际 critical-path 时间；
9. 更新对应 `(bucket, M)` 的 EWMA 比例和残差方差；
10. 下一次同类 bucket 决策使用新的 calibrated/robust cost。

反馈延迟一个 step 消费，是为了避免 M=1 时下游 rank 仍在等待正常 PP 中间张量而造成死锁。

该版本 `ParallelConfig` 中的
`adaptive_ubatch_feedback_interval_steps` 默认是 `1`，原 adaptive 脚本没有单独覆盖它。因此默认不是低频抽样，而是基本每个 eligible step 都会测量；最初的最小观测期、发生切换和受控探索时也会强制测量。

## 13. 两类 trace 不要混淆

### PP timeline trace

```bash
VLLM_ASCEND_PP_TRACE=1
```

输出：

```text
vllm_ascend_pp_trace.jsonl
```

记录 forward、send、recv、bubble 和 overlap。

### adaptive decision trace

脚本通过：

```bash
--adaptive-ubatch-trace-path \
  gradient_results/<run>/adaptive_ubatch_decisions.jsonl
```

记录：

- 每 step 的 workload bucket；
- M 候选；
- prior、calibrated、uncertainty、robust cost；
- selected M、previous M、切换原因；
- observation、bad execution、cooldown、failure。

adaptive 的在线校准来自内部反馈计时，不依赖 PP timeline trace 文件。

## 14. 该版本的局限

### 14.1 bucket 较粗

3B 和 7B 都被划入 `medium`；token 只分 `<256`、`256~1023`、`>=1024` 三档。同一 bucket 内仍可能包含差异很大的真实 step。

### 14.2 没有直接优化请求级指标

目标是 step critical-path cost，并未直接输入：

- TTFT；
- TPOT；
- queue drain rate；
- waiting time；
- 请求完成数。

所以 step-time 好不一定自动等价于端到端请求时延一定好。

### 14.3 反馈测量会扰动被测系统

显式 NPU synchronize 和阻塞式 CPU MAX 归约本身会增加开销，尤其在 3B、1K、medium 等轻场景中占比更高。

### 14.4 默认仍是均匀连续切分

即使 M 选对了，各 micro-batch 的实际计算成本也可能不均衡。这正是后续 compute-aware 要解决的问题。

## 15. 一句话总结

`vllm-ascend_adaptive` 是一个“解析先验 + 分 workload bucket 在线校准 + 风险惩罚 + 稳定切换”的 M selector：PP0 每 step 从 1/2/4 中选 M，广播给所有 stage，执行连续均匀切分，再用跨 stage 的实测 critical-path 时间校准下一次决策。
