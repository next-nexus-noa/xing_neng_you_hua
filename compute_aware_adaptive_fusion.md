# Compute-aware 的实现及其与 Adaptive M 的融合

## 1. 版本范围

compute-aware 不在原始 `vllm-ascend_adaptive` 中，而是在以下组合版本中：

- Ascend 执行层：`vllm-ascend_adapt_const`
- vLLM 分组算法：`vllm_adapt_const`

核心文件：

- `vllm_adapt_const/vllm/v1/worker/compute_aware_ubatch.py`
- `vllm-ascend_adapt_const/vllm_ascend/worker/model_runner_v1.py`
- `vllm-ascend_adapt_const/vllm_ascend/envs.py`

它与 adaptive 解决的是两个不同层次的问题：

```text
Adaptive M  : 当前 step 应该切成几份，即 M=1/2/4
Compute-aware: M 已确定后，每一份应放哪些 token
```

## 2. uniform 切分的问题

原始 uniform 按扁平 token 序列连续、等 token 数切分：

```text
原始 token: [请求A][请求B][请求C][请求D]

M=2:
mb0 = 前一半 token
mb1 = 后一半 token
```

虽然两份 token 数相同，但计算成本不一定相同：

- 请求上下文越长，后续 token 的 attention 成本通常越高；
- 某一份可能集中包含长上下文请求；
- 某一份可能主要是 decode token，另一份主要是长 prefill；
- PP critical path 取决于最慢的 micro-batch，而不是平均成本。

compute-aware 的目标是：

```text
保持每份 token 数基本相同，
但重新组合来自不同请求的 token segment，
让各 micro-batch 的预测计算成本更接近。
```

## 3. token 代价模型

对请求 `r` 中本 step 的第 `i` 个 token，先计算其绝对上下文位置：

```text
position(r, i) =
    num_computed_tokens[r] + i + 1
```

再用当前 batch 的最大 position 归一化：

```text
cost(r, i) =
    1 + position(r, i) / maximum_position
```

因此单 token 预测成本大致在 1 到 2 之间：

- 上下文短、位置靠前的 token 成本较低；
- 上下文长、位置靠后的 token 成本较高。

这不是对 NPU kernel 的真实毫秒级拟合，而是一个轻量、单调的上下文长度代理。

## 4. token 数仍然均匀

compute-aware 首先计算与 uniform 相同的 capacity：

```text
base, remainder = divmod(total_tokens, M)
```

前 `M-1` 份为 `base` 个 token，最后一份接收 remainder。

例如：

```text
T=2048, M=2 -> [1024, 1024]
T=2050, M=2 -> [1025, 1025]
T=2051, M=2 -> [1025, 1026]
```

所以 compute-aware 不通过 45/55 之类的非均匀 token capacity 来平衡成本。它保持 token 数均匀，通过改变“每份由哪些请求片段组成”来平衡。

这也避免了某份 token 数越过 512、1024 等离散 shape 边界后，NPU kernel 成本突然上升的问题。

## 5. 分组算法

### 5.1 基本单位是 request token segment

代码不会无约束地随机打乱单个 token，而是用：

```python
RequestTokenSegment
```

表示某请求中的连续区间：

```text
request_index
request_token_start
request_token_stop
global_token_start
global_token_stop
```

### 5.2 按 quantum 贪心分配

默认：

```text
quantum = 8 tokens
```

构造每个 micro-batch 时：

1. 根据剩余总成本和剩余 micro-batch 数计算本份 target cost；
2. 根据剩余 capacity 计算当前希望得到的单位 token 成本；
3. 检查每个还有 token 的请求；
4. 从该请求取最多一个 quantum 的连续 token；
5. 选择单位成本最接近目标的请求块；
6. 重复直到本 micro-batch 的 token capacity 填满。

它会把高成本和低成本请求片段混合，使每份的总预测成本接近。

### 5.3 保持单请求依赖顺序

虽然不同请求的 token 可以跨 micro-batch 重排，但同一请求内部的 segment 只能按递增 micro-batch id 分配：

```text
请求 A 的较早 token 不会在较晚 token 之后执行
```

这是必要的，因为后续 prefill token 依赖前面 token 已经写入的 KV cache。

## 6. 什么时候真正采用 compute-aware

算法同时构造：

- 原始 contiguous uniform groups；
- compute-aware candidate groups。

分别计算各组预测成本，并把最慢一组视为 critical cost：

```text
uniform_critical   = max(uniform_group_costs)
candidate_critical = max(candidate_group_costs)

predicted_gain =
    (uniform_critical - candidate_critical)
    / uniform_critical
```

只有同时满足以下条件才应用重排：

```text
candidate 确实改变了 token 顺序
predicted_gain >= min_predicted_gain
```

默认参数为：

```bash
COMPUTE_AWARE_MIN_TOKENS=512
COMPUTE_AWARE_MIN_GAIN_PCT=5
COMPUTE_AWARE_QUANTUM=8
```

如果不满足条件，就回退到原始 uniform 分组。

## 7. token 重排如何落到真实执行

compute-aware plan 生成两个映射：

```text
permutation
inverse_permutation
```

### 7.1 forward 前

对每个 micro-batch，根据 `token_indices` 对以下数据做 `index_select`：

- `input_ids`
- `positions`
- `inputs_embeds`
- 上一 PP stage 的 intermediate tensors

attention metadata 也不是简单切连续区间，而是按每个 group 中的 request segment 重新构造：

- request-local query start；
- sequence length；
- computed token 数；
- block table；
- slot mapping；
- position；
- prefill/decode 状态。

因此 KV cache 的地址和 attention 语义仍对应原请求。

### 7.2 forward 后

最后一个 PP rank 按 micro-batch 执行顺序拼接 hidden states，此时输出顺序已经是 permutation 后的顺序。

代码再使用：

```text
inverse_permutation
```

把 hidden states 恢复成原始扁平 token 顺序，确保后续 logits、sampling 和请求输出逻辑不受影响。

## 8. 支持范围和安全回退

当前 compute-aware 会在以下情况回退 uniform：

- speculative decoding；
- pooling model；
- broadcast PP output；
- PCP/DCP/context parallel；
- compressed attention；
- GDN attention；
- Mamba cache；
- KV sharing fast prefill；
- encoder input；
- total tokens 小于最小门槛；
- 预测收益低于门槛；
- M=1，没有实际分组意义。

回退只改变 grouping，不会关闭服务，也不会改变 adaptive 已选择的 M。

## 9. Adaptive M 与 compute-aware 的融合顺序

融合是分层串联，不是同时搜索 `(M, grouping)`：

```text
第一层：Adaptive controller 选择 M
                  │
                  ▼
第二层：在这个 M 下构造 compute-aware grouping
                  │
                  ▼
         若收益不足则仅 grouping 回退 uniform
                  │
                  ▼
       使用最终 grouping 执行 PP micro-batches
                  │
                  ▼
     实测整个组合的 step 成本，反馈给 M controller
```

具体顺序是：

1. scheduler 给出当前 step 的 request/token 组成；
2. adaptive selector 先从 M=1/2/4 中选择 M；
3. runner 根据该 M 判断是否进入 micro-batch；
4. `MICROBATCH_GROUPING=compute_aware` 时，为这个 M 生成分组计划；
5. 若分组计划 `applied=True`，执行 token permutation；
6. 否则仍按该 M 执行 contiguous uniform；
7. adaptive 反馈测到的是最终实际执行路径的时间。

## 10. 两者是如何互相影响的

### 10.1 直接关系

adaptive 给 compute-aware 提供 M。M 不同：

- group 数不同；
- 每组 capacity 不同；
- 可实现的成本平衡程度不同；
- permutation 和 attention metadata 都不同。

### 10.2 间接闭环

compute-aware 不直接修改 adaptive 的解析公式，也不把 predicted grouping gain 传给 M selector。

但 adaptive 的在线观测是在最终 grouping 执行之后取得的。因此：

- 如果 compute-aware 让 M=2 的实际 step 更快；
- 对应 bucket/M=2 的校准状态会逐步变好；
- 后续同类 workload 中，adaptive 更可能保留或选择 M=2。

所以融合是“先独立决策、再通过实际反馈间接耦合”，不是一次联合优化器。

## 11. 四种实验模式如何在同一代码上表示

| 实验方法 | M 的来源 | Grouping |
|---|---|---|
| fixed-M baseline | CSV 固定 1/2/4 | `uniform` |
| adaptive | adaptive selector | `uniform` |
| compute-aware | CSV 固定 2/4 | `compute_aware` |
| adaptive + compute-aware | adaptive selector | `compute_aware` |

注意 `adapt_const` 当前脚本默认：

```bash
MICROBATCH_GROUPING=scom
```

要运行 compute-aware，必须显式设置：

```bash
export MICROBATCH_GROUPING=compute_aware
```

### fixed-M + compute-aware

CSV 中：

```text
micro_batch=2
```

或：

```text
micro_batch=4
```

并使用固定 M server 脚本。

### adaptive + compute-aware

CSV 中：

```text
micro_batch=adaptive
```

并使用 adaptive server 脚本，同时：

```bash
export MICROBATCH_GROUPING=compute_aware
```

### adaptive only

```bash
export MICROBATCH_GROUPING=uniform
```

这会保留 adaptive M selector，但禁用 token 重排。

## 12. compute-aware 与 composition-aware 的关系

当前 compute-aware 已经比“连续等 token 切分”更进一步，因为它：

- 查看每个请求的已计算 token 数；
- 估计每个 token 的上下文位置成本；
- 允许把不同请求的 segment 组合进同一 micro-batch。

但它还不是完整的 composition-aware 模型。当前标量成本没有显式建模：

- prefill 请求数与 decode 请求数；
- 每份中的 request 数；
- attention kernel 的离散 shape；
- KV block 分布；
- PP 通信张量 shape；
- 实测 NPU kernel latency；
- 某些 prefill/decode 混合组合的特殊开销。

因此更准确的定位是：

```text
compute-aware =
    基于上下文位置代理成本的 request-segment 重组

composition-aware =
    同时感知请求类型、长度分布、KV/shape/通信和真实设备成本
```

## 13. 当前实现的优点与局限

### 优点

- 不改变每份 token capacity，规避明显的 shape 档位膨胀；
- 能跨请求平衡成本，不会必然退化为原始 1024/1024 连续边界；
- 保持单请求 token 依赖顺序；
- 有可逆 permutation，输出语义不变；
- 预测收益不足自动回退 uniform；
- 分组在各 PP rank 根据相同输入确定性本地计算，不在热路径广播 Python plan。

### 局限

- token 成本只是线性上下文位置代理；
- 没有使用真实 NPU latency 表；
- 仍固定等 token capacity；
- 贪心算法不保证全局最优；
- 规划、index_select 和重建 metadata 自身有额外开销；
- adaptive 与 grouping 不是联合搜索，短期内可能先选出不适合该 grouping 的 M；
- 轻负载或小 batch 中，分组收益可能小于置换开销。

## 14. 一句话总结

compute-aware 在 adaptive 选出 M 后，保持每份 token 数均匀，但把不同请求的连续 token segment 重新组合，使各 micro-batch 的上下文位置代理成本更均衡；最终执行时间再反馈给 adaptive，形成“外层选 M、内层优化 composition、实测结果间接校准外层”的组合闭环。
