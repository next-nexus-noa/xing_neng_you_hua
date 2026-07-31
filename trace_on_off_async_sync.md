# Trace-on / Trace-off 中“异步”和“同步”到底是什么

## 1. 结论

在当前代码中，`trace-on` 和 `trace-off` 的差别不能简单概括为“一个同步、一个异步”。实际存在四层相互独立的同步机制：

1. NPU 算子是否由 Host 异步下发；
2. PP 两个 stage 之间的中间张量是否异步传输；
3. PP trace 是否为了计时显式同步 NPU；
4. adaptive 是否为了取得跨 PP stage 的反馈而同步 NPU、归约计时。

最重要的代码结论是：

- 仅设置 `VLLM_ASCEND_PP_TRACE=1`，默认**不会**在每个 forward 前后调用 `torch.npu.synchronize()`；
- 但是 trace-on 仍会在每个 step 末尾执行一次阻塞式 `all_gather_object`，并同步写 JSONL 文件，因此会增加 CPU 控制面屏障和 I/O；
- 如果同时开启 `VLLM_ASCEND_PP_TRACE_SYNC=1`，trace 标记为 `sync_device=True` 的阶段会在开始和结束处同步 NPU；
- 如果开启 `VLLM_ASCEND_PP_TRACE_NPU_TIMING=1`，step 结束时会同步 NPU，以读取 NPU Event 的耗时；
- 实验脚本还固定设置了 `ASCEND_LAUNCH_BLOCKING=1`。它与 trace 开关无关，会让 trace-off 也不是“完全异步的生产执行”；
- `vllm-ascend_adaptive` 中的旧版 adaptive 反馈还会在被测 step 前后显式同步 NPU，并在下一次决策前做阻塞式 CPU `MAX all_reduce`。这同样与 PP trace 是否开启无关。

因此，实验中观察到的巨大 trace-on/off 差异，本质上是多个同步点、PP 排队节奏、文件 I/O 和反馈测量共同改变了执行时序，而不是一个单独的 trace 布尔变量改变了模型计算量。

## 2. 正常的异步执行是什么

NPU 运算通常分成两个时间轴：

- Host 线程负责准备输入、调用算子和向 NPU stream 下发任务；
- NPU stream 在设备侧执行已经排队的任务。

在异步模式下，Host 调用一个 NPU 算子后，不必等待该算子真正完成就可以继续下发后续工作。因此：

```text
Host:  下发 A ─ 下发 B ─ 准备下一步 ─────────────>
NPU :       执行 A ───── 执行 B ───────────────>
```

此时用普通的 `time.perf_counter()` 包围一段 NPU 代码，可能测到的是“Host 下发时间”，不一定是设备完成时间。

显式执行：

```python
torch.npu.synchronize()
```

意味着 Host 必须等待当前设备队列中的工作完成：

```text
Host:  下发 A ─ synchronize（等待）─ 后续工作 ─>
NPU :       执行 A ────────────────>
```

同步可以让计时边界更准确，但也会破坏原本可重叠的 Host、NPU、通信和下一 step 准备过程。

## 3. PP 中间张量传输的异步

两卡 PP 下，前一个 stage 计算出中间张量后，通过 `_PPBroadcastCommunicator` 发送给下一个 stage。

当前发送和接收使用：

```python
torch.distributed.broadcast(..., async_op=True)
```

也就是说，调用返回时传输不一定已经结束，而是返回一个 work handle。发送端把 handle 保存下来，在后续确实需要复用相关资源前再 `wait()`。

正常时序大致是：

```text
PP0: forward(mb0) ─ async send(mb0) ─ forward(mb1) ─ ...
PP1:        recv(mb0) ─ forward(mb0) ─ recv(mb1) ─ ...
```

这种异步 broadcast 是 micro-batch 能形成 pipeline overlap 的基础。它和 `VLLM_ASCEND_PP_TRACE` 不是同一个开关。

关键实现位置：

- `vllm-ascend_adaptive/vllm_ascend/worker/worker.py`
- `vllm-ascend_adaptive/vllm_ascend/worker/model_runner_v1.py`

## 4. PP trace 的三个开关

PP trace 实现在：

`vllm-ascend_adaptive/vllm_ascend/worker/pp_trace.py`

### 4.1 `VLLM_ASCEND_PP_TRACE`

```bash
export VLLM_ASCEND_PP_TRACE=1
```

它控制是否：

- 记录每个 stage 的 Host 时间戳；
- 收集 `runner.forward`、`runner.preprocess`、send、recv、sample 等事件；
- 在 step 末尾生成双 PP stage 的 overlap、bubble 和 makespan 指标；
- 向 JSONL 文件写记录。

当它为 `0` 时，`trace.stage(...)` 直接退化为无操作上下文。

但开启该变量后，即使没有 NPU 显式同步，仍有两项不可忽略的开销：

1. 每个 PP rank 同步写自己的 JSONL 记录；
2. 每个 step 调用阻塞式 `torch.distributed.all_gather_object`，收集两个 PP rank 的 timeline。

第二项会形成一次 CPU 控制面 rendezvous。快的 rank 需要等待慢的 rank，因此它会改变两个 stage 进入下一 step 的相对时间。

### 4.2 `VLLM_ASCEND_PP_TRACE_SYNC`

默认值是 `0`。

```bash
export VLLM_ASCEND_PP_TRACE_SYNC=1
```

开启后，只有以 `sync_device=True` 标记的 trace stage 才在边界执行：

```python
torch.npu.synchronize()
```

当前主要包括：

- `runner.forward`
- `runner.forward.microbatch.N`
- `runner.post_process`

这会把原来连续排队的 NPU 工作切成多个同步区间，使 timeline 更接近真实设备完成时间，但也会明显破坏 pipeline overlap。

### 4.3 `VLLM_ASCEND_PP_TRACE_NPU_TIMING`

默认值也是 `0`。

```bash
export VLLM_ASCEND_PP_TRACE_NPU_TIMING=1
```

它为 forward 创建 NPU Event。Event 本身可以先异步记录，但在 `finish_step()` 中，为读取所有 Event 的 elapsed time，代码仍会执行一次：

```python
torch.npu.synchronize()
```

所以 NPU timing 模式至少会在每个 step 末尾形成一次设备同步。

## 5. 三种 trace 配置的真实含义

| 配置 | NPU stage 边界同步 | step 末 NPU 同步 | 每步 PP CPU 聚合 | JSONL I/O |
|---|---:|---:|---:|---:|
| `TRACE=0` | 否 | 否 | 否 | 否 |
| `TRACE=1, SYNC=0, NPU_TIMING=0` | 否 | 否 | 是 | 是 |
| `TRACE=1, SYNC=1` | 是 | 视配置而定 | 是 | 是 |
| `TRACE=1, NPU_TIMING=1` | 否，除非也开 SYNC | 是 | 是 | 是 |

这里的“否”只表示 trace 模块没有引入该同步，不代表整个服务端不存在其他同步。

## 6. `ASCEND_LAUNCH_BLOCKING=1` 的影响

当前以下脚本都在启动服务时设置了：

```bash
ASCEND_LAUNCH_BLOCKING=1
```

- `vllm-ascend_adaptive/scripts/run_gradient_server.sh`
- `vllm-ascend_adaptive/scripts/run_adaptive_gradient_server.sh`
- 对应的 baseline 和 `adapt_const` 脚本

该变量让 NPU 异步下发更接近阻塞执行，主要用于定位异步报错和获得更明确的调用栈。它独立于 PP trace。

这意味着：

- `TRACE_ENABLED=0` 不等于“恢复了完全异步执行”；
- 如果 baseline 和 adaptive 都由同一类脚本启动，它们都承受该设置，方法间仍可比较；
- 但这类结果代表的是 blocking 配置下的性能，不应直接表述为生产异步模式性能。

## 7. `vllm-ascend_adaptive` 自己还引入了哪些同步

原版 adaptive 为了校准 M 的成本，在选定需要测量的 step 后：

1. forward 前调用 `torch.npu.synchronize()`；
2. 用 `time.perf_counter()` 开始计时；
3. 执行 micro-batch forward；
4. forward 后再次调用 `torch.npu.synchronize()`；
5. 把本 rank 的 elapsed time 暂存；
6. 下一次决策前，在 PP CPU group 上执行阻塞式 `MAX all_reduce`；
7. PP0 用两个 stage 的最大耗时更新 controller。

该目录的 `ParallelConfig` 默认
`adaptive_ubatch_feedback_interval_steps=1`，而原脚本没有覆盖它。配合“切换/探索必须测量”的条件，默认运行基本会对每个 eligible step 执行这套反馈测量，而不是偶尔抽样。

对应代码位于：

- `vllm-ascend_adaptive/vllm_ascend/worker/model_runner_v1.py`
  - `_should_measure_adaptive_pp_critical_path`
  - `_queue_adaptive_pp_microbatch_result`
  - `_flush_adaptive_pp_microbatch_result`

因此旧版 adaptive 的“在线反馈”本身就带有同步开销。PP trace 关闭后，这部分不会随之消失。

后来的 `vllm-ascend_adapt_const` 已将其改为：

- 用 NPU Event 记录被测区间；
- PP send 完成后发起 `async_op=True` 的 CPU `MAX all_reduce`；
- 将归约与下一 step 的输入准备重叠；
- 在真正消费反馈时只等待尚未完成的尾部；
- 不依赖 PP trace 提供反馈。

这是代码版本差异，不能反推到 `vllm-ascend_adaptive` 原目录。

## 8. 为什么 trace 开关会显著改变 TTFT、TPOT 和吞吐

### 8.1 PP rank 被重新对齐

trace-on 每 step 的 `all_gather_object` 让两个 PP rank 在 step 末尾重新会合。它可能：

- 减少某些失控排队和跨 step 尾部；
- 也可能破坏原本有利的 overlap；
- 改变请求首次进入 forward 的时刻，从而明显影响 TTFT。

### 8.2 同步会改变 micro-batch 的相对收益

固定 M 和 adaptive M 的价值来自不同 micro-batch 之间、不同 PP stage 之间的重叠。额外同步会改变 bubble：

- 某个 M 在 trace-on 下可能因为 rank 被对齐而表现更好；
- 同一个 M 在 trace-off 下可能暴露真实的 send tail、队列积累或控制面开销；
- 因而最优 M 也可能发生变化。

### 8.3 文件 I/O 位于每 step 热路径

trace 使用普通 Python 文件追加，而不是后台异步日志线程。短输入、低负载、小模型时，计算时间较短，固定的 JSON 序列化和文件写入占比会更大。

### 8.4 Host 时间和 NPU 时间不是同一个指标

不开 sync 时，trace 的 `duration_ms` 更接近 Host 代码段持续时间；开启 NPU timing 后，`npu_duration_ms` 才是 Event 测得的设备区间。不能把两者混作同一个时间口径。

## 9. 实验口径建议

论文主性能实验应满足：

- baseline、fixed-M、adaptive 和 compute-aware 使用完全相同的 trace 配置；
- 主结果优先使用 trace-off；
- PP trace-on 只用于机理分析、bubble 分解和时间线可视化；
- `VLLM_ASCEND_PP_TRACE_SYNC` 与 `VLLM_ASCEND_PP_TRACE_NPU_TIMING` 必须显式记录，不能只记录 `trace_enabled`；
- `ASCEND_LAUNCH_BLOCKING` 也必须作为实验配置记录；
- 决策 trace 与 PP trace 分开控制。

其中决策 trace 指：

```text
adaptive_ubatch_decisions.jsonl
```

它记录 M 的候选分数、选择、观测和回退，不等同于 `vllm_ascend_pp_trace.jsonl`。开启决策 trace 主要增加 JSON 序列化和文件写入，不会自动开启 PP stage 的 NPU 同步。

## 10. 一句话总结

`trace-off` 只是删除 PP timeline 的收集、跨 rank 聚合和写盘；`trace-on` 是否真正同步 NPU，还取决于 `PP_TRACE_SYNC` 和 `PP_TRACE_NPU_TIMING`。与此同时，`ASCEND_LAUNCH_BLOCKING=1` 和旧版 adaptive 的反馈测量又会独立引入同步，所以必须按层拆开分析。
