# SCOM：带 Shape 档位约束的阶段向量与通信感知有序 Micro-batch 映射方案

## 1. 文档目的

本文档用于指导 Codex 在现有 vLLM/vllm-ascend 双卡 Pipeline Parallelism（PP）代码中实现 SCOM：

> **Stage-Vector and Communication-Aware Ordered Micro-batch Mapping**
>
> **阶段向量与通信感知的有序 Micro-batch 映射**

当前系统已经完成：

1. 请求准入，即确定本轮参与执行的请求或请求片段；
2. 自适应选择 micro-batch 数量 \(m\)；
3. 使用等 Token 容量作为当前 micro-batch 容量基线；
4. 使用 scalar compute-aware grouping，根据 scheduled Token 数和
   computed Token 位置估算标量成本，并在等 Token 容量下重排成员；
5. 使用 `_PPBroadcastCommunicator` 在相邻 PP rank 之间传递中间激活。
   其底层对元数据调用 `broadcast_object_list`，对激活张量调用
   `torch.distributed.broadcast(..., async_op=True)`。现有 trace 中的
   `send.microbatch`/`recv.microbatch` 是逻辑阶段名称，不代表底层使用
   Send/Recv 通信原语。

此前的非均匀切分实验已经表明，当前 NPU 的执行成本不是 Token 数的平滑函数：
当单个 micro-batch 跨过 512、1024 等算子 shape 档位时，计算成本可能发生
离散跳升。因此，SCOM 不能把任意比例的非均匀 Token 切分当作默认能力。

修订后的 SCOM 负责：

> 在请求准入结果和 \(m\) 保持不变的条件下，先生成少量不跨越不利 shape
> 档位的容量候选，再决定每个任务单元进入哪个有序 micro-batch slot，
> 使完整双卡 Broadcast PP 流水线的预计完成时间最短。

容量搜索是受约束、可回退的可选步骤：

- 默认基线仍是等 Token 容量；
- 若总 Token 数已经把所有 slot 顶到同一个 shape 上界，例如
  \(2048\rightarrow1024+1024\) 或
  \(2048\rightarrow512\times4\)，容量不得为了“非均匀”而跨档；
- 这类饱和 step 中，SCOM 只优化“每份放什么”和“各份按什么顺序执行”；
- 只有存在同档位内的容量余量，或者实测 profile 明确证明跨档后的全流水线
  收益仍为正时，才允许产生非均匀容量。

本文只实现和验证当前代码真实使用的 Broadcast 通信路径，不实现
Send/Recv。SCOM 名称保持“通信感知”，因为方法显式建模当前平台的
异步 Broadcast、同步等待和通信/计算重叠。

---

## 2. 实现边界

### 2.1 本次需要实现

- 将本轮已准入任务转换为可映射任务单元；
- 从实测 profile 构造 NPU shape 档位表；
- 以等 Token 容量为基线，生成有限数量的 bucket-safe 容量候选；
- 分别预测候选 micro-batch 在 Rank 0 和 Rank 1 上的执行时间；
- 预测候选 micro-batch 的 Broadcast 时间；
- 模拟一个候选有序映射的完整 PP 时序；
- 以 FCFS 映射为基线，执行贪心成员放置；
- 对相邻 slot 进行有限次数的成员交换；
- 在预测收益不足、\(m=1\) 或预测不可信时自动回退；
- 输出成员映射、预测时序、实际时序和回退原因日志；
- 通过功能测试保证请求、Token、KV Cache 和采样元数据不会错位。

### 2.2 本次不实现

- 不修改原始请求准入算法；
- 不修改 adaptive \(m\) 的选择逻辑；
- 不使用 45/55、22/24/26/28 等固定比例作为默认容量；
- 不在没有实测依据时允许任一 slot 跨过等容量基线所在的 shape 档位；
- 不实现 Send/Recv 或异步激活缓冲区；
- 不训练神经网络预测器；
- 不跨调度 step 移动 KV Cache；
- 不改变请求在 Prefill、Decode 阶段的因果关系；
- 不将 SLO、KV Cache 或上下文长度单独包装成新的创新点。

---

## 3. 术语与决策变量

### 3.1 任务单元

SCOM 的最小成员不是完整请求，而是本轮已经由原调度器选中的可执行任务单元：

- 一个 Prefill chunk；
- 一个 Decode 请求在当前 step 中的一个 Token；
- 现有代码已经生成的其他可执行请求片段。

定义：

\[
u_r=
(
\text{request\_id},
\text{phase},
n_r,
L_r,
K_r,
o_r
)
\]

其中：

- \(n_r\)：该任务单元本轮执行的 Token 数；
- \(L_r\)：当前上下文长度；
- \(K_r\)：已有 KV Cache 长度或 KV block 数；
- \(o_r\)：原始 FCFS 顺序；
- `phase`：`prefill` 或 `decode`。

### 3.2 有序 slot

第 \(j\) 个 slot 表示第 \(j\) 个进入 PP 流水线的 micro-batch：

\[
\text{slot}_1\rightarrow\text{slot}_2\rightarrow\cdots\rightarrow\text{slot}_m
\]

等 Token 基线容量为：

\[
\boldsymbol{\tau}^{(0)}
=
\operatorname{UniformCapacity}(T,m)
\]

SCOM 可以从该基线附近生成有限的 bucket-safe 容量候选：

\[
\boldsymbol{\tau}
=
[\tau_1,\tau_2,\ldots,\tau_m]
\]

并满足：

\[
\sum_{j=1}^{m}\tau_j=T
\]

定义 \(b(\tau_j)\) 为该容量对应的 NPU 执行 shape 档位。第一版默认要求：

\[
b(\tau_j)\leq b\left(\max_k\tau_k^{(0)}\right),\quad\forall j
\]

即候选容量不能进入比等容量基线更昂贵的 shape 档位。

SCOM 的联合决策变量为容量向量 \(\boldsymbol{\tau}\) 和成员映射：

\[
\pi(u_r)=j
\]

表示任务单元 \(u_r\) 被放入第 \(j\) 个有序 slot。

---

## 4. 总体处理流程

```text
原始 Scheduler 完成请求准入
              ↓
Adaptive 模块确定 m
              ↓
生成等 Token 基线容量 τ⁽⁰⁾
              ↓
生成有限 bucket-safe 容量候选集合 𝒯
              ↓
构造 ScheduledUnit 列表
              ↓
对每个 τ∈𝒯 生成 FCFS 映射
              ↓
执行 SCOM 贪心有序映射和相邻 slot 交换
              ↓
用完整 Broadcast PP 时序比较 (τ,π)
              ↓
收益门控和安全检查
              ↓
输出 OrderedMicroBatch 列表
              ↓
Model Runner 按新顺序打包
              ↓
双卡 Broadcast PP 执行
              ↓
记录预测时间和实际时间
```

SCOM 必须位于“请求准入和 \(m\) 确定之后、Model Runner 输入张量打包之前”。
当前 scalar compute-aware planner 应保留为实验基线和安全回退，不应先执行一次
scalar 重排、再由 SCOM 进行第二次重排。

---

## 5. 数据结构

建议新增以下数据结构。具体文件位置应根据当前仓库结构调整，不要强行创建与项目风格冲突的目录。

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ScheduledUnit:
    request_id: str
    phase: Literal["prefill", "decode"]
    scheduled_tokens: int
    context_length: int
    kv_blocks: int
    original_order: int

    # 指向现有调度结果的稳定索引或句柄。
    # 不要复制 KV Cache，只保留重新打包所需的关联信息。
    source_index: int

    # 可选安全信息
    waiting_ms: float = 0.0
    slo_slack_ms: float | None = None


@dataclass
class StageCostVector:
    rank0_ms: float
    rank1_ms: float
    confidence: float


@dataclass
class OrderedMicroBatch:
    slot_id: int
    token_capacity: int
    shape_bucket: int
    units: list[ScheduledUnit] = field(default_factory=list)

    predicted_stage: StageCostVector | None = None
    predicted_broadcast_ms: float = 0.0
    predicted_finish_ms: float = 0.0

    @property
    def used_tokens(self) -> int:
        return sum(unit.scheduled_tokens for unit in self.units)


@dataclass
class PipelineSimulationResult:
    objective_ms: float
    finish_ms: float
    rank0_idle_ms: float
    rank1_idle_ms: float
    rank0_broadcast_wait_ms: float
    rank1_broadcast_wait_ms: float
    slot_finish_ms: list[float]
    prediction_confidence: float


@dataclass
class SCOMDecision:
    selected_mapping: list[OrderedMicroBatch]
    baseline_capacities: list[int]
    selected_capacities: list[int]
    selected_shape_buckets: list[int]
    baseline_objective_ms: float
    selected_objective_ms: float
    predicted_gain: float
    mapping_overhead_ms: float
    used_scom: bool
    fallback_reason: str | None
```

要求：

- `ScheduledUnit` 应引用现有调度对象，不能复制或迁移 KV Cache；
- `source_index` 必须能够唯一关联原始输入张量、block table 和采样元数据；
- 输出 slot 顺序必须成为后续 Model Runner 的真实执行顺序。

---

## 6. Micro-batch 特征提取

对候选 micro-batch \(M_j\) 提取：

\[
\mathbf f(M_j)=
[
P_j,
D_j,
N_j,
T_j,
\widetilde T_j,
B_j,
L_j^{\mathrm{sum}},
L_j^{\mathrm{max}},
K_j^{\mathrm{sum}}
]
\]

其中：

- \(P_j\)：Prefill Token 数；
- \(D_j\)：Decode 任务单元数；
- \(N_j\)：成员总数；
- \(T_j\)：本轮调度总 Token 数；
- \(\widetilde T_j\)：进入实际算子或编译图的 padded Token 数；
- \(B_j\)：当前 NPU 执行 shape 档位；
- \(L_j^{\mathrm{sum}}\)：上下文长度之和；
- \(L_j^{\mathrm{max}}\)：最大上下文长度；
- \(K_j^{\mathrm{sum}}\)：KV block 数或 KV 长度之和。

建议实现：

```python
@dataclass(frozen=True)
class MicroBatchFeatures:
    prefill_tokens: int
    decode_count: int
    member_count: int
    total_tokens: int
    padded_tokens: int
    shape_bucket: int
    context_sum: int
    context_max: int
    kv_blocks_sum: int


def extract_features(
    units: list[ScheduledUnit],
) -> MicroBatchFeatures:
    ...
```

空 slot 必须返回全零特征，预测器需要能处理该情况。

`shape_bucket` 不能仅通过理论公式猜测，应由实测 profile 和当前实际
padding/graph 选择逻辑共同确定。跨档位的两个样本之间禁止直接做普通线性插值。

---

## 7. 逐流水级执行时间预测

### 7.1 预测目标

不要为每个请求生成一个标量 cost 后直接相加，而应直接预测候选 micro-batch 在两个 rank 上的执行时间：

\[
\hat{\mathbf C}(M_j)
=
[
\hat C_{0,j},
\hat C_{1,j}
]
\]

实现接口：

```python
class StageCostPredictor:
    def predict(
        self,
        features: MicroBatchFeatures,
    ) -> StageCostVector:
        ...
```

### 7.2 第一版预测器

第一版使用“离线 Profile 查表＋邻近点插值”，不要引入神经网络。

实现分两步上线：

1. 功能与pilot阶段允许使用
   `analytical_shape_aware` 解析先验，验证容量、成员重排、Broadcast时序和
   正确性；该阶段的预测收益不能直接作为论文结论；
2. 扩展完整实验矩阵前必须接入离线Profile查表，并用shadow mode报告
   Rank 0、Rank 1和Broadcast路径的预测误差。

预测器必须显式保留 shape cliff。推荐先按
`(model, rank, shape_bucket, phase_mix_bucket)` 查桶，再只在同一桶内部插值；
不能用一条连续回归曲线把 512 与 1024、1024 与 2048 等档位平滑连接。

建议 Profile 字段：

```text
model_name
rank_id
prefill_tokens
decode_count
member_count
total_tokens
padded_tokens
shape_bucket
context_sum
context_max
kv_blocks_sum
compute_time_ms
sample_count
std_ms
```

建议优先覆盖以下分桶：

| 参数 | 建议取值 |
|---|---|
| Prefill Token 数 | 0、128、256、512、1024、2048、3072 |
| Decode 数量 | 0、1、2、4、8、16、32 |
| 最大上下文长度 | 128、512、1024、2048、4096、6144 |
| 成员数量 | 1、2、4、8、16、32 |
| 模型 | 当前实际使用的 3B、7B、14B |
| rank | 0、1 |

若查不到足够接近的 Profile：

1. 使用最近邻或分段线性回归；
2. 降低 `confidence`；
3. 低于置信度阈值时触发 FCFS 回退。

### 7.3 边际代价

将任务 \(u_r\) 放入 slot \(j\) 时，重新预测整个候选 micro-batch：

\[
\Delta\mathbf C(u_r,j)
=
\hat{\mathbf C}(M_j\cup\{u_r\})
-
\hat{\mathbf C}(M_j)
\]

不能假设：

\[
\hat{\mathbf C}(M_j)
=
\sum_{u_r\in M_j}\hat{\mathbf c}_r
\]

原因是 batch 算子效率、最长上下文和 Prefill/Decode 混合会导致执行时间非线性。

---

## 8. Broadcast 时间模型

### 8.1 输入

Broadcast 模型至少使用：

- 激活字节数；
- 总 Token 数；
- 成员数量；
- Prefill/Decode 组成。

当前实现的真实通信路径为：

1. `_PPBroadcastCommunicator.send_to_next()` 对元数据执行
   `broadcast_object_list`；
2. 对每个中间激活张量执行
   `torch.distributed.broadcast(..., async_op=True)`；
3. Rank 0 保存异步 handle，并在下一次 worker step 开始时等待尚未完成的
   send；
4. Rank 1 返回 `AsyncIntermediateTensors`，在模型真正读取 `.tensors`
   时等待通信完成。

因此，`send.microbatch` 主要覆盖通信发起过程，`recv.microbatch` 主要覆盖
接收注册过程，两者都不一定等于完整传输时间。SCOM 的 Broadcast profile
必须区分“Host发起时间”“设备通信完成时间”和“Rank 1可消费时间”。

激活字节数的初始估计：

\[
A_j=T_j\times H\times B
\]

其中：

- \(T_j\)：micro-batch Token 数；
- \(H\)：模型隐藏层维度；
- \(B\)：每个激活元素字节数。

### 8.2 接口

```python
class BroadcastCostModel:
    def predict_ms(
        self,
        features: MicroBatchFeatures,
        activation_bytes: int,
    ) -> tuple[float, float]:
        """返回 (predicted_ms, confidence)。"""
        ...
```

### 8.3 第一版模型

优先使用实测 Profile 查表。数据不足时退化为：

\[
\hat D_j
=
\alpha_{\mathrm{bc}}
+
\beta_{\mathrm{bc}}A_j
\]

注意：日志中的 `send.microbatch` 时长可能包含等待对端进入集合通信的时间，不能直接作为纯数据传输时间。Profile 时应尽量分别记录：

- Broadcast API 调用开始；
- 对端准备完成；
- Broadcast API 调用结束；
- 激活字节数。

若暂时无法分离纯传输和同步等待，允许第一版使用端到端 Broadcast 调用时间，但必须在日志和论文中将其称为“Broadcast 路径开销”，而不是“纯传输时间”。

---

## 9. 双卡 Broadcast PP 时序模拟器

### 9.1 时序变量

对第 \(j\) 个 micro-batch，定义：

- \(S_{0,j}\)：Rank 0 开始计算时间；
- \(F_{0,j}\)：Rank 0 计算完成时间；
- \(L_{B,j}\)：Rank 0 发起异步 Broadcast 的时间；
- \(S_{B,j}\)：设备侧 Broadcast 真正开始时间；
- \(E_{B,j}\)：Broadcast 结束时间；
- \(S_{1,j}\)：Rank 1 开始计算时间；
- \(F_{1,j}\)：Rank 1 计算完成时间。

初始化：

\[
S_{0,1}=0,\qquad F_{1,0}=0,\qquad E_{B,0}=0
\]

Rank 0 计算完成：

\[
F_{0,j}
=
S_{0,j}+\hat C_{0,j}
\]

Rank 0 在计算结束后发起异步 Broadcast：

\[
L_{B,j}=F_{0,j}+\hat H_{B,j}
\]

其中 \(\hat H_{B,j}\) 是 Host 侧元数据处理和 collective 发起开销。若同一
edge 上的 Broadcast collective 按顺序执行，则：

\[
S_{B,j}=\max(L_{B,j},E_{B,j-1})
\]

设备侧 Broadcast 完成：

\[
E_{B,j}=S_{B,j}+\hat D_j
\]

当前代码使用 `async_op=True`，Rank 0 发起通信后不在每个 micro-batch
边界等待 handle。因此，第一版递推应使用：

\[
S_{0,j+1}=F_{0,j}+\hat H_{B,j}
\]

而不是错误地使用 \(S_{0,j+1}=E_{B,j}\)。通信与下一份 Rank 0 计算可能
重叠，但共享 NPU 资源造成的干扰必须通过实测 profile 校准。

Rank 1 必须同时等待上一份计算完成和本份激活可用：

\[
S_{1,j}=\max(F_{1,j-1},E_{B,j})
\]

\[
F_{1,j}=S_{1,j}+\hat C_{1,j}
\]

Rank 1 因 Broadcast 产生的等待为：

\[
W_{1,j}=\max(0,E_{B,j}-F_{1,j-1})
\]

Rank 0 的通信影响分为 Host 发起开销、通信与计算资源竞争，以及下一
scheduler step 开始时对残留 send handle 的等待。模拟器需额外输出：

\[
W_{\mathrm{carry}}
=
\max(0,E_{B,m}-F_{0,m})
\]

完整 batch 的完成时间和稳态目标分别为：

\[
T_{\mathrm{finish}}=F_{1,m}
\]

\[
J_{\mathrm{steady}}
=
T_{\mathrm{finish}}+\lambda_{\mathrm{carry}}W_{\mathrm{carry}}
\]

若 trace 证明最后一个 Broadcast 总在 Rank 1 完成前结束，可令
\(\lambda_{\mathrm{carry}}=0\)；否则不能把通信开销转移到下一 step 后忽略。

### 9.2 模拟器接口

```python
class BroadcastPipelineSimulator:
    def simulate(
        self,
        microbatches: list[OrderedMicroBatch],
        initial_rank0_ready_ms: float = 0.0,
        initial_rank1_ready_ms: float = 0.0,
    ) -> PipelineSimulationResult:
        ...
```

### 9.3 重要要求

已经从代码确认：

- 激活传输底层是异步 Broadcast，不是 Send/Recv；
- Rank 0 不在每个 micro-batch 后等待 Broadcast handle；
- Rank 0 在下一 worker step 开头统一等待上一步尚未完成的 send handle；
- Rank 1 通过 `AsyncIntermediateTensors` 延迟到实际读取激活时等待；
- Host API 返回不代表设备通信已经完成。

仍需通过新增事件或 NPU profiler 确认：

- 同一 edge 上多个 Broadcast 是否严格串行；
- Broadcast 使用的 stream 与模型计算 stream 的依赖关系；
- 通信与计算同时进行时的资源竞争系数；
- `handle.wait()` 的完成时间和 Rank 1 首次消费激活的时间。

模拟器必须按实测语义校准，不能把 trace 的 `send`/`recv` 阶段时长直接当作
纯通信时间。

---

## 10. 优化目标

第一版采用联合容量与成员映射目标：

\[
\min_{\boldsymbol{\tau},\pi}J(\boldsymbol{\tau},\pi)
\]

\[
J(\boldsymbol{\tau},\pi)
=
\hat T_{\mathrm{finish}}(\boldsymbol{\tau},\pi)
+\lambda_{\mathrm{carry}}\hat W_{\mathrm{carry}}
\]

即在 shape-safe 容量候选中，直接最小化完整双卡 PP 的预计稳态完成时间。
非均匀不是单独目标；若等容量映射最好，就必须保留等容量。

Broadcast 等待和 rank 空闲已经体现在完成时间内，不建议第一版重复设置大量人工权重。

以下内容作为硬约束或次级比较指标：

- Token 容量；
- 显存安全；
- 请求因果关系；
- 最大重排距离；
- SLO 紧急请求保护；
- 总 Broadcast 等待；
- Rank 0/Rank 1 空闲时间。

若两个候选方案的完成时间差小于容差，可按以下顺序打破平局：

1. Broadcast 总等待更少；
2. 最大请求重排距离更小；
3. 更接近 FCFS；
4. 保留原始顺序。

---

## 11. 约束条件

### 11.1 Token 容量

\[
\sum_{u_r\in M_j}n_r
\leq\tau_j,\quad\forall j
\]

同时要求：

\[
\sum_{j=1}^{m}\tau_j=T
\]

第一版使用 shape 档位硬约束。令
\(\tau_{\max}^{(0)}=\max_j\tau_j^{(0)}\)，则：

\[
b(\tau_j)\leq b(\tau_{\max}^{(0)}),\quad\forall j
\]

这条约束直接吸收此前非均匀切分实验观察到的离散成本膨胀：

- 当 \(T=2048,m=2\) 且当前安全上界为 1024 时，
  \(\tau_1+\tau_2=2048\) 且两者都不能超过 1024，因此唯一安全容量就是
  \([1024,1024]\)；
- 当 \(T=2048,m=4\) 且当前安全上界为 512 时，唯一安全容量就是
  \([512,512,512,512]\)；
- 此时 SCOM 不进行容量非均匀化，只优化成员组成和有序 slot 映射；
- 当 \(T\) 未填满档位，例如两个 slot 都能保持在 1024 桶内时，才允许在
  `capacity_quantum` 粒度下产生同档位非均匀候选。

后续版本可以允许跨档，但必须同时满足：

1. 跨档点存在足够的实测 profile；
2. profile 置信度达到阈值；
3. 完整 PP 模拟收益高于普通收益门槛和额外跨档安全裕量；
4. 实测 shadow 结果没有出现预测外的 shape cliff。

### 11.2 唯一映射

\[
\sum_{j=1}^{m}x_{r,j}=1,\quad\forall r
\]

### 11.3 完整覆盖

所有上游已准入任务必须出现在输出映射中，不能遗漏或新增。

### 11.4 显存安全

\[
M_{\mathrm{model}}
+
M_{\mathrm{KV}}
+
M_{\mathrm{activation}}
\leq M_{\mathrm{safe}}
\]

如果上游已经完成严格显存准入，SCOM 至少应保证重新组合不会提高激活峰值到安全阈值以上。

### 11.5 因果顺序

同一请求的后续 Prefill chunk 不能排在其前置 chunk 之前。若当前 step 中同一请求只存在一个任务单元，该约束自然满足。

### 11.6 最大重排距离

\[
\left|
\mathrm{slot}_{\mathrm{new}}(u_r)
-
\mathrm{slot}_{\mathrm{FCFS}}(u_r)
\right|
\leq W
\]

第一版建议：

\[
W=1
\]

即成员只允许在原 slot 或相邻 slot 之间移动。

---

## 12. 映射算法

使用“bucket-safe 容量候选＋FCFS 初始化＋关键任务贪心放置＋相邻 slot
有限交换”。

### 12.0 Bucket-safe 容量候选

从等容量基线 \(\boldsymbol{\tau}^{(0)}\) 开始，只在相邻 slot 之间按
`capacity_quantum` 转移容量：

\[
(\tau_j,\tau_{j+1})
\rightarrow
(\tau_j-q,\tau_{j+1}+q)
\]

每次转移后必须满足：

- 总容量不变；
- 每个 slot 能容纳至少一个合法任务单元；
- 不进入比等容量基线更高的 shape 桶；
- 候选数量不超过 `max_capacity_candidates`；
- 去重后始终包含等容量基线。

第一版不预设“前重后轻”或“先小后大”。slot 顺序由完整 Broadcast PP
模拟器决定。若已有证据表明某一方向始终更优，可以只作为搜索优先级，
不能作为绕过时序评估的硬编码结论。

### 12.1 FCFS 基准

按照原始调度顺序和等 Token 基线容量
\(\boldsymbol{\tau}^{(0)}\) 生成：

\[
\pi_{\mathrm{base}}
\]

计算：

\[
J_{\mathrm{base}}
=
J(\boldsymbol{\tau}^{(0)},\pi_{\mathrm{base}})
\]

Uniform-FCFS 映射必须始终保留，任何异常都回退到该结果。对某一固定容量
候选执行内层成员搜索时，可简写为 \(J(\pi)\)；跨容量候选比较时必须使用
完整的 \(J(\boldsymbol{\tau},\pi)\)。

### 12.2 关键任务排序

可为任务计算一个只用于搜索顺序的启发值：

\[
H_r
=
\max_s\Delta C_{s,r}
+
\eta
\left|
\Delta C_{0,r}-\Delta C_{1,r}
\right|
\]

其中 \(\Delta C_{s,r}\) 可通过将任务暂时放入空 slot 或其原始 slot 得到。

该启发值不直接决定最终 slot，只决定优先尝试哪些任务。最终位置仍由完整 PP 模拟结果决定。

### 12.3 贪心放置

对任务 \(u_r\)：

1. 枚举满足容量、显存、因果关系和重排窗口的 slot；
2. 暂时将任务放入候选 slot；
3. 重新预测受影响 slot 的两个 rank 时间；
4. 重新预测对应 Broadcast 时间；
5. 模拟完整 PP 时序；
6. 选择 \(J(\pi)\) 最小的 slot；
7. 平局时优先选择更接近 FCFS 的 slot。

形式化表示：

\[
j^\star
=
\arg\min_{j\in\mathcal F(r)}
J\left(\pi\oplus(u_r,j)\right)
\]

### 12.4 相邻 slot 交换

贪心映射完成后，只在相邻 slot 之间尝试交换：

\[
u_a\in M_j,\qquad
u_b\in M_{j+1}
\]

若交换后所有约束满足，且：

\[
J(\pi_{\mathrm{swap}})
<
J(\pi)-\epsilon_{\mathrm{swap}}
\]

则接受交换。

第一版默认：

```text
max_swaps = 4
swap_improvement_epsilon = 0.5%
```

### 12.5 伪代码

```python
def build_bucket_safe_scom(
    units,
    total_tokens,
    num_ubatches,
    shape_profile,
    stage_predictor,
    broadcast_model,
    simulator,
    config,
):
    baseline_capacities = uniform_capacities(
        total_tokens,
        num_ubatches,
    )
    capacity_candidates = generate_bucket_safe_capacities(
        baseline_capacities,
        shape_profile,
        quantum=config.capacity_quantum,
        max_candidates=config.max_capacity_candidates,
        allow_bucket_crossing=config.allow_bucket_crossing,
    )

    best_decision = None
    for capacities in capacity_candidates:
        decision = build_scom_mapping(
            units,
            capacities,
            stage_predictor,
            broadcast_model,
            simulator,
            config,
        )
        if is_better_global_decision(decision, best_decision):
            best_decision = decision

    return gate_against_uniform_fcfs(
        best_decision,
        baseline_capacities,
        config,
    )


def build_scom_mapping(
    units,
    capacities,
    stage_predictor,
    broadcast_model,
    simulator,
    config,
):
    baseline = build_fcfs_mapping(units, capacities)

    if len(capacities) <= 1:
        return fallback(baseline, "single_microbatch")

    if len(units) <= 1:
        return fallback(baseline, "insufficient_units")

    baseline_result = evaluate_mapping(
        baseline,
        stage_predictor,
        broadcast_model,
        simulator,
    )

    if baseline_result.prediction_confidence < config.min_confidence:
        return fallback(baseline, "low_prediction_confidence")

    candidate = empty_ordered_slots(capacities)
    ordered_units = sort_units_for_search(units)

    for unit in ordered_units:
        best_trial = None
        best_result = None

        for slot_id in feasible_slots(
            unit,
            candidate,
            capacities,
            max_reorder_distance=config.max_reorder_distance,
        ):
            trial = place_unit(candidate, unit, slot_id)
            result = evaluate_mapping(
                trial,
                stage_predictor,
                broadcast_model,
                simulator,
            )

            if is_better(result, best_result, unit, slot_id):
                best_trial = trial
                best_result = result

        if best_trial is None:
            return fallback(baseline, "no_feasible_slot")

        candidate = best_trial

    candidate = improve_with_adjacent_swaps(
        candidate,
        max_swaps=config.max_swaps,
        improvement_epsilon=config.swap_epsilon,
    )

    candidate_result = evaluate_mapping(
        candidate,
        stage_predictor,
        broadcast_model,
        simulator,
    )

    gain = (
        baseline_result.objective_ms
        - candidate_result.objective_ms
    ) / baseline_result.objective_ms

    if gain < config.min_predicted_gain:
        return fallback(baseline, "insufficient_predicted_gain")

    return accept(candidate, baseline_result, candidate_result)
```

---

## 13. 收益门控和自动回退

预测收益：

\[
G
=
\frac{
J_{\mathrm{base}}-J_{\mathrm{SCOM}}
}{
J_{\mathrm{base}}
}
\]

初始阈值：

```text
min_predicted_gain = 0.03
min_prediction_confidence = 0.70
max_mapping_overhead_ms = 1.0
max_reorder_distance = 1
max_swaps = 4
capacity_quantum = 64
max_capacity_candidates = 8
allow_bucket_crossing = false
bucket_crossing_extra_gain = 0.05
```

以下情况必须回退 FCFS：

- SCOM 功能开关关闭；
- \(m=1\)；
- 任务单元少于 2；
- 没有其他合法映射；
- 预测器置信度不足；
- 预计收益低于 3%；
- 映射开销超过 1 ms；
- 模拟器出现非有限值或异常；
- 输出成员集合与输入不一致；
- 任意 slot 超出 Token 容量；
- 容量候选跨过禁止的 shape 档位；
- 找不到对应 shape 桶的可靠 profile；
- 后续张量重排安全检查失败。

所有回退都必须记录明确 `fallback_reason`。

---

## 14. 配置项

建议增加配置：

```python
@dataclass
class SCOMConfig:
    enabled: bool = False
    profile_path: str | None = None
    broadcast_profile_path: str | None = None
    shape_bucket_profile_path: str | None = None

    min_predicted_gain: float = 0.03
    min_prediction_confidence: float = 0.70
    max_mapping_overhead_ms: float = 1.0

    optimize_capacities: bool = True
    capacity_quantum: int = 64
    max_capacity_candidates: int = 8
    allow_bucket_crossing: bool = False
    bucket_crossing_extra_gain: float = 0.05

    max_reorder_distance: int = 1
    max_swaps: int = 4
    swap_epsilon: float = 0.005

    log_decisions: bool = True
    shadow_mode: bool = False
```

`shadow_mode=True` 时：

- 计算 SCOM 方案；
- 记录预测收益；
- 实际仍执行 FCFS；
- 用于先验证预测器和模拟器，不影响正确性。

建议上线顺序：

1. `enabled=False`，确认无行为变化；
2. `shadow_mode=True`，只记录决策；
3. 小规模测试开启真实映射；
4. 通过正确性和性能测试后再运行完整梯度实验。

---

## 15. Model Runner 重排要求

SCOM 不能只重排 request ID。必须以 `source_index` 为关联键，同步重排所有与任务单元相关的数据：

- request ID；
- scheduled token 数量；
- input IDs；
- positions；
- sequence lengths；
- query lengths；
- block tables；
- KV Cache slot mapping；
- multimodal placeholders（若当前模型使用）；
- sampling metadata；
- Prefill/Decode 标志；
- logits 索引；
- PP 发送激活与接收端请求顺序。

建议增加一致性检查：

```python
def validate_repacked_batch(
    original_units,
    ordered_microbatches,
):
    assert same_unit_multiset(original_units, ordered_microbatches)
    assert no_duplicate_source_indices(ordered_microbatches)
    assert capacities_satisfied(ordered_microbatches)
    assert causal_order_satisfied(ordered_microbatches)
```

如果检查失败，禁止继续执行优化映射，立即使用原始 FCFS 打包结果。

---

## 16. 日志与可观测性

### 16.1 成员映射日志

```text
step_id
slot_id
request_id
source_index
phase
scheduled_tokens
context_length
kv_blocks
original_slot
mapped_slot
```

### 16.2 预测日志

```text
step_id
slot_id
rank0_predicted_ms
rank1_predicted_ms
broadcast_predicted_ms
rank0_predicted_wait_ms
rank1_predicted_wait_ms
slot_predicted_finish_ms
prediction_confidence
```

### 16.3 实际执行日志

```text
step_id
slot_id
rank0_actual_compute_ms
rank1_actual_compute_ms
broadcast_actual_path_ms
rank0_actual_wait_ms
rank1_actual_wait_ms
slot_actual_finish_ms
```

### 16.4 决策日志

```text
step_id
m
baseline_capacities
selected_capacities
baseline_shape_buckets
selected_shape_buckets
capacity_changed
baseline_objective_ms
optimized_objective_ms
predicted_gain
mapping_overhead_ms
used_scom
fallback_reason
```

还应记录：

```text
broadcast_launch_ms
broadcast_complete_ms
rank1_activation_ready_ms
carryover_send_wait_ms
capacity_candidate_count
shape_bucket_rejection_count
```

### 16.5 预测准确性

至少计算：

\[
\mathrm{MAPE}_s
=
\frac{1}{N}
\sum_{j=1}^{N}
\left|
\frac{
C_{s,j}^{\mathrm{actual}}
-
C_{s,j}^{\mathrm{pred}}
}{
C_{s,j}^{\mathrm{actual}}
}
\right|
\]

分别报告 Rank 0、Rank 1 和 Broadcast 路径的 MAPE。

---

## 17. 测试要求

### 17.1 单元测试

必须覆盖：

1. \(m=1\) 时严格回退；
2. 空任务列表；
3. 单任务列表；
4. 所有任务恰好填满容量；
5. 至少一个任务无法放入任何 slot；
6. 重排窗口为 0 时输出等于 FCFS；
7. 低置信度回退；
8. 预测收益不足回退；
9. 相邻交换改善目标；
10. 相邻交换违反容量时被拒绝；
11. 输出任务多重集合与输入完全一致；
12. 模拟器公式与手工计算结果一致。
13. \(T=2048,m=2\) 且安全上界为 1024 时只能生成
    \([1024,1024]\)；
14. \(T=2048,m=4\) 且安全上界为 512 时只能生成
    \([512,512,512,512]\)；
15. 存在同桶容量余量时能够生成非均匀候选；
16. 默认拒绝跨 shape 桶候选；
17. Broadcast 异步发起模型不会错误地令 Rank 0 每份都等待通信完成；
18. `optimize_capacities=False` 时仅优化成员和顺序。

### 17.2 正确性测试

固定随机种子，比较 SCOM 开关前后：

- 每个请求生成的 Token 一致；
- 请求输出数量一致；
- 请求与 KV Cache 映射一致；
- 不出现重复请求或遗漏请求；
- 不出现 block table 越界；
- 不出现 Broadcast 调用顺序不一致或死锁；
- 不出现额外 OOM；
- 关闭 SCOM 后恢复原行为。

### 17.3 Shadow Mode 测试

在不实际改变执行顺序时，至少运行：

- 3B、7B、14B；
- 短输入、长输入；
- 低 QPS、高 QPS；
- \(m=1\)、\(m=2\)、\(m>2\)。

比较预测 makespan 与实际 FCFS makespan，先验证模拟器。

---

## 18. 性能实验与 Baseline

### 18.1 Baseline

| 编号 | 方法 | 说明 |
|---|---|---|
| B0 | Uniform-FCFS | 等 Token 容量、原始成员与顺序 |
| B1 | Token-balanced | 等 Token 容量，仅按 Token 数量均衡成员 |
| B2 | Current Compute-Aware | 当前 scalar cost＋8-token quantum 成员重排 |
| B3 | Context-homogeneous | 按上下文长度相近分组 |
| B4 | SCOM-FixedCapacity | 等 Token 容量＋阶段向量＋Broadcast PP 时序优化 |
| B5 | SCOM-BucketSafe | bucket-safe 容量搜索＋完整 SCOM |

### 18.2 消融实验

| 名称 | 去除内容 |
|---|---|
| SCOM-Scalar | 将 Rank 0/Rank 1 向量退化为单一 cost |
| SCOM-NoBroadcast | 不使用 Broadcast 时间与等待，只考虑计算时间 |
| SCOM-NoOrder | 仅均衡成员，不优化有序 slot |
| SCOM-NoSwap | 去掉相邻成员交换 |
| SCOM-FixedCapacity | 关闭容量候选，只优化成员和顺序 |
| SCOM-AllowCliff | 允许跨 shape 桶，仅用于证明 shape 约束必要性 |
| SCOM-NoGate | 去掉收益门控，仅用于证明门控必要性 |

### 18.3 指标

- request throughput；
- output/total token throughput；
- TTFT；
- TPOT；
- E2E latency；
- batch makespan；
- Rank 0/Rank 1 idle ratio；
- PP bubble ratio；
- Broadcast 路径耗时；
- Broadcast 同步等待；
- SCOM 调度开销；
- 预测器 MAPE；
- 触发率与回退原因分布。

---

## 19. 创新可行性预检

在大规模改代码前，应先利用现有日志或 Shadow Mode 结果判断成员映射是否存在足够空间。

### 19.1 Rank 时间相关性

\[
\rho
=
\operatorname{corr}
\left(
C_{0,j},
C_{1,j}
\right)
\]

如果不同 micro-batch 在两个 rank 上的时间始终严格成比例，阶段向量会退化为标量。

### 19.2 随机映射敏感度

固定：

- 同一任务集合；
- 同一个 \(m\)；
- 同一组容量。

随机生成 100 个合法映射，计算：

\[
\Delta_{\mathrm{mapping}}
=
\frac{
T_{\max}-T_{\min}
}{
T_{\mathrm{FCFS}}
}
\]

建议判断：

| 结果 | 结论 |
|---|---|
| \(\Delta_{\mathrm{mapping}}<3\%\) | 成员映射空间较小，不应投入大量实现工作 |
| \(3\%\leq\Delta_{\mathrm{mapping}}<8\%\) | 可作为辅助优化 |
| \(\Delta_{\mathrm{mapping}}\geq8\%\) | 具备独立创新潜力 |

### 19.3 容量搜索空间与 Shape Cliff 预检

分别统计每个 \((T,m)\) 下：

- 等容量基线所在 shape 桶；
- 不跨桶时可生成的合法容量向量数量；
- 最优同桶非均匀容量相对等容量的预测收益；
- 允许跨桶后预测收益与实测收益的误差；
- 因 shape 桶约束被拒绝的候选比例。

若合法容量候选只有等容量一种，不表示 SCOM 失效，而表示该 step 的容量自由度
为零；此时只评估成员组成和 slot 顺序收益。论文应分别报告：

\[
G_{\mathrm{capacity}},\quad
G_{\mathrm{composition}},\quad
G_{\mathrm{order}}
\]

避免把成员重排收益误归因于非均匀 Token 数。

### 19.4 适用场景

SCOM 主要在以下条件下可能有效：

- \(m\geq2\)；
- 请求长度和 Prefill/Decode组成存在差异；
- Rank 0、Rank 1 对不同batch组成的响应不完全一致；
- Broadcast同步等待在总时延中占比较高；
- 高QPS或长输入产生足够多可重排任务单元。

对于大部分时间 \(m=1\) 的14B场景，SCOM应自动回退，不应人为制造收益。

---

## 20. 推荐实现顺序

### 阶段一：只读分析

1. 定位当前请求准入、micro-batch切分和Model Runner打包位置；
2. 从现有实验和补充 profile 中确定 3B、7B、14B 的 shape 桶边界；
3. 确认当前 `_PPBroadcastCommunicator` 的异步发起、延迟等待和跨 step
   handle 回收语义；
4. 确认需要同步重排的所有张量和元数据；
5. 不修改执行行为。

### 阶段二：模拟器与 Shadow Mode

1. 实现 `ScheduledUnit` 和特征提取；
2. 实现 FCFS 映射；
3. 实现 shape 桶表和 bucket-safe 容量候选器；
4. 实现阶段查表预测器；
5. 实现异步 Broadcast cost model；
6. 实现 Broadcast PP 模拟器；
7. 开启 Shadow Mode；
8. 验证预测准确性。

### 阶段三：真实成员映射

1. 实现贪心有序映射；
2. 实现相邻 slot 交换；
3. 先在 `optimize_capacities=False` 下验证仅成员/顺序映射；
4. 再开启 bucket-safe 容量候选；
5. 实现收益门控；
6. 实现安全回退；
7. 同步重排 Model Runner 输入；
8. 完成功能测试。

### 阶段四：实验

1. FCFS对照；
2. Token-balanced对照；
3. 当前 Compute-Aware 对照；
4. SCOM-FixedCapacity；
5. SCOM-BucketSafe；
6. 消融实验；
7. 分别记录容量、成员组成和顺序三部分收益；
8. 记录调度开销、预测误差和实际收益。

---

## 21. 完成标准

Codex 完成实现后，应满足：

- [ ] 默认关闭SCOM时行为与原代码一致；
- [ ] \(m=1\) 自动回退；
- [ ] SCOM不改变请求准入结果；
- [ ] 默认不跨越等容量基线所在的 shape 桶；
- [ ] 无同桶容量自由度时保持等容量；
- [ ] 关闭容量优化时仅改变成员映射和 slot 顺序；
- [ ] 所有任务单元恰好出现一次；
- [ ] 所有相关张量和元数据同步重排；
- [ ] 双rank Broadcast调用顺序一致；
- [ ] 模拟器符合异步 Broadcast 和延迟等待语义；
- [ ] 不出现死锁、请求错位或KV Cache错位；
- [ ] 支持Shadow Mode；
- [ ] 支持FCFS安全回退；
- [ ] 输出预测与实际时序日志；
- [ ] 映射开销可配置且默认不超过1 ms；
- [ ] 预计收益不足3%时不启用优化映射；
- [ ] 通过单元测试、正确性测试和性能测试。

---

## 22. 论文中的方法表述

可以表述为：

> 本文提出带shape档位约束的阶段向量与通信感知有序微批映射方法SCOM。
> 在请求准入和micro-batch数量确定后，SCOM首先以等Token切分为安全基线，
> 仅在不进入不利NPU算子shape档位的范围内生成有限容量候选；随后不再仅根据
> Token数量或单一综合cost均衡micro-batch，而是分别预测候选micro-batch
> 在不同流水级上的执行时间，并结合目标平台实测得到的异步Broadcast通信路径
> 模型，重建不同容量、成员组成和有序映射对应的完整PP执行时序。在shape档位、
> 显存、因果关系和公平性约束下，SCOM选择预计稳态完成时间最短的方案，从而
> 避免盲目非均匀切分导致的离散计算成本膨胀，同时缩短PP关键路径并减少通信等待。

需要严格保持以下创新边界：

- Sarathi-Serve和gLLM主要决定batch中Prefill/Decode Token的数量和比例；
- AlignedServe、BucketServe主要根据上下文或KV长度进行同质分组；
- KunServe使用单一执行cost构造均衡micro-batch；
- 当前compute-aware实现是SCOM的Scalar-cost基线，不作为可串联的独立规划器；
- SCOM的核心是“shape-safe容量候选＋逐流水级代价向量＋有序成员映射＋
  完整异步Broadcast PP时序评价”。

本文只基于Broadcast对SCOM进行实现和验证，不声称已经实现Send/Recv版本。
