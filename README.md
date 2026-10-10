# bw1100-bench

面向 Hygon BW1100 / gfx938 的 12 题正确性工作集：SOL-ExecBench **L1 7 题、L2 5 题，
完整保留 192 个原始 workload**。用于后续 AITER/Triton/HIP 算子实现与迁移研究。
默认每个 workload 生成 10 轮新输入；完整一轮题集共 1920 个正确性 round。

这是独立的 BW1100 正确性入口。它复用固定版本 SOL-ExecBench 的原始参考实现、
输入生成器和数值比较器，不提供 NVIDIA SOL 分数、CUDA/CUPTI 计时或模型性能结论。
题目与工作量的规范来源是 [suite.json](suite.json) 和
[sources.lock.json](sources.lock.json)。

原题 `reference.py` 只是真值来源，**新加的两项 GEMM 还没有完成 gfx938 baseline 资格验证**。
逐题社区实现、语义缺口和当前设备资格见
[baselines/README.md](baselines/README.md)。已接入的适配器可作为下文
`--candidate` 文件接受完全相同的原始正确性检查；尚未通过整题设备门的实现
不能充当强基线或速度比的分母。

## 选题

难度 1–5 是对迁移和优化工作的人工分层，不是实测排名；L1/L2 是上游分类，
不等于所有 L1 都比 L2 容易。每题均保留 16 个 workload。

| 题目 ID | 实际内容 | 难度 | 覆盖的挑战 |
|---|---|---:|---|
| L1/069 | Residual + RMSNorm | 1 | 内存带宽、归约、BF16 舍入 |
| L1/011 | Llama3 RoPE 频率缩放 | 2 | 整数位置、分段缩放、三角函数 |
| L1/048 | 双投影 + GELU-tanh 门控 | 3 | GEMM 与 epilogue 融合；题名虽含 swiglu，参考实现不是 SiLU |
| L1/058 | 专家稳定排序与前缀和 | 3 | 整数精确性、稳定顺序、histogram/scan |
| L1/001 | GQA attention backward | 4 | softmax/dropout 反向、跨组归约、两个梯度输出 |
| L2/035 | ConvNeXtV2 + GRN | 3 | FP32 视觉负载、深度卷积、布局转换、多种归约 |
| L2/018 | 变长视觉 attention | 4 | cu_seqlens、head_dim=72、RoPE 和投影 |
| L2/024 | 256 专家 MoE dispatch/compute/combine | 4 | top-8、不规则 gather/scatter、grouped GEMM、大权重 |
| L2/060 | Chunk gated delta-rule attention | 5 | chunk=64、尾部 padding、三角更新、递归状态 |
| L2/056 | 完整 decoder layer backward | 5 | 十个梯度输出、混合 dtype、attention/MLP/norm 反向组合 |
| L1/003 | BF16 LM head GEMM | 2 | K=2048、N=102400，大词表规则N和不规则M |
| L1/077 | FP16 Whisper output GEMM | 3 | K=1280、N=51866，decode小M、N尾块及大M |

两项 GEMM 的原始尺寸、精度边界及资格状态见 [GEMM additions](docs/GEMM-ADDITIONS-2026-10-09.md)。

完整题名、选择理由和最小输入规模的原始 smoke workload UUID 在 `suite.json`。
Smoke 使用上游真实 shape，不缩小固定维度。L2/024 单份输入约 **12 GiB**，
正确性验证还需要参考副本、输出及中间结果；应在设备上单独安排。

## 数据与安装

原始题库使用 NVIDIA Evaluation Dataset License，评测器代码使用 Apache-2.0。
因此本仓库只保存来源标识、选题清单和测试工具。定义、reference、workload 和
Parquet 下载到被 Git 忽略的 `.data/`；不上传到 GitHub，包括私有仓库。
具体边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。

在 Python 3.12 的准备环境中：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-prepare.txt
.venv/bin/python scripts/prepare.py
.venv/bin/python bwbench.py audit --output .local/audit.json
```

`prepare.py` 固定源码与数据修订，在 `.deps/sol-execbench` 安装源码，在 `.data/benchmark/`
生成 12 道题的完整输入定义与参考程序。重复运行核对已有内容，遇到不同内容会停止，
不会重置已有仓库。离线准备可使用 `--cache-dir <包含 L1.parquet、L2.parquet 的目录>`。
Suite v2 的准备回执为 `.data/materialization-suite-v2.json`，已有十题的
`.data/materialization.json` 保留原样，扩展题集不会因旧回执的任务数量不同而失败。

宿主机仅查看题单不需要 Torch：

```bash
python3 bwbench.py list
```

完整上游 `sol-execbench` CLI 要求 Python ≥3.12，并依赖 CUDA 13、CUPTI 等 NVIDIA 包。
本项目导入的是其 correctness/data 子集，不在 DTK 环境中执行 `uv sync` 或替换海光版
PyTorch。运行正确性入口需要 Python ≥3.10、可用的 PyTorch 和 Pydantic v2；厂商环境
应先做 import/CPU 预检，不能把上游整包视为已经获得 Hygon 支持。

## 正确性使用

候选是一个 Python 文件，导出 `run(*inputs)`，参数顺序、返回值名称/顺序与该题
`definition.json` 一致。它可以在内部调用 Triton、AITER 或已构建的 HIP 扩展。
本入口采用 return-value ABI；没有另造一份 kernel 语义。

```bash
# 在 CPU 环境检查参考程序与测试入口，不能作为 GPU 资格结果
python bwbench.py check --task L1/069_rms_norm --device cpu \
  --reference-selfcheck --workloads smoke --rounds 2 \
  --output results/rmsnorm-cpu-smoke.json

# 在已获分配的 gfx938 环境检查候选：16 个 workload × 10 轮
python bwbench.py check --task L1/069_rms_norm --device cuda:0 \
  --candidate /path/to/candidate.py --output results/rmsnorm-device.json
```

`--workloads` 接受 `all`、`smoke` 或单个原始 UUID；默认 `all`。`--rounds` 默认 10。
输出路径必须新建；遇到首个生成、reference、候选执行或比较错误即停止并保存原因。
只有候选在 gfx938 上跑完本题全部原始 workload 和 10 轮，才写
`full_device_correctness=true`。CPU、参考程序自检、smoke 和少轮测试都不会获得这个标记。
某一题通过不等于整套题集全部通过。

测试规则、已知阈值字段差异及补充的整数门见 [docs/CORRECTNESS.md](docs/CORRECTNESS.md)。

## 独立 DTK/HCU 执行入口

本仓库的 GPU 路径由 `scripts/dtk.sh` 和 `scripts/hcu_run.py` 自己管理，
不依赖其他项目的源码、Executor 或 admission。CPU 与 GPU 均使用所在机器已经核对过的
DTK 镜像及临时 `docker run --rm`；GPU 模式以新 JSON 收据记录本任务本地锁、
选定 HCU、镜像 digest、执行命令、容器终态和运行前后占用。详细合同见
[docs/DTK-ADMISSION.md](docs/DTK-ADMISSION.md)。

```bash
# CPU 审计，不申请 HCU
bash scripts/dtk.sh cpu IMAGE python3 bwbench.py audit --output .local/container-audit.json

# 仅在实时确认 HCU1 空闲后，执行一项有界正确性检查
HIP_VISIBLE_DEVICES=1 BWBENCH_TIMEOUT=180 \
  bash scripts/dtk.sh gpu IMAGE results/rmsnorm-admission.json \
  python3 bwbench.py check --task L1/069_rms_norm --device cuda:0 \
  --candidate /work/examples/l1_069_torch_baseline.py \
  --workloads smoke --rounds 2 --output results/rmsnorm-device.json
```

GPU 模式仍需 DTK runtime、root/privileged 容器身份及 `/dev/kfd`、`/dev/dri`、
`/dev/mkfd` 和只读 `/opt/hyhal` 挂载；这解决的是当前镜像的设备入口，
不是全机器排他分配。启动前脚本检查所选 HCU 显存和可见 KFD 进程，并用当前用户
的本地锁串行化本套件作业；其他用户活动仍须标记为 `not_excluded`。
遇到无终态收据、SSH 中断或设备未释放时先观察实际容器与 HCU，不能盲目重试。
历史设备检查的原始路径和结论仍保留在
[docs/DEVICE-VALIDATION-2026-10-01.md](docs/DEVICE-VALIDATION-2026-10-01.md)，
它们不自动转成新入口的验证结果。
新入口在 `bw1100-1` 的独立设备检查见
[docs/STANDALONE-DEVICE-VALIDATION-2026-10-01.md](docs/STANDALONE-DEVICE-VALIDATION-2026-10-01.md)。
后续 Ralph 算子优化若要判断 GPU 瓶颈，按
[docs/RALPH-PROFILING.md](docs/RALPH-PROFILING.md) 通过本仓库入口采集
`rocprof` 诊断收据；计时分数仍使用无 profiler 的配对测量。

## 固定版本的新一轮搜索

[Fresh Bench launch](docs/FRESH-BENCH-LAUNCH.md) 说明如何将 evolve 冻结的十二题、
两个 Compiler 条件准备为独立 fresh author Run，并通过现有 HCU gateway 执行。
该入口只运行本轮 Bench；结果审阅后由 evolve owner 选择版本并启动开发任务。

[Compiler tools for hmz](docs/COMPILER-TOOLS.md) 说明新冻结 Run 如何从各自 Compiler
自动获得变换清单、调用已有 Lab 解析器，并保留可回放的父候选、参数与生成结果。
Bench 和 53 题开发共用这条作者接口，继续使用各自原有评测和终态。

## 软件验证

在独立 CPU 开发环境安装测试依赖并运行：

```bash
uv pip install --python .venv/bin/python -r requirements-cpu-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖数值错误、NaN/Inf、最大误差上限、shape/dtype、输出名称、整数精确比较、
参考输入隔离、多轮新输入、首错停止及 CPU/部分测试不能升级为设备资格。
实际验证范围见 [docs/VALIDATION.md](docs/VALIDATION.md)。
