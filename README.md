# bw1100-bench

面向 Hygon BW1100 / gfx938 的 10 题正确性工作集：SOL-ExecBench **L1、L2 各 5 题，
完整保留 160 个原始 workload**。用于后续 AITER/Triton/HIP 算子实现与迁移研究。
默认每个 workload 生成 10 轮新输入；完整一轮题集共 1600 个正确性 round。

这是独立的 BW1100 正确性入口。它复用固定版本 SOL-ExecBench 的原始参考实现、
输入生成器和数值比较器，不提供 NVIDIA SOL 分数、CUDA/CUPTI 计时或模型性能结论。
题目与工作量的规范来源是 [suite.json](suite.json) 和
[sources.lock.json](sources.lock.json)。

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
生成 10 道题的完整输入定义与参考程序。重复运行核对已有内容，遇到不同内容会停止，
不会重置已有仓库。离线准备可使用 `--cache-dir <包含 L1.parquet、L2.parquet 的目录>`。

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
某一题通过不等于十题全部通过。

测试规则、已知阈值字段差异及补充的整数门见 [docs/CORRECTNESS.md](docs/CORRECTNESS.md)。

## 使用现有 DTK 镜像与 Cake 经验

Cake 已记录 `docker exec` 的 HCU socket 拒绝与临时 `docker run --rm` 的区别，
也已有 Hygon 本地串行 admission。先复用现有镜像和这条执行方式；详见
[docs/CAKE.md](docs/CAKE.md)。`scripts/dtk.sh` 是临时容器入口，不是 GPU 分配器。

```bash
# IMAGE 使用所在机器已有、核对过的厂商镜像
bash scripts/dtk.sh cpu IMAGE python3 bwbench.py audit --output .local/container-audit.json

# 设备已由现有分配流程确定后，将可见设备传给临时容器
HIP_VISIBLE_DEVICES="$ALLOCATED_HIP_DEVICE" bash scripts/dtk.sh gpu IMAGE \
  python3 bwbench.py check --task L1/069_rms_norm --device cuda:0 \
  --candidate /work/my_candidate.py --output results/device-rmsnorm.json
```

容器使用当前宿主 UID 写工作目录、只读根文件系统和只读 hyhal 挂载；临时目录可执行。
它不运行模型服务，不自动分配或抢占设备，也不把本地锁表述为物理独占。

## 软件验证

在独立 CPU 开发环境安装测试依赖并运行：

```bash
uv pip install --python .venv/bin/python -r requirements-cpu-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖数值错误、NaN/Inf、最大误差上限、shape/dtype、输出名称、整数精确比较、
参考输入隔离、多轮新输入、首错停止及 CPU/部分测试不能升级为设备资格。
实际验证范围见 [docs/VALIDATION.md](docs/VALIDATION.md)。
