# Sources and distribution boundary

- Evaluation framework: NVIDIA/SOL-ExecBench, Apache-2.0. The exact revision is
  recorded in `sources.lock.json`. Its unmodified correctness and input-generation
  modules are loaded from the separately prepared checkout in `.deps/sol-execbench`.
- Dataset: `nvidia/SOL-ExecBench` on Hugging Face, NVIDIA Evaluation Dataset License.
  The pinned license URL is in `sources.lock.json`. Sections 1 and 3 distinguish
  internal evaluation/benchmarking from redistribution or hosting of the dataset.
  This repository does not include the dataset, references, input specifications,
  workloads, or Parquet files. `scripts/prepare.py` obtains them from their owner
  into ignored local storage and retains the original license there.
- Selection descriptions and difficulty estimates are this project's editorial
  choices. IDs and URLs identify upstream resources; no measured performance or
  official NVIDIA leaderboard qualification is claimed.
- Community operator baseline: FlagOS/FlagGems 5.4.0dev, Apache-2.0, taken
  from the existing FlagRelease Hygon image. `scripts/prepare_flag_gems.py`
  accepts only the recorded source-archive hash and materializes it under
  ignored `.deps/`. This repository commits only its own thin adapter, source
  hashes and verification metadata; it does not redistribute FlagGems source.

Do not commit `.data/`, `.deps/`, materialized problems, raw input tensors, model
weights, or private machine inspection output. Publishing benchmark results does
not grant permission to redistribute the underlying dataset.
