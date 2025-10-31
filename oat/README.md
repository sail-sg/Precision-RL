
## Introduction

This directory contains code built upon [**Oat**](https://github.com/sail-sg/oat) for studying the impact of **numerical precision** in **LLM reinforcement learning (RL)**.
Oat is implemented with **DeepSpeed**, which natively supports **FP16 training** via a simple configuration change.

The provided `main.py` script implements several RL algorithms, including:

* **IS-PG** – Vanilla policy gradient with importance sampling
* **Token-TIS** – Token-level truncated importance sampling
* **Seq-MIS** – Sequence-level masked importance sampling
* **GSPO** – Group Sequence Policy Optimization
* **(Dr)GRPO** – *Done-right* Group Relative Policy Optimization

We provide fine-grained argument configurations, enabling flexible combinations of algorithmic components to study their trade-offs in **stability**, **efficiency**, and **performance**.

**Relevant flags:**

| Category           | Flags / Options                              |
| ------------------ | -------------------------------------------- |
| Precision          | `--bf16`, `--no-bf16`, `--fp16`, `--no-fp16` |
| Sequence-level IS  | `--use_seq_is`, `--seq_is_clip_max`          |
| Token-level IS     | `--use_token_is`, `--token_is_clip_max`      |
| GSPO | `--use_gspo`                                 |


## Getting Started

We recommend using a **Conda environment** with **Python 3.10**.

```bash
pip install vllm==0.8.4
pip install -U oat-llm

# Patch LD_LIBRARY_PATH to avoid dependency errors.
# You may add this line to your .bashrc or .zshrc for convenience:
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
```

After installation, you can start running experiments from the `/scripts` directory.
