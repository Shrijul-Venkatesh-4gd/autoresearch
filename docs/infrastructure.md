# Infrastructure Requirements And Dependencies

This document explains what this repo needs from your machine and software environment before it can run correctly. In simple terms, `autoresearch` is not just a Python script that can run anywhere. It expects a fairly specific setup because it is built around short, high-throughput single GPU training runs.

## Hardware requirements

The main hardware requirement here is a **single NVIDIA GPU**. The README explicitly says the repo was tested on an **H100**, which means the default settings are tuned for a high end GPU and will likely be too heavy for smaller machines without modification.

That said, the repo is still written to run on non-Hopper NVIDIA GPUs as well. In `train.py`, the code checks the GPU capability and chooses an attention kernel accordingly. Hopper GPUs use one Flash Attention 3 path, while non-Hopper NVIDIA GPUs fall back to another kernel source. So the hard requirement is not specifically "an H100", but rather "a CUDA capable NVIDIA GPU".

Along with the GPU, you should also expect the need for:

- enough VRAM to hold the default model and batch sizes
- enough system RAM to handle dataset download, parquet reading and tokenization
- enough local disk space for cached dataset shards and tokenizer artifacts

If your machine is smaller than the default target setup, the README already suggests reducing values like `DEPTH`, `MAX_SEQ_LEN`, `TOTAL_BATCH_SIZE` and `EVAL_TOKENS`.

## Software requirements

The minimum software requirements are straightforward:

- Python `3.10+`
- `uv` as the project and dependency manager
- NVIDIA drivers and a working CUDA setup compatible with the installed PyTorch build

The repo installs PyTorch through the `pytorch-cu128` index in `pyproject.toml`, which means this project expects a CUDA-enabled PyTorch build and not a CPU-only install.

## Python dependencies

The Python dependencies for this repo are defined in `pyproject.toml`. These are:

- `torch==2.9.1`
- `kernels>=0.11.7`
- `numpy>=2.2.6`
- `pandas>=2.3.3`
- `matplotlib>=3.10.8`
- `pyarrow>=21.0.0`
- `requests>=2.32.0`
- `rustbpe>=0.1.0`
- `tiktoken>=0.11.0`

At a high level, these dependencies serve the following roles:

- `torch` powers the model, optimizer and training loop
- `kernels` is used to fetch the Flash Attention style kernel used by the training code
- `pyarrow` is used to read the parquet dataset shards
- `requests` is used to download the dataset shards
- `rustbpe` is used to train the tokenizer
- `tiktoken` is used as the runtime tokenizer interface
- `numpy`, `pandas` and `matplotlib` are mainly useful for analysis and result inspection

## Storage and caching

This repo stores its downloaded and generated assets outside the repo itself in:

- `~/.cache/autoresearch/data`
- `~/.cache/autoresearch/tokenizer`

This means you need:

- permission to write into your user cache directory
- enough disk space for the dataset shards you choose to download
- enough disk space for tokenizer artifacts such as `tokenizer.pkl` and `token_bytes.pt`

The repo does not bundle the dataset in Git. It downloads the data on first setup and then reuses the cached copy.

## Network requirements

You need network access during setup because `prepare.py` downloads the dataset from Hugging Face. The training run itself mainly works from the local cache once the dataset and tokenizer are already prepared.

There is also an indirect network expectation during dependency installation because `uv sync` needs to fetch the Python packages defined in `pyproject.toml`.

## Runtime assumptions

There are a few infrastructural assumptions built into the code:

- training is **single GPU only**
- the code path assumes CUDA, not CPU or MPS
- training runs are designed around a fixed 5 minute wall clock budget
- the repo expects prepared tokenizer and dataset artifacts to already exist before `train.py` runs

So in practical terms, the usual setup flow is:

1. install `uv`
2. run `uv sync`
3. run `uv run prepare.py`
4. run `uv run train.py`

If step 3 has not been done yet, the training phase will not have the data or tokenizer artifacts it depends on.

## Optional but important

There is one more dependency that is not a Python package dependency in the normal sense: an **external coding agent**. The autonomous research loop described in `program.md` assumes you will point a tool like Codex or Claude at this repo. The repo itself does not spawn that agent automatically. So if you only want to run a single training experiment manually, the agent is optional. If you want the full autoresearch workflow, the agent becomes part of the broader infrastructure.

## In short

To run this repo comfortably, you should think in terms of the following stack:

- a CUDA capable NVIDIA GPU
- Python 3.10+
- `uv`
- the dependencies from `pyproject.toml`
- network access for installation and initial dataset download
- writable local disk space for the cache
- optionally, an external coding agent for the autonomous experiment loop
