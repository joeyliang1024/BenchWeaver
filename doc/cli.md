# CLI Usage

This document provides an overview of the BenchWeaver CLI commands and their usage.

## Show CLI Usage

To display the help message and see all available commands, use:
```bash
bench-weaver-cli help
```

```
+----------------------------+-----------------------------------------+
| Command                    | Description                             |
+============================+=========================================+
| bench-weaver-cli eval -h   | Evaluate models                         |
+----------------------------+-----------------------------------------+
| bench-weaver-cli webui     | Launch Gradio Webui for eval (planning) |
+----------------------------+-----------------------------------------+
| bench-weaver-cli version   | Show version info                       |
+----------------------------+-----------------------------------------+
| bench-weaver-cli benchmark | Show supported benchmarks               |
+----------------------------+-----------------------------------------+
| bench-weaver-cli env       | Show dependency info                    |
+----------------------------+-----------------------------------------+
| bench-weaver-cli help      | Show CLI usage                          |
+----------------------------+-----------------------------------------+
```

## Evaluation Models

To evaluate models using BenchWeaver, you can use the following command:
```bash
CONFIG_PATH="example.yaml"
bench-weaver-cli eval --config $CONFIG_PATH
```
Parameters:
- `--config`: Path to the configuration file. In this example, `example.yaml` is the configuration file used.

Make sure to replace `example.yaml` with the path to your actual configuration file.

## Launch Gradio Webui (planning)

To launch the Gradio WebUI for evaluation (currently in planning), use:
```bash
bench-weaver-cli webui
```

## Show Supported Benchmarks

To display the list of supported benchmarks, use:
```bash
bench-weaver-cli benchmark
```

```
+-----------------------+------------+--------------------+---------------------+-------+
| Supported Benchmark   | Language   | Evaluators         |   Suggest num shots | Cot   |
+=======================+============+====================+=====================+=======+
| mmlu                  | en         | mcqa-prob, mcqa-oq |                   5 | True  |
+-----------------------+------------+--------------------+---------------------+-------+
| arc_challenge         | en         | mcqa-prob, mcqa-oq |                   5 | True  |
+-----------------------+------------+--------------------+---------------------+-------+
| gpqa                  | en         | opqa               |                   5 | True  |
+-----------------------+------------+--------------------+---------------------+-------+
| gsm8k                 | en         | opqa               |                   5 | True  |
+-----------------------+------------+--------------------+---------------------+-------+
| truthfulqa            | en         | mix                |                   0 | True  |
+-----------------------+------------+--------------------+---------------------+-------+
| big_bench_hard        | en         | mix                |                   3 | True  |
+-----------------------+------------+--------------------+---------------------+-------+
```

## Show Version Info

To display the current version of BenchWeaver, use:
```bash
bench-weaver-cli version
```

```
+------------------------------------------------------------+
| Welcome to BenchWeaver, version 0.0.0                      |
|                                                            |
| Project page: https://github.com/joeyliang1024/BenchWeaver |
+------------------------------------------------------------+
```

## Show Dependency Info

To display information about the dependencies used by BenchWeaver, use:
```bash
bench-weaver-cli env
```

```
- `BenchWeaver` version: 0.0.0
- Platform: Linux-5.15.0-107-generic-x86_64-with-glibc2.35
- Python version: 3.11.11
- PyTorch version: 2.5.1+cu124 (GPU)
- Transformers version: 4.47.1
- Datasets version: 3.2.0
- OpenAI version: 1.58.1
- PyYAML version: 6.0.2
- vLLM version: 0.6.6.post1
- Bitsandbytes version: 0.45.2
- GPU type: NVIDIA L40S
```
