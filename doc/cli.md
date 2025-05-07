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
+--------------------------+------------+--------------------+---------------------+-------+
| Supported Benchmark      | Language   | Evaluators         |   Suggest num shots | Cot   |
+==========================+============+====================+=====================+=======+
| mmlu                     | en         | mcqa-prob, mcqa-oq |                   5 | True  |
+--------------------------+------------+--------------------+---------------------+-------+
| arc-challenge            | en         | mcqa-prob, mcqa-oq |                   5 | True  |
+--------------------------+------------+--------------------+---------------------+-------+
| gpqa                     | en         | opqa               |                   5 | True  |
+--------------------------+------------+--------------------+---------------------+-------+
| gsm8k                    | en         | opqa               |                   5 | True  |
+--------------------------+------------+--------------------+---------------------+-------+
| truthfulqa               | en         | mix                |                   0 | True  |
+--------------------------+------------+--------------------+---------------------+-------+
| big-bench-hard           | en         | mix                |                   3 | True  |
+--------------------------+------------+--------------------+---------------------+-------+
| click                    | ko         | mcqa-oq            |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| hae-rae-bench            | ko         | mix                |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| kmmlu                    | ko         | mcqa-prob, mcqa-oq |                   5 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| kmmlu-hard               | ko         | mcqa-prob, mcqa-oq |                   5 | True  |
+--------------------------+------------+--------------------+---------------------+-------+
| tmmluplus                | zh-tw      | mcqa-prob, mcqa-oq |                   5 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| tmlu                     | zh-tw      | mcqa-oq            |                   5 | True  |
+--------------------------+------------+--------------------+---------------------+-------+
| drcd                     | zh-tw      | opqa               |                   5 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| awesome-taiwan-knowledge | zh-tw      | opqa               |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| cmmlu                    | zh         | mcqa-prob, mcqa-oq |                   5 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| taide-bench              | zh-tw      | opqa               |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| ccpm                     | zh         | mcqa-prob, mcqa-oq |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| cmath                    | zh         | opqa               |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| cif-bench                | zh         | opqa               |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| c3                       | zh         | mcqa-oq            |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| chinese-safety-qa        | zh         | mcqa-oq            |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| mt-bench-tw              | zh-tw      | multi-turn         |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| hellaswag                | en         | mcqa-prob, mcqa-oq |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| ifeval                   | en         | opqa               |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| flores-plus              | en         | trans              |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| mbpp                     | en         | code               |                   3 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| xnli                     | en         | opqa               |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| logiqa                   | en         | mcqa-oq            |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
| humaneval-xl             | en         | code               |                   0 | False |
+--------------------------+------------+--------------------+---------------------+-------+
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
