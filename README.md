# BenchWeaver 🏆🚀🐍
Welcome to BenchWeaver! 🎉🔬 This Python project provides a specialized benchmarking pipeline, supporting various models and benchmarks. ⚙️🔧📈

## Installation 💻⚡

| Method        | Description                                               | Status          |
|---------------|-----------------------------------------------------------|-----------------|
| Python (pip)  | Clean installation using conda environment and pip        | ✅ Ready        |
| Conda         | One-step installation using environment.yaml              | 🚧 In Progress  |

### Python Installation 🐍
Create a new conda environment and install the package:
```bash
conda create --name BenchWeaver python=3.11 -y
pip install unbabel-comet
pip install -e .
```
> [!WARNING]  
> Package `unbabel-comet` should be installed before `pip install -e .`

### Conda Installation (In Progress) 🏗️
Direct installation using conda environment file:
```bash
conda env create -f environment.yaml
```

## Documentation 📚📝
Access detailed documentation through these links:

| Component      | Description                         | Link                                         |
|----------------|-------------------------------------|----------------------------------------------|
| CLI            | Command-line interface guide        | [CLI](./doc/cli.md)                          |
| Config         | Evaluation configuration details    | [Config](./doc/config_doc.md)                |
| Evaluation     | Methods and metrics explanation     | [Evaluation Method](./doc/evaluation_method.md) |
| Benchmarks     | List of supported benchmarks        | [Support Benchmark](./doc/supported_benchmark.md) |
| Benchmark Type | List of benchmark types             | [Benchmark Classification](./doc/benchmark_classification.md) |
| Add benchmark  | Details how to add benchmark        | [Add Benchmark](./doc/add_benchmark.md)      |
| Problem Record | Record the problem occured          | [Problem Record](./doc/problem_record.md)    |


