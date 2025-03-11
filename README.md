# BenchWeaver

## Installation
### python installation
```bash
conda create --name BenchWeaver python=3.11 -y
pip install -e .
```
### conda installation (Not Done)
This installation will create a env as well
```bash
conda env create -f environment.yaml
```

## CLI Usage
### Evaluation models
To evaluate models using BenchWeaver, you can use the following command:
```bash
bench-weaver-cli eval \
    --task mmlu \
    --mode $MODE \
    --pipeline $PIPELINE \
    --config $CONFIG_PATH
```
Parameters:
- `--task`: Specifies the task to evaluate. In this example, `mmlu` is the task.
- `--mode`: Specifies the mode of evaluation. You can check by [this](#show-supported-benchmarks).
- `--config`: Path to the configuration file. In this example, `example.yaml` is the configuration file used.
- `--pipeline`: Indicate whether to run the same language or different language evaluation.

Make sure to replace `example.yaml` with the path to your actual configuration file.

### Launch Gradio Webui (planning):
```bash
bench-weaver-cli webui
```
### Show Supported Benchmarks
```bash
bench-weaver-cli benchmark
```
### Show Version Info
```bash
bench-weaver-cli version
```
### how Dependency Info
```bash
bench-weaver-cli env
```
### Show CLI Usage
```bash
bench-weaver-cli help
```
## Evaluation Data

### Upload Format

### Load from HuggingFace