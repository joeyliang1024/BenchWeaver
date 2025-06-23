# BenchWeaver 🏆🚀🐍
Welcome to BenchWeaver! 🎉🔬 This Python project provides a specialized benchmarking pipeline, supporting various models and benchmarks. ⚙️🔧📈

This is the official repository of the master's thesis:  

**"BenchWeaver: An Automated Multilingual Evaluation Framework with LLM-as-Judge and Translation Prompting for Low-Resource Languages"**

## Pipeline Overview
The pipeline overview is demonstrate as follows:
![Pipeline Overview](/assets/img/overall_pipeline.png)
> [!WARNING]  
> Some of the benchmarks have custome settings, check configs or support benchmarks for more details.

## Installation 💻⚡
Create a new conda environment and install the package:
```bash
conda create --name BenchWeaver python=3.11 -y
pip install unbabel-comet
pip install -e .
```
> [!WARNING]  
> Package `unbabel-comet` should be installed before `pip install -e .`

> [!NOTE]
> After installation, remember to create `env/tokens.env` for saving your HuggingFace and OpenAI's configuration.
>
> For example:
> ```
> HF_TOKEN=""
> AZURE_ENDPOINT_URL=""
> AZURE_OPENAI_API_KEY=""
> AZURE_API_VERSION=""
> OPENAI_API_KEY=""
> OPENAI_ORGANIZATION=""
> OPENAI_PROJECT=""
> ```

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


## Reproducibility of Results
For pipeline execution, you can run the configurations for each part as listed below:

| Chapter         | Experiment/Detail        | Configuration Link           |
| :-------------- | :----------------------- | :--------------------------- |
| Main Result     | -                        | [Main Result](/config/main_pipeline/) |
| Ablation Study  | Translation Prompt       | [Translation Prompt](/config/trans_template_exp/) |
| Ablation Study  | Compare with P-MMEval    | [Compare with P-MMEval](/config/pmmeval_exp/) |

For checking the translation quality, you can execute the following code for reproduce:
```bash
bash scripts/bash/eval_trans.sh
```