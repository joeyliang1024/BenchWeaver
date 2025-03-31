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
pip install -e .
```

### Conda Installation (In Progress) 🏗️
Direct installation using conda environment file:
```bash
conda env create -f environment.yaml
```

## Documentation 📚📝
Access detailed documentation through these links:

| Component   | Description                         | Link                                         |
|-------------|-------------------------------------|----------------------------------------------|
| CLI         | Command-line interface guide        | [CLI](./doc/cli.md)                          |
| Config      | Evaluation configuration details    | [Config](./doc/config_doc.md)               |
| Evaluation  | Methods and metrics explanation     | [Evaluation Method](./doc/evaluation_method.md) |
| Benchmarks  | List of supported benchmarks        | [Support Benchmark](./doc/supported_benchmark.md) |

## Other Tips
### Chat template
1. Origin ` google/gemma-2-27b-it` do not support system prompt, please edit the `tokenizer_config.json` template column as follows:
   ```jinja
   {{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n' }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != ((loop.index0 + 1) % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{% if message['role'] != 'system' %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}
   ```