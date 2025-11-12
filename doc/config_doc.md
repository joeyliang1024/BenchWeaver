# Configuration Documentation

This document details all available arguments for configuring the inference, and evaluation processes. They are grouped by the Python files that define them.

## Model Arguments

Arguments pertaining to the model, config, tokenizer, quantization, and export.
Check [model_args.py](/src/BenchWeaver/hparams/model_args.py) for more details.

---
### Main Model Arguments
| Argument                      | Type                                                      | Default         | Description                                                                                                     |
|--------------------------------|-----------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------------------|
| output_reasoning               | bool                                                      | False            | Whether or not to output the reasoning content in generation.                                                   |
| dtype                          | Literal["float16", "bfloat16", "float32"]                | "bfloat16"       | Data type for model weights and activations at inference.                                                       |
| print_param_status             | bool                                                      | False            | For debugging purposes, print the status of the parameters in the model.                                        |

---
### vLLM Arguments

| Argument                | Type                  | Default | Description                                                                                     |
|--------------------------|-----------------------|----------|-------------------------------------------------------------------------------------------------|
| vllm_maxlen              | int                   | 4096     | Maximum sequence (prompt + response) length of the vLLM engine.                                 |
| vllm_gpu_util            | float                 | 0.95     | The fraction of GPU memory in (0,1) to be used for the vLLM engine.                             |
| vllm_enforce_eager       | bool                  | False    | Whether or not to disable CUDA graph in the vLLM engine.                                        |
| vllm_max_lora_rank       | int                   | 32       | Maximum rank of all LoRAs in the vLLM engine.                                                   |
| vllm_max_concurrency     | int                   | 100      | Maximum number of concurrent requests for the vLLM server.                                      |
| vllm_disable_log_requests| bool                  | True     | Whether or not to disable logging of requests.                                                  |
| vllm_disable_log_stats   | bool                  | True     | Whether or not to disable logging of statistics.                                                |
| vllm_trust_remote_code   | bool                  | True     | Whether or not to trust the remote code when loading the model.                                 |
| vllm_reasoning_parser    | Optional[str]         | None     | The name of the reasoning parser to use in the vLLM engine.                                     |
| vllm_chunked_prefill     | bool                  | False    | Whether or not to enable chunked prefill in the vLLM engine.                                    |
| vllm_swap_space          | float                 | 0.0      | Size of the CPU swap space per GPU (in GiB). Disables the use of CPU swap space.                |
| vllm_engine_ver          | Literal[0, 1]         | 0        | Version of the vLLM engine to use.                                                              |

---
### Inference Arguments
| Argument                      | Type                                           | Default   | Description                                                                                                                   |
|-------------------------------|------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------------------------|
| inference_model_name_or_path  | Optional[str]                                  | None       | (Required) Path to the inference model weight or identifier from huggingface.co/models or modelscope.cn/models.               |
| inference_model_endpoint      | Optional[str]                                  | None       | The endpoint of the inference model.                                                                                          |
| inference_mode                | Literal["api", "local", "endpoint"]            | "local"    | Mode for the inference model.                                                                                                 |


---
### Checker Arguments
| Argument                     | Type                                           | Default   | Description                                                                                                         |
|-------------------------------|------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------|
| checker_model_name_or_path    | Optional[str]                                  | None       | Path to the checker model weight or identifier from huggingface.co/models or modelscope.cn/models.                 |
| checker_model_endpoint        | Optional[str]                                  | None       | The endpoint of the checker model.                                                                                  |
| check_mode                    | Literal["api", "local", "endpoint"]            | "local"    | Mode for the checker model.                                                                                         |

---
### Translator Arguments
| Argument                     | Type                                           | Default   | Description                                                                                                         |
|-------------------------------|------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------|
| translation_model_name_or_path| Optional[str]                                  | None       | Path to the translator model weight or identifier from huggingface.co/models or modelscope.cn/models.              |
| translation_model_endpoint    | Optional[str]                                  | None       | The endpoint of the translator model.                                                                               |
| translation_mode              | Literal["api", "local", "endpoint"]            | "local"    | Mode for the translator model.                                                                                      |
| transation_templates_name     | Optional[str]                                  | None       | Name of the translation templates.                                                                                  |
| source_lang                   | Optional[str]                                  | None       | Source language for translation.                                                                                    |
| target_lang                   | Optional[str]                                  | None       | Target language for translation.                                                                                    |

> [!NOTE]  
> Only when using **FLORES Benchmark** do the `source_lang` and `target_lang` arguments need to be explicitly set to the corresponding FLORES language codes to determine the translation direction.


---
### OpenAI Arguments
| Argument      | Type                       | Default  | Description                      |
| ------------- | -------------------------- | -------- | -------------------------------- |
| openai_source | Literal["openai", "azure"] | "openai" | The OpenAI source for inference. |

## Generating Arguments

Arguments pertaining to specify the decoding parameters.
Check [generating_args.py](/src/BenchWeaver/hparams/generating_args.py) for more details.

| Argument              | Type          | Default | Description                                                                              |
| --------------------- | ------------- | ------- | ---------------------------------------------------------------------------------------- |
| temperature           | float         | 0       | The value used to modulate the next token probabilities.                                 |
| top_p                 | float         | 1       | The smallest set of most probable tokens with cumulative probabilities ≥ top_p are kept. |
| top_k                 | int           | 100     | The number of highest probability vocabulary tokens to keep for top-k filtering.         |
| max_length            | int           | 4096    | The maximum total length of generated tokens. Can be overridden by `max_new_tokens`.     |
| max_new_tokens        | int           | 4096    | The maximum number of tokens to generate, ignoring the prompt length.                    |
| max_completion_tokens | int           | 100000  | The maximum number of tokens to generate in a single chat completion.                    |
| repetition_penalty    | float         | 1.0     | The penalty factor for repeated tokens; 1.0 means no penalty.                            |
| length_penalty        | float         | 1.0     | Exponential penalty applied to sequence length during beam-based generation.             |
| default_system        | Optional[str] | None    | Default system message to include in chat completions.                                   |

## Evaluation Arguments

Arguments pertaining to specify the evaluation parameters.
Check [evaluation_args.py](/src/BenchWeaver/hparams/evaluation_args.py) for more details.

| Argument               | Type                                                                          | Default                 | Description                                                                      |
| ---------------------- | ----------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------- |
| task                   | str                                                                           | **Required**            | Name of the evaluation task.                                                     |
| ref_task               | Optional[str]                                                                 | None                    | Name of the reference task for few-shot translation.                             |
| task_dir               | str                                                                           | `"evaluation_data"`     | Path to the folder containing evaluation datasets or a HuggingFace dataset name. |
| ref_task_dir           | Optional[str]                                                                 | None                    | Path to the folder containing reference datasets.                                |
| batch_size             | int                                                                           | 4                       | Batch size per GPU used during evaluation.                                       |
| seed                   | int                                                                           | 42                      | Random seed for data loading and reproducibility.                                |
| lang                   | Literal["en", "zh", "zh-tw", "ko"]                                            | `"en"`                  | Language used for the evaluation prompts.                                        |
| n_shot                 | int                                                                           | 5                       | Number of exemplars for few-shot learning.                                       |
| save_dir               | Optional[str]                                                                 | None                    | Directory to save evaluation results.                                            |
| download_mode          | DownloadMode                                                                  | REUSE_DATASET_IF_EXISTS | Mode for downloading or reusing evaluation datasets.                             |
| system_prompt          | str                                                                           | None                    | System prompt for the open question inference model.                             |
| user_prompt            | str                                                                           | None                    | User-provided prompt or query for the open question inference model.             |
| criteria_system_prompt | str                                                                           | None                    | System prompt for the open question checker model.                               |
| criteria_prompt        | str                                                                           | None                    | User-defined evaluation criteria or guidelines for the checker model.            |
| cot                    | bool                                                                          | False                   | Enable or disable  to use Chain-of-Thought (CoT) template for inference.                |
| benchmark_mode         | Literal["trans", "code", "multi-turn", "mcqa-prob", "mcqa-oq", "opqa", "mix"] | `"opqa"`                | Evaluation benchmark mode.                                                       |
| pipeline               | Literal["same", "diff"]                                                       | None                    | Indicates whether to run same-language or cross-language evaluation.             |
| testing_size           | Optional[int]                                                                 | 1_000_000_000           | Number of examples to evaluate. If None, evaluates the full dataset.             |
| record_all             | bool                                                                          | False                   | Record all intermediate reasoning steps.                                         |
| debug                  | bool                                                                          | False                   | Enable debug mode for verbose logging and error tracking.                        |

> [!NOTE]  
> The `task_dir` and `ref_task_dir` arguments can now also be set to a **HuggingFace dataset repository**.
> Check [support benchmark](/doc/supported_benchmark.md) for corresponding repo ID.