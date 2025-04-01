### OpenAI Source

| Parameter       | Value                                                     |
|-----------------|-----------------------------------------------------------|
| openai_source   | `azure` or `openai`                                       |

### Inference Model

| Parameter                      | Value                                      |
|--------------------------------|--------------------------------------------|
| inference_model_name_or_path   | taide/Llama3-TAIDE-LX-8B-Chat-Alpha1       |
| inference_mode                 | `local` or `api`                           |
| cot                            | `true` for chain for thought               |

### Checker Model

| Parameter                      | Value                                      |
|--------------------------------|--------------------------------------------|
| checker_model_name_or_path     | Qwen/Qwen2.5-32B-Instruct                  |
| check_mode                     | `local` or `api`                           |

### Translation Model

| Parameter                      | Value                                      |
|--------------------------------|--------------------------------------------|
| translation_model_name_or_path | gpt-4o-mini                                |
| translation_mode               | `local` or `api`                           |
| transation_templates_name      | test (name in [translation_prompt.json](../prompt/translation_prompt.json))     |
| source_lang                    | en                                         |
| target_lang                    | zh-tw                                      |

### Dataset

| Parameter                      | Value                                      |
|--------------------------------|--------------------------------------------|
| task                           | mmlu_test                                  |
| task_dir                       | evaluation_data                            |
| template                       | fewshot                                    |
| lang                           | en (template name for benchmark)           |
| n_shot                         | 5                                          |
| benchmark_mode                 | mcqa-oq (check it by `bench-weaver-cli benchmark`) |
| pipeline                       | `diff` or `same`                           |
| batch_size                     | 4                                          |

### Output

| Parameter                      | Value                                      |
|--------------------------------|--------------------------------------------|
| save_dir                       | score/mmlu/diff_lang/pipeline-testing      |

### VLLM

| Parameter                      | Value                                      |
|--------------------------------|--------------------------------------------|
| vllm_maxlen                    | 8192                                       |
| vllm_max_concurrency           | 100 (asynco max concurrenc for local vllm) |
| dtype                          | bfloat16                                   |

### Debug

| Parameter                      | Value                                      |
|--------------------------------|--------------------------------------------|
| testing_size                   | 5                                          |
| record_all                     | `true` for record all the tings            |
| print_param_status             | false (commented out)                      |