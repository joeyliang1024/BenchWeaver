# Supported Benchmark

The table below lists the benchmarks supported by our system along with various features and configurations. Here's a brief explanation of each column:

- **Supported Benchmark**: The benchmark name should use in the config.
- **Language**: The language of the origin benchmark, fill in to the `lang` column in the config.
- **Evaluators**: The `benchmark_mode` values correspond to the evaluation method used for each benchmark.
  - **mcqa-prob**: Uses the probability of the first token to predict the answer for multiple-choice questions (MCQA).
  - **mcqa-oq**: Applies LLM-as-a-judge to evaluate the full output of MCQA responses.
  - **opqa**: Used for benchmarks with open-ended answers.
  - **mix**: Combines both MCQA (mcqa-oq) and open-ended (opqa) question types.
  - **trans**: Used for benchmarks focused on translation quality.
  - **code**: Designed for evaluating code-related benchmarks.
- **Suggest num shots**: The number of shots (examples) suggest to use for few-shot inference.
- **Cot (Chain of thought)**: Indicates if the benchmark supports chain of thought reasoning.



| Supported Benchmark       | Language | Evaluators              | Suggest num shots | Cot   |
|---------------------------|----------|--------------------------|-------------------|-------|
| mmlu                      | en       | mcqa-prob, mcqa-oq       | 5                 | True  |
| arc-challenge             | en       | mcqa-prob, mcqa-oq       | 5                 | True  |
| gpqa                      | en       | opqa                     | 5                 | True  |
| gsm8k                     | en       | opqa                     | 5                 | True  |
| truthfulqa                | en       | mix                      | 0                 | True  |
| big-bench-hard            | en       | mix                      | 3                 | True  |
| hellaswag                 | en       | mcqa-prob, mcqa-oq       | 0                 | False |
| ifeval                    | en       | opqa                     | 0                 | False |
| flores-plus               | en       | trans                    | 0                 | False |
| mbpp                      | en       | code                     | 3                 | False |
| xnli                      | en       | opqa                     | 0                 | False |
| logiqa                    | en       | mcqa-oq                  | 0                 | False |
| humaneval-xl              | en       | code                     | 0                 | False |
| click                     | ko       | mcqa-oq                  | 0                 | False |
| hae-rae-bench             | ko       | mix                      | 0                 | False |
| kmmlu                     | ko       | mcqa-prob, mcqa-oq       | 5                 | False |
| kmmlu-hard                | ko       | mcqa-prob, mcqa-oq       | 5                 | True  |
| cmmlu                     | zh       | mcqa-prob, mcqa-oq       | 5                 | False |
| ccpm                      | zh       | mcqa-prob, mcqa-oq       | 0                 | False |
| cmath                     | zh       | opqa                     | 0                 | False |
| cif-bench                 | zh       | opqa                     | 0                 | False |
| c3                        | zh       | mcqa-oq                  | 0                 | False |
| chinese-safety-qa         | zh       | mcqa-oq                  | 0                 | False |
| tmmluplus                 | zh-tw    | mcqa-prob, mcqa-oq       | 5                 | False |
| tmlu                      | zh-tw    | mcqa-oq                  | 5                 | True  |
| drcd                      | zh-tw    | opqa                     | 5                 | False |
| awesome-taiwan-knowledge  | zh-tw    | opqa                     | 0                 | False |
| mt-bench-tw               | zh-tw    | multi-turn               | 0                 | False |
| taide-bench               | zh-tw    | opqa                     | 0                 | False |
