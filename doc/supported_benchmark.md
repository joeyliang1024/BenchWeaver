# Supported Benchmark

The table below lists the benchmarks supported by our system along with various features and configurations. Here's a brief explanation of each column:

- **Benchmark**: The name of the benchmark.
- **Support**: Indicates whether the benchmark is supported (✅) or not (❌).
- **MCQA-OQ**: Multi-Choice Question Answering with Open Questions support.
- **MCQA-Prob**: Multi-Choice Question Answering with Probability support.
- **OPQA**: Open-Ended Question Answering support.
- **Mix**: Mixed type questions support.
- **Chain of thought**: Indicates if the benchmark supports chain of thought reasoning.
- **Num Shot**: The number of shots (examples) used for few-shot learning.

| Benchmark      | Support | MCQA-OQ | MCQA-Prob | OPQA | Mix | Chain of thought | Num Shot |
|----------------|:-------:|:-------:|:---------:|:----:|:---:|:----------------:|:--------:|
| MMLU           |    ✅   |    ✅    |     ✅     |  ❌  |  ❌  |        ✅        |     5    |
| ARC Challenge  |    ✅   |    ✅    |     ✅     |  ❌  |  ❌  |        ✅        |     5    |
| GPQA           |    ✅   |    ❌    |     ❌     |  ✅  |  ❌  |        ✅        |     5    |
| GSM8K          |    ✅   |    ❌    |     ❌     |  ✅  |  ❌  |        ✅        |     5    |
| TruthfulQA     |    ✅   |    ❌    |     ❌     |  ❌  |  ✅  |        ❌        |     0    |
| Big Bench Hard |    ✅   |    ❌    |     ❌     |  ❌  |  ✅  |        ❌        |     3    |

