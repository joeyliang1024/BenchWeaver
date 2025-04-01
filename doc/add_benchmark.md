# Adding a Benchmark  

To add a new benchmark to the pipeline, you need to complete the following steps:

1. [Formatting Your Data](#formatting-your-data)
2. [New Evaluation Template](#new-evaluation-template)
3. [New Evaluators](#new-evaluators)

## Formatting Your Data  

### 1. Create a Benchmark Folder  
- Inside the [`evaluation_data`](../evaluation_data/) directory, create a new folder named after your benchmark.  

### 2. Organize Your Dataset  
- Your dataset should be split into the following structure:  

  ```bash
  benchmark_name/
  ├── data/
  │   ├── dev/   # Optional
  │   ├── val/   # Optional
  │   └── test/
  ```  

- Each dataset file should follow the naming convention:  
  ```bash
  {subject_name}_{split}.csv
  ```  
  **Example:**  
  ```bash
  data/dev/accounting_dev.csv
  data/val/accounting_val.csv
  data/test/accounting_test.csv
  ```

### 3. Compress Your Dataset  
- Zip the `data/` folder inside your benchmark folder. (You can delete the `data` folder after zipped.) 

### 4. Create a `mapping.json` File  
- This file should define metadata for each subject in your benchmark.  
- The format follows:  

  ```json
  {
    "abstract_algebra": {
      "name": "abstract algebra",
      "category": "STEM"
    },
    "anatomy": {
      "name": "anatomy",
      "category": "Other"
    }
  }
  ```
  - **`name`**: Customizable subject name.  
  - **`category`**: Used for grouping in the final output score.  

### 5. Implement a Dataset Loader  
- Create a Python script named `{benchmark_name}.py` inside the benchmark folder.  
- Use the [`datasets`](https://pypi.org/project/datasets/) library to load your benchmark data.  

### Final Folder Structure  
After completing the steps, your benchmark folder should look like this:  

```bash
evaluation_data/
└── benchmark_name/
    ├── mapping.json
    ├── benchmark_name.zip
    └── benchmark_name.py
```

## New Evaluation Template

Some evaluations have their own evaluation templates and data loading formats. If the default evaluation template is not supported, you can define your own.

### Basic Evaluation Templates

- **[MCQA_Template](../src/BenchWeaver/eval/template/eval/mcqa_template.py)**: Inherit this class if your benchmark follows the MCQA format.
- **[OPQA_Template](../src/BenchWeaver/eval/template/eval/opqa_template.py)**: Inherit this class if your benchmark follows the OPQA format.
- **[Template](../src/BenchWeaver/eval/template/template.py)**: Use this if your data format is neither MCQA nor OPQA.

### Adding a New Benchmark Template to Config

To register a new evaluation template, update the `EVAL_TEMPLATE_CONFIG` dictionary as follows:

```python
EVAL_TEMPLATE_CONFIG = {
    "mmlu": {
        "class": MCQA_Template,
        "func": get_mmlu_eval_template,
    },
}
```
- **`class`**: The benchmark template class.
- **`func`**: The function used to load the benchmark template based on the template name (typically defined by the template language).

## New Evaluators

Similar to evaluation templates, you can define your own evaluator if the default ones are not supported.

### Basic Evaluators

#### MCQA Evaluators
- **[Open Question Output](../src/BenchWeaver/eval/evaluator/mcqa/oq_evaluator.py)**: Inherit this class if your benchmark follows the MCQA format and the model's output fully answers the question.
- **[Probability Output](../src/BenchWeaver/eval/evaluator/mcqa/prob_evaluator.py)**: Inherit this class if your benchmark follows the MCQA format and the model outputs logits.

#### Other Evaluators
- **[OPQA Evaluator](../src/BenchWeaver/eval/evaluator/opqa/opqa_evaluator.py)**: Inherit this class if your benchmark follows the OPQA format.
- **[Evaluator](../src/BenchWeaver/eval/evaluator/evaluator.py)**: Use this if your data format is neither MCQA nor OPQA.

### `load_data` Method

If the default data loading method does not support your benchmark, override or define the `load_data` method in your evaluator class.

```python
def load_data(self, mode: Literal['inference', 'check', 'translation'], choices: List[str], responses_trans: bool = False, check_source: Literal['original', 'translated'] = "original") -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Load data based on the specified mode. This should be defined by your benchmark.

    Args:
        mode (Literal['inference', 'check', 'translation']): The mode of data loading.
        choices (List[str]): A list of choices relevant to data selection.

    Returns:
        Union[Tuple[None, Dict[str, list]], Tuple[Dict[str, list], Dict[str, list]]]:
            - If mode is "inference": (None, inference_messages)
            - If mode is "check": (checked_answers, checked_messages)
            - If mode is "translation": (translation_ground_truth, translation_messages)
    """
    pass
```

### Adding a New Benchmark Evaluator to Config

To register a new evaluator, update the [`BENCHMARK_CONFIG`](../src/BenchWeaver/eval/benchmarks/configs.py#L5) dictionary as follows:

```python
BENCHMARK_CONFIG = {
    "mmlu": {
        "language": "en",
        "mode": ["mcqa-prob", "mcqa-oq"],
        "display_scores": ["Average", "STEM", "Social Sciences", "Humanities", "Other"],
        "mcqa_choices": ["A", "B", "C", "D"],
        "sugguest_num_shots": 5,
        "support_chain_of_thought": True,
    }
}
```
- **`language`**: Original language of the benchmark.
- **`mode`**: Supported evaluation modes.
- **`display_scores`**: Categories for final score display, should match values in **category** from `mapping.json`.
- **`mcqa_choices`**: Option codes for MCQA benchmarks. Use `OPTION_CODES` for variable-length MCQA or `None` for OPQA.
- **`sugguest_num_shots`**: Default number of shots for the benchmark.
- **`support_chain_of_thought`**: Indicates whether Chain of Thought (CoT) reasoning is supported.

### Updating `get_evaluators`

Also, update the [`get_evaluators`](../src/BenchWeaver/eval/benchmarks/configs.py#L129) function:

```python
def get_evaluators(task_name: str) -> Dict[str, Any]:
    """
    Dynamically import evaluators for the specified task. This is used only for CLI to display supported benchmarks.
    """
    if task_name == "mmlu":
        from .en.mmlu import MMLUOQEvaluator, MMLUProbEvaluator
        return {
            "mcqa-prob": MMLUProbEvaluator,
            "mcqa-oq": MMLUOQEvaluator,
        }
```