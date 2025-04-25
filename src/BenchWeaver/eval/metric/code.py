import os
from typing import List
from evaluate import load
from BenchWeaver.extras.constants import PROJECT_BASE_PATH

def eval_code(predictions: List[str], references: List[List[str]], k: int = 1)->dict:
    """
    Evaluates code generation using the CodeEval metric.
    Args:
        predictions (List[str]): List of generated code strings.
        references (List[List[str]]): List of reference code strings.
        k (int): Number of top-k predictions to consider.
    Returns:
        dict: Evaluation results including accuracy and other metrics.
    """
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    code_eval = load(os.path.join(PROJECT_BASE_PATH, "src/BenchWeaver/eval/metric/code_utils.py"))
    return code_eval.compute(predictions=predictions, references=references, k=[k])