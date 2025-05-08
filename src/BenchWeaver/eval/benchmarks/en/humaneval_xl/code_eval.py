import os
from typing import Any, Dict, List
from datasets import load_dataset
from tqdm import tqdm
from ....evaluator.code.code_evaluator import CodeEvaluator
from ....template import get_humaneval_xl_eval_template
from ....metric.mxeval import evaluate_functional_correctness
from .....extras.constants import PROJECT_BASE_PATH

class HumanEvalXLEvaluator(CodeEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_humaneval_xl_eval_template(self.eval_args.lang)
        
    def format_scoring_input(self, response_result: Dict[str, List[Any]]):
        '''
        Sample Input: list of dict with keys: 'task_id', 'completion', 'language'
        Problem Input: list of dict with keys: 'task_id', 'prompt',  'entry_point', 'test', 'description', 'language', 'canonical_solution'
        '''
        sample_input, problem_input = [], []
        for subject in tqdm(self.categories.keys(), desc="Formatting MXEval Input"):
            dataset = load_dataset(
                    path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.eval_task),
                    name=subject,
                    cache_dir=self.model_args.cache_dir,
                    download_mode=self.eval_args.download_mode,
                    token=self.hf_token,
                    trust_remote_code=True,
                )
            for idx in range(min(len(dataset[self.eval_split]), self.testing_size)):
                sample_input.append({
                    "task_id": dataset[self.eval_split][idx]["task_id"],
                    "completion": response_result[subject][idx][0],
                    "language": str(dataset[self.eval_split][idx]["language"]).lower(),
                })
                problem_input.append({
                    "task_id": dataset[self.eval_split][idx]["task_id"],
                    "prompt": dataset[self.eval_split][idx]["prompt"],
                    "entry_point": dataset[self.eval_split][idx]["entry_point"],
                    "test": dataset[self.eval_split][idx]["test"],
                    "description": dataset[self.eval_split][idx]["description"],
                    "language": str(dataset[self.eval_split][idx]["language"]).lower(),
                    "canonical_solution": dataset[self.eval_split][idx]["canonical_solution"]
                })
        return sample_input, problem_input
    
    def comput_score(self, test_codes: Dict[str, List[Any]], response_result: Dict[str, List[Any]], k: int = 1):
        '''
        Compute the score of all programs languages
        '''
        # Format the data to desired format
        sample_input, problem_input = self.format_scoring_input(response_result)
        return evaluate_functional_correctness(
            sample_file=sample_input,
            problem_file=problem_input,
            output_dir=self.save_folder,
            k=[k]
        )
        