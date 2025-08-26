import numpy as np
from tqdm.auto import tqdm
from typing import Any, Dict, List
from .....evaluator import OPQAEvaluator
from .....template import get_industryinstruction_law_eval_template

class IndustryInstructionLawEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_industryinstruction_law_eval_template(self.eval_args.lang)

    def comput_score(self, check_results: Dict[str, List[Any]], subjects: List[str], checked_answers=None) -> Dict[str, float]:
        category_corrects = {subj: [] for subj in subjects}

        for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
            category_name = self.categories[subject]["category"]
            for check_string in check_results[subject]:
                score = self.retrieve_answer(text=check_string, numerical=True)
                category_corrects[category_name].append(score)
                category_corrects['Average'].append(score)
         
        # average score
        for subject in category_corrects.keys():
            category_corrects[subject] = np.mean(category_corrects[subject])

        return {category_name: round(score, 4) for category_name, score in category_corrects.items()}