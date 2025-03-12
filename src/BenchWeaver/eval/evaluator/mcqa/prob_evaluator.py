import os
import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, List
from transformers import PreTrainedModel
from ..evaluator import Evaluator
from ...template.eval.mcqa_template import MCQA_Template
from ....data.template import get_template_and_fix_tokenizer
from ....extras.constants import PROJECT_BASE_PATH
from ....inference.transformers.prob_model import load_model
from ....inference.transformers.tokenizer import load_tokenizer

if TYPE_CHECKING:
    from numpy.typing import NDArray
    
class ProbEvaluator(Evaluator):
    choice_inputs: List[int]
    eval_template: MCQA_Template
    model: PreTrainedModel
    def __init__(self, args):
        super().__init__(args=args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.model_args, self.finetuning_args)
        
    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, "torch.Tensor"]) -> List[str]:
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]
    
    @staticmethod
    def compute_score(category_corrects: Dict[str, "NDArray"]):
        return {category_name: round(100 * np.mean(category_array), 4) 
                for category_name, category_array in category_corrects.items()}
    
    def save_results(self, category_corrects: Dict[str, "NDArray"], results: Dict[str, Dict[int, str]]) -> None:
        if self.eval_args.save_dir is not None:
            os.makedirs(os.path.join(PROJECT_BASE_PATH, self.eval_args.save_dir), exist_ok=False)
            self.save_data(results, os.path.join(PROJECT_BASE_PATH, self.eval_args.save_dir, "results.json"))
            self.save_data(self.compute_score(category_corrects), os.path.join(PROJECT_BASE_PATH, self.eval_args.save_dir, "scores.json"))
        
        