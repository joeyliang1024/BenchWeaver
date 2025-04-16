import ast
from ..template import EvalTemplate
from ....data.data_utils import Role
from typing import Dict, List, Tuple

class Multi_Turn_Template(EvalTemplate):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str, **kwargs):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.criteria_prompt = criteria_prompt
        self.response = response
        
    def _parse_example(self, example: Dict[str, str], **kwargs) -> Tuple[list, list]:
        question_turns = ast.literal_eval(example["question_turns"])
        answer_turns = ast.literal_eval(example["answer_turns"])
        return question_turns, answer_turns
    
    def get_inference_quesitons(self, example: Dict[str, str], **kwargs) -> List[str]:
        """
        Get the inference questions from the example.
        """
        question_turns, _ = self._parse_example(example=example)
        return question_turns
    
    def format_inference_example(
        self, target_data: Dict[str, str], history: List[dict], **kwargs
    ) -> List[Dict[str, str]]:
        """
        Format the inference example for multi-turn evaluation.
        """
        # count the number of turns in the history
        turns_idx = 0 if (history == [] or history is None) else sum([1 for turn in history if turn["role"] == Role.ASSISTANT])
        # get the question and answer turns
        question_turns, _ = self._parse_example(example=target_data)
        history.append({
            "role": Role.USER,
            "content": question_turns[turns_idx]
        })
        return history
        

    def format_checker_example(
        self, target_data: Dict[str, str], history: List[dict], **kwargs
    ) -> List[Dict[str, str]]:
        assert self.criteria_prompt is not None, ValueError("`criteria_prompt` should not be empty.")
        assert "{ref_block}" in self.criteria_prompt and \
               "{assistant_block}" in self.criteria_prompt \
                , ValueError("Criteria prompt format incorrect, must contain '{ref_block}' and '{assistant_block}'")
                
        # get the question and answer turns
        question_turns, answer_turns = self._parse_example(example=target_data)
        # get the assistant response
        assistant_turns = [turn["content"] for turn in history if turn["role"] == Role.ASSISTANT]
        # format ref_block
        ref_block = "\n".join(
            f"User:\n{q}\n\nReference answer:\n{r}" for q, r in zip(question_turns, answer_turns)
        )

        assistant_block = "\n".join(
            f"User:\n{q}\n\nAssistant:\n{a}" for q, a in zip(question_turns, assistant_turns)
        )
        return [
                {
                    "role": Role.USER.value, 
                    "content": self.criteria_prompt.format(
                        ref_block=ref_block,
                        assistant_block=assistant_block
                        )
                }
            ]