from typing import Dict, List, Sequence, Tuple
from ..template import EvalTemplate
from ....data.data_utils import Role

class OPQA_Template(EvalTemplate):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.criteria_prompt = criteria_prompt
        self.response = response
        
    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", "answer", ...}
        output: a tuple of (prompt, response)
        """
        # format question
        question = (example["question"] + (self.cot if use_cot else self.answer)).strip()
        # format answer
        answer = ((example.get("explanation") if use_cot and example.get("explanation") else "") + "\n" + 
                  self.response.format(answer=example.get("answer"))).strip()

        return question, answer
    
    def format_inference_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], subject_name: str, user_prompt: str, use_cot: bool
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        # for few shot
        if support_set is not None:
            for k in range(len(support_set)):
                prompt, response = self._parse_example(support_set[k], use_cot)
                messages.append({"role": Role.USER.value, "content": prompt})
                messages.append({"role": Role.ASSISTANT.value, "content": response})
        # format object question
        prompt, response = self._parse_example(target_data, use_cot)
        messages.append({"role": Role.USER.value, "content": prompt})
        # for additional settings
        subject = target_data.get("category") if target_data.get("category") else subject_name
        if user_prompt is not None:
            assert "{subject}" in user_prompt, ValueError("User prompt must contain '{subject}'")
            messages[0]["content"] = user_prompt.format(subject=subject) + messages[0]["content"]
        else:
            messages[0]["content"] = self.system.format(subject=subject) + messages[0]["content"]
        return messages

    def format_checker_example(
        self, target_data: Dict[str, str], llm_response: str, criteria_prompt:str
    ) -> List[Dict[str, str]]:
        if criteria_prompt:
            assert "{answer}" in criteria_prompt and \
                   "{question}" in criteria_prompt and \
                   "{llm_response}" in criteria_prompt \
                ,ValueError("Criteria prompt format incorrect, must contain '{answer}', '{question}', and '{llm_response}'")
            self.criteria_prompt = criteria_prompt
        assert self.criteria_prompt is not None, ValueError("`criteria_prompt` should not be empty.")
        return [
                {
                    "role": Role.USER.value, 
                    "content": self.criteria_prompt.format(
                        answer=target_data['answer'],
                        question=target_data['question'],
                        llm_response=llm_response,
                        )
                }
            ]