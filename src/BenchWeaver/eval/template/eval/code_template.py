from typing import Dict, List, Sequence, Tuple
from ..template import EvalTemplate
from ....data.data_utils import Role

class Code_Template(EvalTemplate):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        super().__init__(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt, response=response)
     
    def _parse_example(self, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        'text', 'code', 'test_list', 'test_setup_code'
        """
        prompt = self.system.format(
            prompt=example['text'],
            tests="\n".join(example["test_list"]),
        )
        # format question
        answer = "\n".join(["[BEGIN]", example["code"], "[DONE]"])
        return prompt, answer
    
    def format_inference_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        # for few shot
        if support_set is not None:
            for k in range(len(support_set)):
                prompt, response = self._parse_example(support_set[k])
                messages.append({"role": Role.USER.value, "content": prompt})
                messages.append({"role": Role.ASSISTANT.value, "content": response})
        # format object question
        prompt, response = self._parse_example(target_data)
        messages.append({"role": Role.USER.value, "content": prompt})
        return messages, target_data['test_list']
    