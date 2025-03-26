import ast
from ..template import EvalTemplate
from ....data.data_utils import Role
from typing import Dict, List, Sequence, Tuple

class MCQA_Template(EvalTemplate):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.criteria_prompt = criteria_prompt
        self.response = response
        
    def _parse_example(self, example: Dict[str, str], choices: List[str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
        output: a tuple of (prompt, response)
        """
        question = "".join(['Question:\n' + 
                            example["question"]] + 
                           '\nOptions:\n' + 
                           [self.choice.format(choice=ch, content=example[ch]) for ch in choices if ch in example] +
                           [self.cot if use_cot else self.answer]
                           ).strip()
        answer = (((example.get("explanation") if use_cot and example.get("explanation") else "") + "\n" + 
                  self.response.format(answer=example.get("answer"))
                  ).strip() if use_cot else example["answer"])
        return question, answer

    def format_example(
        self, target_data: Dict[str, str], choices: List[str], support_set: Sequence[Dict[str, str]], subject_name: str
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        for k in range(len(support_set)):
            prompt, response = self._parse_example(support_set[k], choices)
            messages.append({"role": Role.USER.value, "content": prompt})
            messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data, choices)
        messages.append({"role": Role.USER.value, "content": prompt})
        messages.append({"role": Role.ASSISTANT.value, "content": response})
        messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages

    def format_inference_example(
        self, target_data: Dict[str, str], choices: List[str], support_set: Sequence[Dict[str, str]], subject_name: str, user_prompt: str, use_cot: bool
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        if support_set is not None:
            for k in range(len(support_set)):
                prompt, response = self._parse_example(support_set[k], choices)
                messages.append({"role": Role.USER.value, "content": prompt})
                messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data, choices, use_cot=use_cot)
        messages.append({"role": Role.USER.value, "content": prompt})
        if user_prompt is not None:
            assert "{subject}" in user_prompt, ValueError("User prompt must contain '{subject}'")
            messages[0]["content"] = user_prompt.format(subject=subject_name) + messages[0]["content"]
        else:
            messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages
    
    def format_checker_example(
        self, target_data: Dict[str, str], choices: List[str], llm_response: str, criteria_prompt:str
    ) -> Tuple[List[List[Dict[str, str]]], List[str]]:
        check_msg_list = []
        answer_list = []
        if criteria_prompt:
            assert "{option}" in criteria_prompt and \
                   "{question}" in criteria_prompt and \
                   "{llm_response}" in criteria_prompt \
                ,ValueError("Criteria prompt format incorrect, must contain '{option}', '{question}', and '{llm_response}'")
            self.criteria_prompt = criteria_prompt
        assert self.criteria_prompt is not None, ValueError("`criteria_prompt` should not be empty.")
        parsed_question, _ = self._parse_example(target_data, choices)
        if "choices" in target_data.keys():
            for candidate in ast.literal_eval(target_data["choices"]):
                check_msg_list.append([
                    {
                        "role": Role.USER.value, 
                        "content": self.criteria_prompt.format(
                            option=candidate,
                            question=parsed_question,
                            llm_response=llm_response,
                            )
                    }
                ])
                answer_list.append(
                    "True".lower() if candidate == target_data["answer"] else "False".lower()
                )
        else:
            for idx in range(len(choices)):

                check_msg_list.append([
                    {
                        "role": Role.USER.value, 
                        "content": self.criteria_prompt.format(
                            option=target_data[chr(ord("A") + idx)],
                            question=parsed_question,
                            llm_response=llm_response,
                            )
                    }
                ])
                answer_list.append(
                    "True".lower() if chr(ord("A") + idx) == target_data["answer"] else "False".lower()
                )
        return check_msg_list, answer_list