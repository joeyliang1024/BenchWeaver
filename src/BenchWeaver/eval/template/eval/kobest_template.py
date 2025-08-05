import json
import ast
from typing import Dict, List, Literal, Sequence, Tuple, Union
from ..template import EvalTemplate, _register_eval_template
from ....data.data_utils import Role
from ....extras.constants import OPTION_CODES

class KoBest_Template(EvalTemplate):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.mcqa_criteria_prompt = "당신은 평가 모델이며, 하나의 객관식 질문, LLM 응답, 그리고 하나의 선택지 {option}을 받게 됩니다.  \n당신의 임무는 질문에 답하는 것이 아니라, LLM 응답에서 {option}이 명확하게 선택된 답변 중 하나인지 판단하는 것입니다.  \n\n질문: {question}  \n\nLLM 응답: {llm_response}  \n\n판단 기준:  \n\n{option}이 LLM 응답에서 명확하고 직접적으로 선택된 답변으로 표현되었거나, LLM 응답이 선택지(A, B, C, D 등) 또는 {option}만 포함하는 경우 'True'를 답하십시오.  \n\n{option}이 선택되지 않았거나, LLM 응답이 명확한 답을 표현하지 않았다면 'False'를 답하십시오.  \n\nLLM 응답이 비어 있거나, {option}이 선택된 답변인지 판단할 수 없다면 'Unknown'을 답하십시오."
        self.opqa_criteria_prompt = 'LLM 응답이 질문에 올바르게 답하는지 판단하십시오.\n\n질문: {question}\n\n참고 정답: {answer}\n\nLLM 응답: {llm_response}\n\nLLM 응답이 올바르면 \'True\'를, 그렇지 않으면 \'False\'를 응답하십시오.'
        self.response = response

    def _parse_boolq(self, example: Dict[str, str], judgement:bool=False) -> Tuple[str, str]:
        question = (
            example["paragraph"] +
            "질문:\n" +
            example["question"] + 
            "답변:\n"   
            )
        answer = example["answer"]
        return question, answer.lower().strip() if judgement else answer
    
    def _parse_copa(self, example: Dict[str, str], judgement:bool=False) -> Tuple[str, str]:
        question = (
            example['question'] +
            f"\n (A) {example['A']}" +
            f"\n (B) {example['B']}" +
            "답변: "
        )
        answer = self.response.format(answer=example["answer"], content=example[example["answer"]])
        return question, example["answer"] if judgement else answer

    def _parse_hellaswag(self, example: Dict[str, str], judgement:bool=False) -> Tuple[str, str]:
        question = (
            example['question'] +
            f"\n (A) {example['A']}" +
            f"\n (B) {example['B']}" +
            f"\n (C) {example['C']}" +
            f"\n (D) {example['D']}" +
            "답변: "
        )
        answer = self.response.format(answer=example["answer"], content=example[example["answer"]])
        return question, example["answer"] if judgement else answer

    def _parse_wic(self, example: Dict[str, str], judgement:bool=False) -> Tuple[str, str]:
        question = example['question']
        answer  = example['answer']
        return question, answer.lower().strip() if judgement else answer

    def _parse_sentineg(self, example: Dict[str, str], judgement:bool=False) -> Tuple[str, str]:
        question = (
            example['question'] +
            "\n (A) Positive" +
            "\n (B) Negative"
        )
        option_code = "A" if example['answer']== "Positive" else "B"
        answer = self.response.format(answer=option_code, content=example['answer'])
        return question, option_code if judgement else answer

    def _parse_example(self, type:str, example: Dict[str, str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        if type == "boolq":
            return self._parse_boolq(example, judgement=kwargs.get("judgement", False))
        elif type == "copa":
            return self._parse_copa(example, judgement=kwargs.get("judgement", False))
        elif type == "hellaswag":
            return self._parse_hellaswag(example, judgement=kwargs.get("judgement", False))
        elif type == "wic":
            return self._parse_wic(example, judgement=kwargs.get("judgement", False))
        elif type == "sentineg":
            return self._parse_sentineg(example, judgement=kwargs.get("judgement", False))
        else:
            raise ValueError(f"Value type `{type}` is not supported.")
    
    def format_inference_example(
         self, type:str, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], user_prompt: str, use_cot: bool, **kwargs
    ) -> List[Dict[str, str]]:
        
        messages = []
        if support_set is not None:
            for k in range(len(support_set)):
                prompt, response = self._parse_example(type=type, example=support_set[k], use_cot=use_cot)
                messages.append({"role": Role.USER.value, "content": prompt})
                messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, _ = self._parse_example(type=type, example=target_data, use_cot=use_cot)
        messages.append({"role": Role.USER.value, "content": prompt})
        return messages
    
    def format_checker_example(
        self, type:str, target_data: Dict[str, str], llm_response: str, **kwargs
    ) -> Union[List[Dict[str, str]],
               Tuple[List[List[Dict[str, str]]], List[str]]]:
        # MCQA format: copa, hellaswag, sentineg
        # OPQA format: boolq, wic
        if type in ["boolq", "wic"]:
               return [
                    {
                        "role": Role.USER.value, 
                        "content": self.opqa_criteria_prompt.format(
                            answer=target_data['answer'],
                            question=target_data['question'],
                            llm_response=llm_response,
                            )
                    }
                ]
        
        elif type in ["copa", "hellaswag", "sentineg"]:
            check_msg_list = []
            answer_list = []
            num_choices = 4 if type == "hellaswag" else 2
            for idx in range(num_choices):
                _, response = self._parse_example(type=type, example=target_data, judgement=True)
                check_msg_list.append([
                    {
                        "role": Role.USER.value, 
                        "content": self.mcqa_criteria_prompt.format(
                            option=OPTION_CODES[idx],
                            question=target_data['question'],
                            llm_response=llm_response
                        )
                    }
                ])
                answer_list.append("True".lower() if response == chr(ord("A") + idx) else "False".lower())
            return check_msg_list, answer_list
        
        else:
           raise ValueError(f"Value type `{type}` is not supported.")
        

kobest_eval_templates: Dict[str, "KoBest_Template"] = {}

def get_kobest_eval_template(name: str) -> "KoBest_Template":
    eval_template = kobest_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


_register_eval_template(
    name="ko",
    system="질문: \n\n",
    choice="\n{choice}. {content}",
    answer="\n정답:",
    cot="\n차근차근 생각해 봅시다.\n정답:",
    templates=kobest_eval_templates,
    template_class=KoBest_Template,
    criteria_prompt="",
    response="정답은 ({answer}) {content}입니다."
)