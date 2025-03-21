import json
import ast
from typing import Dict, List, Sequence, Tuple, Union
from ..template import EvalTemplate, _register_eval_template
from ....data.data_utils import Role

class HAE_RAE_BENCH_Template(EvalTemplate):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.mcqa_criteria_prompt = '다음 LLM 응답이 "{option}"을(를) 객관식 질문의 답으로 포함하는지 판단하세요.\n\n질문: {question}\n\nLLM 응답: {llm_response}\n\n"{option}"이(가) LLM 응답에서 명확하게 선택된 답이면 \'True\'를, 그렇지 않으면 \'False\'를 답하세요.'
        self.opqa_criteria_prompt = 'LLM 응답이 질문에 올바르게 답하는지 판단하십시오.\n\n질문: {answer}\n\n참고 정답: {question}\n\nLLM 응답: {llm_response}\n\nLLM 응답이 올바르면 \'True\'를, 그렇지 않으면 \'False\'를 응답하십시오.'
        
    def _parse_example(self, example: Dict[str, str], use_cot: bool=False) -> Tuple[str, str]:
        return example["question"], example["answer"]
    
    def format_inference_example(
         self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], user_prompt: str, use_cot: bool
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        if support_set is not None:
            for k in range(len(support_set)):
                prompt, response = self._parse_example(example=support_set[k], use_cot=use_cot)
                messages.append({"role": Role.USER.value, "content": prompt})
                messages.append({"role": Role.ASSISTANT.value, "content": response})
        
        prompt, response = self._parse_example(example=target_data, use_cot=use_cot)
        messages.append({"role": Role.USER.value, "content": prompt})
        return messages
    
    def format_checker_example(
        self, target_data: Dict[str, str], choices:List[str], llm_response: str) -> Union[
            List[Dict[str, str]],
            Tuple[List[List[Dict[str, str]]], List[str]]
            ]:
        # mcqa
        if target_data.get('categories', None) == 'mcqa':
            check_msg_list = []
            answer_list = []
            for idx in range(len(choices)):
                check_msg_list.append([
                    {
                        "role": Role.USER.value, 
                        "content": self.mcqa_criteria_prompt.format(
                            option=target_data[chr(ord("A") + idx)],
                            question=target_data['question'],
                            llm_response=llm_response,
                            )
                    }
                ])
                answer_list.append(
                    "True".lower() if chr(ord("A") + idx) == target_data["answer"] else "False".lower()
                )
            return check_msg_list, answer_list
        # opqa
        else:
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
           
hae_rae_bench_eval_templates: Dict[str, "HAE_RAE_BENCH_Template"] = {}

def get_hae_rae_bench_eval_template(name: str) -> "HAE_RAE_BENCH_Template":
    eval_template = hae_rae_bench_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="ko",
    system="",
    choice="",
    answer="",
    cot="",
    templates=hae_rae_bench_eval_templates,
    template_class=HAE_RAE_BENCH_Template,
    criteria_prompt="",
)