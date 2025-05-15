import ast
from typing import Dict, List, Sequence, Tuple
from ..template import _register_eval_template
from ...template import OPQA_Template
from ....data.data_utils import Role

class XNLI_Template(OPQA_Template):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.criteria_prompt = criteria_prompt
        self.response = response
    
    def _parse_example(self, example: Dict[str, str], choices: List[str], use_cot: bool=False, **kwargs) -> Tuple[str, str]:
        # example inscluding: premise, statement, answer
        question = self.system.format(
            premise=example['premise'],
            statement=example['statement']
        ).strip()
        answer = self.response.format(answer=example['answer']).strip()
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
        question, answer = self._parse_example(target_data, choices=[], use_cot=False)
        return [
                {
                    "role": Role.USER.value, 
                    "content": self.criteria_prompt.format(
                        answer=answer,
                        question=question,
                        llm_response=llm_response,
                        )
                }
            ]
        
xnli_eval_templates: Dict[str, "XNLI_Template"] = {}

def get_xnli_eval_template(name: str) -> "XNLI_Template":
    eval_template = xnli_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="Take the following as truth: {premise}\nThen the following statement: '{statement}' is\nOptions:\nA.true\nB. inconclusive\nC. false\nSelect the correct option from A, B, and C.",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=xnli_eval_templates,
    template_class=XNLI_Template,
    criteria_prompt="Determine whether the LLM Response correctly answer the question.\n\nQuestion: {question}\n\nReference Answer: {answer}\n\nLLM Response: {llm_response}\n\nIf the LLM Response correct, just response 'True', else response 'False'.",
    response="The correct answer is ({answer})."
)
    
_register_eval_template(
    name="zh",
    system="假设以下内容为真：{premise}\n考虑以下陈述：“{statement}”\n该陈述是：\n选项：\nA. 真实的\nB.无法确定\nC. 虚假的\n从 A, B 或者C 中选择正确的选项。",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n让我们一步一步地思考。\n答案：",
    templates=xnli_eval_templates,
    template_class=XNLI_Template,
    criteria_prompt="请判断大语言模型的回答是否正确回答了问题。\n\n问题：{question}\n\n参考答案：{answer}\n\nLLM 回答：{llm_response}\n\n如果回答正确，请仅回答 'True'，否则回答 'False'。",
    response="正确答案是（{answer}）。"
)

_register_eval_template(
    name="ko",
    system="다음 내용을 사실로 가정합니다: {premise}\n그런 다음 다음 진술을 고려하십시오: '{statement}'\n진술은:\n옵션:\nA. 진실\nB. 결론을 내릴 수 없음\nC. 거짓\nA, B 또는 C에서 올바른 옵션을 선택하십시오.",
    choice="\n{choice}. {content}",
    answer="\n답변:",
    cot="\n단계별로 생각해 보겠습니다.\n답변:",
    templates=xnli_eval_templates,
    template_class=XNLI_Template,
    criteria_prompt="LLM 응답이 질문에 올바르게 답변했는지 확인하십시오.\n\n질문: {question}\n\n참조 답변: {answer}\n\nLLM 응답: {llm_response}\n\nLLM 응답이 올바른 경우 'True'로 응답하고, 그렇지 않으면 'False'로 응답합니다.",
    response="정답은 ({answer})입니다."
)
        