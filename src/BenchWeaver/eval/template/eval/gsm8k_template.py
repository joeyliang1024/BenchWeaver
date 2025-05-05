from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

gsm8k_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_gsm8k_eval_template(name: str) -> "OPQA_Template":
    eval_template = gsm8k_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="Problem: \n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=gsm8k_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="Determine whether the LLM Response correctly answer the question.\n\nQuestion: {question}\n\nReference Answer: {answer}\n\nLLM Response: {llm_response}\n\nIf the LLM Response correct, just response 'True', else response 'False'.",
    response="The correct answer is ({answer})."
)

_register_eval_template(
    name="zh",
    system="問題：\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n讓我們一步一步地思考。\n答案：",
    templates=gsm8k_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="判斷 LLM 回應是否正確回答了問題。\n\n問題: {question}\n\n參考答案: {answer}\n\nLLM 回應: {llm_response}\n\n如果 LLM 回應正確，請回答 \'True\'，否則回答 \'False\'。",
    response="正確答案是 ({answer})."
)

_register_eval_template(
    name="ko",
    system="문제: \n\n",
    choice="\n{choice}. {content}",
    answer="\n답:",
    cot="\n단계별로 생각해 보자.\n답:",
    templates=gsm8k_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="LLM 응답이 질문에 올바르게 답변했는지 확인하십시오.\n\n질문: {question}\n\n참조 답변: {answer}\n\nLLM 응답: {llm_response}\n\nLLM 응답이 올바른 경우 'True'라고만 응답하고, 그렇지 않으면 'False'라고 응답합니다.",
    response="정답은 ({answer})입니다.",
)

