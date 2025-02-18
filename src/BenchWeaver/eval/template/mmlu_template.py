from typing import Dict
from .mcqa_template import MCQA_Template
from .template import _register_eval_template

mmlu_eval_templates: Dict[str, "MCQA_Template"] = {}

def get_mmlu_eval_template(name: str) -> "MCQA_Template":
    eval_template = mmlu_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


_register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
    cot="\nLet's think step by step.\nAnswer:",
    templates=mmlu_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt='Determine whether the following response correctly identifies "{option}" as the answer to the multiple-choice question.\n\nQuestion: {question}\n\nLLM Response: {llm_response}\n\nIf the LLM Response is correct, just response \'True\', else response \'False\'.',
)

_register_eval_template(
    name="zh-tw",
    system="以下是台灣關於{subject}考試的單項選擇題，請選出其中的正確答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n讓我們一步一步來思考。\n答案：",
    templates=mmlu_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt='判斷以下 LLM 回應是否正確識別 "{option}" 為多項選擇題的答案。\n\n問題: {question}\n\nLLM 回應: {llm_response}\n\n如果 LLM 回應正確，請回答 \'True\'，否則回答 \'False\'。',
)

_register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n让我们一步一步来思考。\n答案：",
    templates=mmlu_eval_templates,
    template_class=MCQA_Template,
    criteria_prompt='判断以下 LLM 回答是否正确识别 "{option}" 为多项选择题的答案。\n\n问题: {question}\n\nLLM 回答: {llm_response}\n\n如果 LLM 回答正确，请回答 \'True\'，否则回答 \'False\'。',
)
