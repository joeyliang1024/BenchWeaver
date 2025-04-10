from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

cmath_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_cmath_eval_template(name: str) -> "OPQA_Template":
    eval_template = cmath_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="zh",
    system="问题：\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n让我们一步一步地思考。\n答案：",
    templates=cmath_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="请判断大语言模型（LLM）的回答是否正确。\n\n问题：{answer}\n\n参考答案：{question}\n\nLLM 回答：{llm_response}\n\n如果 LLM 回答正确，请回复 “True”，否则回复 “False”。",
    response="正确答案是（{answer}）。"
)

