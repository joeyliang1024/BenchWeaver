from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

awesome_taiwan_knowledge_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_awesome_taiwan_knowledge_eval_template(name: str) -> "OPQA_Template":
    eval_template = awesome_taiwan_knowledge_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="zh-tw",
    system="問題：\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    cot="\n讓我們一步一步來思考。\n答案：",
    templates=awesome_taiwan_knowledge_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="判斷 LLM 回應是否正確回答了問題。\n\n問題: {question}\n\n參考答案: {answer}\n\nLLM 回應: {llm_response}\n\n如果 LLM 回應正確，請回答 \'True\'，否則回答 \'False\'。",
    response="{answer}"
)
