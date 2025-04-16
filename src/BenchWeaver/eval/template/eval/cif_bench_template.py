from typing import Dict
from ..template import _register_eval_template
from .opqa_template import OPQA_Template

cif_bench_eval_templates: Dict[str, "OPQA_Template"] = {}

def get_cif_bench_eval_template(name: str) -> "OPQA_Template":
    eval_template = cif_bench_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="zh",
    system="问题：\n\n",
    choice="\n{choice}. {content}",
    answer="\n回答：",
    cot="\n让我们一步一步思考。\n回答：",
    templates=cif_bench_eval_templates,
    template_class=OPQA_Template,
    criteria_prompt="请确认LLM回答是否正确完成评估指南所要求的任务。\n\n问题：{question}\n\n参考答案：{answer}\n\nLLM回答：{llm_response}\n\n如果LLM回答正确，请仅回复'True'，否则回复'False'。",
    response="正确答案是({answer})。"
)

