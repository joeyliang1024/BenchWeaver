from typing import Dict
from ..template import _register_eval_template
from .code_template import Code_Template

humaneval_xl_eval_templates: Dict[str, "Code_Template"] = {}

def get_humaneval_xl_eval_template(name: str) -> "Code_Template":
    eval_template = humaneval_xl_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="{prompt} Your code should pass these tests:\n\n{tests}",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=humaneval_xl_eval_templates,
    template_class=Code_Template,
    criteria_prompt="",
    response=""
)

_register_eval_template(
    name="zh",
    system="{prompt} 你的代码应该通过这些测试：\n\n{tests}",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=humaneval_xl_eval_templates,
    template_class=Code_Template,
    criteria_prompt="",
    response=""
)

_register_eval_template(
    name="ko",
    system="{prompt} 귀하의 코드는 다음 테스트를 통과해야 합니다:\n\n{tests}",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=humaneval_xl_eval_templates,
    template_class=Code_Template,
    criteria_prompt="",
    response=""
)