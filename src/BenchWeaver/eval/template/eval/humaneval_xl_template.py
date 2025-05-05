from typing import Dict
from ..template import _register_eval_template
from .code_template import Code_Template

humaneval_xl_eval_templates: Dict[str, "Code_Template"] = {}

def get_mbpp_eval_template(name: str) -> "Code_Template":
    eval_template = humaneval_xl_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="You are an expert programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}",
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
    system="你是一个专家程序员，你的任务是：{prompt} 你的代码应该通过这些测试：\n\n{tests}",
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
    system="당신은 전문가 프로그래머이며, 당신의 작업은 다음과 같습니다: {prompt} 귀하의 코드는 다음 테스트를 통과해야 합니다:\n\n{tests}",
    choice="\n{choice}. {content}",
    answer="",
    cot="",
    templates=humaneval_xl_eval_templates,
    template_class=Code_Template,
    criteria_prompt="",
    response=""
)