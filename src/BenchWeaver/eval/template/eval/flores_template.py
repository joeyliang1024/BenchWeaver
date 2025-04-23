from typing import Dict
from .trans_template import Trans_Template
from ..template import _register_eval_template

flores_eval_templates: Dict[str, "Trans_Template"] = {}

def get_flores_eval_template(name: str) -> "Trans_Template":
    eval_template = flores_eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

_register_eval_template(
    name="en",
    system="Translate this sentence from {source_lang} to {target_lang}.\n\n{source_sentence}\n",
    choice="",
    answer="",
    cot="",
    templates=flores_eval_templates,
    template_class=Trans_Template,
    criteria_prompt="",
    response="",
)

_register_eval_template(
    name="ko",
    system="이 문장을 {source_lang}에서 {target_lang}로 번역하십시오.\n\n{source_sentence}\n",
    choice="",
    answer="",
    cot="",
    templates=flores_eval_templates,
    template_class=Trans_Template,
    criteria_prompt="",
    response="",
)

_register_eval_template(
    name="zh",
    system="将这个句子从{source_lang}翻译成{target_lang}。\n\n{source_sentence}\n",
    choice="",
    answer="",
    cot="",
    templates=flores_eval_templates,
    template_class=Trans_Template,
    criteria_prompt="",
    response="",
)

_register_eval_template(
    name="zh-tw",
    system="將這個句子從{source_lang}翻譯成{target_lang}。\n\n{source_sentence}\n",
    choice="",
    answer="",
    cot="",
    templates=flores_eval_templates,
    template_class=Trans_Template,
    criteria_prompt="",
    response="",
)