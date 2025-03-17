import json
from typing import Dict
from ..template import _register_trans_template
from .advance_template import AdvancedTransTemplate
from ....extras.constants import TRANSLATION_PROMPT_PATH

def get_translation_template(name: str) -> "AdvancedTransTemplate":
    translation_template = translation_templates.get(name, None)
    assert translation_template is not None, "Template {} does not exist.".format(name)
    return translation_template

with open(TRANSLATION_PROMPT_PATH, "r") as f:
    translation_prompts: Dict[str, dict] = json.load(f)
    
translation_templates: Dict[str, "AdvancedTransTemplate"] = {}

for name, config in translation_prompts.items():
    _register_trans_template(
        name=name,
        template_lang=config.get('language'),
        trans_prompt=config.get('trans_prompt'),
        system_prompt=config.get('system_prompt'),
        guide_line=config.get('guide_line'),
        templates=translation_templates,
        template_class=AdvancedTransTemplate
    )
