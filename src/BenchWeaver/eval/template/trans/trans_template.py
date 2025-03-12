from typing import Dict
from ..template import _register_trans_template
from .advance_template import AdvancedTransTemplate
from ....extras.constants import PROJECT_BASE_PATH

def get_translation_template(name: str) -> "AdvancedTransTemplate":
    translation_template = translation_templates.get(name, None)
    assert translation_template is not None, "Template {} does not exist.".format(name)
    return translation_template

translation_templates: Dict[str, "AdvancedTransTemplate"] = {}

