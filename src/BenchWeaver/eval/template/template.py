from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalTemplate:
    system: str
    choice: str
    answer: str
    cot: str
    criteria_prompt: str

@dataclass
class TransTemplate(EvalTemplate):
    template_lang: str
    trans_prompt: str
    system_prompt: str
    guide_line: str
    
def _register_eval_template(name: str, 
                            system: str, 
                            choice: str, 
                            answer: str, 
                            cot: str, 
                            criteria_prompt: str,
                            templates: dict, 
                            template_class,
                            ) -> None:
    """
    Registers an evaluation template with the given parameters.
    Args:
        name (str): The name of the template to register.
        system (str): The system description for the template.
        choice (str): The choice description for the template.
        answer (str): The answer description for the template.
        cot (str): The chain of thought description for the template.
        templates (dict): The dictionary to store the registered templates.
        template_class: The class of the template to be registered.
    Returns:
        None
    """
    templates[name] = template_class(system=system, choice=choice, answer=answer, cot=cot, criteria_prompt=criteria_prompt)

def _register_trans_template(name: str,
                             template_lang: str,
                             trans_prompt: str,
                             system_prompt: str,
                             guide_line: str,
                             templates: dict, 
                             template_class,
                             ) -> None:
    """
    Registers a translation template  from `template_lang` to `target_lang` with the given parameters. 
    Args:
        name (str): The name of the template to register.
        template_lang (str): The language of the template.
        system_prompt (str): The system prompt for the template.
        guide_line (str): The guideline for the template.
        templates (dict): The dictionary to store the registered templates.
        template_class: The class of the template to be registered.
    Returns:
        None
    """
    templates[name] = template_class(template_lang=template_lang, trans_prompt=trans_prompt, system_prompt=system_prompt, guide_line=guide_line)