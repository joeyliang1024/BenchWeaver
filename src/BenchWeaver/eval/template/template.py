from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalTemplate:
    system: str
    choice: str
    answer: str
    cot: str
    criteria_prompt: str
    
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
        name (str): The language of the template to register.
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
