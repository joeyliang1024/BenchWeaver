from typing import Dict, List, Sequence, Tuple
from ..template import EvalTemplate
from ....data.data_utils import Role

class Trans_Template(EvalTemplate):
    def __init__(self, system: str, choice: str, answer: str, cot: str, criteria_prompt:str, response:str):
        self.system = system
        self.choice = choice
        self.answer = answer
        self.cot = cot
        self.criteria_prompt = criteria_prompt
        self.response = response
        
    def _parse_example(self, 
                       source_example: Dict[str, str], 
                       target_example: Dict[str, str], 
                       **kwargs) -> Tuple[str, str]:
        return source_example['text'], target_example['text']
    
    def format_inference_example(self, 
                                 source_example: Dict[str, str], 
                                 target_example: Dict[str, str], 
                                 source_lang: str,
                                 target_lang: str,
                                 user_prompt: str,
                                 **kwargs
    ) -> Tuple[List[Dict[str, str]], str]:
        r"""
        Converts dataset examples to messages.
        Args:
            source_example: The source example.
            target_example: The target example.
            user_prompt: The user prompt to format the message.
        
        """
        source_sentence, target_sentence = self._parse_example(source_example, target_example)
        prompt = user_prompt.format(source_sentence=source_sentence,
                                    source_lang=source_lang,
                                    target_lang=target_lang) \
                    if user_prompt is not None else \
                        self.system.format(source_sentence=source_sentence,
                                           source_lang=source_lang,
                                           target_lang=target_lang)

        return (
            [{
                "role": Role.USER.value, 
                "content": prompt
            }],
            target_sentence
            )
        