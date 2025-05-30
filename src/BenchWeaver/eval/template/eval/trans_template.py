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
                                 source_support_set: Sequence[Dict[str, str]],
                                 target_support_set: Sequence[Dict[str, str]],
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
        messages = []
        # for few shot
        if source_support_set is not None and target_support_set is not None:
            assert len(source_support_set) == len(target_support_set), \
                ValueError("Source and target support sets must have the same length.")
            for k in range(len(source_support_set)):
                source_sentence, target_sentence = self._parse_example(source_support_set[k], target_support_set[k])
                prompt = user_prompt.format(source_sentence=source_sentence,
                                    source_lang=source_lang,
                                    target_lang=target_lang) \
                    if user_prompt is not None else \
                        self.system.format(source_sentence=source_sentence,
                                           source_lang=source_lang,
                                           target_lang=target_lang)
                messages.append({"role": Role.USER.value, "content": prompt})
                messages.append({"role": Role.ASSISTANT.value, "content": target_sentence})
        # for current example
        source_sentence, correct_target_sentence = self._parse_example(source_example, target_example)
        prompt = user_prompt.format(source_sentence=source_sentence,
                                    source_lang=source_lang,
                                    target_lang=target_lang) \
                    if user_prompt is not None else \
                        self.system.format(source_sentence=source_sentence,
                                           source_lang=source_lang,
                                           target_lang=target_lang)
        messages.append({"role": Role.USER.value, "content": prompt})
        return (
            messages,
            correct_target_sentence
            )
        