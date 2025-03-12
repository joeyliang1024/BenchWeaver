from typing import Dict, List, Literal, Sequence
from ..template import TransTemplate
from ....data.data_utils import Role

class AdvancedTransTemplate(TransTemplate):
    def __init__(self, template_lang: str, trans_prompt: str, system_prompt: str, guide_line: str):
        self.template_lang = template_lang
        self.trans_prompt = trans_prompt
        self.system_prompt = system_prompt
        self.guide_line = guide_line
    
    def _parse_ref_question(self, example: Dict[str, str], choices: List[str], use_cot: bool=False) -> str:
        if choices:
            question = "".join([example["question"]] +
                           [self.choice.format(choice=ch, content=example[ch]) for ch in choices if ch in example] +
                           [self.cot if use_cot else self.answer]
                           ).strip()
        else:
            question = (example["question"] + (self.cot if use_cot else self.answer)).strip()
        return question
    
    def _parse_ref_answer(self, example: Dict[str, str], use_cot: bool=False) -> str:
        return ((example.get("explanation") if use_cot and example.get("explanation") else "") + 
                "\n" + 
                "The correct answer is ({answer}).".format(answer=example.get("answer"))).strip()
    
    def format_translation_example(self, 
                                   trans_source:str,
                                   source_type: Literal['question', 'response'],
                                   source_lang, 
                                   target_lang, 
                                   choices: List[str], 
                                   support_set: Sequence[Dict[str, str]], 
                                   use_cot:bool
                                   ) -> List[Dict[str, str]]:
        """
        Format a few-shot translation example.
        Few-shot translation example will be in-context of a single turn conversation.
        Args:
            trans_source: the source sentence of the translation
            source_lang: the source language of the translation
            target_lang: the target language of the translation
            choices: the choices of the MCQA question, for OPQA, just None.
            support_set: the support set for the translation
            use_cot: whether to use the chain of thought in the question
        Returns:
            A messages list of translation example.
        """
        messages = []
        in_context_examples = ""
        if support_set is not None:
            for k in range(len(support_set)):
                if source_type == "question":
                    example = self._parse_ref_question(support_set[k], choices, use_cot)
                else:
                    example = self._parse_ref_answer(support_set[k], use_cot)
                in_context_examples += f"Q{k+1}:\n{example}\n"
                
        if self.system_prompt:
            messages.append({"role": Role.SYSTEM.value, "content": self.system_prompt})
        messages.append({"role": Role.USER.value, "content": 
                "\n".join([self.guide_line, 
                           in_context_examples,
                           self.trans_prompt.format(trans_source=trans_source, source_lang=source_lang, target_lang=target_lang)
                           ])
            })
        