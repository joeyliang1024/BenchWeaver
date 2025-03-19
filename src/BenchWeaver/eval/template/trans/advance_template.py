import uuid
from typing import Dict, List, Literal, Sequence, Tuple
from ..template import TransTemplate
from ....data.data_utils import Role
from ....extras.constants import OPTION_CODES

class AdvancedTransTemplate(TransTemplate):
    def __init__(self, template_lang: str, trans_prompt: str, system_prompt: str, guide_line: str):
        self.template_lang = template_lang
        self.trans_prompt = trans_prompt
        self.system_prompt = system_prompt
        self.guide_line = guide_line
    
    def _parse_ref_question(self, example: Dict[str, str], choices: List[str], use_cot: bool=False) -> str:
        # needs to use origin template class
        if choices and choices != OPTION_CODES:
            # MCQA with certain choices
            question = "".join([example["question"]] +
                           [self.choice.format(choice=ch, content=example[ch]) for ch in choices if ch in example] +
                           [self.cot if use_cot else self.answer]
                           ).strip()
        elif "choices" in example or "mc1_choices" in example or "mc2_choices" in example:
            # MCQA wtth uncertain choices
            question = " ".join(
                [
                    example.get("paragraph", ""),
                    example["question"],
                ] +
                [
                    "\n{choice}.  {content}".format(choice=option, content=content) for option, content in zip(OPTION_CODES, (example.get("choices", []) or example.get("mc1_choices", []) or example.get("mc2_choices", [])))
                ] +
                [
                    self.cot if use_cot else self.answer
                ]
            ).strip()
        else:
            # OPQA
            question = (example["question"] + (self.cot if use_cot else self.answer)).strip()
        return question
    
    def _parse_ref_answer(self, example: Dict[str, str], use_cot: bool=False) -> str:
        return ((example.get("explanation") if use_cot and example.get("explanation") else "") + 
                "\n" + 
                "The correct answer is ({answer}).".format(answer=example.get("answer"))).strip()
    
    def format_translation_example(self, 
                               trans_source: str | List[Dict[str, str]],
                               source_type: Literal['question', 'response'],
                               source_lang, 
                               target_lang, 
                               choices: List[str], 
                               support_set: Sequence[Dict[str, str]], 
                               use_cot: bool
                               ) -> List[Dict[str, str]] | List[List[Dict[str, str]]]:
        """
        Format a few-shot translation example.
        Few-shot translation example will be in-context of a single turn conversation.
        Args:
            trans_source: the source sentence or messages of the translation
                - If str: a single text to translate
                - If List[Dict[str, str]]: a list of conversation messages with "role" and "content" keys
            source_type: the source language of the translation
            source_lang: the source language of the translation
            target_lang: the target language of the translation
            choices: the choices of the MCQA question, for OPQA, just None.
            support_set: the support set for the translation
            use_cot: whether to use the chain of thought in the question
        Returns:
            If trans_source is str: A messages list of translation example.
            
            If trans_source is List[Dict[str, str]]: A list of messages lists,
                where each messages list is a translation request for one message in the conversation.
        """
        # Prepare in-context examples
        in_context_examples = ""
        if support_set is not None:
            for k in range(len(support_set)):
                if source_type == "question":
                    example = self._parse_ref_question(support_set[k], choices, use_cot)
                else:
                    example = self._parse_ref_answer(support_set[k], use_cot)
                in_context_examples += f"Q{k+1}:\n{example}\n"
    
        # Format the translation prompt based on the type of trans_source
        if isinstance(trans_source, str):
            # Handle single text string case
            messages = []
            if self.system_prompt:
                messages.append({"role": Role.SYSTEM.value, "content": self.system_prompt})
                
            trans_prompt = self.trans_prompt.format(
                trans_source=trans_source, 
                source_lang=source_lang, 
                target_lang=target_lang
            )
            
            messages.append({
                "role": Role.USER.value, 
                "content": "\n".join([
                    self.guide_line, 
                    in_context_examples,
                    trans_prompt
                ])
            })
            
            return messages
        else:
            # Handle list of conversation messages case
            # Create a separate translation request for each message
            list_of_messages = []
            msg_uuid = str(uuid.uuid4())
            for i, msg in enumerate(trans_source):
                messages = []
                # Generate a unique ID for each message
                
                if self.system_prompt:
                    messages.append({"role": Role.SYSTEM.value, "content": self.system_prompt})
                
                trans_prompt = self.trans_prompt.format(
                    trans_source=msg["content"], 
                    source_lang=source_lang, 
                    target_lang=target_lang
                )
                
                messages.append({
                    "idx": i,
                    "uuid": msg_uuid,
                    "origin_role": msg["role"],
                    "role": Role.USER.value, 
                    "content": "\n".join([
                        self.guide_line, 
                        in_context_examples,
                        trans_prompt
                    ])
                })
                
                list_of_messages.append(messages)
            
            # Return both the list of message requests and the metadata for reconstruction
            return list_of_messages