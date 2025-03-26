import uuid
import random
from typing import Dict, List, Literal, Sequence, Tuple
from ..template import TransTemplate
from ....data.data_utils import Role
from ....extras.constants import OPTION_CODES
from ...template import MCQA_Template, OPQA_Template, EvalTemplate

class AdvancedTransTemplate(TransTemplate):
    def __init__(self, template_lang: str, trans_prompt: str, system_prompt: str, guide_line: str, proper_noun_examples:str):
        self.template_lang = template_lang
        self.trans_prompt = trans_prompt
        self.system_prompt = system_prompt
        self.guide_line = guide_line
        self.proper_noun_examples = proper_noun_examples
    
    def format_translation_example(self, 
                               trans_source: str | List[Dict[str, str]],
                               source_type: Literal['question', 'response'],
                               source_lang, 
                               target_lang, 
                               choices: List[str], 
                               support_set: Sequence[Dict[str, str]], 
                               support_set_template: MCQA_Template | OPQA_Template | EvalTemplate,
                               support_set_choices: List[str],
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
                question, answer = support_set_template._parse_example(
                    type=random.sample(["generation", "mcqa-mc1", "mcqa-mc2"]), 
                    example=support_set[k], 
                    choices=support_set_choices,
                    use_cot=use_cot,
                    )
                if source_type == "question":
                    in_context_examples += f"Q{k+1}:\n{question}\n"
                else:
                    in_context_examples += f"A{k+1}:\n{answer}\n"
                
    
        # Format the translation prompt based on the type of trans_source
        if isinstance(trans_source, str):
            # Handle single text string case
            messages = []
            if self.system_prompt:
                messages.append({"role": Role.SYSTEM.value, "content": self.system_prompt})
            
            # only question needs in-context examples
            if source_type == "question":  
                trans_prompt = self.trans_prompt.format(
                    trans_source=trans_source, 
                    source_lang=source_lang, 
                    target_lang=target_lang,
                    in_context_examples=in_context_examples,
                    proper_noun_examples=self.proper_noun_examples,
                )
                
            messages.append({
                "role": Role.USER.value, 
                "content": self.guide_line + "\n" + trans_prompt
            })
            
            return messages
        else:
            # Handle list of conversation messages case
            # Create a separate translation request for each message
            list_of_messages = []
            # Generate a unique ID for each message
            msg_uuid = str(uuid.uuid4())
            for i, msg in enumerate(trans_source):
                messages = []
                
                # system prompt
                if self.system_prompt:
                    messages.append({"role": Role.SYSTEM.value, "content": self.system_prompt.strip()})
                
                # user prompt
                messages.append({
                    "idx": i,
                    "uuid": msg_uuid,
                    "origin_role": msg["role"],
                    "role": Role.USER.value, 
                    "content": (
                        self.guide_line + 
                        "\n" + 
                        self.trans_prompt.format(
                            trans_source=msg["content"], 
                            source_lang=source_lang, 
                            target_lang=target_lang,
                            in_context_examples=in_context_examples if msg['role'] == Role.USER.value else "Not needed.",
                            proper_noun_examples=self.proper_noun_examples,
                        )
                    ).strip()
                })
                
                list_of_messages.append(messages)
            
            # Return both the list of message requests and the metadata for reconstruction
            return list_of_messages