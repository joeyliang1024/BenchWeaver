from argparse import Namespace
import json
import os
import ast
import random
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple
from datasets import load_dataset
from tqdm.auto import tqdm
from .....data.data_utils import Role
from ....evaluator.multi_turn.pipeline_eval import MultiTurnEvaluator
from ....template import get_logickor_eval_template
from ....template.source.logickor_template import PROMPT_STRATEGY
from .....extras.constants import PROJECT_BASE_PATH
from .....extras.logging import get_logger

logger = get_logger(__name__)

class LogicKorEvaluator(MultiTurnEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.translated_few_shot_examples = self.get_few_shot_examples()
    
    @staticmethod
    def retrieve_responses(inference_results: Dict[str, List[Any]], translated_few_shot_examples:List[dict]) -> Dict[str, List[Any]]:
        """Retrieve responses from inference results."""
        responses = {subj: [] for subj in inference_results.keys()}
        len_few_shot = len(translated_few_shot_examples)
        for subj in inference_results.keys():
            responses[subj] = [[msg['content'] for msg in message[len_few_shot:] if msg['role'] == Role.ASSISTANT.value] for message in inference_results[subj]]
        return responses
    
    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Get few-shot examples for the LogicKor evaluation.
        Returns a list of dictionaries containing the example prompts.
        """
        if getattr(self.eval_args, "n_shot") == 1:
            messages = PROMPT_STRATEGY["cot-1-shot"] if getattr(self.eval_args, "cot") is True else PROMPT_STRATEGY["1-shot"]
            return messages
        else:
            return []
        
    def encode_trans_prompt(self, origin_messages: List[Dict[str, str]]):
        if self.ref_task is not None:
            ref_dataset = load_dataset(
                path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.ref_task),
                name=random.choice(list(self.ref_categories.keys())),
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.hf_token,
                trust_remote_code=True,
            )
            support_set = (
                    ref_dataset["test"]
                    .shuffle()
                    .select(range(min(self.eval_args.n_shot, len(ref_dataset["test"]))))
                )
        else:
            support_set = None
            
        return self.trans_template.format_translation_example(
                trans_source=origin_messages,
                source_type="question",
                source_lang=self.model_args.source_lang,
                target_lang=self.model_args.target_lang,
                choices=None,
                support_set=support_set,
                support_set_template=self.ref_template,
                support_set_choices=self.ref_choices,
                use_cot=self.eval_args.cot,
            )
        
    async def translate_few_shot_example(self, translation_process) -> List[Dict[str, str]]:
        self.translated_examples = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data={"ex": self.encode_trans_prompt(origin_messages=self.get_few_shot_examples())},
            output_path="translated_few_shot_ex_record.json",
            progress_desc="Trans Few-Shot Ex Progress",
            terminate_server=False,  
            tmp_catagories = ["ex"]  # Use a temporary category for few-shot examples
        )
        return self.translated_examples['ex'][0] if 'ex' in self.translated_examples else []
            
    def comput_score(self, check_results: Dict[str, List[Any]], subjects: List[str], checked_answers=None) -> Dict[str, float]:
        #TODO: add muti-turn and single turn score
        category_corrects = {subj: [] for subj in subjects}
        for subj in check_results.keys():
            for idx, check_string in enumerate(check_results[subj]):
                score = self.retrieve_answer(text=check_string, numerical=True)
                category_corrects[subj].append(score)
                if idx % 2 == 0:
                    category_corrects['Single Turn'].append(score)
                else:
                    category_corrects['Multi Turn'].append(score)
                category_corrects['Average'].append(score)
        # average score
        for subj in category_corrects.keys():
            category_corrects[subj] = np.mean(category_corrects[subj])
        
        return category_corrects
    
    # Override
    async def multi_turn_generate(
        self,
        model: str,
        system_prompt: Optional[str],
        questions: List[str],
        idx: int, 
        generating_args: Namespace,
    ) -> Tuple[List[dict], int]:
        assert self.translated_few_shot_examples is not None or self.eval_args.pipeline == "same", "Few-shot examples is not translated yet."
        # messages = self.translated_few_shot_examples
        messages = []
        messages += self.translated_few_shot_examples
        for question in questions:
            messages.append({
                "role": Role.USER.value, 
                "content": question
            })
            messages.append({
                "role": Role.ASSISTANT.value, 
                "content": await self.client.generate(
                               model=model,
                               system_prompt=system_prompt,
                               example=messages,
                               generating_args=generating_args,
                           )
            })
        return messages, idx
    
    def load_data(self, 
                  mode = Literal['inference', 'check', 'translation'],
                  histories: Dict[str, list] = None,
                  choices: list = None, 
                  responses_trans: bool = False,
                  check_source: Literal['original', 'translated'] = "original",
                  ) -> Tuple[Dict[str, list], Dict[str, list]]:
        """Load and format data for evaluation."""
        # init data
        inference_questions = {subj: [] for subj in self.categories.keys()}
        checker_prompts     = {subj: [] for subj in self.categories.keys()}
        translate_prompts   = {subj: [] for subj in self.categories.keys()}
        # Load datasets
        for subject in tqdm(self.categories.keys(), desc="Loading subjects"):
            # load dataset from folder
            dataset = load_dataset(
                path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.eval_task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.hf_token,
                trust_remote_code=True,
            )
            # load template
            eval_template = get_logickor_eval_template(name=subject)
            # Prepare examples for evaluation
            if mode == "inference":
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)): 
                    # format inference example
                    questions = eval_template.get_inference_quesitons(example=dataset[self.eval_split][i])
                    inference_questions[subject].append(questions)
            
            elif mode == "check":
                # answers are already in the check prompts
                assert self.inference_results is not None
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                    check_msg_list = eval_template.format_checker_example(
                        target_data=dataset[self.eval_split][i],
                        history=histories[subject][i] if histories is not None else None,
                    )
                    checker_prompts[subject].extend(check_msg_list)

            elif mode == "translation":
                # check is question or repsponse translation
                source_type = "response" if responses_trans else "question"
                # load object benchmark examples
                if self.ref_task is not None:
                    ref_dataset = load_dataset(
                        path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.ref_task),
                        name=random.choice(list(self.ref_categories.keys())),
                        cache_dir=self.model_args.cache_dir,
                        download_mode=self.eval_args.download_mode,
                        token=self.hf_token,
                        trust_remote_code=True,
                    )
                    support_set = (
                            ref_dataset["test"]
                            .shuffle()
                            .select(range(min(self.eval_args.n_shot, len(ref_dataset["test"]))))
                        )
                else:
                    support_set = None
                
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                    # format translation example
                    if source_type == "question":
                        trans_messages = self.trans_template.format_translation_example(
                            trans_source=ast.literal_eval(dataset[self.eval_split][i]['question_turns']),
                            source_type=source_type,
                            source_lang=self.model_args.source_lang,
                            target_lang=self.model_args.target_lang,
                            choices=choices,
                            support_set=support_set,
                            support_set_template=self.ref_template,
                            support_set_choices=self.ref_choices,
                            use_cot=self.eval_args.cot,
                        )
                        # list of messages
                        translate_prompts[subject] += trans_messages
                    elif source_type == "response":
                        trans_messages = self.trans_template.format_translation_example(
                            trans_source=self.retrieved_responses[subject][i],
                            source_type=source_type,
                            source_lang=self.model_args.target_lang,
                            target_lang=self.model_args.source_lang,
                            choices=choices,
                            support_set=support_set,
                            support_set_template=self.ref_template,
                            support_set_choices=self.ref_choices,
                            use_cot=self.eval_args.cot,
                        )
                        # message list
                        translate_prompts[subject] += trans_messages
            else:
                raise ValueError(f"Input mode {mode} is invalid. Please specify one of 'inference' or 'check' instead.")
        
        if mode == "inference":
            return None, inference_questions
        elif mode == "check":
            return None, checker_prompts
        elif mode == "translation":
            return None, translate_prompts
    
    async def diff_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        # specific evaluation pipeline
        # need: trranslator model, inference model, checker model
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        logger.info(f"Data path created: {self.save_folder}")
        ######################################## translation ########################################
        logger.info("============ Start question translation process. ============")
        _, inference_questions = self.load_data(mode="inference", choices=choices)
        _, ques_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=False)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=inference_questions, output_path=os.path.join(self.save_folder, "inference_questions.json"))
            self.save_data(data=ques_trans_prompts, output_path=os.path.join(self.save_folder, "ques_trans_prompts.json"))
        
        if self.translation_mode == "local":
            translation_process = await self.server.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            logger.info("Local vLLM server setup complete.")
        else:
            translation_process = None
            logger.info("Using OpenAI API for inference.")

        self.set_client(mode="translation")
        logger.info("Client setup complete.")
        # translate few-shot examples first
        self.translated_few_shot_examples = await self.translate_few_shot_example(translation_process=translation_process)
        # translate questions
        self.translated_questions = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=ques_trans_prompts,
            output_path="translated_question_record.json",
            progress_desc="Trans Question Progress",
        )

        logger.info("Question translated complete.")
        
        ######################################### inference #########################################
        logger.info("============ Start inference process. ============")
        if self.inference_mode == "local":
            inference_process = await self.server.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            logger.info("Local vLLM server setup complete.")
        else:
            inference_process = None
            logger.info("Using OpenAI API for inference.")

        self.set_client(mode="inference")
        logger.info("Client setup complete.")

        self.inference_results = await self.multi_turn_inference(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.inference_model_name_or_path,
            data=self.translated_questions,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )

        logger.info("Inference complete.")
        ######################################## translation ########################################
        logger.info("============ Start response translation process. ============")
        self.retrieved_responses = self.retrieve_responses(inference_results=self.inference_results, translated_few_shot_examples=self.translated_few_shot_examples)
        _, resp_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=True)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.retrieved_responses, output_path=os.path.join(self.save_folder, "retrieved_responses.json"))
            self.save_data(data=resp_trans_prompts, output_path=os.path.join(self.save_folder, "resp_trans_prompts.json"))
            
        if self.translation_mode == "local":
            translation_process = await self.server.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            logger.info("Local vLLM server setup complete.")
        else:
            translation_process = None
            logger.info("Using OpenAI API for inference.")

        self.set_client(mode="translation")
        logger.info("Client setup complete.")
        
        self.translated_responses = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=resp_trans_prompts,
            output_path="translated_response_record.json",
            progress_desc="Trans Response Progress",
        )
        # recover translated inference results
        translated_inference_results = self.recover_translated_inference_results(
            system_prompt=getattr(self.eval_args, "system_prompt"),
            translated_questions=self.translated_questions,
            translated_responses=self.translated_responses,
        )
        if getattr(self.eval_args, "record_all", False):
            self.save_data(data=translated_inference_results, output_path=os.path.join(self.save_folder, "translated_inference_results.json"))
        logger.info("Question translated complete.")
        ########################################### check ###########################################
        logger.info("============ Start checking process. ============")
        checked_answers, checked_prompts = self.load_data(mode="check", choices=choices, check_source="translated", histories=translated_inference_results)
        if getattr(self.eval_args, "record_all", False):
            self.save_data(data=checked_prompts, output_path=os.path.join(self.save_folder, "checked_prompts.json"))
            self.save_data(data=checked_answers, output_path=os.path.join(self.save_folder, "checked_answers.json"))
        
        if self.check_mode == "local":
            checker_process = await self.server.setup_server(
                model_path=self.model_args.checker_model_name_or_path,
                model_name=self.checker_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            logger.info("Local vLLM server setup complete.")
        else:
            checker_process = None
            logger.info("Using OpenAI API for checking.")

        self.set_client(mode="check")
        logger.info("Client setup complete.")

        check_results = await self.process_subjects(
            server_process=checker_process,
            model_name=self.checker_model_name if self.check_mode == "local" else self.model_args.checker_model_name_or_path,
            data=checked_prompts,
            prompt_key="criteria_system_prompt",
            output_path="check_results.json",
            progress_desc="Check Progress",
        )

        logger.info("Check complete.")
        ####################################### compute score #######################################
        logger.info("============ Computing Score ============")
        score_dict = self.comput_score(checked_answers=checked_answers, check_results=check_results, subjects=subjects)
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))