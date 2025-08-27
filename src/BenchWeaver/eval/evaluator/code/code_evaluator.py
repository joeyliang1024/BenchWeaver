import asyncio
import json
import random
import re
import os
from datasets import load_dataset
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm
from ..evaluator import Evaluator
from ...template import Code_Template
from ....extras.constants import PROJECT_BASE_PATH
from ...metric.code import eval_code
from ....extras.logging import get_logger

logger = get_logger(__name__)


class CodeEvaluator(Evaluator):
    eval_template: Code_Template
    categories: dict
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args = args)
    
    @staticmethod
    def _postprocess_generation(text: str) -> str:
        # Try to match [BEGIN] ... [DONE] with flexible spacing and casing
        pattern_begin_done = r'\[\s*begin\s*\](.*?)\[\s*done\s*\]'
        match = re.search(pattern_begin_done, text, re.IGNORECASE | re.DOTALL)
    
        if match:
            return match.group(1).strip().rstrip('```').lstrip('```')
    
        # If no [BEGIN]...[DONE], try to match text between triple backticks
        pattern_code_block = r'```(?:\w*\n)?(.*?)```'
        match = re.search(pattern_code_block, text, re.DOTALL)
    
        if match:
            return match.group(1).strip().rstrip('```').lstrip('```')
     
        return text.strip()
    
    def post_process_response(self, response_result: Dict[str, Any]):
        """
        Covert the format to pedcitions (Dict[str, List[List[str]]])
        """
        for subject, response_list in response_result.items():
            for i, response in enumerate(response_list):
                # Remove the [BEGIN] and [DONE] tags
                response_result[subject][i] = [self._postprocess_generation(response)]
        return response_result
                
    def comput_score(self, test_codes: Dict[str, List[Any]], response_result: Dict[str, List[Any]], k: int = 1):
        result_dict = {"Average": 0.0}
        for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
            category_name = self.categories[subject]["category"]
            result_dict[category_name] = eval_code(predictions=response_result[subject], references=test_codes[subject], k=k)
        result_dict["Average"] = np.mean([result_dict[_][0][f"pass@{k}"] for _ in result_dict.keys() if _ != "Average"])
        return result_dict
        
        #results = code_eval.compute(predictions=predictions, references=references, k=[1])
    def load_data(self, 
                  mode = Literal['inference', 'check', 'translation'],
                  choices = None, 
                  responses_trans: bool = False,
                  check_source: Literal['original', 'translated'] = "original",
                  ) -> Tuple[Dict[str, list], Dict[str, list]]:
        """Load and format data for evaluation."""
        # init data
        inference_prompts = {subj: [] for subj in self.categories.keys()}
        test_codes = {subj: [] for subj in self.categories.keys()}
        translate_prompts = {subj: [] for subj in self.categories.keys()}
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
            # Prepare examples for evaluation
            if mode == "inference":
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)): 
                    if dataset.get("train"):
                        support_set = (
                            dataset["train"]
                            .shuffle()
                            .select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                        )
                    else:
                        support_set = None
                    # format inference example
                    messages, test_code = self.eval_template.format_inference_example(
                        target_data=dataset[self.eval_split][i],
                        support_set=support_set,
                    )
                    inference_prompts[subject].append(messages)
                    test_codes[subject].append(test_code)
            elif mode == "translation":
                # check is question or repsponse translation
                if responses_trans:
                    assert self.inference_results is not None
                    source_type = "response"
                else:
                    source_type = "question"
                    
                # load object benchmark examples
                if self.ref_task is not None:
                    ref_dataset = load_dataset(
                        path=os.path.join(PROJECT_BASE_PATH, self.eval_args.ref_task_dir, self.ref_task),
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
                            trans_source=self.inference_prompts[subject][i],
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
                            trans_source=self.inference_results[subject][i],
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
                        translate_prompts[subject].append(trans_messages)
            else:
                raise ValueError(f"Input mode {mode} is invalid. Please specify one of 'inference' or 'check' instead.")
        
        if mode == "inference":
            return inference_prompts, test_codes
        elif mode == "translation":
            return None, translate_prompts
    
    async def translation(
        self,
        server_process: asyncio.subprocess.Process,
        model_name: str,
        data: Dict[str, List[Any]],
        output_path: str,
        progress_desc: str,
    ) -> Dict[str, List[Any]]:
        """
        Process subjects using the specified client and data with concurrency control.
        """
        if isinstance(self.categories, dict):
            catogories = self.categories.keys()
        elif isinstance(self.categories, list):
            catogories = self.categories
        results = {subj: [] for subj in catogories}
            
        total_progress_bar = tqdm(catogories, desc=progress_desc)

        # Define maximum concurrency
        max_concurrency = getattr(self.model_args, "vllm_max_concurrency", 100)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single_item(idx:int, messages: List[dict], subject: str, progress_bar: tqdm):
            """Processes a single item with semaphore-based concurrency control."""
            async with semaphore:
                try:
                    if messages[-1].get("origin_role", None) == "assistant":
                        # Not translate Code
                        translation_text = messages[-1].get("origin_content", None)
                    else:
                        # for question
                        translation_text, _ = await self.generate(
                            model=model_name,
                            system_prompt=None,
                            example=messages,
                            idx=idx,
                            generating_args=self.generating_args,
                        )
                    progress_bar.update(1) 
                    return idx, translation_text, messages[-1].get("origin_role", None), messages[-1].get("idx", None), messages[-1].get("uuid", None)
                except Exception as e:
                    progress_bar.update(1)
                    print(f"Error translating messages {json.dumps(messages, ensure_ascii=False, indent=2)}\n Error: {e}")
                    # return original text before translation
                    return idx, messages[-1].get("content", None), messages[-1].get("origin_role", None), messages[-1].get("idx", None), messages[-1].get("uuid", None)

        try:
            for subject in catogories:
                subject_results = [None] * len(data[subject])

                with tqdm(
                    total=len(data[subject]),
                    desc=catogories[subject]["name"] if isinstance(catogories, dict) else subject,
                    dynamic_ncols=True,
                ) as subject_progress_bar:

                    # Create tasks for all items under a subject
                    tasks = [
                        asyncio.create_task(process_single_item(idx, messages, subject, subject_progress_bar))
                        for idx, messages in enumerate(data[subject])
                    ]

                    # Collect results as tasks complete
                    for task in asyncio.as_completed(tasks):
                        origin_idx, translation_text, role, sen_idx, sen_uuid = await task
                        # for question
                        if sen_idx is not None and sen_uuid is not None:
                            subject_results[origin_idx] = {
                                "role": role,
                                "content": translation_text,
                                "idx": sen_idx,
                                "uuid": sen_uuid,
                            }
                        # for answer
                        elif translation_text is not None:
                            subject_results[origin_idx] = translation_text
                            
                # TODO: handle not grouped question results
                results[subject] = self.recover_trans_messages(subject_results)
                total_progress_bar.update(1)

        finally:
            # Ensure cleanup and save results
            await self.terminate_server(process=server_process)
            self.client = None
            self.save_data(data=results, output_path=os.path.join(self.save_folder, output_path))
            total_progress_bar.close()

        return results
    
    async def same_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        # general evaluation pipeline
        # need: inference model, checker model
        """Perform evaluation using inference and checker models with a progress bar."""
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Data path created: {self.save_folder}")
        ######################################### inference #########################################
        self.inference_prompts, test_codes = self.load_data(mode="inference", choices=choices)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.inference_prompts, output_path=os.path.join(self.save_folder, "inference_prompts.json"))
            self.save_data(data=test_codes, output_path=os.path.join(self.save_folder, "test_codes.json"))
            
        if self.inference_mode == "local":
            inference_process = await self.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            inference_process = None
            print("Using OpenAI API for inference.")

        self.set_client(mode="inference")
        print("Client setup complete.")

        self.inference_results = await self.process_subjects(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.inference_model_name_or_path,
            data=self.inference_prompts,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )
        print("Inference complete.")
        ####################################### compute score #######################################
        self.inference_results = self.post_process_response(self.inference_results)
        score_dict = self.comput_score(test_codes=test_codes, response_result=self.inference_results, k=1)
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
        
    async def diff_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Data path created: {self.save_folder}")
        ######################################## translation ########################################
        logger.info("============ Start question translation process. ============")
        self.inference_prompts, test_codes = self.load_data(mode="inference", choices=choices)
        _, ques_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=False)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.inference_prompts, output_path=os.path.join(self.save_folder, "inference_prompts.json"))
            self.save_data(data=ques_trans_prompts, output_path=os.path.join(self.save_folder, "ques_trans_prompts.json"))
        
        if self.translation_mode == "local":
            translation_process = await self.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            translation_process = None
            print("Using OpenAI API for inference.")

        self.set_client(mode="translation")
        print("Client setup complete.")
        
        self.translated_questions = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=ques_trans_prompts,
            output_path="translated_question_record.json",
            progress_desc="Trans Question Progress",
        )

        print("Question translated complete.")
        
        ######################################### inference #########################################
        logger.info("============ Start inference process. ============")
        if self.inference_mode == "local":
            inference_process = await self.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            inference_process = None
            print("Using OpenAI API for inference.")

        self.set_client(mode="inference")
        print("Client setup complete.")

        self.inference_results = await self.process_subjects(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.inference_model_name_or_path,
            data=self.translated_questions,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )
        print("Inference complete.")
        ####################################### compute score #######################################
        logger.info("============ Computing Score ============")
        self.inference_results = self.post_process_response(self.inference_results)
        score_dict = self.comput_score(test_codes=test_codes, response_result=self.inference_results, k=1)
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
        