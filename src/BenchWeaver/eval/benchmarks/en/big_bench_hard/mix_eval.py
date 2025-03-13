import asyncio
import json
import os
import random
from typing import Any, Dict, List, Literal, Tuple, Union
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
from ....evaluator import Evaluator
from .....extras.constants import PROJECT_BASE_PATH, BIG_BENCH_HARD_SUBJECTS
from ....template import BigBenchHard_Template, get_big_bench_hard_eval_template

class BigBenchHardEvaluator(Evaluator):
    server_process: asyncio.subprocess.Process
    eval_template: BigBenchHard_Template
    def __init__(self, args):
        super().__init__(args=args)
    
    def load_data(self, 
                  mode = Literal['inference', 'check', 'translation'],
                  choices = None,
                  responses_trans: bool = False,
                  check_source: Literal['original', 'translated'] = "original"
                  ) -> Tuple[Dict[str, list], Dict[str, list]]:
        # init data
        inference_prompts = {subj: [] for subj in self.categories.keys()}
        checker_answers = {subj: [] for subj in self.categories.keys()}
        checker_prompts = {subj: [] for subj in self.categories.keys()}
        translate_prompts = {subj: [] for subj in self.categories.keys()}
        # Load datasets
        for subject in tqdm(self.categories.keys(), desc="Loading subjects"):
            costume_eval_template = get_big_bench_hard_eval_template(name=subject)
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
                for i in range(len(dataset[self.eval_split])): 
                    if dataset.get("train"):
                        support_set = (
                            dataset["train"]
                            .shuffle()
                            .select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                        )
                    else:
                        support_set = None
                    messages = costume_eval_template.format_inference_example(
                        target_data=dataset[self.eval_split][i],
                        support_set=support_set,
                        user_prompt=self.eval_args.user_prompt,
                        use_cot=self.eval_args.cot,
                    )
                    inference_prompts[subject].append(messages)
                    
            elif mode == "check":
                assert self.inference_results is not None
                # opqa
                if subject in ['formal_fallacies', 'object_counting']:
                    for i in range(len(dataset[self.eval_split])):
                        check_msg_list = costume_eval_template.format_checker_example(
                            target_data=dataset[self.eval_split][i],
                            is_mcqa=False,
                            llm_response=self.inference_results[subject][i] if check_source == "original" else self.translated_responses[subject][i],
                        )
                        checker_prompts[subject].append(check_msg_list)
                # mcqa
                else:
                    for i in range(len(dataset[self.eval_split])):
                        check_msg_list, answer_list = costume_eval_template.format_checker_example(
                            target_data=dataset[self.eval_split][i],
                            is_mcqa=True,
                            llm_response=self.inference_results[subject][i] if check_source == "original" else self.translated_responses[subject][i],
                        )
                        checker_answers[subject] += answer_list
                        checker_prompts[subject] += check_msg_list
                        
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
                        path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.ref_task),
                        name=random.choice(self.ref_categories.keys()),
                        cache_dir=self.model_args.cache_dir,
                        download_mode=self.eval_args.download_mode,
                        token=self.hf_token,
                        trust_remote_code=True,
                    )
                    support_set = (
                            dataset["test"]
                            .shuffle()
                            .select(range(min(self.eval_args.n_shot, len(ref_dataset["test"]))))
                        )
                else:
                    support_set = None
                
                for i in range(len(dataset[self.ref_split])):
                    # format translation example
                    if source_type == "question":
                        trans_messages = self.trans_template.format_translation_example(
                            trans_source=self.inference_prompts[subject][i],
                            source_type=source_type,
                            source_lang=self.eval_args.source_lang,
                            target_lang=self.eval_args.target_lang,
                            choices=choices,
                            support_set=support_set,
                            use_cot=self.eval_args.cot,
                        )
                        # list of messages
                        translate_prompts[subject] += trans_messages
                    elif source_type == "response":
                        trans_messages = self.trans_template.format_translation_example(
                            trans_source=self.inference_results[subject][i],
                            source_type=source_type,
                            source_lang=self.eval_args.source_lang,
                            target_lang=self.eval_args.target_lang,
                            choices=choices,
                            support_set=support_set,
                            use_cot=self.eval_args.cot,
                        )
                        # message list
                        translate_prompts[subject].append(trans_messages)
            else:
                raise ValueError(f"Input mode {mode} is invalid. Please specify one of 'inference' or 'check' instead.")
    
        if mode == "inference":
            return None, inference_prompts
        elif mode == "check":
            return checker_answers, checker_prompts
        elif mode == "translation":
            return None, translate_prompts
         
    def comput_score(self, checked_answers: Dict[str, List[Any]], check_results: Dict[str, List[Any]], subjects: List[str]) -> Dict[str, float]:
        category_corrects = {subj: np.array([], dtype="bool") for subj in subjects}

        for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
            if subject in ['formal_fallacies', 'object_counting']:
                corrects = np.array(['true'] * len(check_results[subject])) == np.array([self.retrieve_answer(answer) for answer in check_results[subject]])
            else:
                corrects = np.array(checked_answers[subject]) == np.array([self.retrieve_answer(answer) for answer in check_results[subject]])
            category_corrects[subject] = np.concatenate([category_corrects[subject], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)

        return {category_name: round(100 * np.mean(category_array), 4) 
                for category_name, category_array in category_corrects.items()}
    
    # async def eval(self):
    #     os.makedirs(self.save_folder, exist_ok=True)
    #     print(f"Data path created: {self.save_folder}")
    #     
    #     # inference
    #     inference_data = self.load_data(mode="inference")
    #     
    #     if self.inference_mode == "local":
    #         print("Setting server...")
    #         inference_process = await self.server.setup_server(
    #             model_path=getattr(self.model_args, "model_name_or_path"),
    #             model_name=self.inference_model_name,
    #             max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
    #         )
    #         print("Server setup complete.")
    #         self.set_client(mode="inference")
    #         print("Client setup complete.")
    #         self.inference_results  = await self.process_subjects(
    #             server_process=inference_process,
    #             model_name=self.inference_model_name,
    #             data=inference_data,
    #             prompt_key="system_prompt",
    #             output_path="inference_results.json",
    #             progress_desc="Inference Progress",
    #         )
    #     else:
    #         self.set_client(mode="inference")
    #         print("Client setup complete.")
    #         self.inference_results  = await self.process_subjects(
    #             server_process=None,
    #             model_name=getattr(self.model_args, "model_name_or_path"),
    #             data=inference_data,
    #             prompt_key="system_prompt",
    #             output_path="inference_results.json",
    #             progress_desc="Inference Progress",
    #         )
    #     print("Inference complete.")
    #     
    #     # check
    #     checked_answers, checked_prompts = self.load_data(mode="check")
    # 
    #     if self.check_mode == "local":
    #         checker_process = await self.server.setup_server(
    #             model_path=getattr(self.model_args, "checker_model_name_or_path"),
    #             model_name="checker_model",
    #             max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
    #         )
    #         print("Server setup complete.")
    #         self.set_client(mode="check")
    #         print("Client setup complete.")
    #         check_results = await self.process_subjects(
    #             server_process=checker_process,
    #             model_name="checker_model",
    #             data=checked_prompts,
    #             prompt_key="criteria_system_prompt",
    #             output_path="check_results.json",
    #             progress_desc="Check Progress",
    #         )
    #     else:
    #         self.set_client(mode="check")
    #         print("Client setup complete.")
    #         check_results = await self.process_subjects(
    #             server_process=None,
    #             model_name=getattr(self.model_args, "checker_model_name_or_path"),
    #             data=checked_prompts,
    #             prompt_key="criteria_system_prompt",
    #             output_path="check_results.json",
    #             progress_desc="Check Progress",
    #         )
    #     print("Check complete.")
    #     # compute score
    #     score_dict = self.comput_score(checked_answers=checked_answers, check_results=check_results, subjects=BIG_BENCH_HARD_SUBJECTS)
    #     self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
        
        