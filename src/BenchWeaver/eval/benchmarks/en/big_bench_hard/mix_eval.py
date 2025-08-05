import asyncio
import os
import random
from typing import Any, Dict, List, Literal, Tuple, Union
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
from ....evaluator import Evaluator
from .....extras.constants import PROJECT_BASE_PATH
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
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)): 
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
                    for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                        check_msg_list = costume_eval_template.format_checker_example(
                            target_data=dataset[self.eval_split][i],
                            is_mcqa=False,
                            llm_response=self.inference_results[subject][i] if check_source == "original" else self.translated_responses[subject][i],
                        )
                        checker_prompts[subject].append(check_msg_list)
                # mcqa
                else:
                    for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
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
            return None, inference_prompts
        elif mode == "check":
            return checker_answers, checker_prompts
        elif mode == "translation":
            return None, translate_prompts
    
    def comput_score(self, checked_answers: Dict[str, List[Any]], check_results: Dict[str, List[Any]], subjects: List[str]) -> Dict[str, float]:
        category_corrects = {score: {"corrects": 0, "true_mask_count": 0} for score in subjects}
        for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
            category_name = self.categories[subject]["category"]
            if subject in ['formal_fallacies', 'object_counting']:
                # OPQA
                corrects = np.array(['true'] * len(check_results[subject])) == np.array([self.retrieve_answer(answer) for answer in check_results[subject]])
                true_mask = np.array([True] * len(check_results[subject]))
            else:
                # MCQA
                answers = np.array(checked_answers[subject])
                predictions = np.array([self.retrieve_answer(ans) for ans in check_results[subject]])
                true_mask: np.ndarray = answers == 'true' # Mask for when the answer is 'true'
                # Compare predictions and answers, only where answer is 'true'
                corrects: np.ndarray = (predictions == 'true') & true_mask  # correct when answer is 'true' and prediction is 'true'
            category_corrects[category_name]["corrects"] += corrects.sum()
            category_corrects[category_name]["true_mask_count"] += true_mask.sum()
            category_corrects["Average"]['corrects'] += corrects.sum()
            category_corrects["Average"]['true_mask_count'] += true_mask.sum()
            
        return {
            category_name: round(100 * (record_dict['corrects'] / record_dict['true_mask_count']), 4)
                for category_name, record_dict in category_corrects.items() if record_dict['true_mask_count'] > 0
        }
    
