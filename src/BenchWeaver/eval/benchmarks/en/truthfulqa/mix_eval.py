import asyncio
import json
import os
import random
from typing import Any, Dict, List, Literal, Tuple, Union
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
from ....evaluator import Evaluator
from .....extras.constants import PROJECT_BASE_PATH, TRUTHFULQA_SCORES
from ....template import get_truthfulqa_eval_template

class TruthfulQAEvaluator(Evaluator):
    server_process: asyncio.subprocess.Process
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_truthfulqa_eval_template(self.eval_args.lang)
        self.categories:List[str] = TRUTHFULQA_SCORES[1:] # override
    
    def load_data(self, 
                  mode = Literal['inference', 'check', 'translation'],
                  choices = None,
                  responses_trans: bool = False,
                  check_source: Literal['original', 'translated'] = "original"
                  ) -> Tuple[Dict[str, list], Dict[str, list]]:
        """Load and format data for evaluation."""
        # init data
        inference_prompts = {subj: [] for subj in self.categories}
        checker_answers = {subj: [] for subj in self.categories}
        checker_prompts = {subj: [] for subj in self.categories}
        translate_prompts = {subj: [] for subj in self.categories}
        # Load datasets
        for data_type in tqdm(self.categories, desc="Loading subjects"):
            # load dataset from folder
            dataset = load_dataset(
                path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.eval_task),
                name="merge",
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
                    messages = self.eval_template.format_inference_example(
                        target_data=dataset[self.eval_split][i],
                        type=data_type,
                        support_set=support_set,
                        user_prompt=self.eval_args.user_prompt,
                        use_cot=self.eval_args.cot,
                    )
                    inference_prompts[data_type].append(messages)
            
            elif mode == "check":
                assert self.inference_results is not None
                if data_type == "generation":
                    for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                        check_msg_list = self.eval_template.format_checker_example(
                            target_data=dataset[self.eval_split][i],
                            type=data_type,
                            llm_response=self.inference_results[data_type][i] if check_source == "original" else self.translated_responses[data_type][i],
                        )
                        checker_prompts[data_type].append(check_msg_list)
                else:
                    for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                        check_msg_list, answer_list = self.eval_template.format_checker_example(
                            target_data=dataset[self.eval_split][i],
                            type=data_type,
                            llm_response=self.inference_results[data_type][i] if check_source == "original" else self.translated_responses[data_type][i],
                        )
                        checker_answers[data_type] += answer_list
                        checker_prompts[data_type] += check_msg_list
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
                
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                    # format translation example
                    if source_type == "question":
                        trans_messages = self.trans_template.format_translation_example(
                            trans_source=self.inference_prompts[data_type][i],
                            source_type=source_type,
                            source_lang=self.model_args.source_lang,
                            target_lang=self.model_args.target_lang,
                            choices=choices,
                            support_set=support_set,
                            use_cot=self.eval_args.cot,
                        )
                        # list of messages
                        translate_prompts[data_type] += trans_messages
                    elif source_type == "response":
                        trans_messages = self.trans_template.format_translation_example(
                            trans_source=self.inference_results[data_type][i],
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
                        translate_prompts[data_type].append(trans_messages)
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

        for subject in tqdm(self.categories, desc="Compute subjects"):
            if subject == "generation":
                corrects = np.array(['true'] * len(check_results[subject])) == np.array([self.retrieve_answer(answer) for answer in check_results[subject]])
            else:
                corrects = np.array(checked_answers[subject]) == np.array([self.retrieve_answer(answer) for answer in check_results[subject]])
            category_corrects[subject] = np.concatenate([category_corrects[subject], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)

        return {category_name: round(100 * np.mean(category_array), 4) 
                for category_name, category_array in category_corrects.items()}
    
    async def process_subjects(
        self,
        server_process: asyncio.subprocess.Process,
        model_name: str,
        data: Dict[str, List[Any]],
        prompt_key: str,
        output_path: str,
        progress_desc: str,
    ) -> Dict[str, List[Any]]:
        """Process subjects using the specified client and data."""
        results = {subj: [] for subj in self.categories}
        total_progress_bar = tqdm(self.categories, desc=progress_desc)

        # Define maximum concurrency
        max_concurrency = 32
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_item(idx: int, messages: Any, subject: str, progress_bar: tqdm):
            """Process a single item using semaphore."""
            async with semaphore:
                try:
                    result, origin_idx = await self.generate(
                        model=model_name,
                        system_prompt=getattr(self.eval_args, prompt_key),
                        example=messages,
                        idx=idx,
                        generating_args=self.generating_args,
                    )
                    progress_bar.update(1)
                    return origin_idx, result
                except Exception as e:
                    progress_bar.update(1)
                    print(f"Error processing item {idx} in subject {subject}: {e}")
                    return idx, None

        try:
            for subject in self.categories:
                subject_results = [None] * len(data[subject])
                with tqdm(
                    total=len(data[subject]),
                    desc=subject,
                    dynamic_ncols=True,
                ) as subject_progress_bar:
                    # Create tasks for all items under the subject
                    tasks = [
                        asyncio.create_task(process_item(idx, messages, subject, subject_progress_bar))
                        for idx, messages in enumerate(data[subject])
                    ]

                    # Process tasks as they complete
                    for coro in asyncio.as_completed(tasks):
                        origin_idx, result = await coro
                        if result is not None:
                            subject_results[origin_idx] = result

                results[subject] = subject_results
                total_progress_bar.update(1)

        finally:
            print(f"Terminating server process: {server_process}")
            await self.terminate_server(process=server_process)
            print(f"Server process terminated: {server_process}")
            self.save_data(data=results, output_path=os.path.join(self.save_folder, output_path))
            total_progress_bar.close()

        return results
