import asyncio
import random
from typing import Any, Dict, List, Literal, Tuple
import numpy as np
from ....data.huggingface_utils import load_hf_or_local_dataset
from tqdm.auto import tqdm
from ..evaluator import Evaluator
from ...template import MCQA_Template

class OQEvaluator(Evaluator):
    eval_template: MCQA_Template
    server_process: asyncio.subprocess.Process
    def __init__(self, args):
        super().__init__(args=args)
        self.options_per_question = None
        
    def comput_score(self, checked_answers: Dict[str, List[Any]], check_results: Dict[str, List[Any]], subjects: List[str]) -> Dict[str, float]:
        category_corrects = {score: {"corrects": 0, "true_mask_count": 0, "total_questions": 0} for score in subjects}
        if "Average" not in category_corrects:
            category_corrects["Average"] = {"corrects": 0, "true_mask_count": 0, "total_questions": 0}

        for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
            category_name = self.categories[subject]["category"]
            if category_name not in category_corrects:
                category_corrects[category_name] = {"corrects": 0, "true_mask_count": 0, "total_questions": 0}

            answers = np.array(checked_answers[subject])
            predictions = np.array([self.retrieve_answer(ans) for ans in check_results[subject]])

            # --- 情況 A: 單選題模式 (self.options_per_question 不為 None) ---
            if getattr(self, 'options_per_question', None) is not None:
                num_options = len(answers)
                num_questions = num_options // self.options_per_question

                # Reshape 成 (題目數, 選項數)
                reshaped_answers = answers[:num_questions * self.options_per_question].reshape(num_questions, self.options_per_question)
                reshaped_preds = predictions[:num_questions * self.options_per_question].reshape(num_questions, self.options_per_question)

                correct_count = 0
                for i in range(num_questions):
                    true_indices_in_ans = np.where(reshaped_answers[i] == 'true')[0]
                    true_indices_in_pred = np.where(reshaped_preds[i] == 'true')[0]

                    # 嚴格判定：模型必須「只選一個」且「選對那個」
                    if len(true_indices_in_pred) == 1 and len(true_indices_in_ans) == 1:
                        if true_indices_in_pred[0] == true_indices_in_ans[0]:
                            correct_count += 1

                category_corrects[category_name]["corrects"] += correct_count
                category_corrects[category_name]["total_questions"] += num_questions
                category_corrects["Average"]['corrects'] += correct_count
                category_corrects["Average"]['total_questions'] += num_questions

            # --- 情況 B: 原始多選模式 (self.options_per_question 為 None) ---
            else:
                true_mask = (answers == 'true')
                corrects = (predictions == 'true') & true_mask

                category_corrects[category_name]["corrects"] += corrects.sum()
                category_corrects[category_name]["true_mask_count"] += true_mask.sum()
                category_corrects["Average"]['corrects'] += corrects.sum()
                category_corrects["Average"]['true_mask_count'] += true_mask.sum()

        # --- 計算最終得分 ---
        result = {}
        for cat, data in category_corrects.items():
            if getattr(self, 'options_per_question', None) is not None:
                # 單選模式分母是總題數
                if data['total_questions'] > 0:
                    result[cat] = round(100 * (data['corrects'] / data['total_questions']), 4)
            else:
                # 多選模式分母是 true 的總個數
                if data['true_mask_count'] > 0:
                    result[cat] = round(100 * (data['corrects'] / data['true_mask_count']), 4)

        return result
        
    def load_data(self, 
                  mode = Literal['inference', 'check', 'translation'],
                  choices = List[str],
                  responses_trans: bool = False,
                  check_source: Literal['original', 'translated'] = "original"
                  ) -> Tuple[Dict[str, list], Dict[str, list]]:
        """Load and format data for evaluation."""
        # init data
        inference_prompts = {subj: [] for subj in self.categories.keys()}
        checker_answers = {subj: [] for subj in self.categories.keys()}
        checker_prompts = {subj: [] for subj in self.categories.keys()}
        translate_prompts = {subj: [] for subj in self.categories.keys()}
        # Load datasets
        for subject in tqdm(self.categories.keys(), desc="Loading subjects"):
            # load dataset from folder
            dataset = load_hf_or_local_dataset(
                exists_on_hf=self.exists_on_hf,
                path=self.eval_args.task_dir,
                task_name=self.eval_task,
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.hf_token,
                trust_remote_code=True,
            )
            # Prepare examples for evaluation
            if mode == "inference":
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)): 
                    if dataset.get(self.train_split):
                        support_set = (
                            dataset[self.train_split]
                            .select(range(min(self.eval_args.n_shot, len(dataset[self.train_split]))))
                            .shuffle()
                        )
                    else:
                        support_set = None
                    messages = self.eval_template.format_inference_example(
                        target_data=dataset[self.eval_split][i],
                        choices=choices,
                        support_set=support_set,
                        subject_name=self.categories[subject]["name"],
                        user_prompt=self.eval_args.user_prompt,
                        use_cot=self.eval_args.cot,
                    )
                    inference_prompts[subject].append(messages)
            
            elif mode == "check":
                assert self.inference_results is not None
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                    check_msg_list, answer_list = self.eval_template.format_checker_example(
                        choices=choices,
                        target_data=dataset[self.eval_split][i],
                        llm_response=self.inference_results[subject][i] if check_source == "original" else self.translated_responses[subject][i],
                        criteria_prompt=self.eval_args.criteria_prompt,
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
                    ref_dataset = load_hf_or_local_dataset(
                        exists_on_hf=self.exists_on_hf,
                        path=self.eval_args.ref_task_dir,
                        task_name=self.eval_task,
                        name=random.choice(list(self.ref_categories.keys())),
                        cache_dir=self.model_args.cache_dir,
                        download_mode=self.eval_args.download_mode,
                        token=self.hf_token,
                        trust_remote_code=True,
                    )
                    support_set = (
                            ref_dataset[self.eval_split]
                            .shuffle()
                            .select(range(min(self.eval_args.n_shot, len(ref_dataset[self.eval_split]))))
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
        
    #def comput_score(self, checked_answers: Dict[str, List[Any]], check_results: Dict[str, List[Any]], subjects: List[str]) -> Dict[str, float]:
    #    category_corrects = {score: {"corrects": 0, "true_mask_count": 0} for score in subjects}
    #
    #    for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
    #        category_name = self.categories[subject]["category"]
    #        answers = np.array(checked_answers[subject])
    #        predictions = np.array([self.retrieve_answer(ans) for ans in check_results[subject]])
    #        # Mask for when the answer is 'true'
    #        true_mask: np.ndarray = answers == 'true'
    #        # Compare predictions and answers, only where answer is 'true'
    #        corrects: np.ndarray = (predictions == 'true') & true_mask  # correct when answer is 'true' and prediction is 'true'
    #        # Update the corrects and true_mask counts
    #        if category_name not in category_corrects:
    #            category_corrects[category_name] = {"corrects": 0, "true_mask_count": 0}
    #        category_corrects[category_name]["corrects"] += corrects.sum()
    #        category_corrects[category_name]["true_mask_count"] += true_mask.sum()
    #        category_corrects["Average"]['corrects'] += corrects.sum()
    #        category_corrects["Average"]['true_mask_count'] += true_mask.sum()
    #    
    #    return {
    #        category_name: round(100 * (record_dict['corrects'] / record_dict['true_mask_count']), 4)
    #            for category_name, record_dict in category_corrects.items() if record_dict['true_mask_count'] > 0
    #    }