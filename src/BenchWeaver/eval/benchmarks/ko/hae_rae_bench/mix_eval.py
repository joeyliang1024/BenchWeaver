import asyncio
import random
from typing import Any, Dict, List, Literal, Tuple
from .....data.huggingface_utils import load_hf_or_local_dataset
import numpy as np
from tqdm.auto import tqdm
from ....evaluator import Evaluator
from ....template import get_hae_rae_bench_eval_template

class HAE_RAE_BENCHEvaluator(Evaluator):
    server_process: asyncio.subprocess.Process
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_hae_rae_bench_eval_template(self.eval_args.lang)
        self.options_per_questionuestion = 5
        
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
                    if dataset.get(self.train_split) is not None:
                        support_set = (
                            dataset[self.train_split]
                            .select(range(min(self.eval_args.n_shot, len(dataset[self.train_split]))))
                            .shuffle()
                        )
                    else:
                        support_set = None
                    messages = self.eval_template.format_inference_example(
                        target_data=dataset[self.eval_split][i],
                        support_set=support_set,
                        user_prompt=self.eval_args.user_prompt,
                        use_cot=self.eval_args.cot,
                    )
                    inference_prompts[subject].append(messages)
             
            elif mode == "check":
                assert self.inference_results is not None
                # opqa
                if subject in ['lyrics_denoising', 'proverbs_denoising']:
                    for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                        check_msg_list = self.eval_template.format_checker_example(
                            target_data=dataset[self.eval_split][i],
                            choices=["A", "B", "C", "D", "E"],
                            llm_response=self.inference_results[subject][i] if check_source == "original" else self.translated_responses[subject][i],
                        )
                        checker_prompts[subject].append(check_msg_list)
                # mcqa
                else:
                    for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                        check_msg_list, answer_list = self.eval_template.format_checker_example(
                            target_data=dataset[self.eval_split][i],
                            choices=["A", "B", "C", "D", "E"],
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
                    ref_dataset = load_hf_or_local_dataset(
                        exists_on_hf=self.exists_on_hf,
                        path=self.eval_args.ref_task_dir,
                        task_name=self.ref_task,
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
        # 初始化統計，確保所有 category 都有位置
        category_corrects = {}

        for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
            category_name = self.categories[subject]["category"]
            if category_name not in category_corrects:
                category_corrects[category_name] = {"corrects": 0, "true_mask_count": 0, "total_questions": 0}

            preds_arr = np.array([self.retrieve_answer(ans) for ans in check_results[subject]])
            corrects = (preds_arr == 'true')
            category_corrects[category_name]["corrects"] += corrects.sum()
            category_corrects[category_name]["true_mask_count"] += len(preds_arr)
        
        # 產出結果
        final_results = {}
        avg_num, avg_den = 0, 0

        for cat_name, record in category_corrects.items():
            denominator = record["true_mask_count"]

            if denominator > 0:
                score = round(100 * (record["corrects"] / denominator), 4)
                final_results[cat_name] = score
                avg_num += record["corrects"]
                avg_den += denominator

        if avg_den > 0:
            final_results["Average"] = round(100 * (avg_num / avg_den), 4)

        return final_results
    # def comput_score(self, checked_answers: Dict[str, List[Any]], check_results: Dict[str, List[Any]], subjects: List[str]) -> Dict[str, float]:
    #     # 初始化統計，確保所有 category 都有位置
    #     category_corrects = {}
    # 
    #     options_per_q = getattr(self, 'options_per_question', 5) # 建議在外面設定好
    # 
    #     for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
    #         category_name = self.categories[subject]["category"]
    #         if category_name not in category_corrects:
    #             category_corrects[category_name] = {"corrects": 0, "true_mask_count": 0, "total_questions": 0}
    # 
    #         raw_preds = [self.retrieve_answer(ans) for ans in check_results[subject]]
    # 
    #         if subject in ['lyrics_denoising', 'proverbs_denoising']:
    #             # --- OPQA 邏輯 (逐項計分) ---
    #             preds_arr = np.array(raw_preds)
    #             corrects = (preds_arr == 'true')
    #             category_corrects[category_name]["corrects"] += corrects.sum()
    #             category_corrects[category_name]["true_mask_count"] += len(preds_arr)
    #         else:
    #             # --- MCQA 邏輯 (按題計分) ---
    #             answers = np.array(checked_answers[subject])
    #             predictions = np.array(raw_preds)
    # 
    #             num_questions = len(answers) // options_per_q
    #             reshaped_ans = answers.reshape(num_questions, options_per_q)
    #             reshaped_preds = predictions.reshape(num_questions, options_per_q)
    # 
    #             q_correct_count = 0
    #             for i in range(num_questions):
    #                 ans_idx = np.where(reshaped_ans[i] == 'true')[0]
    #                 pred_idx = np.where(reshaped_preds[i] == 'true')[0]
    # 
    #                 # 嚴格判定：僅當模型只選一個且選對時
    #                 if len(pred_idx) == 1 and len(ans_idx) == 1:
    #                     if pred_idx[0] == ans_idx[0]:
    #                         q_correct_count += 1
    # 
    #             category_corrects[category_name]["corrects"] += q_correct_count
    #             category_corrects[category_name]["total_questions"] += num_questions
    # 
    #     # 產出結果
    #     final_results = {}
    #     avg_num, avg_den = 0, 0
    # 
    #     for cat_name, record in category_corrects.items():
    #         # 自動判斷分母：如果有 total_questions (MCQA) 則優先使用，否則使用 true_mask_count (OPQA)
    #         if record["total_questions"] > 0:
    #             denominator = record["total_questions"]
    #         else:
    #             denominator = record["true_mask_count"]
    # 
    #         if denominator > 0:
    #             score = round(100 * (record["corrects"] / denominator), 4)
    #             final_results[cat_name] = score
    #             avg_num += record["corrects"]
    #             avg_den += denominator
    # 
    #     if avg_den > 0:
    #         final_results["Average"] = round(100 * (avg_num / avg_den), 4)
    # 
    #     return final_results

    # def compute_score(self, checked_answers: Dict[str, List[Any]], check_results: Dict[str, List[Any]], subjects: List[str]) -> Dict[str, float]:
    #     category_corrects = {score: {"corrects": 0, "true_mask_count": 0} for score in subjects}
    #     for subject in tqdm(self.categories.keys(), desc="Compute subjects"):
    #         category_name = self.categories[subject]["category"]
    #         if subject in ['lyrics_denoising', 'proverbs_denoising']:
    #             # OPQA
    #             corrects = np.array(['true'] * len(check_results[subject])) == np.array([self.retrieve_answer(answer) for answer in check_results[subject]])
    #             true_mask = np.array([True] * len(check_results[subject]))
    #         else:
    #             # MCQA
    #             answers = np.array(checked_answers[subject])
    #             predictions = np.array([self.retrieve_answer(ans) for ans in check_results[subject]])
    #             true_mask: np.ndarray = answers == 'true' # Mask for when the answer is 'true'
    #             # Compare predictions and answers, only where answer is 'true'
    #             corrects: np.ndarray = (predictions == 'true') & true_mask  # correct when answer is 'true' and prediction is 'true'
    #         category_corrects[category_name]["corrects"] += corrects.sum()
    #         category_corrects[category_name]["true_mask_count"] += true_mask.sum()
    #         category_corrects["Average"]['corrects'] += corrects.sum()
    #         category_corrects["Average"]['true_mask_count'] += true_mask.sum()
    #         
    #     return {
    #         category_name: round(100 * (record_dict['corrects'] / record_dict['true_mask_count']), 4)
    #             for category_name, record_dict in category_corrects.items() if record_dict['true_mask_count'] > 0
    #     }
    
    