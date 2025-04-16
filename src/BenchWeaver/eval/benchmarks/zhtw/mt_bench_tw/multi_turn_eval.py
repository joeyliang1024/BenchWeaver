import os
import ast
import random
import numpy as np
from typing import Any, Dict, List, Literal, Tuple
from datasets import load_dataset
from tqdm.auto import tqdm
from ....evaluator.multi_turn.pipeline_eval import MultiTurnEvaluator
from ....template import get_mt_bench_tw_eval_template
from .....extras.constants import PROJECT_BASE_PATH

class MTBenchTWEvaluator(MultiTurnEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        
    def comput_score(self, check_results: Dict[str, List[Any]], subjects: List[str], checked_answers=None) -> Dict[str, float]:
        category_corrects = {subj: [] for subj in subjects}
        for subj in check_results.keys():
            for check_string in check_results[subj]:
                score = self.retrieve_answer(text=check_string, numerical=True)
                category_corrects[subj].append(score)
                category_corrects['Average'].append(score)
        # average score
        for subj in category_corrects.keys():
            category_corrects[subj] = np.mean(category_corrects[subj])

        return category_corrects
    
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
            eval_template = get_mt_bench_tw_eval_template(name=subject)
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