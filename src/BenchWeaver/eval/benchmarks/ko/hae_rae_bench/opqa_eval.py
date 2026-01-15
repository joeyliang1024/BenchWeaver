import random
from typing import Dict, Literal, Tuple
from tqdm import tqdm
from .....data.huggingface_utils import load_hf_or_local_dataset
from ....template import get_hae_rae_bench_eval_template
from ....evaluator import OPQAEvaluator

class HAE_RAE_BENCHEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_hae_rae_bench_eval_template(self.eval_args.lang)
    
    def load_data(self, 
                  mode = Literal['inference', 'check', 'translation'],
                  choices = None,
                  responses_trans: bool = False,
                  check_source: Literal['original', 'translated'] = "original"
                  ) -> Tuple[Dict[str, list], Dict[str, list]]:
        # init data
        inference_prompts = {subj: [] for subj in self.categories.keys()}
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
                for i in range(min(len(dataset[self.eval_split]), self.testing_size)):
                    check_msg_list = self.eval_template.format_checker_example(
                        target_data=dataset[self.eval_split][i],
                        choices=["A", "B", "C", "D", "E"],
                        llm_response=self.inference_results[subject][i] if check_source == "original" else self.translated_responses[subject][i],
                    )
                    checker_prompts[subject].append(check_msg_list)
                
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
            return None, checker_prompts
        elif mode == "translation":
            return None, translate_prompts