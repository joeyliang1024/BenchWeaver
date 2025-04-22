import os
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from typing import Any, Dict, List, Optional
from ..evaluator import Evaluator
from ....extras.constants import PROJECT_BASE_PATH
from ....extras.logging import get_logger

logger = get_logger(__name__)

class TransEvaluator(Evaluator):
    categories: dict
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args = args)
        self.re_init_categories()
        
    def re_init_categories(self) -> None:
        source_lang = self.eval_args.source_lang 
        new_categories = {
            f"{source_lang}-{lang}":{
                "name": "Translation from {} to {}".format(source_lang, lang),
                "category": f"{source_lang}-{lang}",
            }
            for lang in self.categories.keys()
        }
        # Add the inverse direction as well
        inverse_categories = {
            f"{lang}-{source_lang}": {
                "name": f"Translation from {lang} to {source_lang}",
                "category": f"{lang}-{source_lang}",
            }
            for lang in self.categories.keys()
        }
        # Merge the two dictionaries
        new_categories.update(inverse_categories)
        self.categories = new_categories
        logger.info("Re-initialized categories.")
        
    def compute_tans_score(self, checked_answers: Dict[str, List[Any]], check_results: Dict[str, List[Any]], subjects: List[str]) -> Dict[str, float]:
        category_corrects = {subj: np.array([], dtype="bool") for subj in subjects}
    
    def load_data(self, mode, choices, responses_trans = False, check_source = "original", **kwargs):
        inference_prompts = {subj: [] for subj in self.categories.keys()}
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
                    messages = self.eval_template.format_inference_example(
                        target_data=dataset[self.eval_split][i],
                        support_set=support_set,
                        subject_name=self.categories[subject]["name"],
                        user_prompt=self.eval_args.user_prompt,
                        use_cot=self.eval_args.cot,
                    )
                    inference_prompts[subject].append(messages)
    
    
    async def tp_eval(self, choices: List[str], subjects: List[str]) -> None:
        """
        Evaluates the translation performance of the pipeline.
        The scores include: BLEU, CHRF++, and GPT score.
        """
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Data path created: {self.save_folder}")
        ######################################### inference #########################################
        _, self.inference_prompts = self.load_data(mode="inference", choices=choices)
        
        if self.inference_mode == "local":
            inference_process = await self.server.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
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
        
