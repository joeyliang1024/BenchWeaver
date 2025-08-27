import os
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Process, Queue
from datasets import load_dataset
from typing import Any, Dict, List, Optional
from ..evaluator import Evaluator
from ...template import Trans_Template
from ....extras.constants import PROJECT_BASE_PATH
from ....extras.logging import get_logger
from ...metric.translate import (
    eval_bleu,
    eval_chrf,
    eval_spbleu
)

logger = get_logger(__name__)

            
class TransEvaluator(Evaluator):
    eval_template: Trans_Template
    categories: dict
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args = args)
        self.re_init_categories()
        
    def re_init_categories(self) -> None:
        source_lang = self.model_args.source_lang 
        new_categories = {
            f"{source_lang}-{lang}":{
                "name": "Translation from {} to {}".format(source_lang, lang),
                "category": f"{source_lang}-{lang}",
            }
            for lang in self.categories.keys() if lang != source_lang
        }
        # Add the inverse direction as well
        inverse_categories = {
            f"{lang}-{source_lang}": {
                "name": f"Translation from {lang} to {source_lang}",
                "category": f"{lang}-{source_lang}",
            }
            for lang in self.categories.keys() if lang != source_lang
        }
        # Merge the two dictionaries
        new_categories.update(inverse_categories)
        self.categories = new_categories
        logger.info("Re-initialized categories.")
        
    def compute_tans_score(self, trans_result: Dict[str, List[Any]], trans_groundtruth: Dict[str, List[Any]]) -> Dict[str, Any]:    
        score_dict = {
            "Average": {
                "BLEU": [],
                "CHRF++": [],
                "SPBLEU": [],
            }
        } 
        for subject in self.categories.keys():
            target_lang = subject.split("-")[1]
            bleu = eval_bleu(predictions=trans_result[subject], references=trans_groundtruth[subject], lang=target_lang)
            chrf = eval_chrf(predictions=trans_result[subject], references=trans_groundtruth[subject])
            spbleu = eval_spbleu(predictions=trans_result[subject], references=trans_groundtruth[subject])
 
            score_dict[subject] = {
                "BLEU": round(bleu['bleu'] * 100, 4),
                "CHRF++": round(chrf['score'], 4),
                "SPBLEU": round(spbleu['bleu'] * 100, 4),
            }
            score_dict["Average"]["BLEU"].append(round(bleu['bleu'] * 100, 4))
            score_dict["Average"]["CHRF++"].append(round(chrf['score'], 4))
            score_dict["Average"]["SPBLEU"].append(round(spbleu['bleu'] * 100, 4))

        score_dict["Average"]["BLEU"] = round(np.mean(score_dict["Average"]["BLEU"]), 4)
        score_dict["Average"]["CHRF++"] = round(np.mean(score_dict["Average"]["CHRF++"]), 4)
        score_dict["Average"]["SPBLEU"] = round(np.mean(score_dict["Average"]["SPBLEU"]), 4)

        return score_dict

    
    def load_data(self, **kwargs):
        inference_prompts = {subj: [] for subj in self.categories.keys()}
        trans_groundtruth = {subj: [] for subj in self.categories.keys()}
        # Load datasets
        
        for subject in tqdm(self.categories.keys(), desc="Loading subjects"):
            # load dataset from folder
            source_lang, target_lang = subject.split("-")
            source_dataset = load_dataset(
                path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.eval_task),
                name=source_lang,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.hf_token,
                trust_remote_code=True,
            )
            target_dataset = load_dataset(
                path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.eval_task),
                name=target_lang,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.hf_token,
                trust_remote_code=True,
            )
            # prepare support set
            if source_dataset.get("train") and target_dataset.get("train"):
                # Determine how many examples you can safely sample
                n_shot = min(self.eval_args.n_shot, len(source_dataset["train"]), len(target_dataset["train"]))
                # Shuffle the source dataset to decide which IDs to sample
                shuffled_source = source_dataset["train"].shuffle()
                selected_ids = set(shuffled_source.select(range(n_shot))["id"])
                # Filter both datasets to only those selected IDs
                source_support_set = source_dataset["train"].filter(lambda example: example["id"] in selected_ids)
                target_support_set = target_dataset["train"].filter(lambda example: example["id"] in selected_ids)
                # Optionally: sort both sets by id to maintain alignment
                source_support_set = source_support_set.sort("id")
                target_support_set = target_support_set.sort("id")
            else:
                source_support_set = None
                target_support_set = None
                
            for i, (source_example, target_example) in enumerate(zip(source_dataset[self.eval_split], target_dataset[self.eval_split])):
                # format inference example
                if i < min(len(source_dataset[self.eval_split]), self.testing_size):
                    # format messages
                    messages, groundtruth = self.eval_template.format_inference_example(
                        source_example=source_example,
                        target_example=target_example,
                        source_support_set=source_support_set,
                        target_support_set=target_support_set,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        user_prompt=self.eval_args.user_prompt,
                    )
                    inference_prompts[subject].append(messages)
                    trans_groundtruth[subject].append([groundtruth])
        
        return inference_prompts, trans_groundtruth
    
    async def trans_eval(self, choices: List[str], subjects: List[str]) -> None:
        """
        Evaluates the translation performance of the pipeline.
        The scores include: BLEU, CHRF++, and GPT score.
        """
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        logger.info(f"Data path created: {self.save_folder}")
        ######################################### inference #########################################
        self.inference_prompts, trans_groundtruth = self.load_data()
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.inference_prompts, output_path=os.path.join(self.save_folder, "inference_prompts.json"))
            self.save_data(data=trans_groundtruth, output_path=os.path.join(self.save_folder, "trans_groundtruth.json"))
        
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
        logger.info("Client setup complete.")

        self.inference_results = await self.process_subjects(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.inference_model_name_or_path,
            data=self.inference_prompts,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )

        logger.info("Inference complete.")
        ####################################### compute score #######################################
        logger.info("============ Computing Score ============")
        score_dict = self.compute_tans_score(trans_result=self.inference_results, trans_groundtruth=trans_groundtruth)
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))