import os
import numpy as np
from typing import Any, Dict, List, Optional
from ..evaluator import Evaluator

class TransPipelineEvaluator(Evaluator):
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args = args)
    
    def compute_tans_score(self, checked_answers: Dict[str, List[Any]], check_results: Dict[str, List[Any]], subjects: List[str]) -> Dict[str, float]:
        category_corrects = {subj: np.array([], dtype="bool") for subj in subjects}
        
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
        
