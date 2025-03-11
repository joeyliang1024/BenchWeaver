import os
from .....extras.constants import GSM8K_SUBJECTS
from ....evaluator import OPQAEvaluator
from ....template import get_gsm8k_eval_template

class GSM8KEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_gsm8k_eval_template(self.eval_args.lang)
        
    # async def eval(self):
    #     """Perform evaluation using inference and checker models with a progress bar."""
    #     # ensure save folder exists
    #     os.makedirs(self.save_folder, exist_ok=False)
    #     print(f"Data path created: {self.save_folder}")
    #     
    #     # inference
    #     inference_data = self.load_data(mode="inference")
    #     
    #     if self.inference_mode == "local":
    #         inference_process = await self.server.setup_server(
    #             model_path=getattr(self.model_args, "model_name_or_path"),
    #             model_name="inference_model",
    #             max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
    #         )
    #         print("Server setup complete.")
    #         self.set_client(mode="inference")
    #         print("Client setup complete.")
    #         self.inference_results  = await self.process_subjects(
    #             server_process=inference_process,
    #             model_name="inference_model",
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

    #     # check
    #     checked_prompts = self.load_data(mode="check")
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
    #     score_dict = self.comput_score(check_results=check_results, subjects=GSM8K_SUBJECTS)
    #     self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
        
    
        
    
    
    