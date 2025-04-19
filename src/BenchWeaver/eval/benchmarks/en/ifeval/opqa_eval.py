import os
from typing import List, Tuple, Dict
from datasets import load_dataset, Dataset
from .source_code.evaluation_main import evaluate_instruction_following
from ....evaluator import OPQAEvaluator
from ....template import get_ifeval_eval_template
from .....extras.constants import PROJECT_BASE_PATH
from .....extras.logging import get_logger

logger = get_logger(__name__)

class IFEvalEvaluator(OPQAEvaluator):
    def __init__(self, args):
        super().__init__(args=args)
        self.eval_template = get_ifeval_eval_template(self.eval_args.lang)
        
    def format_source_code_input(self, response_dict: Dict[str, list]) -> Tuple[Dataset, Dict[str, list]]:
        """
        Format the input same as the format as the source code.
        """
        # load the input 
        # min(len(dataset[self.eval_split]), self.testing_size)
        dataset_dict = load_dataset(
                path=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, self.eval_task),
                name="all",
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.hf_token,
                trust_remote_code=True,
            )
                # Extract the specified split (e.g., "test")
        inputs = dataset_dict[self.eval_split]

        # Select the top N examples, if testing_size is specified
        if self.testing_size is not None:
            inputs = inputs.select(range(min(self.testing_size, len(inputs))))
        # load the response
        new_response_dict = {
            'all': [
                {
                    "prompt": row['question'],
                    "response": response,
                }
                for row, response in zip(inputs, response_dict['all'])
            ]
        }
        return inputs, new_response_dict
    
    async def same_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        # general evaluation pipeline
        # need: inference model, checker model
        """Perform evaluation using inference and checker models with a progress bar."""
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Data path created: {self.save_folder}")
        ######################################### inference #########################################
        _, self.inference_prompts = self.load_data(mode="inference", choices=choices)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.inference_prompts, output_path=os.path.join(self.save_folder, "inference_prompts.json"))
        
        if self.inference_mode == "local":
            inference_process = await self.server.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
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
        input_data, input_response_data = self.format_source_code_input(response_dict=self.inference_results)
        score_dict = evaluate_instruction_following(
            origin_dataset=input_data,
            response_dict=input_response_data,
        )
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
            
    async def diff_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        # specific evaluation pipeline
        # need: trranslator model, inference model, checker model
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Data path created: {self.save_folder}")
        ######################################## translation ########################################
        logger.info("============ Start question translation process. ============")
        _, self.inference_prompts = self.load_data(mode="inference", choices=choices)
        _, ques_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=False)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.inference_prompts, output_path=os.path.join(self.save_folder, "inference_prompts.json"))
            self.save_data(data=ques_trans_prompts, output_path=os.path.join(self.save_folder, "ques_trans_prompts.json"))
        
        if self.translation_mode == "local":
            translation_process = await self.server.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            print("Local vLLM server setup complete.")
        else:
            translation_process = None
            print("Using OpenAI API for inference.")

        self.set_client(mode="translation")
        print("Client setup complete.")
        
        self.translated_questions = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=ques_trans_prompts,
            output_path="translated_question_record.json",
            progress_desc="Trans Question Progress",
        )

        print("Question translated complete.")
        
        ######################################### inference #########################################
        logger.info("============ Start inference process. ============")
        if self.inference_mode == "local":
            inference_process = await self.server.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
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
            data=self.translated_questions,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )

        print("Inference complete.")
        ######################################## translation ########################################
        logger.info("============ Start response translation process. ============")
        _, resp_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=True)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=resp_trans_prompts, output_path=os.path.join(self.save_folder, "resp_trans_prompts.json"))
        
        if self.translation_mode == "local":
            translation_process = await self.server.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            print("Local vLLM server setup complete.")
        else:
            translation_process = None
            print("Using OpenAI API for inference.")

        self.set_client(mode="translation")
        print("Client setup complete.")
        
        self.translated_responses = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=resp_trans_prompts,
            output_path="translated_response_record.json",
            progress_desc="Trans Response Progress",
        )

        print("Question translated complete.")
        ####################################### compute score #######################################
        input_data, input_response_data = self.format_source_code_input(response_dict=self.translated_responses)
        score_dict = evaluate_instruction_following(
            origin_dataset=input_data,
            response_dict=input_response_data,
        )
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
