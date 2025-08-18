from argparse import Namespace
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
from ....data.data_utils import Role
from ..evaluator import Evaluator
from ....extras.logging import get_logger
logger = get_logger(__name__)

class MultiTurnEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args=args)
    
    @staticmethod
    def retrieve_responses(inference_results: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Retrieve responses from inference results."""
        responses = {subj: [] for subj in inference_results.keys()}
        for subj in inference_results.keys():
            responses[subj] = [[msg['content'] for msg in message if msg['role'] == Role.ASSISTANT.value] for message in inference_results[subj]]
        return responses
    
    @staticmethod
    def recover_translated_inference_results(system_prompt: str, translated_questions: Dict[str, List[Any]], translated_responses: Dict[str, List[Any]]):
        """Recover translated inference results."""
        record = {subj: [] for subj in translated_questions.keys()}
        for subj in translated_questions.keys():
            for question_list, response_list in zip(translated_questions[subj], translated_responses[subj]):
                message = [] if system_prompt is None else [{"role": Role.SYSTEM.value, "content": system_prompt}]
                for question, response in zip(question_list, response_list):
                    message.append({
                        "role": Role.USER.value,
                        "content": question
                    })
                    message.append({
                        "role": Role.ASSISTANT.value,
                        "content": response
                    })
                record[subj].append(message)
        return record
    
    async def multi_turn_generate(
        self,
        model: str,
        system_prompt: Optional[str],
        questions: List[str],
        idx: int, 
        generating_args: Namespace,
    ) -> Tuple[List[dict], int]:
        messages = []
        for question in questions:
            messages.append({
                "role": Role.USER.value, 
                "content": question
            })
            messages.append({
                "role": Role.ASSISTANT.value, 
                "content": await self.client.generate(
                               model=model,
                               system_prompt=system_prompt,
                               example=messages,
                               generating_args=generating_args,
                           )
            })
        return messages, idx
    
    async def multi_turn_inference(
        self,
        server_process: asyncio.subprocess.Process,
        model_name: str,
        data: Dict[str, List[Any]],
        prompt_key: str,
        output_path: str,
        progress_desc: str,
    ) -> Dict[str, List[Any]]:
        """Process subjects using the specified client and data with concurrency control."""
        results = {subj: [] for subj in self.categories.keys()}
        total_progress_bar = tqdm(self.categories.keys(), desc=progress_desc)

        # Define maximum concurrency
        max_concurrency = getattr(self.model_args, "vllm_max_concurrency", 100)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single_item(idx: int, questions: List[str], subject: str, progress_bar: tqdm):
            """Processes a single item with semaphore-based concurrency control."""
            async with semaphore:
                try:
                    result, origin_idx = await self.multi_turn_generate(
                        model=model_name,
                        system_prompt=getattr(self.eval_args, prompt_key),
                        questions=questions,
                        idx=idx,
                        generating_args=self.generating_args,
                    )
                    progress_bar.update(1)
                    return origin_idx, result
                except Exception as e:
                    progress_bar.update(1)
                    print(f"Error processing item {idx} in subject {subject}: {e}")
                    # show exact error message trace back
                    import traceback
                    traceback.print_exc()
                    return idx, None

        try:
            for subject in self.categories.keys():
                subject_results = [None] * len(data[subject])

                with tqdm(
                    total=len(data[subject]),
                    desc=self.categories[subject]["name"],
                    dynamic_ncols=True,
                ) as subject_progress_bar:

                    # Create tasks for all items under a subject
                    tasks = [
                        asyncio.create_task(process_single_item(idx, questions, subject, subject_progress_bar))
                        for idx, questions in enumerate(data[subject])
                    ]

                    # Collect results as tasks complete
                    for task in asyncio.as_completed(tasks):
                        origin_idx, result = await task
                        if result is not None:
                            subject_results[origin_idx] = result

                results[subject] = subject_results
                total_progress_bar.update(1)

        finally:
            # Ensure cleanup and save results
            await self.terminate_server(process=server_process)
            self.client = None
            self.save_data(data=results, output_path=os.path.join(self.save_folder, output_path))
            total_progress_bar.close()

        return results
    
    async def same_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        # general evaluation pipeline
        # need: inference model, checker model
        """Perform evaluation using inference and checker models with a progress bar."""
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Data path created: {self.save_folder}")
        ######################################### inference #########################################
        _, inference_questions = self.load_data(mode="inference", choices=choices)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=inference_questions, output_path=os.path.join(self.save_folder, "inference_questions.json"))
        
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

        self.inference_results = await self.multi_turn_inference(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.inference_model_name_or_path,
            data=inference_questions,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )

        print("Inference complete.")
        ########################################### check ###########################################
        checked_answers, checked_prompts = self.load_data(mode="check", choices=choices, histories=self.inference_results)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=checked_prompts, output_path=os.path.join(self.save_folder, "checked_prompts.json"))
            self.save_data(data=checked_answers, output_path=os.path.join(self.save_folder, "checked_answers.json"))
        
        if self.check_mode == "local":
            checker_process = await self.server.setup_server(
                model_path=self.model_args.checker_model_name_or_path,
                model_name=self.checker_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            print("Local vLLM server setup complete.")
        else:
            checker_process = None
            print("Using OpenAI API for checking.")

        self.set_client(mode="check")
        print("Client setup complete.")

        check_results = await self.process_subjects(
            server_process=checker_process,
            model_name=self.checker_model_name if self.check_mode == "local" else self.model_args.checker_model_name_or_path,
            data=checked_prompts,
            prompt_key="criteria_system_prompt",
            output_path="check_results.json",
            progress_desc="Check Progress",
        )

        print("Check complete.")
        ####################################### compute score #######################################
        score_dict = self.comput_score(checked_answers=checked_answers, check_results=check_results, subjects=subjects)
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
    
    async def diff_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        # specific evaluation pipeline
        # need: trranslator model, inference model, checker model
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        logger.info(f"Data path created: {self.save_folder}")
        ######################################## translation ########################################
        logger.info("============ Start question translation process. ============")
        _, inference_questions = self.load_data(mode="inference", choices=choices)
        _, ques_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=False)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=inference_questions, output_path=os.path.join(self.save_folder, "inference_questions.json"))
            self.save_data(data=ques_trans_prompts, output_path=os.path.join(self.save_folder, "ques_trans_prompts.json"))
        
        if self.translation_mode == "local":
            translation_process = await self.server.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            logger.info("Local vLLM server setup complete.")
        else:
            translation_process = None
            logger.info("Using OpenAI API for inference.")

        self.set_client(mode="translation")
        logger.info("Client setup complete.")
        
        self.translated_questions = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=ques_trans_prompts,
            output_path="translated_question_record.json",
            progress_desc="Trans Question Progress",
        )

        logger.info("Question translated complete.")
        
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
            logger.info("Local vLLM server setup complete.")
        else:
            inference_process = None
            logger.info("Using OpenAI API for inference.")

        self.set_client(mode="inference")
        logger.info("Client setup complete.")

        self.inference_results = await self.multi_turn_inference(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.inference_model_name_or_path,
            data=self.translated_questions,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )

        logger.info("Inference complete.")
        ######################################## translation ########################################
        logger.info("============ Start response translation process. ============")
        self.retrieved_responses = self.retrieve_responses(inference_results=self.inference_results)
        _, resp_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=True)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.retrieved_responses, output_path=os.path.join(self.save_folder, "retrieved_responses.json"))
            self.save_data(data=resp_trans_prompts, output_path=os.path.join(self.save_folder, "resp_trans_prompts.json"))
            
        if self.translation_mode == "local":
            translation_process = await self.server.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            logger.info("Local vLLM server setup complete.")
        else:
            translation_process = None
            logger.info("Using OpenAI API for inference.")

        self.set_client(mode="translation")
        logger.info("Client setup complete.")
        
        self.translated_responses = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=resp_trans_prompts,
            output_path="translated_response_record.json",
            progress_desc="Trans Response Progress",
        )
        # recover translated inference results
        translated_inference_results = self.recover_translated_inference_results(
            system_prompt=getattr(self.eval_args, "system_prompt"),
            translated_questions=self.translated_questions,
            translated_responses=self.translated_responses,
        )
        if getattr(self.eval_args, "record_all", False):
            self.save_data(data=translated_inference_results, output_path=os.path.join(self.save_folder, "translated_inference_results.json"))
        logger.info("Question translated complete.")
        ########################################### check ###########################################
        logger.info("============ Start checking process. ============")
        checked_answers, checked_prompts = self.load_data(mode="check", choices=choices, check_source="translated", histories=translated_inference_results)
        if getattr(self.eval_args, "record_all", False):
            self.save_data(data=checked_prompts, output_path=os.path.join(self.save_folder, "checked_prompts.json"))
            self.save_data(data=checked_answers, output_path=os.path.join(self.save_folder, "checked_answers.json"))
        
        if self.check_mode == "local":
            checker_process = await self.server.setup_server(
                model_path=self.model_args.checker_model_name_or_path,
                model_name=self.checker_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
                dtype=getattr(self.model_args, "dtype", "bfloat16"),
            )
            logger.info("Local vLLM server setup complete.")
        else:
            checker_process = None
            logger.info("Using OpenAI API for checking.")

        self.set_client(mode="check")
        logger.info("Client setup complete.")

        check_results = await self.process_subjects(
            server_process=checker_process,
            model_name=self.checker_model_name if self.check_mode == "local" else self.model_args.checker_model_name_or_path,
            data=checked_prompts,
            prompt_key="criteria_system_prompt",
            output_path="check_results.json",
            progress_desc="Check Progress",
        )

        logger.info("Check complete.")
        ####################################### compute score #######################################
        logger.info("============ Computing Score ============")
        score_dict = self.comput_score(checked_answers=checked_answers, check_results=check_results, subjects=subjects)
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))