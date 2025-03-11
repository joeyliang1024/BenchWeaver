import asyncio
import argparse
import yaml
from ..eval.evaluator.evaluator import Evaluator
from ..eval.benchmarks.configs import BENCHMARK_CONFIG

async def eval():
    args = parse_args()

    # Load configuration from the provided YAML file
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Select the appropriate evaluator
    if args.task not in BENCHMARK_CONFIG.keys() or args.mode not in BENCHMARK_CONFIG[args.task]['evaluators'].keys():
        raise ValueError(f"Task '{args.task}' with mode '{args.mode}' is not supported.")

    evaluator_class = BENCHMARK_CONFIG[args.task]['evaluators'][args.mode]
    evaluator: Evaluator = evaluator_class(args=config)
    # Run the evaluation
    if args.mode == "mcqa-prob":
        evaluator.eval()
    else:
        if args.pipeline == "same":
            # same language evaluation
            await evaluator.same_lang_eval(
                choices=BENCHMARK_CONFIG[args.task]['mcqa_choices'],
                subjects=BENCHMARK_CONFIG[args.task]['display_scores'],
            ) 
        else:
            # different language evaluation
            await evaluator.diff_lang_eval(
                choices=BENCHMARK_CONFIG[args.task]['mcqa_choices'],
                subjects=BENCHMARK_CONFIG[args.task]['display_scores'],
            )
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation using a config file.")
    parser.add_argument(
        "-t", "--task", 
        type=str, 
        choices=list(BENCHMARK_CONFIG.keys()),
        required=True, 
        help="Evaluation task to run."
    )
    parser.add_argument(
        "-m", "--mode", 
        type=str, 
        choices=["mcqa-prob", "mcqa-oq", "opqa", "mix"], 
        required=True, 
        help="Evaluation mode."
    )
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, 
        help="Path to the config.yaml file."
    )
    parser.add_argument(
        "-p", "--pipeline",
        type=str,
        choices=["same", "diff"],
        required=True,
        help="Indicate whether to run the same language or different language evaluation."
    )
    return parser.parse_args()


def run_eval():
    """
    excute function for cli
    """
    asyncio.run(eval())
    
  
if __name__ == "__main__":
    asyncio.run(eval())
