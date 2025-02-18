import asyncio
import argparse
import yaml
from ..eval.benchmarks.en.mmlu import MMLUOQEvaluator, MMLUProbEvaluator
from ..eval.benchmarks.en.arc_challenge import ArcChallengeOQEvaluator, ArcChallengeProbEvaluator
from ..eval.benchmarks.en.gpqa import GPQAEvaluator
from ..eval.benchmarks.en.gsm8k import GSM8KEvaluator
from ..eval.benchmarks.en.truthfulqa import TruthfulQAEvaluator
from ..eval.benchmarks.en.big_bench_hard.mix_eval import BigBenchHardEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation using a config file.")
    parser.add_argument(
        "-t", "--task", 
        type=str, 
        choices=["mmlu", "arc_challenge", "gpqa", "gsm8k", "truthfulqa", "big_bench_hard"],
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
    return parser.parse_args()

async def eval():
    args = parse_args()

    # Load configuration from the provided YAML file
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Map task to evaluator classes
    task_evaluators = {
        "mmlu": {"mcqa-prob": MMLUProbEvaluator, "mcqa-oq": MMLUOQEvaluator},
        "arc_challenge": {"mcqa-prob": ArcChallengeProbEvaluator, "mcqa-oq": ArcChallengeOQEvaluator},
        "gpqa": {"opqa": GPQAEvaluator},
        "gsm8k": {"opqa": GSM8KEvaluator},
        "truthfulqa": {"mix": TruthfulQAEvaluator},
        "big_bench_hard": {"mix": BigBenchHardEvaluator},
    }

    # Select the appropriate evaluator
    if args.task not in task_evaluators or args.mode not in task_evaluators[args.task]:
        raise ValueError(f"Task '{args.task}' with mode '{args.mode}' is not supported.")

    evaluator_class = task_evaluators[args.task][args.mode]
    evaluator = evaluator_class(args=config)
    # Run the evaluation
    if args.mode == "mcqa-prob":
        evaluator.eval() # def eval():...
    else:
        await evaluator.eval() # async def eval():...

def run_eval():
    asyncio.run(eval())
    
    
if __name__ == "__main__":
    asyncio.run(eval())
