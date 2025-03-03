import os
import random
import subprocess
import sys
from enum import Enum, unique
import logging
from ..extras.print_env import VERSION, print_env
from .eval_score import run_eval
# this is a temp cli usage
USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   bench-weaver-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   bench-weaver-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   bench-weaver-cli eval -h: evaluate models                        |\n"
    + "|   bench-weaver-cli export -h: merge LoRA adapters and export model |\n"
    + "|   bench-weaver-cli train -h: train models                          |\n"
    + "|   bench-weaver-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   bench-weaver-cli webui: launch LlamaBoard                        |\n"
    + "|   bench-weaver-cli version: show version info                      |\n"
    + "|   bench-weaver-cli env: show dependency info                       |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 62
    + "\n"
    + f"| Welcome to BenchWeaver, version {VERSION}"
    + " " * (27 - len(VERSION))
    + "|\n|"
    + " " * 60
    + "|\n"
    + "| Project page: https://github.com/joeyliang1024/BenchWeaver |\n"
    + "-" * 62
)

logger = logging.getLogger(__name__)

@unique
class Command(str, Enum):
    ENV = "env"
    VER = "version"
    HELP = "help"
    WEBUI = "webui"
    EVAL = "eval"
    OTH_LANG_EVAL = "ol_eval"
    
def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP

    if command == Command.ENV:
        print_env()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    elif command == Command.WEBUI:
        # not implemented yet
        pass
    elif command == Command.EVAL:
        run_eval()
    elif command == Command.OTH_LANG_EVAL:
        # not implemented yet
        pass
    else:
        raise NotImplementedError(f"Unknown command: {command}.")

if __name__ == "__main__":
    main()