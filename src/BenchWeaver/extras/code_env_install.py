import argparse
import os
import subprocess
import shutil
from pathlib import Path
from .logging import get_logger
from .constants import PROJECT_BASE_PATH


logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--env", 
        type=str, 
        default="ubuntu", 
        choices=["ubuntu", "amazon_linux"], 
        help="Target environment setup"
        )
    return parser.parse_args()

def install_code_env():
    """
    Set up the mxeval environment in PROJECT_BASE_PATH for a given OS environment.
    """
    args = parse_args()
    original_dir = Path.cwd()
    repo_url = "https://github.com/amazon-science/mxeval.git"
    clone_dir = Path(PROJECT_BASE_PATH) / "mxeval"

    try:
        # Clone the repository
        logger.info(f"Cloning mxeval repository from {repo_url} to {clone_dir}")
        # remove the existing directory if it exists
        if clone_dir.exists():
            logger.info(f"Removing existing directory: {clone_dir}")
            shutil.rmtree(clone_dir)
        subprocess.run(["git", "clone", repo_url, str(clone_dir)], check=True)

        # Install the package in editable mode
        logger.info("Installing mxeval in editable mode")
        subprocess.run(["pwd"], check=True)
        subprocess.run(["pip", "install", "-e", "mxeval"], check=True)

        # Change to the mxeval directory
        logger.info(f"Changing directory to {clone_dir}")
        os.chdir(clone_dir)
        
        # Run the appropriate environment setup script
        if args.env == "amazon_linux":
            setup_script = "language_setup/amazon_linux_ami.sh"
        elif args.env == "ubuntu":
            setup_script = "language_setup/ubuntu.sh"
        else:
            raise ValueError(f"Unsupported environment: {args.env}")

        logger.info(f"Running setup script: {setup_script}")
        subprocess.run(["bash", setup_script], check=True)

    finally:
        # Change back to the original directory
        logger.info(f"Changing back to the original directory: {original_dir}")
        os.chdir(original_dir)