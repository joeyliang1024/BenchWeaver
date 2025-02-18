import os
from dotenv import load_dotenv
from .constants import PROJECT_BASE_PATH

def load_env_variables(env_file_path: str = "env/tokens.env"):
    env_path = os.path.join(PROJECT_BASE_PATH, env_file_path)
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
    else:
        print(f"Environment file not found at {env_path}.")
        print("Please make sure to create a `.env` file with the required environment variables.")
        print("Exiting...")
        exit(1)